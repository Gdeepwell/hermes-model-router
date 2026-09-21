"""One usage guard for every delegation account.

Both accounts run through the same suite: what differs is only how a reading is
fetched. Anthropic's /api/oauth/usage reports utilization as a percentage (live
2026-09-18: 5.0 / 13.0), read raw because Hermes scales values <= 1 by 100.
Codex comes through Hermes's fetch_account_usage, whose Weekly/Session windows
are already percentages.
"""

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import model_router as model_router_module
from model_router import usage_guard
from model_router.usage_guard import GuardOutcome, Reading

ACCOUNTS = ("anthropic", "openai-codex")
STEP = {"anthropic": ("opus5", "sonnet5"), "openai-codex": ("sol", "terra")}


class _HttpError(Exception):
    """An HTTP failure shaped like httpx's: the guard reads .response.status_code."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


def _unauthorized(status):
    return _HttpError(status)


def _cfg(state_path=""):
    return {"usage_guard": {
        "cache_seconds": 300, "state_path": state_path,
        "accounts": {
            "anthropic": {"soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
            "openai-codex": {"soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"}},
        },
    }}


def _reading(weekly, session=10.0, fetched_at=None):
    return Reading(weekly, session, None, None, time.time() if fetched_at is None else fetched_at)


class ConfigTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual([usage_guard.account_label(a) for a in ("openai-codex", "anthropic", "qwen-token", "x")],
                         ["Codex", "Claude", "Qwen", "x"])

    def test_an_unconfigured_account_is_not_guarded(self):
        self.assertFalse(usage_guard.guarded("anthropic", {}))
        self.assertIsNone(usage_guard.account_limits("qwen-token", _cfg()))
        self.assertEqual(usage_guard.state("anthropic", {}, _reading(99)), "unknown")
        self.assertEqual(usage_guard.apply("anthropic", "opus5", {}, _reading(99)), GuardOutcome("opus5"))

    def test_fetchers_exist_for_both_accounts_only(self):
        self.assertTrue(usage_guard.has_fetcher("anthropic"))
        self.assertTrue(usage_guard.has_fetcher("openai-codex"))
        self.assertFalse(usage_guard.has_fetcher("qwen-token"))


class GuardRuleTests(unittest.TestCase):
    def test_every_account_follows_the_same_rules(self):
        for account in ACCOUNTS:
            heavy, lighter = STEP[account]
            with self.subTest(account=account):
                self.assertEqual(usage_guard.apply(account, heavy, _cfg(), _reading(50)).tier, heavy)
                soft = usage_guard.apply(account, heavy, _cfg(), _reading(75))
                self.assertEqual(soft.tier, lighter)
                self.assertEqual(soft.adjusted, f"{heavy}→{lighter} (weekly usage 75%)")
                self.assertEqual(usage_guard.apply(account, lighter, _cfg(), _reading(75)).tier, lighter)
                label = usage_guard.account_label(account)
                self.assertIn(f"{label} delegation closed: weekly usage 95%",
                              usage_guard.apply(account, lighter, _cfg(), _reading(95)).refused)
                self.assertIn("5-hour session usage 92%",
                              usage_guard.apply(account, lighter, _cfg(), _reading(40, 92)).refused)
                self.assertEqual(usage_guard.apply(account, heavy, _cfg(), None), GuardOutcome(heavy))

    def test_state(self):
        for account in ACCOUNTS:
            with self.subTest(account=account):
                self.assertEqual(usage_guard.state(account, _cfg(), None), "unknown")
                self.assertEqual(usage_guard.state(account, _cfg(), _reading(50)), "open")
                self.assertEqual(usage_guard.state(account, _cfg(), _reading(75)), "soft")
                self.assertEqual(usage_guard.state(account, _cfg(), _reading(90)), "closed")

    def test_open_usage_is_reported_and_an_unlisted_tier_is_unaffected(self):
        """An "open" reading carries the usage string, and a tier absent from
        ``step_down`` (haiku on anthropic, luna on openai-codex) is never touched,
        even past the soft limit."""
        unlisted = {"anthropic": "haiku", "openai-codex": "luna"}
        for account in ACCOUNTS:
            with self.subTest(account=account):
                heavy, _lighter = STEP[account]
                self.assertEqual(usage_guard.apply(account, heavy, _cfg(), _reading(50)).usage, "50%")
                tier = unlisted[account]
                outcome = usage_guard.apply(account, tier, _cfg(), _reading(75))
                self.assertEqual(outcome.tier, tier)
                self.assertEqual(outcome.adjusted, "")


class CacheTests(unittest.TestCase):
    def setUp(self):
        usage_guard._reset_cache()
        self.addCleanup(usage_guard._reset_cache)

    def _fetchers(self, **readings):
        return patch.dict(usage_guard.FETCHERS, {a: MagicMock(return_value=r) for a, r in readings.items()})

    def test_one_fetch_per_account_per_period(self):
        with self._fetchers(**{"anthropic": _reading(40), "openai-codex": _reading(20)}):
            for account in ACCOUNTS:
                usage_guard.read(account, _cfg(), now=1000.0)
                usage_guard.read(account, _cfg(), now=1010.0)
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 1)
            self.assertEqual(usage_guard.FETCHERS["openai-codex"].call_count, 1)
            usage_guard.read("anthropic", _cfg(), now=1400.0)
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 2)

    def test_a_failed_fetch_is_not_retried_within_the_period(self):
        with self._fetchers(anthropic=None):
            self.assertIsNone(usage_guard.read("anthropic", _cfg(), now=1000.0))
            self.assertIsNone(usage_guard.read("anthropic", _cfg(), now=1010.0))
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 1)

    def test_an_account_without_a_fetcher_reads_nothing(self):
        self.assertIsNone(usage_guard.read("qwen-token", _cfg(), now=1000.0))

    def test_no_state_path_means_no_file(self):
        cfg = _cfg()
        self.assertIsNone(usage_guard._state_path(cfg))
        with self._fetchers(anthropic=_reading(61)), \
             patch.object(usage_guard.os, "replace") as replace, \
             patch.object(Path, "write_text") as write_text:
            reading = usage_guard.read("anthropic", cfg, now=1000.0)
        self.assertEqual(reading.weekly, 61)
        replace.assert_not_called()
        write_text.assert_not_called()

    def test_a_configured_state_file_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            cfg = _cfg(str(path))
            with self._fetchers(anthropic=_reading(61)):
                usage_guard.read("anthropic", cfg, now=1000.0)
            usage_guard._reset_cache()  # a new process
            fetch = MagicMock()
            with patch.dict(usage_guard.FETCHERS, {"anthropic": fetch}):
                reading = usage_guard.read("anthropic", cfg, now=1010.0)
        self.assertEqual(reading.weekly, 61)
        fetch.assert_not_called()

    def test_peek_returns_a_fresh_reading_without_refreshing(self):
        refresh = MagicMock()
        with self._fetchers(anthropic=_reading(40)):
            usage_guard.read("anthropic", _cfg())
        with patch.object(usage_guard, "_start_refresh", refresh):
            self.assertEqual(usage_guard.peek("anthropic", _cfg()).weekly, 40)
        refresh.assert_not_called()

    def test_one_state_file_holds_every_account_and_is_read_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            with self._fetchers(**{"anthropic": _reading(13), "openai-codex": _reading(41)}):
                usage_guard.read("anthropic", _cfg(str(path)), now=1000.0)
                usage_guard.read("openai-codex", _cfg(str(path)), now=1000.0)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((stored["anthropic"]["weekly"], stored["openai-codex"]["weekly"]), (13, 41))
            usage_guard._reset_cache()  # another process, e.g. the dashboard
            self.assertEqual(usage_guard.cached("openai-codex", _cfg(str(path))).weekly, 41)
            stored["openai-codex"]["weekly"] = 55  # Hermes wrote a newer reading
            path.write_text(json.dumps(stored), encoding="utf-8")
            self.assertEqual(usage_guard.cached("openai-codex", _cfg(str(path))).weekly, 55)

    def test_cached_never_fetches_or_refreshes(self):
        refresh = MagicMock()
        with self._fetchers(anthropic=_reading(1)), patch.object(usage_guard, "_start_refresh", refresh):
            self.assertIsNone(usage_guard.cached("anthropic", _cfg()))
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 0)
        refresh.assert_not_called()

    def test_peek_never_fetches_and_starts_one_refresh_per_account(self):
        refresh = MagicMock()
        with self._fetchers(anthropic=_reading(1)), patch.object(usage_guard, "_start_refresh", refresh):
            self.assertIsNone(usage_guard.peek("anthropic", _cfg()))
            usage_guard.peek("anthropic", _cfg())
            usage_guard.peek("openai-codex", _cfg())
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 0)
        self.assertEqual([c.args[0] for c in refresh.call_args_list], ["anthropic", "openai-codex"])

    def test_peek_resets_the_refreshing_flag_when_starting_the_refresh_thread_raises(self):
        """M13: if _start_refresh itself raises (e.g. the thread fails to
        start), the account's `refreshing` flag must not be left stuck at
        True forever -- that would permanently block every future refresh."""
        with self._fetchers(anthropic=_reading(1)), \
             patch.object(usage_guard, "_start_refresh", side_effect=RuntimeError("boom")):
            self.assertIsNone(usage_guard.peek("anthropic", _cfg()))
            self.assertFalse(usage_guard._slot("anthropic")["refreshing"])

        # And a later peek (with a working _start_refresh) can refresh again.
        refresh = MagicMock()
        with self._fetchers(anthropic=_reading(1)), patch.object(usage_guard, "_start_refresh", refresh):
            usage_guard.peek("anthropic", _cfg())
        refresh.assert_called_once_with("anthropic", _cfg())

    def test_concurrent_persist_does_not_lose_accounts(self):
        """Verify that concurrent _persist calls on different accounts don't lose data."""
        import threading
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            cfg = _cfg(str(path))

            # Patch _load_file to sleep briefly, making the race condition reproducible.
            # Both threads will read the file before either writes, so without the lock
            # the second write would overwrite the first account's data.
            original_load_file = usage_guard._load_file
            def slow_load_file(config):
                result = original_load_file(config)
                time.sleep(0.005)
                return result

            # Run the test multiple times to increase likelihood of hitting the race
            for round_num in range(50):
                # Reset the file for each round
                if path.exists():
                    path.unlink()

                # Create readings with different values each round
                reading_anthropic = Reading(10.0 + round_num, 5.0, None, None, time.time())
                reading_codex = Reading(20.0 + round_num, 15.0, None, None, time.time())

                with patch.object(usage_guard, "_load_file", side_effect=slow_load_file):
                    # Use threads to simulate concurrent _persist calls
                    errors = []
                    def persist_anthropic():
                        try:
                            usage_guard._persist(cfg, "anthropic", reading_anthropic)
                        except Exception as e:
                            errors.append(e)

                    def persist_codex():
                        try:
                            usage_guard._persist(cfg, "openai-codex", reading_codex)
                        except Exception as e:
                            errors.append(e)

                    t1 = threading.Thread(target=persist_anthropic)
                    t2 = threading.Thread(target=persist_codex)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()

                # Assert no exceptions occurred
                self.assertFalse(errors, f"Errors in round {round_num}: {errors}")

                # Assert both accounts are in the final file
                stored = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("anthropic", stored, f"anthropic missing in round {round_num}")
                self.assertIn("openai-codex", stored, f"openai-codex missing in round {round_num}")
                self.assertEqual(stored["anthropic"]["weekly"], 10.0 + round_num)
                self.assertEqual(stored["openai-codex"]["weekly"], 20.0 + round_num)


def _hermes_importable():
    try:
        import agent.account_usage  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_hermes_importable(), "Hermes is not importable in this interpreter")
class FetcherTests(unittest.TestCase):
    def test_anthropic_is_read_raw_as_percentages_with_resets(self):
        payload = {"five_hour": {"utilization": 1.0, "resets_at": "2026-09-18T17:00:00+00:00"},
                   "seven_day": {"utilization": 0.8, "resets_at": "2026-09-24T16:00:00+00:00"}}
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok"), \
             patch("agent.account_usage._get_json", return_value=payload) as get_json:
            reading = usage_guard.FETCHERS["anthropic"]()
        self.assertEqual((reading.weekly, reading.session), (0.8, 1.0))
        self.assertEqual(reading.weekly_resets_at, "2026-09-24T16:00:00+00:00")
        self.assertEqual(reading.session_resets_at, "2026-09-18T17:00:00+00:00")
        self.assertEqual(get_json.call_args.args[0], "https://api.anthropic.com/api/oauth/usage")

    def test_anthropic_without_a_token_reads_nothing(self):
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=""), \
             patch("agent.account_usage._get_json") as get_json:
            self.assertIsNone(usage_guard.FETCHERS["anthropic"]())
        get_json.assert_not_called()

    def test_anthropic_with_no_windows_reads_nothing(self):
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok"), \
             patch("agent.account_usage._get_json", return_value={}):
            self.assertIsNone(usage_guard.FETCHERS["anthropic"]())

    def test_the_usage_endpoint_is_asked_with_the_oauth_token(self):
        payload = {"five_hour": {"utilization": 5.0}, "seven_day": {"utilization": 13.0}}
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok"), \
             patch("agent.account_usage._get_json", return_value=payload) as get_json:
            usage_guard.FETCHERS["anthropic"]()
        headers = get_json.call_args.args[1]
        self.assertEqual(headers["Authorization"], "Bearer tok")

    def test_a_rejected_token_falls_back_to_the_refreshing_resolver(self):
        """A stale pool row must not make the account permanently unreadable.

        resolve_anthropic_token reads the pool with refresh=False, so a
        manual:hermes_pkce row can shadow valid Claude Code credentials with an
        expired token. Observed on this host 2026-09-21: a 401 while the Claude
        Code credentials were good for another six hours.
        """
        payload = {"five_hour": {"utilization": 16.0}, "seven_day": {"utilization": 2.0}}
        answers = {"stale": _unauthorized(401), "fresh": payload}

        def answer(url, headers, **kwargs):
            result = answers[headers["Authorization"].removeprefix("Bearer ")]
            if isinstance(result, Exception):
                raise result
            return result

        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="stale"), \
             patch("agent.anthropic_credentials._resolve_claude_code_token_from_credentials",
                   return_value="fresh", create=True), \
             patch("agent.account_usage._get_json", side_effect=answer) as get_json:
            reading = usage_guard.FETCHERS["anthropic"]()
        self.assertIsNotNone(reading)
        self.assertEqual((reading.weekly, reading.session), (2.0, 16.0))
        self.assertEqual(get_json.call_count, 2)

    def test_the_first_token_is_used_when_it_is_accepted(self):
        payload = {"five_hour": {"utilization": 5.0}, "seven_day": {"utilization": 13.0}}
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok"), \
             patch("agent.anthropic_credentials._resolve_claude_code_token_from_credentials",
                   return_value="other", create=True) as fallback, \
             patch("agent.account_usage._get_json", return_value=payload) as get_json:
            usage_guard.FETCHERS["anthropic"]()
        self.assertEqual(get_json.call_count, 1)
        fallback.assert_not_called()

    def test_one_identity_is_never_tried_twice(self):
        """Both resolvers commonly return the same token; that is one attempt, not two."""
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="same"), \
             patch("agent.anthropic_credentials._resolve_claude_code_token_from_credentials",
                   return_value="same", create=True), \
             patch("agent.account_usage._get_json", side_effect=_unauthorized(401)) as get_json:
            self.assertIsNone(usage_guard.FETCHERS["anthropic"]())
        self.assertEqual(get_json.call_count, 1)

    def test_a_non_auth_failure_does_not_try_another_token(self):
        """A network or 5xx failure is not about identity, so it fails the read at once."""
        with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="stale"), \
             patch("agent.anthropic_credentials._resolve_claude_code_token_from_credentials",
                   return_value="fresh", create=True), \
             patch("agent.account_usage._get_json", side_effect=_unauthorized(503)) as get_json:
            self.assertIsNone(usage_guard.FETCHERS["anthropic"]())
        self.assertEqual(get_json.call_count, 1)

    def test_codex_uses_the_weekly_and_session_windows(self):
        reset = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)
        snapshot = SimpleNamespace(available=True, windows=(
            SimpleNamespace(label="Session", used_percent=12.0, reset_at=None),
            SimpleNamespace(label="Weekly", used_percent=41.0, reset_at=reset),
        ))
        with patch("agent.account_usage.fetch_account_usage", return_value=snapshot) as fetch:
            reading = usage_guard.FETCHERS["openai-codex"]()
        fetch.assert_called_once_with("openai-codex")
        self.assertEqual((reading.weekly, reading.session), (41.0, 12.0))
        self.assertEqual(reading.weekly_resets_at, reset.isoformat())
        self.assertIsNone(reading.session_resets_at)

    def test_codex_unavailable_reads_nothing(self):
        with patch("agent.account_usage.fetch_account_usage", return_value=SimpleNamespace(available=False, windows=())):
            self.assertIsNone(usage_guard.FETCHERS["openai-codex"]())


from model_router import (  # noqa: E402
    RouteDecision,
    _target_availability,
    _usage_step_down,
)

ROUTER_CFG = {
    "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
    "callable": {"luna": True, "spark": True, "terra": True, "sol": True, "opus5": True, "sonnet5": True},
    "tier_providers": {"luna": "openai-codex", "spark": "openai-codex", "terra": "openai-codex",
                       "sol": "openai-codex", "opus5": "anthropic", "sonnet5": "anthropic"},
    "effort": {}, **_cfg(),
}


def _peek(**weekly):
    return patch.object(usage_guard, "peek",
                        side_effect=lambda account, cfg: _reading(weekly[account]) if account in weekly else None)


class CodexStepDownTests(unittest.TestCase):
    def _sol(self, **extra):
        return RouteDecision("sol", "gpt-sol", "long work", "medium", **extra)

    def test_the_soft_limit_steps_sol_down_to_terra(self):
        with _peek(**{"openai-codex": 72.0}):
            decision = _usage_step_down(self._sol(kind="long"), ROUTER_CFG)
        self.assertEqual((decision.tier, decision.model), ("terra", "gpt-terra"))
        self.assertIn("usage soft limit: sol→terra (weekly 72%)", decision.reason)
        self.assertTrue(decision.reason.startswith("long work; "))
        self.assertEqual(decision.kind, "long")

    def test_the_hard_weekly_limit_also_steps_sol_down_to_terra(self):
        with _peek(**{"openai-codex": 95.0}):
            decision = _usage_step_down(self._sol(kind="long"), ROUTER_CFG)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("usage hard limit: sol→terra (weekly 95%)", decision.reason)
        self.assertEqual(decision.kind, "long")

    def test_the_hard_session_limit_also_steps_sol_down_to_terra(self):
        with patch.object(usage_guard, "peek",
                          side_effect=lambda account, cfg: _reading(75.0, session=95.0)
                          if account == "openai-codex" else None):
            decision = _usage_step_down(self._sol(kind="long"), ROUTER_CFG)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("usage hard limit: sol→terra", decision.reason)

    def test_the_hard_session_limit_reason_names_the_session_window_not_weekly(self):
        """M10: the hard limit can trigger on either window; the reason must
        say which one actually did, not always claim "weekly"."""
        with patch.object(usage_guard, "peek",
                          side_effect=lambda account, cfg: _reading(75.0, session=95.0)
                          if account == "openai-codex" else None):
            decision = _usage_step_down(self._sol(kind="long"), ROUTER_CFG)
        self.assertIn("usage hard limit: sol→terra (session 95%)", decision.reason)
        self.assertNotIn("weekly 75%", decision.reason)

    def test_the_hard_weekly_limit_reason_still_names_weekly(self):
        with _peek(**{"openai-codex": 95.0}):
            decision = _usage_step_down(self._sol(kind="long"), ROUTER_CFG)
        self.assertIn("usage hard limit: sol→terra (weekly 95%)", decision.reason)

    def test_nothing_changes_below_the_limit_when_unknown_or_unguarded(self):
        for peeked, cfg in (({"openai-codex": 50.0}, ROUTER_CFG), ({}, ROUTER_CFG),
                            ({"openai-codex": 99.0}, {k: v for k, v in ROUTER_CFG.items() if k != "usage_guard"})):
            with self.subTest(peeked=peeked), _peek(**peeked):
                self.assertEqual(_usage_step_down(self._sol(), cfg).tier, "sol")

    def test_a_mandatory_route_is_never_stepped_down(self):
        with _peek(**{"openai-codex": 72.0}):
            self.assertEqual(_usage_step_down(self._sol(mandatory=True), ROUTER_CFG).tier, "sol")

    def test_an_unavailable_target_keeps_the_tier_and_says_why(self):
        cfg = {**ROUTER_CFG, "callable": {**ROUTER_CFG["callable"], "terra": False}}
        with _peek(**{"openai-codex": 72.0}):
            decision = _usage_step_down(self._sol(), cfg)
        self.assertEqual(decision.tier, "sol")
        self.assertIn("usage soft limit: sol→terra skipped (terra unavailable)", decision.reason)

    def test_an_unavailable_target_keeps_the_tier_and_says_why_at_the_hard_limit(self):
        cfg = {**ROUTER_CFG, "callable": {**ROUTER_CFG["callable"], "terra": False}}
        with _peek(**{"openai-codex": 95.0}):
            decision = _usage_step_down(self._sol(), cfg)
        self.assertEqual(decision.tier, "sol")
        self.assertIn("usage hard limit: sol→terra skipped (terra unavailable)", decision.reason)

    def test_a_malformed_guard_fails_open(self):
        cfg = {**ROUTER_CFG, "usage_guard": {"cache_seconds": 300, "accounts": {
            "openai-codex": {"soft_percent": "seventy", "hard_percent": 90, "step_down": {"sol": "terra"}},
        }}}
        with _peek(**{"openai-codex": 72.0}):
            self.assertEqual(_usage_step_down(self._sol(), cfg).tier, "sol")

    def test_a_malformed_guard_logs_once_with_the_traceback(self):
        """M8: the fail-open except must not swallow the failure silently --
        it logs once through the router's own logger, with exc_info so the
        traceback is not lost."""
        cfg = {**ROUTER_CFG, "usage_guard": {"cache_seconds": 300, "accounts": {
            "openai-codex": {"soft_percent": "seventy", "hard_percent": 90, "step_down": {"sol": "terra"}},
        }}}
        with _peek(**{"openai-codex": 72.0}):
            with self.assertLogs("model_router", level="WARNING") as cm:
                decision = _usage_step_down(self._sol(), cfg)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(len(cm.records), 1)
        self.assertIsNotNone(cm.records[0].exc_info)


class AccountMarkTests(unittest.TestCase):
    def test_marks_name_the_account_state(self):
        with _peek(**{"openai-codex": 72.0, "anthropic": 95.0}):
            notes = _target_availability(["sol", "opus5", "luna"], ROUTER_CFG)
        self.assertEqual(notes["sol"], " [Codex soft limit]")
        self.assertEqual(notes["opus5"], " [Claude closed]")
        self.assertEqual(notes["luna"], " [Codex soft limit]")

    def test_no_guard_no_marks(self):
        cfg = {k: v for k, v in ROUTER_CFG.items() if k != "usage_guard"}
        with _peek(**{"openai-codex": 99.0}):
            self.assertEqual(_target_availability(["sol"], cfg), {"sol": ""})

    def test_a_cooling_tier_keeps_both_its_cooldown_and_its_account_mark(self):
        with _peek(**{"openai-codex": 72.0}), \
             patch("model_router._tier_cooldown_remaining", side_effect=lambda name, cfg: 90.0 if name == "sol" else 0.0):
            notes = _target_availability(["sol"], ROUTER_CFG)
        self.assertEqual(notes["sol"], " [unavailable for another 2 min] [Codex soft limit]")

    def test_a_malformed_guard_fails_open(self):
        cfg = {**ROUTER_CFG, "usage_guard": {"cache_seconds": 300, "accounts": {
            "openai-codex": {"soft_percent": "seventy", "hard_percent": 90, "step_down": {"sol": "terra"}},
        }}}
        with _peek(**{"openai-codex": 72.0}):
            self.assertEqual(_target_availability(["sol"], cfg), {"sol": ""})

    def test_account_states_logs_once_on_a_malformed_guard(self):
        cfg = {**ROUTER_CFG, "usage_guard": {"cache_seconds": 300, "accounts": {
            "openai-codex": {"soft_percent": "seventy", "hard_percent": 90, "step_down": {"sol": "terra"}},
        }}}
        with _peek(**{"openai-codex": 72.0}):
            with self.assertLogs("model_router", level="WARNING") as cm:
                states = model_router_module._account_states(cfg)
        self.assertEqual(states, {})
        self.assertEqual(len(cm.records), 1)
        self.assertIsNotNone(cm.records[0].exc_info)

    def test_account_mark_logs_once_when_it_fails(self):
        with patch.object(usage_guard, "account_label", side_effect=RuntimeError("boom")):
            with self.assertLogs("model_router", level="WARNING") as cm:
                mark = model_router_module._account_mark(
                    "sol", ROUTER_CFG, {"openai-codex": "soft"}
                )
        self.assertEqual(mark, "")
        self.assertEqual(len(cm.records), 1)
        self.assertIsNotNone(cm.records[0].exc_info)


if __name__ == "__main__":
    unittest.main()


class ForcedReadTests(unittest.TestCase):
    """A human pressing Refresh is not a background poll.

    ``read`` gates every fetch on ``cache_seconds`` (300s), so the dashboard's
    Refresh button was a silent no-op for five minutes at a time: it returned
    200 with the cached reading and the bars never moved. An explicit refresh
    has to bypass the TTL that exists to throttle *automatic* reads.
    """

    def setUp(self):
        usage_guard._reset_cache()
        self.addCleanup(usage_guard._reset_cache)

    def _fetchers(self, **readings):
        return patch.dict(usage_guard.FETCHERS, {a: MagicMock(side_effect=r) for a, r in readings.items()})

    def test_a_forced_read_fetches_inside_the_cache_window(self):
        with self._fetchers(anthropic=[_reading(40), _reading(55)]):
            first = usage_guard.read("anthropic", _cfg(), now=1000.0)
            self.assertEqual(first.weekly, 40)
            # Three seconds later -- far inside the 300s TTL.
            forced = usage_guard.read("anthropic", _cfg(), now=1003.0, force=True)
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 2)
            self.assertEqual(forced.weekly, 55)

    def test_an_unforced_read_still_honours_the_cache_window(self):
        with self._fetchers(anthropic=[_reading(40), _reading(55)]):
            usage_guard.read("anthropic", _cfg(), now=1000.0)
            usage_guard.read("anthropic", _cfg(), now=1003.0)
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 1)

    def test_a_forced_read_retries_after_a_recent_failure(self):
        """The failure backoff is throttling too; a human asking again overrides it."""
        with self._fetchers(anthropic=[None, _reading(55)]):
            self.assertIsNone(usage_guard.read("anthropic", _cfg(), now=1000.0))
            forced = usage_guard.read("anthropic", _cfg(), now=1003.0, force=True)
            self.assertEqual(usage_guard.FETCHERS["anthropic"].call_count, 2)
            self.assertEqual(forced.weekly, 55)

    def test_a_forced_read_that_fails_keeps_the_last_good_reading(self):
        with self._fetchers(anthropic=[_reading(40), None]):
            usage_guard.read("anthropic", _cfg(), now=1000.0)
            self.assertIsNone(usage_guard.read("anthropic", _cfg(), now=1003.0, force=True))
            # The bar should keep showing the last number it had, not go blank.
            self.assertEqual(usage_guard.cached("anthropic", _cfg()).weekly, 40)

    def test_an_account_without_a_fetcher_is_still_nothing_to_force(self):
        self.assertIsNone(usage_guard.read("qwen-token", _cfg(), now=1000.0, force=True))
