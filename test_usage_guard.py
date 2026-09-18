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

from model_router import usage_guard
from model_router.usage_guard import GuardOutcome, Reading

ACCOUNTS = ("anthropic", "openai-codex")
STEP = {"anthropic": ("opus5", "sonnet5"), "openai-codex": ("sol", "terra")}


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
        with tempfile.TemporaryDirectory() as directory, self._fetchers(anthropic=_reading(61)):
            usage_guard.read("anthropic", _cfg(), now=1000.0)
            self.assertEqual(list(Path(directory).iterdir()), [])

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


if __name__ == "__main__":
    unittest.main()
