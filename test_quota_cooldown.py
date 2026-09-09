"""Quota cooldown duration and scope.

Observed 2026-09-09 11:31: a [terra] leaf died on `HTTP 429 The usage limit has
been reached`, whose body said `resets_in_seconds: 11029` (~3 hours). The router
had benched Terra for the configured 900s; that lapsed at 11:29, Terra was offered
again, and the leaf spent its three retries rediscovering the same wall. Sol had
been benched separately a moment earlier — it shares the account, so that call was
spent learning something already known.
"""

import tempfile
import unittest
from pathlib import Path

from model_router import (
    _account_siblings,
    _record_tier_failure,
    _reset_hint_seconds,
    _tier_cooldown_remaining,
)
from model_router.test_model_router import CALLABLE, MODELS


REAL_429 = (
    "HTTP 429: The usage limit has been reached {'type': 'usage_limit_reached', "
    "'message': 'The usage limit has been reached', 'plan_type': 'plus', "
    "'resets_at': 1788956281, 'eligible_promo': None, 'resets_in_seconds': 11029}"
)


def _cfg(path, **overrides):
    cfg = {
        "models": MODELS,
        # Mirrors a real config: the Claude tiers are switchable even though they
        # are delegation targets rather than routable models.
        "callable": {**CALLABLE, "opus5": True, "sonnet5": True, "qwen": True},
        "tier_providers": {"luna": "openai-codex", "spark": "openai-codex",
                           "terra": "openai-codex", "sol": "openai-codex",
                           "opus5": "anthropic", "sonnet5": "anthropic", "qwen": "qwen-token"},
        "cooldown": {"enabled": True, "path": str(path), "quota_seconds": 900,
                     "quota_max_seconds": 21600, "allowed_fails": 3,
                     "failure_window_seconds": 60, "failure_seconds": 60},
    }
    cfg.update(overrides)
    return cfg


class ResetHintTests(unittest.TestCase):
    def test_resets_in_seconds_is_read_from_the_body(self):
        self.assertEqual(_reset_hint_seconds(RuntimeError(REAL_429)), 11029.0)

    def test_resets_at_epoch_is_converted_to_a_remaining_duration(self):
        import time
        future = int(time.time()) + 1800
        hint = _reset_hint_seconds(RuntimeError(f"HTTP 429 {{'resets_at': {future}}}"))
        self.assertIsNotNone(hint)
        self.assertGreater(hint, 1700)
        self.assertLess(hint, 1900)

    def test_a_past_reset_is_not_a_hint(self):
        self.assertIsNone(_reset_hint_seconds(RuntimeError("HTTP 429 {'resets_at': 1000000000}")))

    def test_an_error_without_a_hint_yields_none(self):
        self.assertIsNone(_reset_hint_seconds(RuntimeError("HTTP 429: usage limit reached")))


class CooldownDurationTests(unittest.TestCase):
    def test_the_providers_own_reset_outranks_the_configured_guess(self):
        """900s against a three-hour reset is what produced the retry loop."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            _record_tier_failure("terra", cfg, quota=True, error=RuntimeError(REAL_429))
            self.assertGreater(_tier_cooldown_remaining("terra", cfg), 10000)

    def test_without_a_hint_the_configured_value_still_applies(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            _record_tier_failure("terra", cfg, quota=True,
                                 error=RuntimeError("HTTP 429: usage limit reached"))
            remaining = _tier_cooldown_remaining("terra", cfg)
            self.assertGreater(remaining, 800)
            self.assertLess(remaining, 1000)

    def test_an_absurd_hint_is_capped(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            _record_tier_failure("terra", cfg, quota=True,
                                 error=RuntimeError("HTTP 429 {'resets_in_seconds': 9999999}"))
            self.assertLessEqual(_tier_cooldown_remaining("terra", cfg), 21600)


class AccountScopeTests(unittest.TestCase):
    def test_siblings_are_grouped_by_account_not_by_routability(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            self.assertEqual(set(_account_siblings("terra", cfg)), {"luna", "spark", "sol"})
            # A delegation-only target still shares the subscription.
            self.assertEqual(_account_siblings("opus5", cfg), ("sonnet5",))
            self.assertEqual(_account_siblings("qwen", cfg), ())

    def test_a_usage_quota_benches_every_tier_on_that_account(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            _record_tier_failure("terra", cfg, quota=True, error=RuntimeError(REAL_429))
            for tier in ("luna", "spark", "sol"):
                self.assertGreater(_tier_cooldown_remaining(tier, cfg), 10000, tier)

    def test_another_account_is_left_alone(self):
        """Benching Anthropic because Codex ran out would defeat the whole point."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            _record_tier_failure("terra", cfg, quota=True, error=RuntimeError(REAL_429))
            for tier in ("opus5", "sonnet5", "qwen"):
                self.assertEqual(_tier_cooldown_remaining(tier, cfg), 0, tier)

    def test_a_longer_existing_cooldown_is_not_shortened(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            _record_tier_failure("luna", cfg, quota=True,
                                 error=RuntimeError("HTTP 429 {'resets_in_seconds': 20000}"))
            _record_tier_failure("terra", cfg, quota=True, error=RuntimeError(REAL_429))
            self.assertGreater(_tier_cooldown_remaining("luna", cfg), 19000)


if __name__ == "__main__":
    unittest.main()
