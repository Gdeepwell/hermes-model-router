"""The Claude wing: Claude workers next to the Codex workforce.

Hermes's delegate_task has one route per process, pinned to Codex on this host.
delegate_claude reaches Claude by calling the same delegate_task with a per-call
route pinned to the anthropic provider. These tests cover the wing without any
network or model call.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_router import claude_wing
from model_router.claude_wing import (
    TARGET_FOR_TIER,
    TIER_FOR_TARGET,
    registration_block,
    target_for_model,
    target_names,
    tier_model,
    wing_config,
)

WING = {
    "enabled": True,
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"},
    "default_tier": "sonnet",
    "usage_guard": {"soft_percent": 70, "hard_percent": 90, "cache_seconds": 300},
}


def _cfg(**overrides):
    cfg = {
        "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True,
                     "opus5": True, "sonnet5": True, "haiku": True},
        "tier_providers": {"opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic"},
        "fallbacks": {"opus5": "sol"},
        "peer_groups": {"heavy": ["terra", "opus5", "sonnet5"], "light": ["luna", "spark", "haiku"]},
        "default_model": "terra",
        "claude_wing": json.loads(json.dumps(WING)),
    }
    cfg.update(overrides)
    return cfg


class WingConfigTests(unittest.TestCase):
    def test_a_config_without_the_block_leaves_the_wing_off(self):
        """Configuring nothing must change nothing."""
        self.assertFalse(wing_config({})["enabled"])
        self.assertFalse(wing_config(None)["enabled"])

    def test_a_partial_guard_keeps_the_other_defaults(self):
        wing = wing_config({"claude_wing": {"enabled": True, "usage_guard": {"soft_percent": 60}}})
        self.assertEqual(wing["usage_guard"]["soft_percent"], 60)
        self.assertEqual(wing["usage_guard"]["hard_percent"], 90)
        self.assertEqual(wing["tiers"]["opus"], "claude-opus-5")

    def test_names_map_both_ways(self):
        self.assertEqual(TARGET_FOR_TIER, {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"})
        self.assertEqual(TIER_FOR_TARGET["opus5"], "opus")

    def test_tier_model(self):
        self.assertEqual(tier_model("haiku", _cfg()), "claude-haiku-4-5-20251001")
        self.assertEqual(tier_model("gpt", _cfg()), "")

    def test_target_names_skip_a_tier_without_a_model(self):
        cfg = _cfg()
        self.assertEqual(target_names(cfg), ("haiku", "opus5", "sonnet5"))
        cfg["claude_wing"]["tiers"]["haiku"] = ""
        self.assertEqual(target_names(cfg), ("opus5", "sonnet5"))

    def test_a_model_maps_back_to_its_target(self):
        self.assertEqual(target_for_model("claude-haiku-4-5-20251001", _cfg()), "haiku")
        self.assertIsNone(target_for_model("gpt-terra", _cfg()))
        self.assertIsNone(target_for_model("", _cfg()))


class RegistrationBlockTests(unittest.TestCase):
    def test_a_disabled_wing_does_not_register(self):
        cfg = _cfg()
        cfg["claude_wing"]["enabled"] = False
        self.assertIn("enabled", registration_block(cfg))

    def test_every_claude_target_switched_off_does_not_register(self):
        cfg = _cfg()
        for target in ("haiku", "sonnet5", "opus5"):
            cfg["callable"][target] = False
        self.assertIn("switched off", registration_block(cfg))

    def test_a_host_without_the_api_does_not_register(self):
        with patch.object(claude_wing, "host_check", return_value=(False, "delegate_task lacks credentials_cfg")):
            self.assertIn("credentials_cfg", registration_block(_cfg()))

    def test_an_enabled_wing_on_a_capable_host_registers(self):
        with patch.object(claude_wing, "host_check", return_value=(True, "")):
            self.assertEqual(registration_block(_cfg()), "")


from model_router.claude_wing import (  # noqa: E402
    GuardOutcome,
    UsageReading,
    apply_guard,
    peek_usage,
    read_usage,
    wing_state,
)


def _reading(weekly, session=10.0, fetched_at=None):
    return UsageReading(weekly, session, time.time() if fetched_at is None else fetched_at)


class GuardTests(unittest.TestCase):
    def test_below_the_soft_limit_everything_runs_as_asked(self):
        for tier in ("haiku", "sonnet", "opus"):
            with self.subTest(tier=tier):
                outcome = apply_guard(tier, _cfg(), _reading(50))
                self.assertEqual((outcome.tier, outcome.refused, outcome.adjusted), (tier, "", ""))
                self.assertEqual(outcome.usage, "50%")

    def test_the_soft_limit_lowers_opus_only(self):
        outcome = apply_guard("opus", _cfg(), _reading(75))
        self.assertEqual(outcome.tier, "sonnet")
        self.assertEqual(outcome.adjusted, "opus→sonnet (weekly usage 75%)")
        for tier in ("haiku", "sonnet"):
            with self.subTest(tier=tier):
                self.assertEqual(apply_guard(tier, _cfg(), _reading(75)).tier, tier)

    def test_the_hard_limit_closes_the_wing(self):
        outcome = apply_guard("haiku", _cfg(), _reading(95))
        self.assertIn("Claude wing closed: weekly usage 95%", outcome.refused)

    def test_a_full_session_window_also_closes_it(self):
        outcome = apply_guard("sonnet", _cfg(), _reading(40, session=92))
        self.assertIn("5-hour session usage 92%", outcome.refused)

    def test_no_reading_fails_open(self):
        """A missing reading must not close the wing; Anthropic's own quota error
        still stops a child, and the router records that as a cooldown."""
        self.assertEqual(apply_guard("opus", _cfg(), None), GuardOutcome("opus"))

    def test_wing_state(self):
        self.assertEqual(wing_state(_cfg(), None), "unknown")
        self.assertEqual(wing_state(_cfg(), _reading(50)), "open")
        self.assertEqual(wing_state(_cfg(), _reading(75)), "soft")
        self.assertEqual(wing_state(_cfg(), _reading(90)), "closed")


class UsageCacheTests(unittest.TestCase):
    def setUp(self):
        claude_wing._reset_usage_cache()
        self.addCleanup(claude_wing._reset_usage_cache)

    def test_one_fetch_per_cache_period(self):
        fetch = MagicMock(return_value=_reading(40))
        with patch.object(claude_wing, "_fetch_reading", fetch):
            read_usage(_cfg(), now=1000.0)
            read_usage(_cfg(), now=1010.0)
            self.assertEqual(fetch.call_count, 1)
            read_usage(_cfg(), now=1400.0)
            self.assertEqual(fetch.call_count, 2)

    def test_a_failed_reading_is_not_retried_within_the_period(self):
        fetch = MagicMock(return_value=None)
        with patch.object(claude_wing, "_fetch_reading", fetch):
            self.assertIsNone(read_usage(_cfg(), now=1000.0))
            self.assertIsNone(read_usage(_cfg(), now=1010.0))
        self.assertEqual(fetch.call_count, 1)

    def test_a_configured_state_file_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            cfg["claude_wing"]["usage_guard"]["state_path"] = str(Path(directory) / "usage.json")
            with patch.object(claude_wing, "_fetch_reading", return_value=_reading(61)):
                read_usage(cfg, now=1000.0)
            claude_wing._reset_usage_cache()  # a new process
            fetch = MagicMock()
            with patch.object(claude_wing, "_fetch_reading", fetch):
                reading = read_usage(cfg, now=1010.0)
        self.assertEqual(reading.weekly, 61)
        fetch.assert_not_called()

    def test_no_state_path_means_no_file(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(claude_wing, "_fetch_reading", return_value=_reading(61)):
            read_usage(_cfg(), now=1000.0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_peek_never_fetches_and_starts_one_refresh_when_stale(self):
        fetch, refresh = MagicMock(), MagicMock()
        with patch.object(claude_wing, "_fetch_reading", fetch), \
             patch.object(claude_wing, "_start_refresh", refresh):
            self.assertIsNone(peek_usage(_cfg()))
            peek_usage(_cfg())
        fetch.assert_not_called()
        self.assertEqual(refresh.call_count, 1)

    def test_peek_returns_a_fresh_reading_without_refreshing(self):
        refresh = MagicMock()
        with patch.object(claude_wing, "_fetch_reading", return_value=_reading(40)):
            read_usage(_cfg())
        with patch.object(claude_wing, "_start_refresh", refresh):
            self.assertEqual(peek_usage(_cfg()).weekly, 40)
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
