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


if __name__ == "__main__":
    unittest.main()
