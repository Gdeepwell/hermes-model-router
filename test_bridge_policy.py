import time
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import _maybe_run_opus5, usage_guard


class BridgePolicyTests(unittest.TestCase):
    def run_bridge(self, label, switches, weekly=10):
        cfg = {"callable": switches, "coding_agent": {"delegated_review": {"enabled": True}},
               "usage_guard": {"accounts": {"anthropic": {
                   "soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}}}}}
        request = {"messages": [{"role": "user", "content": f"[{label}-review] Review parser"}]}
        with patch("model_router._verified_delegated_claude_review", return_value=(Path("/tmp"), label)), \
             patch("model_router.usage_guard.read", return_value=usage_guard.Reading(weekly, 10, None, None, time.time())), \
             patch("model_router._run_opus5_bridge", return_value={"result": "reviewed", "model": "claude-sonnet-5"}) as bridge:
            _maybe_run_opus5(request, cfg, platform="subagent", api_mode="codex_responses")
        return bridge

    def test_sonnet_obeys_its_own_switch(self):
        self.run_bridge("sonnet", {"sonnet5": False, "opus5": True}).assert_not_called()
        self.run_bridge("sonnet", {"sonnet5": True, "opus5": False}).assert_called_once()

    def test_hard_limit_blocks_both_cli_tiers(self):
        for label in ("opus", "sonnet"):
            self.run_bridge(label, {"sonnet5": True, "opus5": True}, weekly=95).assert_not_called()

    def test_soft_limit_selects_only_an_enabled_lighter_tier(self):
        bridge = self.run_bridge("opus", {"sonnet5": True, "opus5": True}, weekly=75)
        self.assertEqual(bridge.call_args.kwargs["model"], "sonnet")
        self.run_bridge("opus", {"sonnet5": False, "opus5": True}, weekly=75).assert_not_called()
