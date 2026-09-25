"""The pre_tool_call gate on delegate_task.

Observed 2026-09-25 10:42: with Qwen switched off in the dashboard, a Terra
worker called ``delegate_task(model="qwen")``. The tool offers every Hermes
``delegation.targets`` entry, the router's llm_request middleware raised, and
Hermes logged that as a warning and sent the request to Qwen anyway -- the child
died on a 403. The gate refuses such a call before anything is spawned.
"""

import unittest
from unittest.mock import MagicMock, patch

import model_router
from model_router import claude_delegation

CFG = {
    "callable": {"luna": True, "terra": True, "sol": True, "qwen": False,
                 "opus5": True, "sonnet5": True, "haiku": True},
    "models": {"luna": "gpt-5.6-luna", "terra": "gpt-5.6-terra",
               "sol": "gpt-5.6-sol", "qwen": "qwen3.7-plus"},
    "tier_providers": {"luna": "openai-codex", "terra": "openai-codex",
                       "sol": "openai-codex", "qwen": "qwen-token"},
    "fallbacks": {"qwen": "terra", "sol": "terra"},
    "peer_groups": {"heavy": ["terra", "opus5", "qwen", "sonnet5"]},
    "default_model": "terra",
}


class DelegateTaskGateTests(unittest.TestCase):
    def gate(self, args, *, cooling=(), claude=True, worker_fallback=False, tool="delegate_task"):
        remaining = lambda tier, cfg: 600.0 if tier in cooling else 0.0
        with patch.object(model_router, "_load_config", return_value=CFG), \
             patch.object(model_router, "_tier_cooldown_remaining", side_effect=remaining), \
             patch.object(claude_delegation, "is_active", return_value=claude), \
             patch.object(model_router, "_hermes_worker_fallback_configured", return_value=worker_fallback):
            return model_router.on_pre_tool_call(tool_name=tool, args=args, task_id="t")

    def test_a_switched_off_call_level_model_is_blocked_with_a_working_route(self):
        result = self.gate({"model": "qwen", "tasks": [{"goal": "look around"}]})
        self.assertEqual(result["action"], "block")
        self.assertIn('"qwen" is switched off in the dashboard', result["message"])
        self.assertIn('model "terra"', result["message"])
        self.assertIn("delegate_claude", result["message"])
        self.assertIn("Nothing was spawned", result["message"])

    def test_a_per_task_model_and_a_full_model_name_are_both_checked(self):
        result = self.gate({"tasks": [{"goal": "a", "model": "terra"},
                                      {"goal": "b", "model": "qwen3.7-plus"}]})
        self.assertIn('"qwen"', result["message"])
        self.assertNotIn('"terra" is', result["message"])

    def test_an_available_target_passes(self):
        self.assertIsNone(self.gate({"model": "terra", "tasks": [{"goal": "g"}]}))
        self.assertIsNone(self.gate({"tasks": [{"goal": "no model at all"}]}))

    def test_a_name_the_router_does_not_know_is_left_to_hermes(self):
        self.assertIsNone(self.gate({"tasks": [{"goal": "g", "model": "someone-elses"}]}))

    def test_other_tools_are_never_touched(self):
        self.assertIsNone(self.gate({"model": "qwen"}, tool="terminal"))

    def test_without_claude_delegation_no_claude_route_is_offered(self):
        result = self.gate({"model": "qwen", "tasks": [{"goal": "g"}]}, claude=False)
        self.assertNotIn("delegate_claude", result["message"])

    def test_a_cooling_target_is_blocked_only_without_a_worker_fallback_chain(self):
        args = {"model": "sol", "tasks": [{"goal": "g"}]}
        blocked = self.gate(args, cooling={"sol"})
        self.assertIn('"sol" is cooling down', blocked["message"])
        self.assertIsNone(self.gate(args, cooling={"sol"}, worker_fallback=True),
                          "Hermes's worker chain can still move the child to another account")

    def test_switched_off_is_blocked_even_with_a_worker_fallback_chain(self):
        result = self.gate({"model": "qwen", "tasks": [{"goal": "g"}]}, worker_fallback=True)
        self.assertEqual(result["action"], "block")

    def test_a_broken_check_fails_open(self):
        with patch.object(model_router, "_load_config", side_effect=RuntimeError("boom")):
            self.assertIsNone(model_router.on_pre_tool_call(tool_name="delegate_task", args={"model": "qwen"}))

    def test_register_wires_the_gate(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "register"):
            model_router.register(ctx)
        hooks = [call.args[0] for call in ctx.register_hook.call_args_list]
        self.assertIn("pre_tool_call", hooks)


if __name__ == "__main__":
    unittest.main()
