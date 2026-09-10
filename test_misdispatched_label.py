"""A goal that names an external target in its text instead of in model:.

[opus5] reads like [sol] and does the opposite of what the dispatcher meant: the
override regex knows only this provider's four tiers, so the prefix is inert, the
goal is classified on its remaining text, and the leaf works to completion on the
very account the dispatcher was trying to spare. These cover stopping it, and the
three things that must not be mistaken for it.
"""

import unittest
from unittest.mock import patch

from model_router import _misdispatched_external_label, _model_param_contract, route_llm_request


MODELS = {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"}

CFG = {
    "enabled": True,
    "provider": "openai-codex",
    "models": MODELS,
    "callable": {"luna": True, "spark": False, "terra": True, "sol": True,
                 "opus5": True, "sonnet5": True, "qwen": False},
    "effort": {"luna": "low", "terra": "medium", "sol": "medium"},
    "preferences": {"code": ["opus5", "terra"]},
    "default_model": "terra",
}

GOAL = ("[opus5] Fejezd be a Booking SaaS customer reliability presentation integrációját "
        "a tiszta worktree-ben.")


def request_for(text, **extra):
    return {
        "model": "gpt-terra",
        "messages": [{"role": "user", "content": text}],
        "tools": [{"name": "terminal"}, {"name": "patch"}],
        "tool_choice": "auto",
        **extra,
    }


class DetectionTests(unittest.TestCase):
    def test_an_external_name_on_this_providers_model_is_a_misdispatch(self):
        self.assertEqual(_misdispatched_external_label(GOAL, "gpt-terra", CFG), "opus5")

    def test_the_short_form_resolves_to_the_target_name(self):
        self.assertEqual(_misdispatched_external_label("[opus] Do it.", "gpt-terra", CFG), "opus5")
        self.assertEqual(_misdispatched_external_label("[sonnet] Do it.", "gpt-sol", CFG), "sonnet5")

    def test_a_leaf_already_on_that_account_is_not_misdispatched(self):
        """There the prefix is redundant, not wrong -- model: did its job."""
        self.assertEqual(_misdispatched_external_label(GOAL, "claude-opus-5", CFG), "")

    def test_a_claude_review_label_is_a_real_label(self):
        for text in ("[opus5-review] Review the diff.", "[sonnet-review] Review the diff."):
            with self.subTest(text=text):
                self.assertEqual(_misdispatched_external_label(text, "gpt-terra", CFG), "")

    def test_a_tier_label_is_untouched(self):
        self.assertEqual(_misdispatched_external_label("[sol] Build it.", "gpt-terra", CFG), "")

    def test_the_name_must_open_the_goal(self):
        """Mentioning a target is not dispatching to one."""
        self.assertEqual(
            _misdispatched_external_label("Port the opus5 bridge to the new API.", "gpt-terra", CFG),
            "",
        )


class StoppedLeafTests(unittest.TestCase):
    def _route(self, request, turn_id="turn:sa-0-abc:def", platform="subagent"):
        with patch("model_router._load_config", return_value=CFG), \
             patch("model_router._log_decision"), \
             patch("model_router._force_terra_supervisor_preflight", return_value=None), \
             patch("model_router._force_shadow_delegation_if_eligible", return_value=None):
            return route_llm_request(
                request=request, provider="openai-codex", model="gpt-terra",
                api_call_count=1, turn_id=turn_id, platform=platform,
            )

    def test_tool_use_is_switched_off(self):
        """The lever that actually bounds the leaf: a middleware exception is
        fail-open here, so refusing the route would send it unrouted instead."""
        result = self._route(request_for(GOAL))
        self.assertEqual(result["request"]["tool_choice"], "none")

    def test_the_toolset_itself_is_left_intact(self):
        """An emptied tools array beside tool_choice: none is what providers
        reject -- a 400 would trade a quiet waste for a noisy crash."""
        self.assertEqual(len(self._route(request_for(GOAL))["request"]["tools"]), 2)

    def test_the_leaf_is_told_to_report_the_correction(self):
        content = self._route(request_for(GOAL))["request"]["messages"][-1]["content"]
        self.assertIn("WRONG DISPATCH MECHANISM", content)
        self.assertIn('delegate_task(model="opus5")', content)
        self.assertIn("Do not begin the work", content)

    def test_the_route_log_says_why(self):
        self.assertIn("is not a route", self._route(request_for(GOAL))["reason"])

    def test_a_root_turn_is_left_alone(self):
        """There the prefix is the operator asking for Opus, not a dispatch bug."""
        result = self._route(request_for(GOAL), turn_id="root-turn", platform="cli")
        self.assertNotEqual(result["request"].get("tool_choice"), "none")
        self.assertNotIn("WRONG DISPATCH", result["request"]["messages"][-1]["content"])

    def test_an_ordinary_leaf_is_left_alone(self):
        result = self._route(request_for("[sol] Implement the approved panel."))
        self.assertNotEqual(result["request"].get("tool_choice"), "none")
        self.assertNotIn("WRONG DISPATCH", result["request"]["messages"][-1]["content"])

    def test_the_callers_own_request_is_not_mutated(self):
        request = request_for(GOAL)
        original = request["messages"][-1]["content"]
        self._route(request)
        self.assertEqual(request["messages"][-1]["content"], original)
        self.assertEqual(len(request["tools"]), 2)


class ContractTests(unittest.TestCase):
    def test_the_contract_names_the_mistake_not_just_the_rule(self):
        """The general rule was already there and lost anyway, seven goals running."""
        with patch("model_router._delegation_target_names",
                   return_value=("luna", "opus5", "sol", "sonnet5", "terra")), \
             patch("model_router._tier_cooldown_remaining", return_value=0.0), \
             patch("model_router._recent_account_load", return_value={}):
            contract = _model_param_contract("terra", CFG)
        self.assertIn("[opus5] and [sonnet5] are not labels", contract)
        self.assertIn("stopped at its first call", contract)


if __name__ == "__main__":
    unittest.main()
