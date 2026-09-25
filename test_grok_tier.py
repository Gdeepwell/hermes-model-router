"""The Grok tier: a SuperGrok subscription reached through Hermes's ``xai-oauth`` provider.

Grok is wired the way Qwen is -- a tier in ``models`` on its own account -- so it
is a delegation target, a preference-chain entry and a fallback, never a model
the router can switch a running turn onto. It ships switched off: it needs a
SuperGrok login that most installations do not have.
"""

import unittest
from unittest.mock import patch

import yaml

import model_router
from model_router import _decision, _misdispatched_external_label, usage_guard
from model_router import web_viewer


def shipped():
    return yaml.safe_load(model_router._CONFIG_PATH.read_text(encoding="utf-8"))


class ShippedConfigTests(unittest.TestCase):
    def test_grok_is_a_separate_account_that_ships_off(self):
        cfg = shipped()
        self.assertEqual(cfg["models"]["grok"], "grok-4.7")
        self.assertEqual(cfg["tier_providers"]["grok"], "xai-oauth")
        self.assertIs(cfg["callable"]["grok"], False)
        self.assertEqual(cfg["fallbacks"]["grok"], "terra")
        self.assertIn("grok", cfg["peer_groups"]["heavy"])

    def test_the_decision_carries_the_grok_model_and_an_effort(self):
        cfg = shipped()
        decision = _decision("grok", "delegated grok worker", cfg)
        self.assertEqual(decision.model, "grok-4.7")
        self.assertEqual(decision.effort, "medium")

    def test_the_account_has_a_label(self):
        self.assertEqual(usage_guard.ACCOUNT_LABELS["xai-oauth"], "Grok")


class LabelTests(unittest.TestCase):
    def test_a_grok_label_on_a_codex_leaf_is_misdispatched(self):
        cfg = shipped()
        self.assertEqual(_misdispatched_external_label("[grok] implement it", "gpt-5.6-terra", cfg), "grok")

    def test_a_grok_label_on_the_grok_leaf_is_just_redundant(self):
        cfg = shipped()
        self.assertEqual(_misdispatched_external_label("[grok] implement it", "grok-4.7", cfg), "")


class GateTests(unittest.TestCase):
    def test_delegate_task_to_a_switched_off_grok_is_refused(self):
        cfg = shipped()
        with patch.object(model_router, "_load_config", return_value=cfg), \
             patch.object(model_router, "_tier_cooldown_remaining", return_value=0.0):
            result = model_router.on_pre_tool_call(
                tool_name="delegate_task", args={"model": "grok", "tasks": [{"goal": "g"}]})
        self.assertEqual(result["action"], "block")
        self.assertIn('"grok" is switched off', result["message"])
        self.assertIn('model "terra"', result["message"])


class DefaultModelSyncTests(unittest.TestCase):
    def test_a_grok_parent_is_written_with_the_responses_api(self):
        """xAI's OAuth route speaks the Codex Responses API; writing chat_completions
        (the branch every non-Codex provider used to take) would break the parent."""
        cfg = shipped()
        hermes = {"model": {"default": "gpt-5.6-terra", "provider": "openai-codex",
                            "api_mode": "codex_responses"}, "providers": {}}
        with patch.object(web_viewer, "_write_hermes_config") as write:
            self.assertIsNone(web_viewer._sync_hermes_default_model("grok", cfg, hermes, None))
        written = write.call_args.args[0]["model"]
        self.assertEqual(written, {"default": "grok-4.7", "provider": "xai-oauth",
                                   "api_mode": "codex_responses"})


if __name__ == "__main__":
    unittest.main()
