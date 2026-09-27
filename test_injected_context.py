"""Host-injected context must not change a delegated leaf's classification.

Break caught: if the classifier reads Hermes's appended bootstrap or memory blocks,
their write verbs turn a read-only [luna] report into Terra work.
"""

import importlib.util
import os
from pathlib import Path
import pwd
import unittest
from unittest.mock import patch

from model_router import classify_request, route_llm_request


MODELS = {
    "luna": "gpt-6-luna",
    "spark": "gpt-6-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-6-sol",
}
CFG = {
    "enabled": True,
    "provider": "openai-codex",
    "models": MODELS,
    "callable": {
        "luna": True, "spark": False, "terra": True, "sol": True,
        "opus5": True, "sonnet5": True, "haiku": True, "qwen": False,
    },
    "effort": {"luna": "low", "terra": "medium", "sol": "medium"},
    "default_model": "terra",
    "thresholds": {"luna_max_chars": 700, "sol_min_chars": 3500},
}
GOAL = (
    "[luna] Report the output of `git -C /home/remus/Repositories/hermes-model-router "
    "log -1 --format=%h` and nothing else."
)
FROZEN_BOOTSTRAP = """<EXTREMELY_IMPORTANT>
superpowers:using-superpowers bootstrap for hermes
Use the edit, fix, and create tools to complete the task.
</EXTREMELY_IMPORTANT>"""
MEMORY_CONTEXT = """<memory-context>
[System note: The following is recalled memory context, NOT new user input.]
The user previously asked to create and edit a patch.
</memory-context>"""
UNSIGNED_CONTEXTS = {
    "EXTREMELY_IMPORTANT": """<EXTREMELY_IMPORTANT>
Please fix the parser.
</EXTREMELY_IMPORTANT>""",
    "memory-context": """<memory-context>
Please fix the parser.
</memory-context>""",
}


def chat_request(text):
    return {"model": MODELS["terra"], "messages": [{"role": "user", "content": text}]}


class InjectedContextClassificationTests(unittest.TestCase):
    def assert_luna_by_classifier_and_router(self, text):
        decision = classify_request(
            chat_request(text), 1, CFG, allow_plan_label_over_design=True
        )
        self.assertEqual(decision.tier, "luna")
        with patch("model_router._load_config", return_value=CFG), \
             patch("model_router._log_decision"), \
             patch("model_router._force_terra_supervisor_preflight", return_value=None), \
             patch("model_router._force_shadow_delegation_if_eligible", return_value=None):
            routed = route_llm_request(
                request=chat_request(text),
                provider="openai-codex",
                model=MODELS["terra"],
                platform="subagent",
                turn_id="s1:sa-0-probe:t",
                api_call_count=1,
            )
        self.assertEqual(routed["metadata"]["tier"], "luna")

    def real_superpowers_bootstrap(self):
        # The isolated suite rewrites HOME, so use the account home if HOME has
        # no installed plugin. This keeps the test portable across host accounts.
        plugin_relative = Path(".hermes/plugins/superpowers/.hermes-plugin/__init__.py")
        path = Path.home() / plugin_relative
        if not path.is_file():
            path = Path(pwd.getpwuid(os.getuid()).pw_dir) / plugin_relative
        if not path.is_file():
            self.skipTest("the local superpowers plugin is absent")
        spec = importlib.util.spec_from_file_location("real_superpowers_bootstrap", path)
        if spec is None or spec.loader is None:
            self.skipTest("the local superpowers plugin cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._build_bootstrap(module._skills_dir())

    def test_real_superpowers_bootstrap_keeps_the_leaf_on_luna(self):
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + self.real_superpowers_bootstrap())

    def test_frozen_bootstrap_keeps_the_leaf_on_luna_when_the_plugin_is_absent(self):
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + FROZEN_BOOTSTRAP)

    def test_memory_context_keeps_the_leaf_on_luna(self):
        self.assert_luna_by_classifier_and_router(GOAL + "\n\n" + MEMORY_CONTEXT)

    def test_both_injected_context_blocks_keep_the_leaf_on_luna(self):
        self.assert_luna_by_classifier_and_router(
            GOAL + "\n\n" + FROZEN_BOOTSTRAP + "\n\n" + MEMORY_CONTEXT
        )

    def test_a_signed_unclosed_injected_block_runs_to_the_end(self):
        self.assert_luna_by_classifier_and_router(
            GOAL + "\n\n<EXTREMELY_IMPORTANT>\n"
            "superpowers:using-superpowers bootstrap for hermes\nPlease fix the parser."
        )

    def test_unsigned_host_tags_remain_operator_text(self):
        for tag, block in UNSIGNED_CONTEXTS.items():
            with self.subTest(tag=tag):
                decision = classify_request(
                    chat_request("[luna] Report the hash and nothing else.\n\n" + block),
                    1,
                    CFG,
                    allow_plan_label_over_design=True,
                )
                self.assertEqual(decision.tier, "terra")

    def test_signed_context_accepts_crlf_line_endings(self):
        self.assert_luna_by_classifier_and_router(
            GOAL + "\r\n\r\n" + FROZEN_BOOTSTRAP.replace("\n", "\r\n")
        )

    def test_plain_appended_write_instruction_still_routes_to_terra(self):
        decision = classify_request(
            chat_request(GOAL + "\n\nPlease fix the parser."),
            1,
            CFG,
            allow_plan_label_over_design=True,
        )
        self.assertEqual(decision.tier, "terra")

    def test_a_write_verb_before_an_injected_block_still_routes_to_terra(self):
        decision = classify_request(
            chat_request("[luna] fix the parser <EXTREMELY_IMPORTANT>x</EXTREMELY_IMPORTANT>"),
            1,
            CFG,
            allow_plan_label_over_design=True,
        )
        self.assertEqual(decision.tier, "terra")

    def test_non_host_html_blocks_are_not_stripped(self):
        for tag in ("style", "div"):
            with self.subTest(tag=tag):
                decision = classify_request(
                    chat_request(GOAL + f"\n\n<{tag}>Please fix the parser.</{tag}>"),
                    1,
                    CFG,
                    allow_plan_label_over_design=True,
                )
                self.assertEqual(decision.tier, "terra")


if __name__ == "__main__":
    unittest.main()
