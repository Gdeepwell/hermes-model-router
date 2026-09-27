"""Deferred routing-core Minors: A-M2, A-M4, A-M5.

A-M2: the host-injected bootstrap mentions delegate_task, so a raw-text reader
of the orchestration preflight gate must not treat that as the operator
explicitly naming a delegation tool.

A-M4: the dispatch-review repository lookup must not stat a directory
(Path.is_dir) while holding the shared repository cache lock.
"""

import importlib.util
import os
from pathlib import Path
import pwd
import tempfile
import threading
import unittest
from unittest.mock import patch

import model_router as router


CFG = {
    "enabled": True,
    "provider": "openai-codex",
    "models": {
        "luna": "gpt-6-luna", "spark": "gpt-6-spark",
        "terra": "gpt-5.6-terra", "sol": "gpt-6-sol",
    },
    "callable": {
        "luna": True, "spark": False, "terra": True, "sol": True,
        "opus5": True, "sonnet5": True, "haiku": True, "qwen": False,
    },
    "effort": {"luna": "low", "terra": "medium", "sol": "medium"},
    "default_model": "terra",
    "thresholds": {"luna_max_chars": 700, "sol_min_chars": 3500},
    "orchestration": {"enabled": True},
}

# Long enough to clear the preflight's min_chars gate (180) on its own, with
# no delegate_task/delegate_claude mention and no design-related words.
LONG_OPERATOR_TEXT = (
    "Investigate why the nightly batch job in the billing reconciliation "
    "service intermittently produces duplicate ledger rows and write up "
    "findings with reproduction steps in a plain text report."
)


def real_superpowers_bootstrap(case: unittest.TestCase) -> str:
    # The isolated suite rewrites HOME, so use the account home if HOME has
    # no installed plugin (mirrors test_injected_context.py).
    plugin_relative = Path(".hermes/plugins/superpowers/.hermes-plugin/__init__.py")
    path = Path.home() / plugin_relative
    if not path.is_file():
        path = Path(pwd.getpwuid(os.getuid()).pw_dir) / plugin_relative
    if not path.is_file():
        case.skipTest("the local superpowers plugin is absent")
    spec = importlib.util.spec_from_file_location("real_superpowers_bootstrap_m2", path)
    if spec is None or spec.loader is None:
        case.skipTest("the local superpowers plugin cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_bootstrap(module._skills_dir())


class OrchestrationSkipReasonBootstrapTests(unittest.TestCase):
    def _kwargs(self, text: str) -> dict:
        request = {
            "messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "delegate_task"}}],
        }
        return {
            "request": request, "api_call_count": 1,
            "platform": "parent", "turn_id": "s1:root",
        }

    def test_real_bootstrap_delegate_task_mention_does_not_fire_the_gate(self):
        bootstrap = real_superpowers_bootstrap(self)
        decision = router._decision(CFG["default_model"], "x", CFG)
        without_bootstrap = router._orchestration_skip_reason(
            self._kwargs(LONG_OPERATOR_TEXT), CFG, decision,
        )
        with_bootstrap = router._orchestration_skip_reason(
            self._kwargs(LONG_OPERATOR_TEXT + "\n\n" + bootstrap), CFG, decision,
        )
        self.assertNotEqual(with_bootstrap, "explicit_delegation_tool")
        # The bootstrap must not change the gate's verdict at all: whatever
        # the operator text alone decided is what the bootstrapped turn gets.
        self.assertEqual(with_bootstrap, without_bootstrap)

    def test_operator_own_delegate_task_mention_still_fires_the_gate(self):
        decision = router._decision(CFG["default_model"], "x", CFG)
        text = LONG_OPERATOR_TEXT + " Please call delegate_task yourself for this."
        reason = router._orchestration_skip_reason(self._kwargs(text), CFG, decision)
        self.assertEqual(reason, "explicit_delegation_tool")


class OrchestrationSkipReasonMinCharsBootstrapTests(unittest.TestCase):
    """A2-N1: the A-M2 fix stripped the tool-name check but not min_chars or
    the bounded-UI check, so a short operator turn plus the real bootstrap
    (which pushes total length over min_chars and the bounded-UI max_chars)
    used to fall through the length gate and force a planner preflight.
    """

    SHORT_OPERATOR_TEXT = "Fix the typo in README.md line 3."

    def _kwargs(self, text: str) -> dict:
        request = {
            "messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "delegate_task"}}],
        }
        return {
            "request": request, "api_call_count": 1,
            "platform": "parent", "turn_id": "s1:root",
        }

    def test_short_operator_text_with_real_bootstrap_still_hits_min_chars(self):
        bootstrap = real_superpowers_bootstrap(self)
        decision = router._decision(CFG["default_model"], "x", CFG)
        with patch("model_router._host_delegation_limits",
                   return_value={"conductor_available": True}), \
             patch("model_router._conductor_tier", return_value="terra"):
            without_bootstrap = router._orchestration_skip_reason(
                self._kwargs(self.SHORT_OPERATOR_TEXT), CFG, decision,
            )
            with_bootstrap = router._orchestration_skip_reason(
                self._kwargs(self.SHORT_OPERATOR_TEXT + "\n\n" + bootstrap), CFG, decision,
            )
        expected = f"prompt_shorter_than_min_chars:{len(self.SHORT_OPERATOR_TEXT)}<180"
        self.assertEqual(without_bootstrap, expected)
        self.assertEqual(with_bootstrap, expected)


class RememberedRepositoryLockTests(unittest.TestCase):
    def setUp(self):
        router._DISPATCH_REVIEW_REPOSITORIES.clear()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_directory_stat_does_not_run_while_the_lock_is_held(self):
        repo = Path(self._tmpdir.name)
        goal = "x" * 25  # >= the 20-char minimum the prefix matcher requires
        router._remember_dispatch_review_repository(goal, repo)

        stat_started = threading.Event()
        release_stat = threading.Event()
        original_is_dir = Path.is_dir

        def slow_is_dir(self_path):
            if self_path == repo.resolve():
                stat_started.set()
                release_stat.wait(timeout=5)
            return original_is_dir(self_path)

        result = {}

        def call():
            with patch.object(Path, "is_dir", slow_is_dir):
                result["repo"] = router._remembered_dispatch_review_repository(goal)

        thread = threading.Thread(target=call)
        thread.start()
        try:
            self.assertTrue(stat_started.wait(timeout=2), "the directory stat never started")
            acquired = router._REPOSITORY_CACHE_LOCK.acquire(timeout=1)
            self.assertTrue(
                acquired,
                "the repository cache lock was still held while Path.is_dir() ran",
            )
            router._REPOSITORY_CACHE_LOCK.release()
        finally:
            release_stat.set()
            thread.join(timeout=5)
        self.assertEqual(result["repo"], repo.resolve())


if __name__ == "__main__":
    unittest.main()
