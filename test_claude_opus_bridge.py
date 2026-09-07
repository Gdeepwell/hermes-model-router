import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from claude_opus_bridge import CANONICAL_OPUS_MODEL, classify_coding_dispatch, classify_review_dispatch, dispatch


class ClaudeOpusBridgeTests(unittest.TestCase):
    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_lifecycle_records_started_then_one_terminal_with_parent_and_precedence(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({
            "subtype": "error_max_turns", "is_error": True,
            "modelUsage": {"claude-sonnet-4": {}}, "num_turns": 8,
        })
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = Path(directory) / "bridge.jsonl"
            with self.assertRaisesRegex(RuntimeError, "max turns"):
                dispatch(
                    "[opus-review] Review only", Path(directory), review=True,
                    parent_session_id="parent-session", parent_turn_id="parent-turn",
                    lifecycle_path=lifecycle,
                )
            events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events], ["started", "terminal"])
        self.assertEqual({event["bridge_run_id"] for event in events}, {events[0]["bridge_run_id"]})
        self.assertEqual(events[0]["parent_session_id"], "parent-session")
        self.assertEqual(events[0]["parent_turn_id"], "parent-turn")
        self.assertEqual(events[1]["state"], "max-turn")
        self.assertNotIn("task", events[0])
        self.assertNotIn("prompt", json.dumps(events))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("claude", 1))
    def test_timeout_is_terminal_and_has_highest_precedence(self, run, log):
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = Path(directory) / "bridge.jsonl"
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                dispatch("[opus-review] Review only", Path(directory), review=True, timeout=1, lifecycle_path=lifecycle)
            events = lifecycle.read_text().splitlines()
            terminal = json.loads(events[-1])
        self.assertEqual(terminal["state"], "timeout")
        self.assertEqual(len(events), 2)

    def test_classifier_is_conservative_and_manual_override_is_available(self):
        self.assertEqual(classify_coding_dispatch("[opus5] Implement the parser")[0], True)
        self.assertEqual(classify_coding_dispatch("[opus5] Implement this bounded CSS label fix")[0], True)
        self.assertEqual(classify_coding_dispatch("Debug the backend parser")[0], True)
        self.assertEqual(classify_coding_dispatch("Write product UI CSS")[0], False)
        self.assertEqual(classify_coding_dispatch("Say hello")[0], False)

    def test_design_veto_applies_to_accented_hungarian(self):
        """The Hungarian design terms are spelled unaccented, so matching raw
        text let real Hungarian design work past the Sol-only veto."""
        for task in (
            "Refaktoráld a wireframe komponens tipográfiáját.",
            "Igazítsd a felület színpalettáját.",
        ):
            with self.subTest(task=task):
                eligible, reason = classify_coding_dispatch(task)
                self.assertFalse(eligible, task)
                self.assertIn("Sol-only", reason)

    def test_explicit_review_accepts_non_coding_and_design_work_read_only(self):
        for task in (
            "[opus-review] Vizsgáld felül ezt a jogosultsági tervet; ne módosíts fájlt.",
            "[opus-review] Review the UI hierarchy and accessibility risks; do not edit files.",
        ):
            with self.subTest(task=task):
                eligible, reason = classify_review_dispatch(task)
                self.assertTrue(eligible, task)
                self.assertIn("review", reason.casefold())
        self.assertFalse(classify_review_dispatch("Review this plan")[0])

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_dispatch_logs_only_the_actual_canonical_opus_model(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "session_id": "verified", "num_turns": 1, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            result = dispatch("[opus] Implement a parser test", Path(directory))
        self.assertEqual(result["effective_model"], CANONICAL_OPUS_MODEL)
        command = run.call_args.args[0]
        self.assertIn("--model", command)
        self.assertIn("opus", command)
        self.assertIn("--tools", command)
        self.assertEqual(command[command.index("--tools") + 1], "Read")
        self.assertIn("--disallowedTools", command)
        self.assertNotIn("--dangerously-skip-permissions", command)
        logged = log.call_args.args[0]
        self.assertEqual((logged.tier, logged.model, logged.effort), ("opus5", CANONICAL_OPUS_MODEL, "external"))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_review_keeps_prompt_off_argv_and_enforces_600_second_floor(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            dispatch("[opus-review] Review only", Path(directory), review=True, timeout=300)
        command = run.call_args.args[0]
        self.assertNotIn("[opus-review] Review only", command)
        self.assertEqual(run.call_args.kwargs["input"].split("\n", 1)[0], "[opus-review] Review only")
        self.assertEqual(run.call_args.kwargs["timeout"], 600)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_the_review_label_selects_the_claude_tier(self, run, log):
        """Two tiers exist so routine review can spend the cheaper one; a single
        tier would burn the separate quota that is the reason to reach Claude."""
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-5": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            out = dispatch("[sonnet-review] Review only", Path(directory), review=True)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "sonnet")
        # Sonnet has no lower tier worth accepting: the result is trusted on the
        # strength of the model that produced it, so a silent drop must not happen.
        self.assertNotIn("--fallback-model", command)
        self.assertEqual(out["effective_model"], "claude-sonnet-5")

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_a_review_that_served_another_tier_is_rejected(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-5": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                dispatch("[opus-review] Review only", Path(directory), review=True)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_explicit_opus_review_accepts_canonical_opus_with_internal_subtask_usage(self, run, log):
        """The requested alias is satisfied by the canonical primary route, not a pure usage map."""
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({
            "modelUsage": {
                CANONICAL_OPUS_MODEL: {"inputTokens": 100, "outputTokens": 50},
                "claude-haiku-4-5": {"inputTokens": 10, "outputTokens": 5},
            },
            "result": "review complete",
        })
        with tempfile.TemporaryDirectory() as directory:
            result = dispatch("[opus-review] Review only", Path(directory), review=True)
        self.assertEqual(result["effective_model"], CANONICAL_OPUS_MODEL)
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--model") + 1], "opus")
        logged = log.call_args.args[0]
        self.assertEqual((logged.tier, logged.model), ("opus5", CANONICAL_OPUS_MODEL))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_dispatch_rejects_alias_or_fallback_as_effective_route(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-4": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "did not serve"):
                dispatch("[opus] Implement a parser test", Path(directory))
        log.assert_not_called()


if __name__ == "__main__":
    unittest.main()