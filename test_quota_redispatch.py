"""A worker that stopped on an account limit, told back to the conductor.

The envelope already reports that a leaf failed and quotes the provider's error.
What it cannot say is that the account rather than the task is what stopped, which
target the operator's order says to use instead, and that the work already
committed is worth continuing from. These cover that gap -- and the two ways of
getting it wrong: firing on an ordinary failure, and firing on a leaf whose own
subject happens to be quotas.
"""

import unittest
from unittest.mock import patch

from model_router import (
    _delegation_failure_reason,
    _failed_delegation_blocks,
    _kind_for_goal,
    _next_available_entry,
    _quota_redispatch_instruction,
    route_llm_request,
)


TARGETS = ("luna", "opus5", "sol", "sonnet5", "terra")

CFG = {
    "enabled": True,
    "provider": "openai-codex",
    "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
    "callable": {"luna": True, "spark": False, "terra": True, "sol": True,
                 "opus5": True, "sonnet5": True, "qwen": False},
    "effort": {"luna": "low", "terra": "medium", "sol": "medium"},
    "preferences": {"code": ["opus5", "terra"], "design": ["sol", "opus5"]},
    "default_model": "terra",
}

GOAL = ("Implement and commit the backend/domain/data/API portion of the customer "
        "reliability ledger in the assigned worktree.")

QUOTA_ERROR = ("(error: Error code: 429 - {'code': 'Throttling.AllocationQuota', "
               "'message': 'quota has been exhausted'})")
OTHER_ERROR = "(error: TypeError: cannot read property 'id' of undefined)"


def envelope(error_line=QUOTA_ERROR, summary="Committed the schema migration; stopped there."):
    return (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_2f5ad483]\n"
        "A background fan-out unit you dispatched earlier has finished.\n"
        "\n--- ✓ TASK 1/2: [sol] Implement the approved OWNER policy panel."
        "  (status=completed, api_calls=11, 240s) ---\n"
        "Panel implemented and committed.\n"
        f"\n--- ✗ TASK 2/2: {GOAL}  (status=error, api_calls=6, 61s) ---\n"
        f"{error_line}\n"
        "Partial output:\n"
        f"{summary}\n"
    )


def request_for(text):
    return {"model": "gpt-terra", "messages": [{"role": "user", "content": text}]}


def _no_cooldown(name, cfg):
    return 0.0


class EnvelopeParsingTests(unittest.TestCase):
    def test_only_the_failed_task_is_picked_up(self):
        blocks = _failed_delegation_blocks(envelope())
        self.assertEqual([goal for goal, _ in blocks], [GOAL])

    def test_the_reason_excludes_the_workers_own_output(self):
        """A leaf reporting *about* quotas must not read as stopped by one."""
        blocks = _failed_delegation_blocks(
            envelope(error_line=OTHER_ERROR, summary="Analysed the 429 quota handling path.")
        )
        self.assertNotIn("429", _delegation_failure_reason(blocks[0][1]))

    def test_the_goal_is_classified_by_the_normal_classifier(self):
        self.assertEqual(_kind_for_goal(GOAL, CFG), "code")


class NextEntryTests(unittest.TestCase):
    def test_the_first_entry_that_is_not_cooling_wins(self):
        with patch("model_router._tier_cooldown_remaining", side_effect=_no_cooldown), \
             patch("model_router._delegation_target_names", return_value=TARGETS):
            self.assertEqual(_next_available_entry("code", CFG), "opus5")

    def test_a_cooling_first_entry_yields_the_next_one(self):
        with patch("model_router._tier_cooldown_remaining",
                   side_effect=lambda name, cfg: 900.0 if name == "opus5" else 0.0), \
             patch("model_router._delegation_target_names", return_value=TARGETS):
            self.assertEqual(_next_available_entry("code", CFG), "terra")

    def test_a_switched_off_entry_is_never_offered(self):
        cfg = {**CFG, "preferences": {"code": ["qwen", "terra"]}}
        with patch("model_router._tier_cooldown_remaining", side_effect=_no_cooldown), \
             patch("model_router._delegation_target_names", return_value=TARGETS + ("qwen",)):
            self.assertEqual(_next_available_entry("code", cfg), "terra")


class RedispatchInstructionTests(unittest.TestCase):
    def _instruction(self, text, cooling=()):
        def remaining(name, cfg):
            return 900.0 if name in cooling else 0.0
        with patch("model_router._tier_cooldown_remaining", side_effect=remaining), \
             patch("model_router._delegation_target_names", return_value=TARGETS):
            return _quota_redispatch_instruction(request_for(text), CFG)

    def test_it_names_the_next_target_for_the_goals_kind(self):
        instruction = self._instruction(envelope())
        self.assertIn("model:opus5", instruction)
        self.assertIn("code work", instruction)
        self.assertIn(GOAL[:60], instruction)

    def test_it_advances_down_the_chain_when_the_first_entry_is_cooling(self):
        self.assertIn("model:terra", self._instruction(envelope(), cooling={"opus5"}))

    def test_it_says_how_long_to_wait_when_every_entry_is_cooling(self):
        instruction = self._instruction(envelope(), cooling={"opus5", "terra"})
        self.assertIn("every configured target is cooling", instruction)
        self.assertIn("min", instruction)
        self.assertNotIn("re-dispatch with model:", instruction)

    def test_it_says_to_continue_rather_than_restart(self):
        instruction = self._instruction(envelope())
        self.assertIn("continue from what the stopped worker already committed", instruction)
        self.assertIn("Do not re-plan", instruction)

    def test_an_ordinary_failure_is_left_alone(self):
        """Only an account limit is safe to re-dispatch unchanged."""
        self.assertEqual(self._instruction(envelope(error_line=OTHER_ERROR)), "")

    def test_a_leaf_whose_subject_is_quotas_is_left_alone(self):
        self.assertEqual(
            self._instruction(envelope(
                error_line=OTHER_ERROR,
                summary="Analysed the quota exhaustion path and the 429 rate limit handling.",
            )),
            "",
        )

    def test_a_batch_with_nothing_failed_is_left_alone(self):
        self.assertEqual(self._instruction(envelope().split("--- ✗")[0]), "")

    def test_an_ordinary_turn_is_left_alone(self):
        self.assertEqual(self._instruction("Folytasd a naptár javítását."), "")


class MiddlewareIntegrationTests(unittest.TestCase):
    def _route(self, request):
        def remaining(name, cfg):
            return 0.0
        with patch("model_router._load_config", return_value=CFG), \
             patch("model_router._log_decision"), \
             patch("model_router._tier_cooldown_remaining", side_effect=remaining), \
             patch("model_router._delegation_target_names", return_value=TARGETS), \
             patch("model_router._force_terra_supervisor_preflight", return_value=None), \
             patch("model_router._force_shadow_delegation_if_eligible", return_value=None):
            return route_llm_request(
                request=request, provider="openai-codex", model="gpt-terra",
                api_call_count=1, turn_id="turn-redispatch",
            )

    def test_the_instruction_reaches_the_routed_request(self):
        result = self._route(request_for(envelope()))
        self.assertIn("A WORKER STOPPED ON AN ACCOUNT LIMIT",
                      result["request"]["messages"][-1]["content"])

    def test_the_callers_own_request_is_not_mutated(self):
        """The plain path is a shallow copy; appending in place would rewrite the
        conversation the caller still holds."""
        request = request_for(envelope())
        original = request["messages"][-1]["content"]
        self._route(request)
        self.assertEqual(request["messages"][-1]["content"], original)


if __name__ == "__main__":
    unittest.main()


class ExternalConductorTests(unittest.TestCase):
    """The conductor itself often runs on Claude — the `code` chain puts it there."""

    def _route(self, request, forced=None):
        with patch("model_router._load_config", return_value=CFG), \
             patch("model_router._log_decision"), \
             patch("model_router._tier_cooldown_remaining", side_effect=_no_cooldown), \
             patch("model_router._delegation_target_names", return_value=TARGETS), \
             patch("model_router._external_target_for_model", return_value="opus5"), \
             patch("model_router._force_terra_supervisor_preflight", return_value=forced):
            return route_llm_request(
                request=request, provider="anthropic", model="claude-opus-5",
                api_call_count=1, turn_id="turn-external",
            )

    def test_the_notice_reaches_a_conductor_on_another_provider(self):
        result = self._route(request_for(envelope()))
        self.assertIsNotNone(result)
        self.assertIn("A WORKER STOPPED ON AN ACCOUNT LIMIT",
                      result["request"]["messages"][-1]["content"])

    def test_the_route_is_still_not_rewritten(self):
        """This branch may add text; it may never move the call to another provider."""
        result = self._route(request_for(envelope()))
        self.assertEqual(result["metadata"]["model"], "claude-opus-5")
        self.assertEqual(result["metadata"]["provider"], "anthropic")
        # Untouched: the middleware cannot move a call across providers, so the
        # request goes back carrying only the added text.
        self.assertEqual(result["request"]["model"], request_for("")["model"])

    def test_an_ordinary_off_provider_turn_still_returns_none(self):
        self.assertIsNone(self._route(request_for("Folytasd.")))

    def test_the_callers_own_request_is_not_mutated(self):
        request = request_for(envelope())
        original = request["messages"][-1]["content"]
        self._route(request)
        self.assertEqual(request["messages"][-1]["content"], original)


SINGLE_FAILURE = """[ASYNC DELEGATION TASK FAILED — deleg_2f5ad483, task 2/2]
One subagent in a background fan-out you dispatched has failed while its siblings are still running.
Task: {goal}
Status: error   Duration: 61s
Error: Error code: 429 - {{'code': 'Throttling.AllocationQuota', 'message': 'quota has been exhausted'}}
"""


class EarlyFailureNoticeTests(unittest.TestCase):
    """The immediate notice is the moment to re-dispatch — not batch end."""

    def _instruction(self, text):
        with patch("model_router._tier_cooldown_remaining", side_effect=_no_cooldown), \
             patch("model_router._delegation_target_names", return_value=TARGETS):
            return _quota_redispatch_instruction(request_for(text), CFG)

    def test_the_early_notice_is_answered_too(self):
        instruction = self._instruction(SINGLE_FAILURE.format(goal=GOAL))
        self.assertIn("model:opus5", instruction)
        self.assertIn(GOAL[:60], instruction)

    def test_an_early_notice_for_an_ordinary_failure_is_left_alone(self):
        text = SINGLE_FAILURE.format(goal=GOAL).replace(
            "Error code: 429 - {'code': 'Throttling.AllocationQuota', 'message': 'quota has been exhausted'}",
            "TypeError: cannot read property 'id' of undefined",
        )
        self.assertEqual(self._instruction(text), "")


DISPATCH_ERROR = (
    '{"error": "Cannot resolve delegation provider \'openai-codex\': Codex provider quota '
    'exhausted (429); retry after 3731s. Credentials are still valid."}'
)


def request_with_tool_error(content):
    return {
        "model": "gpt-terra",
        "messages": [
            {"role": "user", "content": "folytasd"},
            {"role": "assistant", "content": "delegating"},
            {"role": "tool", "name": "delegate_task", "content": content},
        ],
    }


class DispatchFailureTests(unittest.TestCase):
    """A delegation that failed before any worker existed.

    The quota notice reads a delegation outcome, and there is none here: the tool
    returns an error inline, no child runs, and nothing is ever delivered to
    explain it.

    And the error is not a fact about the account it names. delegate_task
    resolves the configured default delegation provider once for the whole call,
    before `_normalize_task_list` has even parsed the tasks, and returns
    tool_error on failure -- so an exhausted default blocks every delegation,
    including a task naming a target on a healthy account, whose model: value is
    never read. Advising a retry with a different model: would loop.
    """

    def _instruction(self, request, cooling=()):
        def remaining(name, cfg):
            return 900.0 if name in cooling else 0.0
        with patch("model_router._tier_cooldown_remaining", side_effect=remaining), \
             patch("model_router._delegation_target_names", return_value=TARGETS):
            from model_router import _dispatch_failure_instruction
            return _dispatch_failure_instruction(request, CFG)

    def test_it_says_the_block_is_at_the_host_not_at_that_account(self):
        instruction = self._instruction(request_with_tool_error(DISPATCH_ERROR))
        self.assertIn("BLOCKED AT THE HOST", instruction)
        self.assertIn("before it reads the tasks", instruction)

    def test_it_does_not_advise_a_retry_that_would_fail_identically(self):
        """The first version of this notice said "re-issue with model: X". Naming
        a target does not bypass the default resolution, so that looped."""
        instruction = self._instruction(request_with_tool_error(DISPATCH_ERROR))
        self.assertIn("will fail identically", instruction)
        self.assertIn("delegation.provider", instruction)

    def test_it_separates_a_healthy_target_from_an_unreachable_one(self):
        instruction = self._instruction(request_with_tool_error(DISPATCH_ERROR), cooling={"sol", "terra"})
        self.assertIn("opus5", instruction)
        self.assertIn("unreachable only because the default route is down", instruction)

    def test_it_still_speaks_when_every_target_is_cooling(self):
        """The host block is the point, and it holds whatever the targets say."""
        instruction = self._instruction(
            request_with_tool_error(DISPATCH_ERROR),
            cooling=set(TARGETS),
        )
        self.assertIn("BLOCKED AT THE HOST", instruction)
        self.assertNotIn("Targets that are themselves fine", instruction)

    def test_another_tools_error_is_left_alone(self):
        request = request_with_tool_error('{"error": "npm ERR! missing script test"}')
        self.assertEqual(self._instruction(request), "")

    def test_an_ordinary_turn_is_left_alone(self):
        self.assertEqual(self._instruction(request_for("folytasd")), "")

    def test_the_notice_reaches_the_routed_request(self):
        with patch("model_router._load_config", return_value=CFG), \
             patch("model_router._log_decision"), \
             patch("model_router._tier_cooldown_remaining", side_effect=_no_cooldown), \
             patch("model_router._delegation_target_names", return_value=TARGETS), \
             patch("model_router._force_terra_supervisor_preflight", return_value=None), \
             patch("model_router._force_shadow_delegation_if_eligible", return_value=None):
            result = route_llm_request(
                request=request_with_tool_error(DISPATCH_ERROR), provider="openai-codex",
                model="gpt-terra", api_call_count=2, turn_id="turn-dispatch-failure",
            )
        self.assertIn("BLOCKED AT THE HOST", result["request"]["messages"][0]["content"])
