"""A leaf must be classified from its goal, not from the contract attached to it.

Observed 2026-09-14, twice in one day: a leaf whose goal began "[spark] Read-only
discovery: map every point where customerNote is filtered..." ran on Sol, logged as
"consequential Spark task requires Sol" — fourteen calls of it.

The leaf carries the routing contract in its own first message, and that contract
necessarily talks about applying, implementing and committing. Measured: the goal
alone reads read-only, the goal plus the contract does not, on the word "apply"
from the contract. "server" in the same contract then made it consequential, so the
label was not merely rejected but promoted to the most expensive tier.
"""

import unittest

from model_router import (
    _is_consequential_spark_request,
    _is_spark_read_only_work,
    _without_router_contract,
)


GOAL = (
    "[spark] Read-only discovery: map every point where customerNote is filtered, "
    "redacted, or gated in the booking-saas codebase on branch development. "
    "Specifically: 1. Find every API route or server-side code that returns bookings "
    "with customerNote. 2. Find the note indicator component and show the exact JSX."
)

CONTRACT = (
    "Set the delegate_task 'model' parameter on every worker to choose its route "
    "(targets: luna, opus5, sol, sonnet5, spark, terra). The rules each label carries "
    "still apply, so a Spark leaf must still be read-only. Give them implementation or "
    "deep review. Name the absolute worktree path, the branch and the commit it builds on."
)


class ContractStrippingTests(unittest.TestCase):
    def test_the_goal_alone_reads_as_read_only_work(self):
        self.assertTrue(_is_spark_read_only_work(GOAL))

    def test_the_attached_contract_alone_would_reverse_that(self):
        """The regression this guards: the contract's own vocabulary decided the route."""
        self.assertFalse(_is_spark_read_only_work(GOAL + "\n\n" + CONTRACT))

    def test_stripping_restores_the_goal_verdict(self):
        self.assertTrue(_is_spark_read_only_work(_without_router_contract(GOAL + "\n\n" + CONTRACT)))

    def test_a_goal_without_a_contract_is_untouched(self):
        self.assertEqual(_without_router_contract(GOAL), GOAL)

    def test_every_marker_cuts(self):
        for marker in ("Set the delegate_task 'model' parameter", "planning conductor.",
                       "[INTERNAL ORCHESTRATOR PREFLIGHT]"):
            with self.subTest(marker=marker):
                self.assertEqual(_without_router_contract(f"{GOAL}\n\n{marker} rest"), GOAL)

    def test_the_earliest_marker_wins(self):
        text = f"{GOAL}\n\nplanning conductor. then Set the delegate_task 'model' parameter"
        self.assertEqual(_without_router_contract(text), GOAL)

    def test_a_message_that_is_only_contract_is_left_alone(self):
        """Cutting to empty would classify a blank string; keep the text instead."""
        self.assertEqual(_without_router_contract(CONTRACT), CONTRACT)

    def test_empty_input_is_safe(self):
        self.assertEqual(_without_router_contract(""), "")

    def test_the_contract_opened_the_door_the_goal_then_walked_through(self):
        """Precisely: the contract did not make the leaf consequential — "server-side" in
        the goal itself does that. What the contract did was flip the read-only verdict,
        and only a failed read-only test consults the consequential check at all. With
        the verdict restored the question is never asked, which is why the leaf routes
        to Spark despite still matching the consequential pattern."""
        self.assertTrue(_is_consequential_spark_request(GOAL))  # "server-side"
        self.assertFalse(_is_spark_read_only_work(GOAL + "\n\n" + CONTRACT))
        self.assertTrue(_is_spark_read_only_work(_without_router_contract(GOAL + "\n\n" + CONTRACT)))


if __name__ == "__main__":
    unittest.main()
