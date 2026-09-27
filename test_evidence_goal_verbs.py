"""Write verbs a read-only evidence goal cannot avoid using.

Observed 2026-09-17: a `[luna]` evidence report on the Szamlazz.hu receipt
integration ran its three calls on Sol, reason "consequential Luna task requires
Sol". The goal was read-only in every line -- and it named its base commit, asked
what a module implements, and promised to write `[REDACTED]` instead of a secret.
Three write verbs, none of them an instruction, and the Luna label was overruled.

The escalation needs both halves: the goal must read as mutating *and* mention a
consequential domain. This covers the first half. The second stays as it is on
purpose -- "credential", "password" and "production" appear here in their
read-only sense, but that guard only ever fires behind a write verb, and a leaf
that really does write to a credential path still belongs on Sol.
"""

import unittest

from model_router import (
    _is_consequential_spark_request,
    _is_spark_read_only_work,
    classify_request,
)


MODELS = {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"}

CFG = {
    "enabled": True,
    "provider": "openai-codex",
    "models": MODELS,
    "callable": {"luna": True, "spark": False, "terra": True, "sol": True,
                 "opus5": True, "sonnet5": True, "qwen": False},
    "effort": {"luna": "low", "terra": "medium", "sol": "medium"},
    "default_model": "terra",
    "thresholds": {"luna_max_chars": 700, "sol_min_chars": 3500},
}

# The goal as dispatched, shortened to the lines that carried the three verbs.
GOAL = (
    "[luna] Produce a factual, read-only evidence report on how the Szamlazz.hu "
    "(Szamla Agent) receipt issuing integration is wired in a Next.js/TypeScript "
    "repository, and on which runtime switch decides between sandbox and real "
    "issuing.\n\n"
    "Repository: absolute path /home/deepwell/booking-saas, git branch "
    "`development`. The relevant commit is 54e23a4ccd7f3c0b0e3796144d605325fa721348 "
    '("feat: autofill Hungarian billing cities", 2026-09-16), which is an ancestor '
    "of the current HEAD 93fd249. Read the working tree as-is.\n\n"
    "1. What does src/integrations/szamlazz-agent.ts actually implement today? Is "
    "it a real HTTP client, a typed stub, or an interface with no network call?\n"
    "2. Exactly which environment variable name(s) or credential fields would "
    "carry a Szamla Agent key / agent user-password, and where are they read?\n"
    "5. What does the runbook say about sandbox vs production issuing?\n\n"
    "Method: read-only inspection only. Do NOT modify, create, or delete any file. "
    "Do NOT open, print, echo, or dump any .env, .env.local, or credential file "
    "contents. Never print a secret value; if you encounter one, write [REDACTED]."
)


def chat_request(text):
    return {"model": MODELS["terra"], "messages": [{"role": "user", "content": text}]}


class ReceiptEvidenceGoalTests(unittest.TestCase):
    """The goal from the incident, end to end."""

    def test_the_dispatched_goal_reads_as_read_only(self):
        self.assertTrue(_is_spark_read_only_work(GOAL))

    def test_the_consequential_domain_guard_is_left_alone(self):
        """Still true -- and now unreachable, because the goal reads read-only."""
        self.assertTrue(_is_consequential_spark_request(GOAL))

    def test_the_luna_label_survives(self):
        decision = classify_request(
            chat_request(GOAL), 1, CFG, allow_plan_label_over_design=True
        )
        self.assertEqual(decision.tier, "luna")
        self.assertNotEqual(decision.reason, "consequential Luna task requires Sol")


class CommitCopulaTests(unittest.TestCase):
    """`commit` linked to its hash by a verb rather than juxtaposition."""

    def test_a_copula_before_the_hash_still_reads_as_a_reference(self):
        for text in (
            "[luna] Report the exports in /home/x. The relevant commit is 7abc123.",
            "[spark] Inspect /home/x. The base commit was 9def4567.",
            "[luna] Read /home/x and list the routes. commit: 54e23a4ccd7f3c0b.",
            "[spark] Map the DTOs in /home/x at commit 7abc123.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_an_instruction_to_commit_is_untouched(self):
        for text in (
            "[spark] Fix the parser and commit the change.",
            "[luna] Inspect /home/x, then commit is what you do last.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))


class RedactionPlaceholderTests(unittest.TestCase):
    """A write verb whose object is the report's own redaction marker."""

    def test_writing_a_placeholder_is_not_writing(self):
        for text in (
            "[luna] List the env var names in /home/x; if you find a value, "
            "write [REDACTED].",
            "[spark] Report the config keys in /home/x and replace it with [MASKED].",
            "[luna] Inspect /home/x and write `[redacted]` in place of any token.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_writing_something_real_is_still_writing(self):
        for text in (
            "[luna] Write the masked config to disk in /home/x.",
            "[spark] Create a redacted copy of .env.example in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))


class DescriptiveQuestionTests(unittest.TestCase):
    """A write verb describing what the code under inspection already does."""

    def test_asking_what_the_code_does_is_reading(self):
        for text in (
            "[luna] What does the nightly sync job update in /home/x?",
            "[spark] How does src/queue/worker.ts delete stale rows in /home/x?",
            "[luna] Which module does the checkout flow patch on success in /home/x?",
            "[spark] Report where does the importer write its output in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_an_instruction_is_not_a_question(self):
        for text in (
            "[luna] Update the nightly sync job in /home/x.",
            "[spark] Inspect the importer, then delete the stale rows in /home/x.",
            "[luna] Update the DTO in /home/x, which does matter for the API.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))


class VerbAsNounTests(unittest.TestCase):
    """A write verb naming what a read-only report is measured against.

    Observed 2026-09-21: a second `[luna]` evidence goal ran on Sol, same reason
    as the receipt one. It asked for "a concrete test case that would fail before
    the requested edit" and closed with "make no edits" -- escalated for the word
    "edit" in a clause that forbids editing, with "authorization" supplying the
    consequential half.
    """

    SECOND_INCIDENT = (
        "[luna] Map the exact existing issuer-mode behavior and its focused "
        "regression-test seam for the staff receipt member editor. Acceptance "
        "criteria: report the relevant source/test paths, the identifiers and "
        "state/authorization flow governing `Ceg / szalon` versus `Sajat "
        "kibocsato`, the smallest executable test command, and a concrete test "
        "case that would fail before the requested edit; make no edits."
    )

    def test_the_second_incident_goal_reads_as_read_only(self):
        self.assertTrue(_is_spark_read_only_work(self.SECOND_INCIDENT))

    def test_the_second_incident_keeps_its_luna_label(self):
        decision = classify_request(
            chat_request(self.SECOND_INCIDENT), 1, CFG, allow_plan_label_over_design=True
        )
        self.assertEqual(decision.tier, "luna")
        self.assertNotEqual(decision.reason, "consequential Luna task requires Sol")

    def test_the_consequential_half_is_still_there(self):
        """As with the receipt goal: true, and unreachable because it reads read-only."""
        self.assertTrue(_is_consequential_spark_request(self.SECOND_INCIDENT))

    def test_a_verb_after_a_temporal_preposition_is_a_noun(self):
        for text in (
            "[luna] Report a test case that fails before the requested edit in /home/x.",
            "[spark] Compare the behavior after the proposed change in /home/x.",
            "[luna] List what breaks prior to the planned migration in /home/x.",
            "[spark] State the behavior before the update in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_an_adjective_makes_it_a_noun_without_a_preposition(self):
        for text in (
            "[luna] Describe the requested edit and its blast radius in /home/x.",
            "[spark] Report the scope of the proposed rewrite in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_a_bare_determiner_is_still_an_instruction(self):
        """"Make the change" must not become read-only: the adjective is the token."""
        for text in (
            "[luna] Make the change in /home/x.",
            "[spark] Apply the requested edit to the router in /home/x.",
            "[luna] Do the update in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))

    def test_the_filler_never_walks_over_a_real_instruction(self):
        """An open filler swallowed the verb after the noun; a closed list cannot."""
        for text in (
            "[luna] Before the audit rewrite the config in /home/x.",
            "[spark] After the review implement the fix in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))


class PluralNounReadOnlyRegressionTests(unittest.TestCase):
    """Plural nouns after read verbs must not be mistaken for write commands."""

    def test_plural_nouns_remain_read_only_and_keep_the_luna_label(self):
        for text in (
            "[luna] Report the recent changes in /home/x/repo.",
            "[luna] List the files with uncommitted changes in /home/x/repo.",
            "[luna] Summarise the updates in the CHANGELOG of /home/x/repo.",
            "[luna] Check which fixes landed since v1.2 in /home/x/repo.",
            "[luna] Find the commits that touch __init__.py in /home/x/repo.",
            "[luna] Read the deletes and adds in the migration log of /home/x/repo.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))
                decision = classify_request(
                    chat_request(text), 1, CFG, allow_plan_label_over_design=True
                )
                self.assertEqual(decision.tier, "luna")


class NegatedVerbTests(unittest.TestCase):
    """An explicit refusal to write, in a shape the prohibition filter misses.

    The prohibition-clause filter knows "do not", "never" and "without". A goal
    saying "make no edits" kept its verb, and escaped only because the plural
    missed a singular pattern.
    """

    def test_refusing_to_write_is_not_writing(self):
        for text in (
            "[luna] Report the schema in /home/x; make no edits.",
            "[spark] Inspect the router in /home/x and make no edit.",
            "[luna] Map the flow in /home/x with no changes to the schema.",
            "[spark] Trace the importer in /home/x, no further edits.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_a_refusal_does_not_hide_the_instruction_beside_it(self):
        """Stripped as a phrase, not as a clause, exactly so this still reads as a write."""
        for text in (
            "[luna] Make no edits but rewrite the config in /home/x.",
            "[spark] No edits and rewrite the schema in /home/x.",
            "[luna] Make no changes except bump the version in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))


class NothingObjectTests(unittest.TestCase):
    """"Change nothing": the refusal written with the verb first.

    Observed 2026-09-27: a [luna] smoke check whose context said "No files in
    scope; change nothing." ran on Terra, reason "Luna is restricted to
    non-design read-only subtasks". The "no <verb>" phrase filter reads only the
    word order "no edits"; the verb-first "<verb> nothing" kept its write verb.
    """

    # The goal and context exactly as dispatched; Hermes joins them into the
    # child's first user message.
    DISPATCHED = (
        "[luna] Report the output of `git -C /home/x/repo log -1 --format=%h` "
        "and nothing else.\n\nCONTEXT:\nRead-only smoke test of routing. "
        "Worktree: /home/x/repo, branch feat/x, base a8334c4. No files in scope; "
        "change nothing. Verify: output is a short commit hash."
    )

    def test_refusing_with_nothing_is_not_writing(self):
        for text in (
            "[luna] Report the schema in /home/x; change nothing.",
            "[spark] Inspect the router in /home/x and modify nothing.",
            "[luna] Map the flow in /home/x, edit nothing, write nothing.",
            "[luna] List the tables in /home/x. Change absolutely nothing.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_is_spark_read_only_work(text))

    def test_the_dispatched_smoke_check_keeps_its_luna_label(self):
        decision = classify_request(
            chat_request(self.DISPATCHED), 1, CFG, allow_plan_label_over_design=True
        )
        self.assertEqual(decision.tier, "luna")

    def test_exception_clauses_after_nothing_are_writes(self):
        for text in (
            "[luna] Change nothing except the timeout in /home/x/config.yaml.",
            "[spark] Modify nothing but the config in /home/x.",
            "[luna] Edit nothing other than the version string in /home/x.",
            "[spark] Make no changes apart from the README in /home/x.",
            "[luna] Change nothing besides the test fixture in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))

    def test_widened_exception_words_after_nothing_are_writes(self):
        """A2-N2: save, aside from, else but, other then join the shared alternation."""
        for text in (
            "[luna] Change nothing save the timeout in /home/x.",
            "[luna] Modify nothing aside from the README in /home/x.",
            "[luna] Change nothing else but the README in /home/x.",
            "[luna] Change nothing other then the timeout in /home/x.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))

    def test_a_plain_but_after_a_refusal_stays_a_write(self):
        """A2-N3 ruling: a plain coordinating 'but' after a refusal is a write,
        the safe (Terra) direction. Pinned so a later reviewer does not flip it.
        """
        self.assertFalse(_is_spark_read_only_work(
            "[luna] Make no edits, but list the files in /home/x."
        ))

    def test_nothing_does_not_hide_a_real_instruction(self):
        """Only the verb directly refusing its object is stripped."""
        for text in (
            "[luna] Change nothing in the tests but rewrite the config in /home/x.",
            "[luna] Fix the parser in /home/x; nothing else.",
            "[spark] Edit the schema in /home/x so that nothing breaks.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_spark_read_only_work(text))


if __name__ == "__main__":
    unittest.main()
