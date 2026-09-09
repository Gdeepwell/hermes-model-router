"""A parent on a fallback account orchestrates like any other.

Observed 2026-09-09: with Codex exhausted, the orchestrator moved to Sonnet via the
Hermes fallback chain and then worked alone for 35 calls. `route_llm_request` returned
None for any model outside its own provider, so the parent never received the
delegation contract — and a second gate only recognised Sol and `default_model` as
orchestrators. Rewriting a model is provider-bound; orchestrating is not.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import _conductor_tier, route_llm_request
from model_router.test_model_router import CALLABLE, MODELS, chat_request

ACTIONABLE = (
    "Javitsd meg a naptar komponens hibajat a repoban: a 15 perces racs akkor is latszik, "
    "amikor minden 15 perces szolgaltatas inaktiv. Irj ra regressziot is."
)


def _cfg(temp_dir, **overrides):
    cfg = {
        "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": dict(CALLABLE),
        "default_model": "terra",
        "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
        "orchestration": {"enabled": True, "max_tasks": 3,
                          "path": str(Path(temp_dir) / "orchestration.jsonl")},
        "shadow": {"enabled": False},
        "fallbacks": {"terra": "luna", "luna": "sol"},
    }
    cfg.update(overrides)
    return cfg


def _delegating_request(model="claude-sonnet-5", tool_name="delegate_task"):
    request = chat_request(ACTIONABLE)
    # The body carries the model actually in use, as it does in a real call.
    request["model"] = model
    request["tools"] = [{
        "type": "function", "name": tool_name,
        "parameters": {"type": "object", "properties": {"goal": {"type": "string"},
                                                        "role": {"type": "string"}}},
    }]
    return request


class ExternalParentOrchestrationTests(unittest.TestCase):
    def _route(self, model, cfg, tool_name="delegate_task"):
        with patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision"), \
             patch("model_router._delegation_target_names", return_value=("sonnet5", "opus5", "qwen")):
            return route_llm_request(
                request=_delegating_request(model, tool_name), provider="anthropic", model=model,
                api_call_count=1, turn_id="external-parent-turn")

    def test_the_claude_code_mcp_prefix_still_counts_as_the_delegate_tool(self):
        """Anthropic OAuth requests are normalised for Claude Code, which renames every
        tool to mcp__<name>. Matching the bare name told a Claude parent it had no
        delegate_task and skipped its preflight — measured live before this fix."""
        with tempfile.TemporaryDirectory() as d:
            routed = self._route("claude-sonnet-5", _cfg(d), tool_name="mcp__delegate_task")
        self.assertIsNotNone(routed, "the mcp__-prefixed tool was not recognised")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])

    def test_a_request_with_no_delegate_tool_is_still_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self._route("claude-sonnet-5", _cfg(d), tool_name="read_file"))

    def test_a_parent_on_another_account_still_gets_the_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            routed = self._route("claude-sonnet-5", _cfg(d))
        self.assertIsNotNone(routed, "an external parent received no orchestration")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])

    def test_the_external_parents_model_is_never_rewritten(self):
        """The router cannot switch providers; only the instructions are ours to add."""
        with tempfile.TemporaryDirectory() as d:
            routed = self._route("claude-sonnet-5", _cfg(d))
        self.assertEqual(routed["request"].get("model", "claude-sonnet-5"), "claude-sonnet-5")

    def test_an_unknown_model_is_still_left_alone(self):
        """Only a configured delegation target counts; anything else is not ours."""
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self._route("some-other-vendor/model", _cfg(d)))

    def test_orchestration_disabled_still_means_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, orchestration={"enabled": False, "path": str(Path(d) / "o.jsonl")})
            self.assertIsNone(self._route("claude-sonnet-5", cfg))


class ConductorTierTests(unittest.TestCase):
    """The forced conductor must not be pinned to an account that has run out."""

    def test_the_code_chain_outranks_the_configured_default(self):
        """Measured need: six conductors in a row ran to their iteration cap on the
        Codex account, spending 36% of a five-hour limit before a leaf did real work.
        The operator's own order decides where planning happens."""
        cfg = {"models": MODELS, "callable": {**CALLABLE, "opus5": True},
               "default_model": "terra", "preferences": {"code": ["opus5", "terra"]}}
        self.assertEqual(_conductor_tier(cfg), "opus5")

    def test_an_uncallable_preference_falls_through_to_the_next(self):
        cfg = {"models": MODELS, "callable": {**CALLABLE, "opus5": False},
               "default_model": "terra", "preferences": {"code": ["opus5", "terra"]}}
        self.assertEqual(_conductor_tier(cfg), "terra")

    def test_without_a_code_chain_the_default_still_wins(self):
        cfg = {"models": MODELS, "callable": dict(CALLABLE), "default_model": "terra",
               "preferences": {"design": ["sol"]}}
        self.assertEqual(_conductor_tier(cfg), "terra")

    def test_the_configured_default_is_used_when_callable(self):
        self.assertEqual(_conductor_tier({"models": MODELS, "callable": dict(CALLABLE),
                                          "default_model": "terra"}), "terra")

    def test_an_exhausted_default_falls_through_its_chain(self):
        cfg = {"models": MODELS, "callable": {**CALLABLE, "terra": False},
               "default_model": "terra", "fallbacks": {"terra": "luna"}}
        self.assertEqual(_conductor_tier(cfg), "luna")

    def test_a_chain_of_exhausted_tiers_reaches_the_first_callable_one(self):
        cfg = {"models": MODELS, "callable": {**CALLABLE, "terra": False, "luna": False},
               "default_model": "terra", "fallbacks": {"terra": "luna", "luna": "sol"}}
        self.assertEqual(_conductor_tier(cfg), "sol")

    def test_with_no_chain_any_callable_tier_beats_a_dead_default(self):
        cfg = {"models": MODELS, "callable": {**{t: False for t in MODELS}, "sol": True},
               "default_model": "terra"}
        self.assertEqual(_conductor_tier(cfg), "sol")

    def test_nothing_callable_degrades_to_the_configured_default(self):
        cfg = {"models": MODELS, "callable": {t: False for t in MODELS}, "default_model": "terra"}
        self.assertEqual(_conductor_tier(cfg), "terra")


class ContractTruthfulnessTests(unittest.TestCase):
    def test_the_conductor_is_not_told_claude_is_unavailable(self):
        """That claim was disproven on 2026-09-09; leaving it in suppressed the very
        delegation the plugin exists to produce."""
        with tempfile.TemporaryDirectory() as d:
            routed = ExternalParentOrchestrationTests()._route("claude-sonnet-5", _cfg(d))
        contract = routed["request"]["tools"][0]["parameters"]["properties"]["context"]["enum"][0]
        self.assertNotIn("does not fund third-party API access", contract)
        self.assertNotIn("Claude delegation target is switched off", contract)


if __name__ == "__main__":
    unittest.main()
