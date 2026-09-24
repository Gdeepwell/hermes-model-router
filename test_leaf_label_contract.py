"""The conductor must be told to label every leaf goal, not only Spark and Sol ones.

Regression for 2026-09-18: two read-only discovery leaves went out unlabelled and
both opened on Sol from their first call, vetoed by the design gate on the bare
word "ui" in "UI components" / "self-service UI".
"""

import unittest
from unittest.mock import patch

from model_router import (
    _leaf_label_contract,
    _prepare_orchestration_delegation,
    _read_only_delegation_clause,
    _read_only_leaf_tier,
    classify_request,
)


MODELS = {
    "luna": "gpt-6-luna",
    "spark": "gpt-5.3-codex-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-6-sol",
}

READ_ONLY_UI_GOAL = (
    "Read-only source mapping for the STAFF issuer-mode correction. Identify exact "
    "issuer-mode route, RBAC/issuer validation logic, UI components and existing "
    "route/UI tests. Report exact paths, behavioral gap, and test commands."
)


def config(**overrides):
    result = {
        "enabled": True,
        "provider": "openai-codex",
        "models": MODELS,
        "default_model": "terra",
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True},
        "effort": {"luna": "low", "spark": "medium", "terra": "medium", "sol": "medium"},
        "orchestration": {"enabled": True, "max_tasks": 2},
    }
    result.update(overrides)
    return result


def delegation_request():
    return {
        "messages": [{"role": "user", "content": "javitsd a staff issuer modot"}],
        "tools": [{
            "type": "function",
            "name": "delegate_task",
            "parameters": {
                "type": "object",
                "properties": {"goal": {"type": "string"}, "model": {"type": "string"}},
                "required": ["goal"],
            },
        }],
    }


def preflight_texts(cfg):
    routed = _prepare_orchestration_delegation(delegation_request(), "plan-1", 2, cfg)
    instruction = routed["messages"][-1]["content"]
    context = routed["tools"][0]["parameters"]["properties"]["context"]["enum"][0]
    return instruction, context


class LeafLabelContractTests(unittest.TestCase):
    def test_label_rule_is_universal_and_states_its_consequence(self):
        contract = _leaf_label_contract(config())
        self.assertIn("Begin every worker goal", contract)
        # The failure was not that the rule was wrong but that it read as advice for
        # two named tiers. Both halves have to be there: the rule and the cost.
        self.assertIn("not optional", contract)
        self.assertIn("forces the leaf onto Sol", contract)

    def test_label_rule_offers_only_tiers_the_operator_left_callable(self):
        contract = _leaf_label_contract(config(
            callable={"luna": True, "spark": False, "terra": True, "sol": True},
        ))
        self.assertNotIn("[spark]", contract)
        for tier in ("[luna]", "[sol]", "[terra]"):
            self.assertIn(tier, contract)

    def test_read_only_leaf_falls_back_to_luna_when_spark_is_off(self):
        cfg = config(callable={"luna": True, "spark": False, "terra": True, "sol": True})
        self.assertEqual(_read_only_leaf_tier(cfg), "luna")
        clause = _read_only_delegation_clause(cfg)
        self.assertIn("[luna]", clause)
        self.assertIn("model:luna", clause)
        self.assertNotIn("Spark", clause)

    def test_read_only_clause_is_dropped_when_no_read_only_tier_is_callable(self):
        cfg = config(callable={"luna": False, "spark": False, "terra": True, "sol": True})
        self.assertEqual(_read_only_leaf_tier(cfg), "")
        self.assertEqual(_read_only_delegation_clause(cfg), "")

    def test_preflight_never_names_a_tier_the_conductor_cannot_spawn(self):
        cfg = config(callable={"luna": True, "spark": False, "terra": True, "sol": True})
        instruction, context = preflight_texts(cfg)
        for text in (instruction, context):
            self.assertNotIn("[spark]", text)
            self.assertNotIn("model:spark", text)
            self.assertIn("Begin every worker goal", text)

    def test_conductor_contract_marker_survives_the_inserted_rule(self):
        # _ROUTER_CONTRACT_MARKERS strips the contract before classification on this
        # exact substring; splitting it would make every leaf classify its rulebook.
        _, context = preflight_texts(config())
        self.assertIn("planning conductor.", context)

    def test_labelled_read_only_ui_leaf_no_longer_lands_on_sol(self):
        cfg = config()
        with patch("model_router._log_decision"):
            unlabelled = classify_request(
                {"messages": [{"role": "user", "content": READ_ONLY_UI_GOAL}]},
                api_call_count=1, config=cfg, allow_plan_label_over_design=True,
            )
            labelled = classify_request(
                {"messages": [{"role": "user", "content": f"[luna] {READ_ONLY_UI_GOAL}"}]},
                api_call_count=1, config=cfg, allow_plan_label_over_design=True,
            )
        # The gate itself is unchanged: an unlabelled goal still reads as design.
        self.assertEqual(unlabelled.tier, "sol")
        self.assertEqual(labelled.tier, "luna")
        self.assertEqual(labelled.reason, "explicit [luna] override")
