"""Per-work-kind model preferences.

The router chooses among its own provider's tiers; Claude and Qwen reach work only
through delegation. These cover both halves of that split, and the rule that matters
most in practice: configuring nothing must change nothing.
"""

import unittest
from unittest.mock import patch

from model_router import (
    WORK_KINDS,
    RouteDecision,
    _apply_preferences,
    _preference_list,
    _preferred_route,
    _preferred_target,
    classify_request,
)


BASE_CFG = {
    "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
    "callable": {"luna": True, "spark": True, "terra": True, "sol": True,
                 "opus5": True, "sonnet5": True, "qwen": False},
    "effort": {},
    "default_model": "terra",
}


def _cfg(**overrides):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CFG.items()}
    cfg.update(overrides)
    return cfg


def _request(text):
    return {"messages": [{"role": "user", "content": text}]}


class PreferenceListTests(unittest.TestCase):
    def test_unset_kind_yields_no_preference(self):
        """Unset means "keep the built-in route" — never an empty preference."""
        self.assertEqual(_preference_list("design", _cfg()), ())

    def test_entries_are_normalised_and_deduplicated(self):
        cfg = _cfg(preferences={"code": ["  Sol ", "sol", "TERRA"]})
        self.assertEqual(_preference_list("code", cfg), ("sol", "terra"))

    def test_malformed_preferences_are_ignored_rather_than_fatal(self):
        for value in (None, "sol", 3, {"a": 1}):
            with self.subTest(value=value):
                self.assertEqual(_preference_list("code", _cfg(preferences={"code": value})), ())

    def test_empty_kind_has_no_preference(self):
        self.assertEqual(_preference_list("", _cfg(preferences={"": ["sol"]})), ())


class PreferredRouteTests(unittest.TestCase):
    def test_first_callable_routable_tier_wins(self):
        cfg = _cfg(preferences={"code": ["sol", "terra"]})
        self.assertEqual(_preferred_route("code", cfg), "sol")

    def test_a_disabled_tier_is_skipped(self):
        cfg = _cfg(preferences={"code": ["sol", "terra"]})
        cfg["callable"]["sol"] = False
        self.assertEqual(_preferred_route("code", cfg), "terra")

    def test_external_targets_are_not_routable(self):
        """The router rewrites models inside one provider; it cannot switch provider."""
        cfg = _cfg(preferences={"design": ["opus5", "sol"]})
        self.assertEqual(_preferred_route("design", cfg), "sol")

    def test_no_callable_entry_yields_none(self):
        cfg = _cfg(preferences={"code": ["sol"]})
        cfg["callable"]["sol"] = False
        self.assertIsNone(_preferred_route("code", cfg))


class PreferredTargetTests(unittest.TestCase):
    def test_leading_external_entry_becomes_delegation_advice(self):
        cfg = _cfg(preferences={"design": ["opus5", "sol"]})
        self.assertEqual(_preferred_target("design", cfg), "opus5")

    def test_a_disabled_external_target_is_skipped(self):
        cfg = _cfg(preferences={"review": ["qwen", "sonnet5", "terra"]})
        self.assertEqual(_preferred_target("review", cfg), "sonnet5")

    def test_routable_only_list_advises_nothing(self):
        cfg = _cfg(preferences={"code": ["sol", "terra"]})
        self.assertEqual(_preferred_target("code", cfg), "")


class ApplyPreferencesTests(unittest.TestCase):
    @staticmethod
    def _decision(tier="terra", kind="code", mandatory=False):
        return RouteDecision(tier=tier, model=f"gpt-{tier}", reason="built-in",
                             kind=kind, mandatory=mandatory)

    def test_no_configuration_leaves_the_decision_untouched(self):
        d = self._decision()
        self.assertIs(_apply_preferences(d, _cfg()), d)

    def test_preference_replaces_the_tier(self):
        cfg = _cfg(preferences={"code": ["sol"]})
        self.assertEqual(_apply_preferences(self._decision(), cfg).tier, "sol")

    def test_preference_overrides_a_mandatory_policy_route(self):
        """The user chose to own this mapping; the built-in policy is the default only."""
        cfg = _cfg(preferences={"design": ["terra"]})
        result = _apply_preferences(self._decision("sol", "design", mandatory=True), cfg)
        self.assertEqual(result.tier, "terra")
        self.assertFalse(result.mandatory)

    def test_external_preference_keeps_the_route_and_adds_advice(self):
        cfg = _cfg(preferences={"design": ["opus5", "sol"]})
        result = _apply_preferences(self._decision("sol", "design", mandatory=True), cfg)
        self.assertEqual(result.tier, "sol")
        self.assertEqual(result.prefer_target, "opus5")

    def test_an_untagged_decision_is_never_touched(self):
        cfg = _cfg(preferences={"code": ["sol"]})
        d = self._decision(kind="")
        self.assertIs(_apply_preferences(d, cfg), d)


class EndToEndTests(unittest.TestCase):
    def test_design_work_can_be_moved_off_sol(self):
        cfg = _cfg(preferences={"design": ["luna"]})
        d = classify_request(_request("Csinald meg a UI-t, a gombok legyenek kerekek"), config=cfg)
        self.assertEqual(d.kind, "design")
        self.assertEqual(d.tier, "luna")

    def test_design_work_stays_on_sol_without_a_preference(self):
        d = classify_request(_request("Csinald meg a UI-t, a gombok legyenek kerekek"), config=_cfg())
        self.assertEqual(d.tier, "sol")
        self.assertTrue(d.mandatory)

    def test_review_is_addressable_and_defaults_to_terra(self):
        d = classify_request(_request("nezd at a kodot es reviewold"), config=_cfg())
        self.assertEqual((d.kind, d.tier), ("review", "terra"))

    def test_review_can_be_pointed_at_an_external_account(self):
        cfg = _cfg(preferences={"review": ["sonnet5", "terra"]})
        d = classify_request(_request("nezd at a kodot es reviewold"), config=cfg)
        self.assertEqual(d.prefer_target, "sonnet5")
        self.assertEqual(d.tier, "terra")  # route unchanged: delegation carries it


class WorkKindTests(unittest.TestCase):
    def test_every_documented_kind_is_reachable(self):
        self.assertIn("design", WORK_KINDS)
        self.assertIn("review", WORK_KINDS)
        self.assertIn("explore", WORK_KINDS)
        self.assertEqual(len(set(WORK_KINDS)), len(WORK_KINDS))


if __name__ == "__main__":
    unittest.main()


class GuidanceSentenceTests(unittest.TestCase):
    """The conductor is the only place a cross-provider preference can be realised."""

    def test_preferences_are_stated_to_the_conductor(self):
        from model_router import _preference_sentence

        cfg = _cfg(preferences={"design": ["opus5", "sol"], "review": ["sonnet5", "terra"]})
        sentence = _preference_sentence(["opus5", "sonnet5", "luna"], cfg)

        self.assertIn("design -> model:opus5", sentence)
        self.assertIn("review -> model:sonnet5", sentence)

    def test_a_target_that_is_not_offered_is_not_advised(self):
        """Advising a switched-off account produces a leaf that never runs."""
        from model_router import _preference_sentence

        cfg = _cfg(preferences={"design": ["opus5", "sol"]})
        self.assertEqual(_preference_sentence(["luna", "terra"], cfg), "")

    def test_routable_only_preferences_say_nothing_here(self):
        """Those are routes, not delegation advice — the router applies them itself."""
        from model_router import _preference_sentence

        cfg = _cfg(preferences={"code": ["sol", "terra"]})
        self.assertEqual(_preference_sentence(["opus5", "luna"], cfg), "")
