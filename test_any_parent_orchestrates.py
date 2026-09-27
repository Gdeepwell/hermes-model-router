"""Every parent tier orchestrates, and by default plans on its own account.

Observed 2026-09-27: a session switched to Grok (2026-09-25) ran every call on
Grok alone. The root pin kept Grok as the parent, but the forced preflight only
fired for Sol and ``default_model``, and Grok -- a tier in ``models`` -- never
reached the external-parent branch that would have preflighted it. The
orchestration log had no entry at all for that session, not even a skip.

Orchestration was also Codex-shaped: even a preflighted parent handed planning
to ``default_model``. The conductor now follows the parent unless the operator
pins one or switches ``orchestration.conductor_follows_parent`` off.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_router as router
from model_router import _conductor_tier, _orchestration_skip_reason, route_llm_request

MODELS = {
    "luna": "gpt-6-luna",
    "spark": "gpt-5.3-codex-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-6-sol",
    "grok": "grok-4.7",
}
PROVIDERS = {"luna": "openai-codex", "spark": "openai-codex", "terra": "openai-codex",
             "sol": "openai-codex", "grok": "xai-oauth"}
TARGETS = {"grok": {"provider": "xai-oauth", "model": "grok-4.7"},
           "luna": {"provider": "openai-codex", "model": "gpt-6-luna"},
           "terra": {"provider": "openai-codex", "model": "gpt-5.6-terra"}}

LONG_TASK = (
    "A publikus foglalasi oldalon a fizetesi lehetosegek kozul egyik sincs alapertelmezetten "
    "kivalasztva. Legyen az elso engedelyezett fizetesi mod bekattintva, amikor az oldal betolt, "
    "irj ra regresszios tesztet, es ellenorizd, hogy a kivalasztas a mentett beallitasokkal is "
    "helyesen mukodik. " * 6
)


def _cfg(temp_dir, **orchestration):
    return {
        "enabled": True, "workflow": "codex", "provider": "openai-codex",
        "models": MODELS, "tier_providers": PROVIDERS,
        "callable": {**{tier: True for tier in MODELS}, "spark": False},
        "default_model": "terra",
        "effort": {"terra": "medium", "sol": "medium", "luna": "low", "grok": "medium"},
        "session_policy": {"pin_root_parent": True},
        "orchestration": {"enabled": True, "min_chars": 1000, "max_tasks": 2,
                          "path": str(Path(temp_dir) / "orchestration.jsonl"), **orchestration},
        "shadow": {"enabled": False},
        "fallbacks": {"grok": "terra", "sol": "terra"},
    }


def _request(model, text):
    return {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        "tools": [{
            "type": "function", "name": "delegate_task",
            "parameters": {"type": "object", "properties": {
                "goal": {"type": "string"}, "role": {"type": "string"},
                "model": {"type": "string", "enum": sorted(TARGETS)},
            }},
        }],
    }


class _Patched(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.cfg = _cfg(self._dir.name)
        for target, value in (
            ("_delegation_targets_detail", TARGETS),
            ("_hermes_delegation_target_names", tuple(sorted(TARGETS))),
            ("_host_delegation_limits", {"conductor_available": True}),
        ):
            patcher = patch.object(router, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(router, "_load_config", side_effect=lambda: self.cfg)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(router, "_log_decision")
        patcher.start()
        self.addCleanup(patcher.stop)

    def route(self, tier, text=LONG_TASK, provider=None):
        model = MODELS[tier]
        return route_llm_request(
            request=_request(model, text), model=model, provider=provider or PROVIDERS[tier],
            api_call_count=1, turn_id=f"any-parent-{tier}", platform="cli")


class AnyParentGetsThePreflightTests(_Patched):
    def assert_preflighted(self, routed, conductor):
        self.assertIsNotNone(routed, "the parent received no orchestration")
        request = routed["request"]
        schema = request["tools"][0]["parameters"]["properties"]
        self.assertEqual(len(request["tools"]), 1, "the preflight must narrow to delegate_task")
        self.assertEqual(request["tool_choice"], "required")
        self.assertEqual(schema["role"]["enum"], ["orchestrator"])
        self.assertEqual(schema["model"]["enum"], [conductor])
        self.assertIn(f"a goal beginning with [{conductor}]", request["input"][-1]["content"][-1]["text"])

    def test_a_grok_parent_orchestrates_and_plans_on_grok(self):
        self.assert_preflighted(self.route("grok"), "grok")

    def test_the_grok_parent_is_not_rewritten(self):
        routed = self.route("grok")
        self.assertEqual(routed["request"]["model"], "grok-4.7")

    def test_a_luna_parent_orchestrates_too(self):
        self.assert_preflighted(self.route("luna"), "luna")

    def test_a_sol_parent_without_its_bridge_uses_the_common_contract(self):
        """Sol's own design preflight needs the Opus bridge; without it Sol used to
        be skipped outright (`sol_preflight_disabled`) and worked alone."""
        self.assert_preflighted(self.route("sol"), "sol")

    def test_a_terra_parent_is_unchanged(self):
        self.assert_preflighted(self.route("terra"), "terra")

    def test_a_short_grok_turn_skips_the_preflight_like_terra(self):
        routed = self.route("grok", text="csinald meg")
        tools = (routed or {}).get("request", {}).get("tools") or []
        self.assertFalse(any((t.get("parameters") or {}).get("properties", {}).get("role", {}).get("enum")
                             for t in tools))

    def test_the_log_names_the_real_conductor(self):
        self.route("grok")
        log = Path(self.cfg["orchestration"]["path"]).read_text(encoding="utf-8")
        events = [json.loads(line) for line in log.splitlines()]
        forced = [e for e in events if e.get("event") == "preflight_forced"]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0]["parent_model"], "grok")
        self.assertEqual(forced[0]["preflight_owner"], "grok")

    def test_the_conductor_follows_the_parent_can_be_switched_off(self):
        self.cfg = _cfg(self._dir.name, conductor_follows_parent=False)
        self.assert_preflighted(self.route("grok"), "terra")


class ConductorFollowsParentTests(unittest.TestCase):
    CFG = {"models": MODELS, "tier_providers": PROVIDERS,
           "callable": {**{tier: True for tier in MODELS}, "opus5": True},
           "default_model": "terra"}

    def test_the_parent_wins_over_the_default(self):
        self.assertEqual(_conductor_tier(self.CFG, parent="grok"), "grok")

    def test_a_pinned_conductor_still_wins_over_the_parent(self):
        cfg = {**self.CFG, "orchestration": {"conductor": "sol"}}
        self.assertEqual(_conductor_tier(cfg, parent="grok"), "sol")

    def test_an_uncallable_parent_falls_back_to_the_default(self):
        cfg = {**self.CFG, "callable": {**self.CFG["callable"], "grok": False}}
        self.assertEqual(_conductor_tier(cfg, parent="grok"), "terra")

    def test_spark_never_conducts(self):
        self.assertEqual(_conductor_tier(self.CFG, parent="spark"), "terra")

    def test_no_parent_keeps_the_old_answer(self):
        self.assertEqual(_conductor_tier(self.CFG), "terra")

    def test_an_off_provider_parent_needs_a_host_target(self):
        """A goal prefix cannot move a planner to xAI; without a `model` field the
        conductor stays on the delegate_task provider."""
        request = _request("grok-4.7", LONG_TASK)
        del request["tools"][0]["parameters"]["properties"]["model"]
        with patch.object(router, "_delegation_targets_detail", return_value=TARGETS):
            self.assertEqual(_conductor_tier(self.CFG, request, parent="grok"), "terra")


class SkipReasonTests(_Patched):
    def test_no_known_tier_is_rejected_as_a_non_orchestrator(self):
        for tier in ("grok", "luna", "sol", "terra"):
            with self.subTest(tier=tier):
                decision = router._decision(tier, "root", self.cfg)
                reason = _orchestration_skip_reason(
                    {"request": _request(MODELS[tier], LONG_TASK), "api_call_count": 1,
                     "turn_id": f"skip-{tier}"}, self.cfg, decision)
                self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
