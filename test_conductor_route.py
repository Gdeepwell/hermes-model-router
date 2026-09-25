"""The conductor's tier must be a route, not only a goal prefix.

Observed 2026-09-15: a "[qwen] Plan and conduct ..." conductor ran 24 calls on
Terra. The preflight names the conductor's tier in prose and in the goal prefix,
but a prefix only renames a model inside the default provider -- so with no
`model` parameter the planner was created on the delegation default, which is
the account the delegation exists to spare.

The prose contract cannot close this by itself: `_model_param_contract` lists
the *other* targets to spread work across, so the conductor's own tier is the
one name it never offers.
"""

import unittest

from model_router import (
    _conductor_tier,
    _load_config,
    _misdispatched_external_label,
    _model_param_contract,
    _prepare_orchestration_delegation,
)


def _request_with_delegate_tool():
    return {
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "csinald meg"}]}],
        "tools": [
            {
                "type": "function",
                "name": "delegate_task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "model": {
                            "type": "string",
                            "enum": ["luna", "opus5", "qwen", "sol", "sonnet5", "terra"],
                        },
                    },
                    "required": ["goal"],
                },
            }
        ],
    }


class ConductorRouteTests(unittest.TestCase):
    def setUp(self):
        self.cfg = _load_config()
        self.models = self.cfg.get("models") or {}

    def test_the_planner_schema_pins_the_conductor_route(self):
        # Asserted against _conductor_tier rather than a literal. The claim here is
        # "the schema carries whatever route the conductor resolves to"; hardcoding
        # one tier made this fail the moment the operator's config named another,
        # reporting a red test where the only thing that had changed was a setting.
        conductor = _conductor_tier(self.cfg)
        routed = _prepare_orchestration_delegation(
            _request_with_delegate_tool(), "plan-test", 3, cfg=self.cfg
        )
        schema = routed["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["model"]["enum"], [conductor])
        self.assertIn("model", schema["required"])

    def test_the_prose_contract_still_omits_the_conductors_own_tier(self):
        # Not a bug to fix here: that list is "other targets to spread across".
        # It is the reason the schema has to carry the conductor's own route.
        conductor = _conductor_tier(self.cfg)
        self.assertNotIn(f"targets: {conductor}", _model_param_contract(conductor, self.cfg))


class QwenMisdispatchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = _load_config()
        self.models = self.cfg.get("models") or {}

    def test_a_qwen_goal_running_on_the_default_provider_is_a_misdispatch(self):
        self.assertEqual(
            _misdispatched_external_label(
                "[qwen] Plan and conduct the booking work", self.models["terra"], self.cfg
            ),
            "qwen",
        )

    def test_a_qwen_goal_already_on_qwen_is_redundant_not_wrong(self):
        self.assertEqual(
            _misdispatched_external_label(
                "[qwen] Complete the DTO pass", self.models["qwen"], self.cfg
            ),
            "",
        )

    def test_the_claude_targets_are_unchanged(self):
        self.assertEqual(
            _misdispatched_external_label("[opus5] review this", self.models["terra"], self.cfg),
            "opus5",
        )
        self.assertEqual(
            _misdispatched_external_label("[sonnet] review this", self.models["sol"], self.cfg),
            "sonnet5",
        )

    def test_a_real_tier_prefix_is_not_a_misdispatch(self):
        self.assertEqual(
            _misdispatched_external_label("[sol] implement", self.models["terra"], self.cfg), ""
        )

class HostCapabilityTests(unittest.TestCase):
    def test_flat_host_skips_forced_conductor(self):
        from unittest.mock import patch
        import model_router as router
        from model_router.test_external_orchestrator import _cfg, _delegating_request
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d)
            kwargs = {'request': _delegating_request(), 'api_call_count': 1, 'turn_id': 'flat-host'}
            decision = router.RouteDecision('sonnet5', 'claude-sonnet-5', 'external', 'external')
            with patch.object(router, '_delegation_target_names', return_value=('sonnet5',)), \
                 patch.object(router, '_host_delegation_limits', return_value={'conductor_available': False}):
                self.assertIn('host_has_no_conductor_depth', router._orchestration_skip_reason(kwargs, cfg, decision))
            with patch.object(router, '_delegation_target_names', return_value=('sonnet5',)), \
                 patch.object(router, '_host_delegation_limits', return_value={'conductor_available': True}):
                self.assertIsNone(router._orchestration_skip_reason(kwargs, cfg, decision))

    def test_limits_follow_host_capability_not_legacy_role(self):
        from unittest.mock import patch
        import model_router as router
        with patch('tools.delegate_tool_config._get_max_spawn_depth', return_value=1):
            self.assertFalse(router._host_delegation_limits()['conductor_available'])
        with patch('tools.delegate_tool_config._get_max_spawn_depth', return_value=2), \
             patch('tools.delegate_tool_config._get_orchestrator_enabled', return_value=False):
            self.assertFalse(router._host_delegation_limits()['conductor_available'])
