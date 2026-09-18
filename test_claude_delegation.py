"""The Claude wing: Claude workers next to the Codex workforce.

Hermes's delegate_task has one route per process, pinned to Codex on this host.
delegate_claude reaches Claude by calling the same delegate_task with a per-call
route pinned to the anthropic provider. These tests cover the wing without any
network or model call.
"""

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_router import claude_delegation
from model_router.claude_delegation import (
    TARGET_FOR_TIER,
    TIER_FOR_TARGET,
    registration_block,
    target_for_model,
    target_names,
    tier_model,
    delegation_config,
)

CLAUDE_DELEGATION = {
    "enabled": True,
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"},
    "default_tier": "sonnet",
}


def _cfg(**overrides):
    cfg = {
        "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True,
                     "opus5": True, "sonnet5": True, "haiku": True},
        "tier_providers": {"opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic",
                          "terra": "openai-codex", "sol": "openai-codex",
                          "spark": "openai-codex", "luna": "openai-codex"},
        "fallbacks": {"opus5": "sol"},
        "peer_groups": {"heavy": ["terra", "opus5", "sonnet5"], "light": ["luna", "spark", "haiku"]},
        "default_model": "terra",
        "claude_delegation": json.loads(json.dumps(CLAUDE_DELEGATION)),
        "usage_guard": {"cache_seconds": 300, "accounts": {
            "anthropic": {"soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
        }},
    }
    cfg.update(overrides)
    return cfg


class DelegationConfigTests(unittest.TestCase):
    def test_a_config_without_the_block_leaves_the_wing_off(self):
        """Configuring nothing must change nothing."""
        self.assertFalse(delegation_config({})["enabled"])
        self.assertFalse(delegation_config(None)["enabled"])

    def test_a_partial_block_keeps_the_other_defaults(self):
        settings = delegation_config({"claude_delegation": {"enabled": True, "default_tier": "haiku"}})
        self.assertEqual(settings["default_tier"], "haiku")
        self.assertEqual(settings["tiers"]["opus"], "claude-opus-5")
        self.assertNotIn("usage_guard", settings)

    def test_names_map_both_ways(self):
        self.assertEqual(TARGET_FOR_TIER, {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"})
        self.assertEqual(TIER_FOR_TARGET["opus5"], "opus")

    def test_tier_model(self):
        self.assertEqual(tier_model("haiku", _cfg()), "claude-haiku-4-5-20251001")
        self.assertEqual(tier_model("gpt", _cfg()), "")

    def test_target_names_skip_a_tier_without_a_model(self):
        cfg = _cfg()
        self.assertEqual(target_names(cfg), ("haiku", "opus5", "sonnet5"))
        cfg["claude_delegation"]["tiers"]["haiku"] = ""
        self.assertEqual(target_names(cfg), ("opus5", "sonnet5"))

    def test_a_model_maps_back_to_its_target(self):
        self.assertEqual(target_for_model("claude-haiku-4-5-20251001", _cfg()), "haiku")
        self.assertIsNone(target_for_model("gpt-terra", _cfg()))
        self.assertIsNone(target_for_model("", _cfg()))


class RegistrationBlockTests(unittest.TestCase):
    def test_a_disabled_wing_does_not_register(self):
        cfg = _cfg()
        cfg["claude_delegation"]["enabled"] = False
        self.assertIn("enabled", registration_block(cfg))

    def test_every_claude_target_switched_off_does_not_register(self):
        cfg = _cfg()
        for target in ("haiku", "sonnet5", "opus5"):
            cfg["callable"][target] = False
        self.assertIn("switched off", registration_block(cfg))

    def test_a_host_without_the_api_does_not_register(self):
        with patch.object(claude_delegation, "host_check", return_value=(False, "delegate_task lacks credentials_cfg")):
            self.assertIn("credentials_cfg", registration_block(_cfg()))

    def test_an_enabled_wing_on_a_capable_host_registers(self):
        with patch.object(claude_delegation, "host_check", return_value=(True, "")):
            self.assertEqual(registration_block(_cfg()), "")


from model_router import usage_guard  # noqa: E402
from model_router.usage_guard import GuardOutcome, Reading  # noqa: E402


def _reading(weekly, session=10.0, fetched_at=None):
    return Reading(weekly, session, None, None, time.time() if fetched_at is None else fetched_at)


from model_router.claude_delegation import build_schema, handle_delegate_claude  # noqa: E402


def _fake_host(parent, result=None):
    """A recording delegate_task and a fixed active parent."""
    calls = []

    def delegate_task(**kwargs):
        calls.append(kwargs)
        return json.dumps(result if result is not None else {"status": "dispatched", "delegation_id": "d1"})

    return calls, (lambda: (delegate_task, lambda: parent))


class SchemaTests(unittest.TestCase):
    def test_the_schema_mirrors_delegate_task_plus_a_tier(self):
        with patch.object(claude_delegation, "_independent_completions", return_value=False):
            schema = build_schema(_cfg())
        self.assertEqual(schema["name"], "delegate_claude")
        properties = schema["parameters"]["properties"]
        self.assertEqual(properties["tier"]["enum"], ["haiku", "sonnet", "opus"])
        self.assertIn("goal", properties["tasks"]["items"]["properties"])
        self.assertEqual(properties["tasks"]["items"]["required"], ["goal"])
        self.assertNotIn("group", properties["tasks"]["items"]["properties"])
        self.assertNotIn("action", properties)
        self.assertIn("delegate_task", schema["description"])


class HandlerTests(unittest.TestCase):
    def setUp(self):
        usage_guard._reset_cache()
        self.addCleanup(usage_guard._reset_cache)

    def _call(self, args, *, cfg=None, parent=None, usage=40.0, result=None):
        parent = parent if parent is not None else SimpleNamespace(_delegate_depth=0)
        calls, host = _fake_host(parent, result)
        reading = None if usage is None else _reading(usage)
        with patch("model_router._load_config", return_value=cfg or _cfg()), \
             patch.object(claude_delegation, "_host", host), \
             patch.object(usage_guard, "read", return_value=reading):
            raw = handle_delegate_claude(args)
        return json.loads(raw), calls

    def test_a_call_runs_on_a_pinned_anthropic_route(self):
        payload, calls = self._call({"tasks": [{"goal": "g", "acp_command": "x"}], "tier": "haiku"})
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["credentials_cfg"], {"provider": "anthropic",
                                                   "model": "claude-haiku-4-5-20251001",
                                                   "fallback_providers": []})
        self.assertTrue(call["background"])
        self.assertEqual(call["tasks"], [{"goal": "g"}])
        self.assertEqual(payload["claude_tier"], "haiku")
        self.assertEqual(payload["delegation_id"], "d1")

    def test_the_default_tier_applies_when_none_is_given(self):
        _payload, calls = self._call({"tasks": [{"goal": "g"}]})
        self.assertEqual(calls[0]["credentials_cfg"]["model"], "claude-sonnet-5")

    def test_an_orchestrator_child_waits_for_its_workers(self):
        """Same rule as Hermes: a child at depth > 0 needs results within its turn."""
        _payload, calls = self._call({"tasks": [{"goal": "g"}]}, parent=SimpleNamespace(_delegate_depth=1))
        self.assertFalse(calls[0]["background"])

    def test_the_caller_is_the_parent(self):
        parent = SimpleNamespace(_delegate_depth=0)
        _payload, calls = self._call({"tasks": [{"goal": "g"}]}, parent=parent)
        self.assertIs(calls[0]["parent_agent"], parent)

    def test_an_unknown_tier_is_refused(self):
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "gpt"})
        self.assertIn("Unknown tier", payload["error"])
        self.assertEqual(calls, [])

    def test_no_active_parent_is_refused(self):
        calls, _host = _fake_host(None)
        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_delegation, "_host", lambda: (lambda **k: calls.append(k), lambda: None)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("agent turn", payload["error"])
        self.assertEqual(calls, [])

    def test_a_switched_off_tier_is_refused_with_a_pointer(self):
        cfg = _cfg()
        cfg["callable"]["haiku"] = False
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "haiku"}, cfg=cfg)
        self.assertIn("switched off", payload["error"])
        self.assertIn("delegate_task with a goal prefixed [luna]", payload["error"])
        self.assertEqual(calls, [])

    def test_the_soft_limit_lowers_opus_and_says_so(self):
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, usage=75.0)
        self.assertEqual(calls[0]["credentials_cfg"]["model"], "claude-sonnet-5")
        self.assertEqual(payload["claude_tier"], "sonnet")
        self.assertEqual(payload["tier_adjusted"], "opus→sonnet (weekly usage 75%)")

    def test_the_hard_limit_refuses_and_names_the_codex_call(self):
        payload, calls = self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, usage=95.0)
        self.assertIn("Claude delegation closed", payload["error"])
        self.assertIn("[sol]", payload["error"])
        self.assertEqual(calls, [])

    def test_an_unknown_usage_is_marked(self):
        payload, _calls = self._call({"tasks": [{"goal": "g"}]}, usage=None)
        self.assertEqual(payload["usage"], "unknown")

    def test_a_host_error_becomes_a_tool_error(self):
        def boom(**_kwargs):
            raise ValueError("Cannot resolve delegation provider 'anthropic'")

        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_delegation, "_host", lambda: (boom, lambda: SimpleNamespace(_delegate_depth=0))), \
             patch.object(usage_guard, "read", return_value=_reading(10)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("Cannot resolve delegation provider", payload["error"])

    def test_a_configured_audit_log_gets_one_line_per_call(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, cfg=cfg, usage=75.0)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["event"], "delegate_claude")
        self.assertEqual((entry["tier_requested"], entry["tier_used"], entry["outcome"]),
                         ("opus", "sonnet", "lowered"))
        self.assertNotIn("tier", entry)  # keeps it out of the router's per-account load

    def test_a_delegate_task_error_is_audited_as_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, cfg=cfg,
                       result={"error": "Delegation depth limit reached"})
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["outcome"], "error")
        self.assertEqual(entry["message"], "Delegation depth limit reached")

    def test_audit_lines_name_the_calling_session_and_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-delegation.jsonl"
            cfg["claude_delegation"]["log_path"] = str(log)
            parent = SimpleNamespace(_delegate_depth=0, session_id="sess-1", _current_turn_id="turn-9")
            self._call({"tasks": [{"goal": "g"}]}, cfg=cfg, parent=parent)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual((entry["session_id"], entry["turn_id"]), ("sess-1", "turn-9"))

    def test_switching_it_off_refuses_on_the_next_call(self):
        cfg = _cfg()
        cfg["claude_delegation"]["enabled"] = False
        payload, calls = self._call({"tasks": [{"goal": "g"}]}, cfg=cfg)
        self.assertEqual(payload["error"], "Claude delegation is switched off in router_config.yaml.")
        self.assertEqual(calls, [])


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, claude_delegation, "_ACTIVE", False)

    def test_an_enabled_wing_registers_in_the_delegation_toolset(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
            self.assertTrue(claude_delegation.register(ctx, _cfg()))
        kwargs = ctx.register_tool.call_args.kwargs
        self.assertEqual((kwargs["name"], kwargs["toolset"]), ("delegate_claude", "delegation"))
        self.assertIs(kwargs["handler"], handle_delegate_claude)
        self.assertTrue(claude_delegation.is_active())

    def test_a_disabled_wing_registers_nothing(self):
        ctx = MagicMock()
        cfg = _cfg()
        cfg["claude_delegation"]["enabled"] = False
        self.assertFalse(claude_delegation.register(ctx, cfg))
        ctx.register_tool.assert_not_called()
        self.assertFalse(claude_delegation.is_active())

    def test_register_tool_returning_none_means_not_registered(self):
        ctx = MagicMock()
        ctx.register_tool.return_value = None
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
            self.assertFalse(claude_delegation.register(ctx, _cfg()))
        self.assertFalse(claude_delegation.is_active())

    def test_register_calls_the_deadline_exemption_only_after_success(self):
        ctx = MagicMock()
        exempt = MagicMock(return_value=True)
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", exempt):
            self.assertTrue(claude_delegation.register(ctx, _cfg()))
        exempt.assert_called_once_with()

    def test_register_does_not_block_when_the_exemption_fails(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=False):
            self.assertTrue(claude_delegation.register(ctx, _cfg()))
        self.assertTrue(claude_delegation.is_active())

    def test_registration_is_logged_either_way(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "claude-delegation.jsonl"
            cfg = _cfg()
            cfg["claude_delegation"]["log_path"] = str(log)
            cfg["claude_delegation"]["enabled"] = False
            claude_delegation.register(MagicMock(), cfg)
            with patch.object(claude_delegation, "host_check", return_value=(True, "")), \
                 patch.object(claude_delegation, "_independent_completions", return_value=False), \
                 patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
                cfg["claude_delegation"]["enabled"] = True
                claude_delegation.register(MagicMock(), cfg)
            lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(l["event"], l["registered"]) for l in lines],
                         [("registration", False), ("registration", True)])
        self.assertIn("enabled is false", lines[0]["reason"])


class SequentialDeadlineExemptionTests(unittest.TestCase):
    def test_a_fake_module_gains_delegate_claude_and_keeps_existing_members(self):
        fake = types.ModuleType("agent.tool_executor")
        fake._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset({"delegate_task", "manage_connections"})
        with patch.dict(sys.modules, {"agent.tool_executor": fake}):
            self.assertTrue(claude_delegation._exempt_from_sequential_deadline())
        self.assertEqual(
            fake._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS,
            frozenset({"delegate_task", "manage_connections", "delegate_claude"}),
        )

    def test_a_missing_attribute_returns_false_without_raising(self):
        fake = types.ModuleType("agent.tool_executor")
        with patch.dict(sys.modules, {"agent.tool_executor": fake}):
            self.assertFalse(claude_delegation._exempt_from_sequential_deadline())

    def test_an_unimportable_module_returns_false_without_raising(self):
        with patch.dict(sys.modules, {"agent.tool_executor": None}):
            self.assertFalse(claude_delegation._exempt_from_sequential_deadline())


def _hermes_importable():
    try:
        import tools.delegate_tool  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_hermes_importable(), "Hermes is not importable in this interpreter")
class RealHostTests(unittest.TestCase):
    """Against the installed Hermes: the guarantees the wing leans on."""

    def test_the_installed_hermes_passes_the_host_check(self):
        self.assertEqual(claude_delegation.host_check(), (True, ""))

    def test_a_leaf_loses_the_delegation_toolset(self):
        from tools.delegate_tool_toolsets import _strip_blocked_tools
        self.assertNotIn("delegation", _strip_blocked_tools(["delegation", "file"]))

    def test_the_exemption_helper_adds_delegate_claude_on_the_real_host(self):
        import agent.tool_executor as tool_executor

        original = tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
        self.addCleanup(setattr, tool_executor, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", original)
        self.assertTrue(claude_delegation._exempt_from_sequential_deadline())
        self.assertIn("delegate_task", tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS)
        self.assertIn("delegate_claude", tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS)

    def test_the_depth_limit_holds_for_delegate_claude(self):
        """Nothing spawns from an agent at max_spawn_depth, whichever tool asked."""
        parent = SimpleNamespace(_delegate_depth=99)
        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_delegation, "_host", lambda: (claude_delegation._host_delegate_task(), lambda: parent)), \
             patch.object(usage_guard, "read", return_value=_reading(10)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("depth limit", payload["error"].lower())


import model_router  # noqa: E402

try:
    import yaml  # noqa: E402
except ImportError:  # pragma: no cover
    yaml = None


class OfferedNamesTests(unittest.TestCase):
    def _hermes_config(self, directory):
        path = Path(directory) / "config.yaml"
        path.write_text(
            "delegation:\n  targets:\n"
            "    opus5: {provider: anthropic, model: claude-opus-5}\n"
            "    sonnet5: {provider: anthropic, model: claude-sonnet-5}\n",
            encoding="utf-8",
        )
        return path

    def test_an_inactive_wing_adds_nothing(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_HERMES_CONFIG_PATH", self._hermes_config(directory)), \
             patch.object(claude_delegation, "_ACTIVE", False):
            self.assertEqual(model_router._delegation_target_names(), ("opus5", "sonnet5"))
            self.assertIsNone(model_router._external_target_for_model("claude-haiku-4-5-20251001"))

    def test_an_active_wing_offers_and_counts_haiku(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_HERMES_CONFIG_PATH", self._hermes_config(directory)), \
             patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._load_config", return_value=_cfg()):
            self.assertEqual(model_router._delegation_target_names(), ("haiku", "opus5", "sonnet5"))
            self.assertEqual(model_router._external_target_for_model("claude-haiku-4-5-20251001"), "haiku")


@unittest.skipIf(yaml is None, "PyYAML missing")
class ShippedConfigTests(unittest.TestCase):
    def setUp(self):
        path = Path(model_router.__file__).resolve().parent / "router_config.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_the_wing_ships_enabled_with_its_files(self):
        settings = self.cfg["claude_delegation"]
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["tiers"], CLAUDE_DELEGATION["tiers"])
        self.assertEqual(settings["default_tier"], "sonnet")
        self.assertNotIn("usage_guard", settings)
        self.assertEqual(settings["log_path"], "~/.hermes/logs/claude-delegation.jsonl")

    def test_the_usage_guard_covers_both_accounts(self):
        guard = self.cfg["usage_guard"]
        self.assertEqual(guard["state_path"], "~/.hermes/state/model-router-usage.json")
        self.assertEqual(guard["accounts"]["anthropic"]["step_down"], {"opus5": "sonnet5"})
        self.assertEqual(guard["accounts"]["openai-codex"]["step_down"], {"sol": "terra"})
        # The limits are the operator's to tune from the dashboard (the live file is
        # this one), so pin their shape, not the shipped 70/90.
        for account in ("anthropic", "openai-codex"):
            soft = guard["accounts"][account]["soft_percent"]
            hard = guard["accounts"][account]["hard_percent"]
            self.assertTrue(0 < soft < hard <= 100, (account, soft, hard))

    def test_haiku_is_a_known_claude_target(self):
        self.assertIs(self.cfg["callable"]["haiku"], True)
        self.assertEqual(self.cfg["tier_providers"]["haiku"], "anthropic")
        self.assertIn("haiku", self.cfg["peer_groups"]["light"])

    def test_the_starting_preferences(self):
        self.assertEqual(self.cfg["preferences"], {
            "design": ["sol", "opus5"],
            "code": ["terra", "sonnet5"],
            "explore": ["spark", "luna", "haiku"],
            "review": ["sonnet5", "opus5", "terra"],
            "sensitive": ["opus5", "sol"],
            "critical": ["opus5", "sol"],
            "long": ["sol", "sonnet5"],
        })


from model_router import (  # noqa: E402
    _claude_target_sentence,
    _dispatch_phrase,
    _host_delegate_has_model,
    _model_param_contract,
    _preference_sentence,
    _prepare_orchestration_delegation,
    _quota_redispatch_instruction,
    _without_router_contract,
)
from model_router.test_external_orchestrator import _delegating_request  # noqa: E402
from model_router.test_quota_redispatch import CFG as REDISPATCH_CFG  # noqa: E402
from model_router.test_quota_redispatch import TARGETS as REDISPATCH_TARGETS  # noqa: E402
from model_router.test_quota_redispatch import envelope, request_for  # noqa: E402

CONTRACT_TARGETS = ("haiku", "luna", "opus5", "sol", "sonnet5", "terra")


def _contract(cfg, *, active, model_param=True):
    with patch.object(claude_delegation, "_ACTIVE", active), \
         patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
         patch("model_router._tier_cooldown_remaining", return_value=0.0), \
         patch("model_router._recent_account_load", return_value={}):
        return _model_param_contract("terra", cfg, model_param=model_param)


class ContractTextTests(unittest.TestCase):
    def test_the_host_schema_decides_whether_a_model_parameter_exists(self):
        request = _delegating_request()
        self.assertFalse(_host_delegate_has_model(request))
        request["tools"][0]["parameters"]["properties"]["model"] = {"type": "string"}
        self.assertTrue(_host_delegate_has_model(request))

    def test_no_model_parameter_means_no_instruction_to_set_one(self):
        contract = _contract(_cfg(), active=True, model_param=False)
        self.assertNotIn("Set the delegate_task 'model' parameter", contract)
        self.assertTrue(contract.startswith("Route choice for delegated workers"))

    def test_an_inactive_wing_keeps_todays_opening_even_without_a_model_parameter(self):
        contract = _contract(_cfg(), active=False, model_param=False)
        self.assertTrue(contract.startswith("Set the delegate_task 'model' parameter"))
        self.assertNotIn("Route choice for delegated workers", contract)

    def test_the_dispatch_phrase_names_the_goal_prefix_route_while_active(self):
        with patch.object(claude_delegation, "_ACTIVE", True):
            self.assertEqual(_dispatch_phrase("terra"), "delegate_task (goal prefix [terra])")
        with patch.object(claude_delegation, "_ACTIVE", False):
            self.assertEqual(_dispatch_phrase("terra"), "model:terra")
            self.assertEqual(_dispatch_phrase("opus5"), "model:opus5")

    def test_the_default_keeps_todays_opening(self):
        self.assertTrue(_contract(_cfg(), active=False).startswith("Set the delegate_task 'model' parameter"))

    def test_the_new_opening_and_the_note_header_are_stripped_from_a_leaf(self):
        goal = "[spark] Read-only discovery of the booking list."
        for marker in ("Route choice for delegated workers", "[ROUTER] This turn classifies as"):
            with self.subTest(marker=marker):
                self.assertEqual(_without_router_contract(f"{goal}\n\n{marker} rest"), goal)

    def test_an_active_wing_names_delegate_claude(self):
        contract = _contract(_cfg(), active=True)
        self.assertIn("[opus5] and [sonnet5] are not labels", contract)
        self.assertIn("stopped at its first call", contract)
        self.assertIn('delegate_claude with tier "haiku", "sonnet" or "opus"', contract)
        self.assertNotIn("Name those targets only in the model parameter", contract)

    def test_an_inactive_wing_keeps_todays_text(self):
        contract = _contract(_cfg(), active=False)
        self.assertIn("Name those targets only in the model parameter", contract)
        self.assertNotIn("delegate_claude", contract)

    def test_the_active_contract_names_the_deferred_tool_hint(self):
        contract = _contract(_cfg(), active=True)
        self.assertIn(
            'If delegate_claude is not in your tool list it is a deferred tool: load it once with '
            'tool_describe, then call it through tool_call with name "delegate_claude".',
            contract,
        )

    def test_an_inactive_contract_has_no_deferred_tool_hint(self):
        contract = _contract(_cfg(), active=False)
        self.assertNotIn("deferred tool", contract)

    def test_the_preference_order_names_the_call(self):
        cfg = _cfg(preferences={"code": ["terra", "sonnet5"]})
        with patch.object(claude_delegation, "_ACTIVE", True):
            active = _preference_sentence(["terra", "sonnet5"], cfg)
        with patch.object(claude_delegation, "_ACTIVE", False):
            inactive = _preference_sentence(["terra", "sonnet5"], cfg)
        self.assertIn("code: terra > sonnet5", active)
        self.assertIn("delegate_claude", active)
        self.assertNotIn("model: parameter", active)
        self.assertIn("model: parameter", inactive)

    def test_the_claude_sentence_covers_haiku_and_the_tool(self):
        with patch.object(claude_delegation, "_ACTIVE", True):
            sentence = _claude_target_sentence(["haiku", "opus5", "sonnet5"], _cfg())
        self.assertIn("haiku", sentence)
        self.assertIn('delegate_claude(tier="haiku"|"sonnet"|"opus")', sentence)
        self.assertIn("Use sonnet5 by default", sentence)

    def test_the_redispatch_notice_names_delegate_claude(self):
        with patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._tier_cooldown_remaining", return_value=0.0), \
             patch("model_router._delegation_target_names", return_value=REDISPATCH_TARGETS):
            instruction = _quota_redispatch_instruction(request_for(envelope()), REDISPATCH_CFG)
        self.assertIn('delegate_claude(tier="opus")', instruction)
        self.assertNotIn("model:opus5", instruction)
        self.assertIn("with the call named above", instruction)

    def test_the_preflight_prefers_a_claude_worker_through_the_tool(self):
        cfg = _cfg(orchestration={"enabled": True, "max_tasks": 2})
        with patch.object(claude_delegation, "_ACTIVE", True), \
             patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
             patch("model_router._recent_account_load", return_value={}):
            routed = _prepare_orchestration_delegation(_delegating_request(), "plan-x", 2, cfg=cfg)
        text = routed["messages"][-1]["content"]
        self.assertIn("prefer a native Claude worker through delegate_claude", text)
        self.assertNotIn("model:opus5 / model:sonnet5", text)
        self.assertNotIn("Set the delegate_task 'model' parameter", text)
        self.assertIn(
            'If delegate_claude is not in your tool list it is a deferred tool: load it once with '
            'tool_describe, then call it through tool_call with name "delegate_claude".',
            text,
        )

    def test_the_preflight_keeps_todays_text_while_the_wing_is_off(self):
        cfg = _cfg(orchestration={"enabled": True, "max_tasks": 2})
        with patch.object(claude_delegation, "_ACTIVE", False), \
             patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
             patch("model_router._recent_account_load", return_value={}):
            routed = _prepare_orchestration_delegation(_delegating_request(), "plan-x", 2, cfg=cfg)
        text = routed["messages"][-1]["content"]
        self.assertIn(
            "For real work prefer a native Claude target via model:opus5 / model:sonnet5 when one is offered.",
            text,
        )
        self.assertIn("Set the delegate_task 'model' parameter", text)
        self.assertNotIn("delegate_claude", text)


from model_router import _routing_note, route_llm_request  # noqa: E402
from model_router.test_external_orchestrator import ACTIONABLE  # noqa: E402
from model_router.test_model_router import chat_request  # noqa: E402

PREFS = {
    "design": ["sol", "opus5"], "code": ["terra", "sonnet5"], "explore": ["spark", "luna", "haiku"],
    "review": ["sonnet5", "opus5", "terra"],
}


def _parent_request(tools=("delegate_task", "delegate_claude")):
    # The same actionable prompt the external-orchestrator tests use, so the
    # forced-preflight case below is not skipped by an unrelated gate.
    request = chat_request(ACTIONABLE)
    request["model"] = "claude-opus-5"
    request["tools"] = [{"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
                        for name in tools]
    return request


def _note(kind="code", weekly=40.0, codex=None, cfg=None, active=True, request=None, **kwargs):
    reading = None if weekly is None else _reading(weekly)
    kwargs = {"api_call_count": 1, "turn_id": "t1", "platform": "cli", **kwargs}
    if cfg is None:
        cfg = _cfg(preferences=PREFS)
        cfg["usage_guard"]["accounts"]["openai-codex"] = {
            "soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"},
        }
    with patch.object(claude_delegation, "_ACTIVE", active), \
         patch.object(usage_guard, "peek", side_effect=lambda account, cfg: (
             reading if account == "anthropic" else (
                 _reading(codex) if codex is not None and account == "openai-codex" else None))), \
         patch("model_router.classify_request", return_value=SimpleNamespace(kind=kind)), \
         patch("model_router._delegation_target_names", return_value=("haiku", "opus5", "sonnet5")), \
         patch("model_router._tier_cooldown_remaining", return_value=0.0):
        return _routing_note(request or _parent_request(), kwargs, cfg)


class RoutingNoteTests(unittest.TestCase):
    def test_the_note_recommends_the_kinds_chain_as_calls(self):
        note = _note()
        self.assertIn("[ROUTER] This turn classifies as: code.", note)
        self.assertIn('If you delegate code work: terra → delegate_task (goal prefix [terra]) > '
                      'sonnet5 → delegate_claude(tier="sonnet").', note)
        self.assertIn("Other kinds:", note)
        self.assertIn("review: sonnet5 > opus5 > terra", note)
        self.assertIn("Usage: Claude weekly 40% (soft 70%, hard 90%); "
                     "Codex weekly unknown (soft 70%, hard 90%).", note)
        self.assertIn("Advisory: if you route differently, say why in one line.", note)

    def test_the_note_names_the_deferred_tool_hint_before_the_advisory_line(self):
        note = _note()
        hint = (
            'If delegate_claude is not in your tool list it is a deferred tool: load it once with '
            'tool_describe, then call it through tool_call with name "delegate_claude".'
        )
        self.assertIn(hint, note)
        self.assertLess(note.index(hint), note.index("Advisory:"))

    def test_the_soft_limit_moves_claude_behind_codex(self):
        note = _note(kind="review", weekly=75.0)
        self.assertIn("If you delegate review work: terra → delegate_task", note)
        self.assertIn('sonnet5 → delegate_claude(tier="sonnet") [Claude soft limit]', note)

    def test_the_hard_limit_marks_claude_closed(self):
        self.assertIn("[Claude closed]", _note(kind="review", weekly=95.0))

    def test_an_unknown_reading_says_so(self):
        self.assertIn("Claude weekly unknown", _note(weekly=None))

    def test_a_codex_soft_limit_moves_codex_behind_claude(self):
        note = _note(kind="code", weekly=40.0, codex=72.0)
        self.assertIn('If you delegate code work: sonnet5 → delegate_claude(tier="sonnet") > '
                      'terra → delegate_task (goal prefix [terra]) [Codex soft limit].', note)

    def test_no_claude_preference_still_announces_the_wing(self):
        note = _note(cfg=_cfg(preferences={"code": ["terra"]}))
        self.assertIn("Claude delegation is available through delegate_claude", note)

    def test_a_kind_without_a_preference_keeps_the_built_in_route(self):
        self.assertIn("delegate_task keeps its built-in route", _note(kind="chat"))

    def test_the_note_is_only_for_a_root_parents_first_call(self):
        cases = {
            "inactive delegation": dict(active=False),
            "subagent": dict(platform="subagent"),
            "mid-loop": dict(api_call_count=2),
            "no delegation tool": dict(request=_parent_request(tools=("read_file",))),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_note(**overrides), "")

    def test_the_claude_code_prefix_still_counts_as_a_delegation_tool(self):
        self.assertIn("[ROUTER]", _note(request=_parent_request(tools=("mcp__delegate_claude",))))

    def test_a_completion_envelope_gets_no_note(self):
        with patch("model_router._is_delegation_outcome_text", return_value=True):
            self.assertEqual(_note(), "")


class RoutingNoteMiddlewareTests(unittest.TestCase):
    def _route(self, orchestration):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cfg = _cfg(preferences=PREFS, enabled=True, provider="openai-codex",
                   orchestration={"enabled": orchestration, "max_tasks": 2,
                                  "path": str(Path(directory.name) / "orchestration.jsonl")},
                   logging={"enabled": False}, shadow={"enabled": False})
        with patch.object(claude_delegation, "_ACTIVE", True), \
             patch.object(usage_guard, "peek", side_effect=lambda account, cfg: _reading(40) if account == "anthropic" else None), \
             patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision"), \
             patch("model_router._orchestration_event"), \
             patch("model_router._delegation_target_names", return_value=("haiku", "opus5", "sonnet5")):
            return route_llm_request(request=_parent_request(), provider="anthropic", model="claude-opus-5",
                                     api_call_count=1, turn_id="root-turn", platform="cli")

    def test_a_claude_parent_gets_the_note_and_keeps_its_model(self):
        routed = self._route(orchestration=False)
        self.assertIsNotNone(routed)
        self.assertIn("[ROUTER] This turn classifies as", routed["request"]["messages"][-1]["content"])
        self.assertEqual(routed["request"]["model"], "claude-opus-5")

    def test_a_forced_preflight_carries_the_contract_instead(self):
        routed = self._route(orchestration=True)
        self.assertNotIn("[ROUTER] This turn classifies as", json.dumps(routed["request"]))

    def test_a_broken_routing_note_does_not_drop_the_redispatch_notice(self):
        """The routing note must never abort the whole response: a raise there
        must not cost the parent its redispatch notice, or return None entirely."""
        with patch("model_router._routing_note", side_effect=RuntimeError("boom")):
            routed = self._route(orchestration=False)
        self.assertIsNone(routed)


class UsageStepDownIntegrationTests(unittest.TestCase):
    """The step-down must land on the *dispatched* tier: the orchestration
    gates (forced preflight, forced shadow) have to see the request's original
    classification, not a tier the usage guard already moved it off of."""

    def test_a_codex_root_request_at_the_soft_limit_steps_down_before_dispatch(self):
        cfg = _cfg(preferences=PREFS, provider="openai-codex",
                   orchestration={"enabled": False}, logging={"enabled": False},
                   shadow={"enabled": False})
        cfg["usage_guard"]["accounts"]["openai-codex"] = {
            "soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"},
        }
        # The same "long request" fixture test_model_router.py uses to prove a
        # long prompt classifies to sol -- non-mandatory, so it is eligible for
        # the step-down.
        request = chat_request("Elemezd részletesen. " + "x" * 4200)
        preflight = MagicMock(wraps=model_router._force_terra_supervisor_preflight)
        log_decision = MagicMock()
        with patch.object(usage_guard, "peek", side_effect=lambda account, cfg: (
                 _reading(72.0) if account == "openai-codex" else None)), \
             patch("model_router._load_config", return_value=cfg), \
             patch("model_router._log_decision", log_decision), \
             patch("model_router._force_terra_supervisor_preflight", preflight):
            routed = route_llm_request(request=request, provider="openai-codex", model="gpt-terra",
                                       api_call_count=1, turn_id="root-turn", platform="cli")
        self.assertIsNotNone(routed)
        self.assertEqual(routed["request"]["model"], "gpt-terra")
        self.assertIn("usage soft limit: sol→terra", routed["reason"])
        logged = log_decision.call_args.args[0]
        self.assertIn("usage soft limit: sol→terra", logged.reason)
        preflight_decision = preflight.call_args.args[2]
        self.assertEqual(preflight_decision.tier, "sol")


if __name__ == "__main__":
    unittest.main()
