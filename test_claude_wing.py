"""The Claude wing: Claude workers next to the Codex workforce.

Hermes's delegate_task has one route per process, pinned to Codex on this host.
delegate_claude reaches Claude by calling the same delegate_task with a per-call
route pinned to the anthropic provider. These tests cover the wing without any
network or model call.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_router import claude_wing
from model_router.claude_wing import (
    TARGET_FOR_TIER,
    TIER_FOR_TARGET,
    registration_block,
    target_for_model,
    target_names,
    tier_model,
    wing_config,
)

WING = {
    "enabled": True,
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"},
    "default_tier": "sonnet",
    "usage_guard": {"soft_percent": 70, "hard_percent": 90, "cache_seconds": 300},
}


def _cfg(**overrides):
    cfg = {
        "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True,
                     "opus5": True, "sonnet5": True, "haiku": True},
        "tier_providers": {"opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic"},
        "fallbacks": {"opus5": "sol"},
        "peer_groups": {"heavy": ["terra", "opus5", "sonnet5"], "light": ["luna", "spark", "haiku"]},
        "default_model": "terra",
        "claude_wing": json.loads(json.dumps(WING)),
    }
    cfg.update(overrides)
    return cfg


class WingConfigTests(unittest.TestCase):
    def test_a_config_without_the_block_leaves_the_wing_off(self):
        """Configuring nothing must change nothing."""
        self.assertFalse(wing_config({})["enabled"])
        self.assertFalse(wing_config(None)["enabled"])

    def test_a_partial_guard_keeps_the_other_defaults(self):
        wing = wing_config({"claude_wing": {"enabled": True, "usage_guard": {"soft_percent": 60}}})
        self.assertEqual(wing["usage_guard"]["soft_percent"], 60)
        self.assertEqual(wing["usage_guard"]["hard_percent"], 90)
        self.assertEqual(wing["tiers"]["opus"], "claude-opus-5")

    def test_names_map_both_ways(self):
        self.assertEqual(TARGET_FOR_TIER, {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"})
        self.assertEqual(TIER_FOR_TARGET["opus5"], "opus")

    def test_tier_model(self):
        self.assertEqual(tier_model("haiku", _cfg()), "claude-haiku-4-5-20251001")
        self.assertEqual(tier_model("gpt", _cfg()), "")

    def test_target_names_skip_a_tier_without_a_model(self):
        cfg = _cfg()
        self.assertEqual(target_names(cfg), ("haiku", "opus5", "sonnet5"))
        cfg["claude_wing"]["tiers"]["haiku"] = ""
        self.assertEqual(target_names(cfg), ("opus5", "sonnet5"))

    def test_a_model_maps_back_to_its_target(self):
        self.assertEqual(target_for_model("claude-haiku-4-5-20251001", _cfg()), "haiku")
        self.assertIsNone(target_for_model("gpt-terra", _cfg()))
        self.assertIsNone(target_for_model("", _cfg()))


class RegistrationBlockTests(unittest.TestCase):
    def test_a_disabled_wing_does_not_register(self):
        cfg = _cfg()
        cfg["claude_wing"]["enabled"] = False
        self.assertIn("enabled", registration_block(cfg))

    def test_every_claude_target_switched_off_does_not_register(self):
        cfg = _cfg()
        for target in ("haiku", "sonnet5", "opus5"):
            cfg["callable"][target] = False
        self.assertIn("switched off", registration_block(cfg))

    def test_a_host_without_the_api_does_not_register(self):
        with patch.object(claude_wing, "host_check", return_value=(False, "delegate_task lacks credentials_cfg")):
            self.assertIn("credentials_cfg", registration_block(_cfg()))

    def test_an_enabled_wing_on_a_capable_host_registers(self):
        with patch.object(claude_wing, "host_check", return_value=(True, "")):
            self.assertEqual(registration_block(_cfg()), "")


from model_router.claude_wing import (  # noqa: E402
    GuardOutcome,
    UsageReading,
    apply_guard,
    peek_usage,
    read_usage,
    wing_state,
)


def _reading(weekly, session=10.0, fetched_at=None):
    return UsageReading(weekly, session, time.time() if fetched_at is None else fetched_at)


class GuardTests(unittest.TestCase):
    def test_below_the_soft_limit_everything_runs_as_asked(self):
        for tier in ("haiku", "sonnet", "opus"):
            with self.subTest(tier=tier):
                outcome = apply_guard(tier, _cfg(), _reading(50))
                self.assertEqual((outcome.tier, outcome.refused, outcome.adjusted), (tier, "", ""))
                self.assertEqual(outcome.usage, "50%")

    def test_the_soft_limit_lowers_opus_only(self):
        outcome = apply_guard("opus", _cfg(), _reading(75))
        self.assertEqual(outcome.tier, "sonnet")
        self.assertEqual(outcome.adjusted, "opus→sonnet (weekly usage 75%)")
        for tier in ("haiku", "sonnet"):
            with self.subTest(tier=tier):
                self.assertEqual(apply_guard(tier, _cfg(), _reading(75)).tier, tier)

    def test_the_hard_limit_closes_the_wing(self):
        outcome = apply_guard("haiku", _cfg(), _reading(95))
        self.assertIn("Claude wing closed: weekly usage 95%", outcome.refused)

    def test_a_full_session_window_also_closes_it(self):
        outcome = apply_guard("sonnet", _cfg(), _reading(40, session=92))
        self.assertIn("5-hour session usage 92%", outcome.refused)

    def test_no_reading_fails_open(self):
        """A missing reading must not close the wing; Anthropic's own quota error
        still stops a child, and the router records that as a cooldown."""
        self.assertEqual(apply_guard("opus", _cfg(), None), GuardOutcome("opus"))

    def test_wing_state(self):
        self.assertEqual(wing_state(_cfg(), None), "unknown")
        self.assertEqual(wing_state(_cfg(), _reading(50)), "open")
        self.assertEqual(wing_state(_cfg(), _reading(75)), "soft")
        self.assertEqual(wing_state(_cfg(), _reading(90)), "closed")


class UsageCacheTests(unittest.TestCase):
    def setUp(self):
        claude_wing._reset_usage_cache()
        self.addCleanup(claude_wing._reset_usage_cache)

    def test_one_fetch_per_cache_period(self):
        fetch = MagicMock(return_value=_reading(40))
        with patch.object(claude_wing, "_fetch_reading", fetch):
            read_usage(_cfg(), now=1000.0)
            read_usage(_cfg(), now=1010.0)
            self.assertEqual(fetch.call_count, 1)
            read_usage(_cfg(), now=1400.0)
            self.assertEqual(fetch.call_count, 2)

    def test_a_failed_reading_is_not_retried_within_the_period(self):
        fetch = MagicMock(return_value=None)
        with patch.object(claude_wing, "_fetch_reading", fetch):
            self.assertIsNone(read_usage(_cfg(), now=1000.0))
            self.assertIsNone(read_usage(_cfg(), now=1010.0))
        self.assertEqual(fetch.call_count, 1)

    def test_a_configured_state_file_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            cfg["claude_wing"]["usage_guard"]["state_path"] = str(Path(directory) / "usage.json")
            with patch.object(claude_wing, "_fetch_reading", return_value=_reading(61)):
                read_usage(cfg, now=1000.0)
            claude_wing._reset_usage_cache()  # a new process
            fetch = MagicMock()
            with patch.object(claude_wing, "_fetch_reading", fetch):
                reading = read_usage(cfg, now=1010.0)
        self.assertEqual(reading.weekly, 61)
        fetch.assert_not_called()

    def test_no_state_path_means_no_file(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(claude_wing, "_fetch_reading", return_value=_reading(61)):
            read_usage(_cfg(), now=1000.0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_peek_never_fetches_and_starts_one_refresh_when_stale(self):
        fetch, refresh = MagicMock(), MagicMock()
        with patch.object(claude_wing, "_fetch_reading", fetch), \
             patch.object(claude_wing, "_start_refresh", refresh):
            self.assertIsNone(peek_usage(_cfg()))
            peek_usage(_cfg())
        fetch.assert_not_called()
        self.assertEqual(refresh.call_count, 1)

    def test_peek_returns_a_fresh_reading_without_refreshing(self):
        refresh = MagicMock()
        with patch.object(claude_wing, "_fetch_reading", return_value=_reading(40)):
            read_usage(_cfg())
        with patch.object(claude_wing, "_start_refresh", refresh):
            self.assertEqual(peek_usage(_cfg()).weekly, 40)
        refresh.assert_not_called()


from model_router.claude_wing import build_schema, handle_delegate_claude  # noqa: E402


def _fake_host(parent, result=None):
    """A recording delegate_task and a fixed active parent."""
    calls = []

    def delegate_task(**kwargs):
        calls.append(kwargs)
        return json.dumps(result if result is not None else {"status": "dispatched", "delegation_id": "d1"})

    return calls, (lambda: (delegate_task, lambda: parent))


class SchemaTests(unittest.TestCase):
    def test_the_schema_mirrors_delegate_task_plus_a_tier(self):
        with patch.object(claude_wing, "_independent_completions", return_value=False):
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
        claude_wing._reset_usage_cache()
        self.addCleanup(claude_wing._reset_usage_cache)

    def _call(self, args, *, cfg=None, parent=None, usage=40.0, result=None):
        parent = parent if parent is not None else SimpleNamespace(_delegate_depth=0)
        calls, host = _fake_host(parent, result)
        reading = None if usage is None else _reading(usage)
        with patch("model_router._load_config", return_value=cfg or _cfg()), \
             patch.object(claude_wing, "_host", host), \
             patch.object(claude_wing, "read_usage", return_value=reading):
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
             patch.object(claude_wing, "_host", lambda: (lambda **k: calls.append(k), lambda: None)):
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
        self.assertIn("Claude wing closed", payload["error"])
        self.assertIn("[sol]", payload["error"])
        self.assertEqual(calls, [])

    def test_an_unknown_usage_is_marked(self):
        payload, _calls = self._call({"tasks": [{"goal": "g"}]}, usage=None)
        self.assertEqual(payload["usage"], "unknown")

    def test_a_host_error_becomes_a_tool_error(self):
        def boom(**_kwargs):
            raise ValueError("Cannot resolve delegation provider 'anthropic'")

        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_wing, "_host", lambda: (boom, lambda: SimpleNamespace(_delegate_depth=0))), \
             patch.object(claude_wing, "read_usage", return_value=_reading(10)):
            payload = json.loads(handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("Cannot resolve delegation provider", payload["error"])

    def test_a_configured_audit_log_gets_one_line_per_call(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _cfg()
            log = Path(directory) / "claude-wing.jsonl"
            cfg["claude_wing"]["log_path"] = str(log)
            self._call({"tasks": [{"goal": "g"}], "tier": "opus"}, cfg=cfg, usage=75.0)
            entry = json.loads(log.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["event"], "delegate_claude")
        self.assertEqual((entry["tier_requested"], entry["tier_used"], entry["outcome"]),
                         ("opus", "sonnet", "lowered"))
        self.assertNotIn("tier", entry)  # keeps it out of the router's per-account load


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, claude_wing, "_ACTIVE", False)

    def test_an_enabled_wing_registers_in_the_delegation_toolset(self):
        ctx = MagicMock()
        with patch.object(claude_wing, "host_check", return_value=(True, "")), \
             patch.object(claude_wing, "_independent_completions", return_value=False):
            self.assertTrue(claude_wing.register(ctx, _cfg()))
        kwargs = ctx.register_tool.call_args.kwargs
        self.assertEqual((kwargs["name"], kwargs["toolset"]), ("delegate_claude", "delegation"))
        self.assertIs(kwargs["handler"], handle_delegate_claude)
        self.assertTrue(claude_wing.is_active())

    def test_a_disabled_wing_registers_nothing(self):
        ctx = MagicMock()
        cfg = _cfg()
        cfg["claude_wing"]["enabled"] = False
        self.assertFalse(claude_wing.register(ctx, cfg))
        ctx.register_tool.assert_not_called()
        self.assertFalse(claude_wing.is_active())


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
        self.assertEqual(claude_wing.host_check(), (True, ""))

    def test_a_leaf_loses_the_delegation_toolset(self):
        from tools.delegate_tool_toolsets import _strip_blocked_tools
        self.assertNotIn("delegation", _strip_blocked_tools(["delegation", "file"]))

    def test_the_depth_limit_holds_for_delegate_claude(self):
        """Nothing spawns from an agent at max_spawn_depth, whichever tool asked."""
        parent = SimpleNamespace(_delegate_depth=99)
        with patch("model_router._load_config", return_value=_cfg()), \
             patch.object(claude_wing, "_host", lambda: (claude_wing._host_delegate_task(), lambda: parent)), \
             patch.object(claude_wing, "read_usage", return_value=_reading(10)):
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
             patch.object(claude_wing, "_ACTIVE", False):
            self.assertEqual(model_router._delegation_target_names(), ("opus5", "sonnet5"))
            self.assertIsNone(model_router._external_target_for_model("claude-haiku-4-5-20251001"))

    def test_an_active_wing_offers_and_counts_haiku(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_HERMES_CONFIG_PATH", self._hermes_config(directory)), \
             patch.object(claude_wing, "_ACTIVE", True), \
             patch("model_router._load_config", return_value=_cfg()):
            self.assertEqual(model_router._delegation_target_names(), ("haiku", "opus5", "sonnet5"))
            self.assertEqual(model_router._external_target_for_model("claude-haiku-4-5-20251001"), "haiku")


@unittest.skipIf(yaml is None, "PyYAML missing")
class ShippedConfigTests(unittest.TestCase):
    def setUp(self):
        path = Path(model_router.__file__).resolve().parent / "router_config.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_the_wing_ships_enabled_with_its_files(self):
        wing = self.cfg["claude_wing"]
        self.assertTrue(wing["enabled"])
        self.assertEqual(wing["tiers"], WING["tiers"])
        self.assertEqual(wing["default_tier"], "sonnet")
        self.assertEqual(wing["usage_guard"]["soft_percent"], 70)
        self.assertEqual(wing["usage_guard"]["hard_percent"], 90)
        self.assertEqual(wing["usage_guard"]["state_path"], "~/.hermes/state/model-router-claude-usage.json")
        self.assertEqual(wing["log_path"], "~/.hermes/logs/claude-wing.jsonl")

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
    with patch.object(claude_wing, "_ACTIVE", active), \
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
        with patch.object(claude_wing, "_ACTIVE", True):
            self.assertEqual(_dispatch_phrase("terra"), "delegate_task (goal prefix [terra])")
        with patch.object(claude_wing, "_ACTIVE", False):
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

    def test_the_preference_order_names_the_call(self):
        cfg = _cfg(preferences={"code": ["terra", "sonnet5"]})
        with patch.object(claude_wing, "_ACTIVE", True):
            active = _preference_sentence(["terra", "sonnet5"], cfg)
        with patch.object(claude_wing, "_ACTIVE", False):
            inactive = _preference_sentence(["terra", "sonnet5"], cfg)
        self.assertIn("code: terra > sonnet5", active)
        self.assertIn("delegate_claude", active)
        self.assertNotIn("model: parameter", active)
        self.assertIn("model: parameter", inactive)

    def test_the_claude_sentence_covers_haiku_and_the_tool(self):
        with patch.object(claude_wing, "_ACTIVE", True):
            sentence = _claude_target_sentence(["haiku", "opus5", "sonnet5"], _cfg())
        self.assertIn("haiku", sentence)
        self.assertIn('delegate_claude(tier="haiku"|"sonnet"|"opus")', sentence)
        self.assertIn("Use sonnet5 by default", sentence)

    def test_the_redispatch_notice_names_delegate_claude(self):
        with patch.object(claude_wing, "_ACTIVE", True), \
             patch("model_router._tier_cooldown_remaining", return_value=0.0), \
             patch("model_router._delegation_target_names", return_value=REDISPATCH_TARGETS):
            instruction = _quota_redispatch_instruction(request_for(envelope()), REDISPATCH_CFG)
        self.assertIn('delegate_claude(tier="opus")', instruction)
        self.assertNotIn("model:opus5", instruction)
        self.assertIn("with the call named above", instruction)

    def test_the_preflight_prefers_a_claude_worker_through_the_tool(self):
        cfg = _cfg(orchestration={"enabled": True, "max_tasks": 2})
        with patch.object(claude_wing, "_ACTIVE", True), \
             patch("model_router._delegation_target_names", return_value=CONTRACT_TARGETS), \
             patch("model_router._recent_account_load", return_value={}):
            routed = _prepare_orchestration_delegation(_delegating_request(), "plan-x", 2, cfg=cfg)
        text = routed["messages"][-1]["content"]
        self.assertIn("prefer a native Claude worker through delegate_claude", text)
        self.assertNotIn("model:opus5 / model:sonnet5", text)
        self.assertNotIn("Set the delegate_task 'model' parameter", text)


if __name__ == "__main__":
    unittest.main()
