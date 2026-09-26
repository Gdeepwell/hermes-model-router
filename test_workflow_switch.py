"""Claude is an account like Qwen or Grok: available exactly when its models are
switched on in ``callable``.

The old ``workflow:`` switch (codex | claude_delegation) and
``claude_delegation.enabled`` are retired. A local file that still carries them
is translated in memory at load time -- never rewritten -- so an operator who had
Claude on keeps it on, and one who was on ``workflow: codex`` keeps it off.
Every Claude gate (delegate_claude, advice to the conductor, the execution guard
for a running child, the CLI review bridge) then reads ``callable`` alone.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import yaml

import model_router
from model_router import claude_delegation, worker_admission
from model_router import RouteDecision, _apply_preferences, _require_callable, classify_request

ROOT = Path(model_router.__file__).resolve().parent
CLAUDE = ("opus5", "sonnet5", "haiku")

# The owner's live router_config.local.yaml at the time the switch was retired:
# the legacy keys, and no callable entry for any Claude model.
OWNER_LOCAL = """\
# Your own router settings, layered over router_config.yaml. Git-ignored:
# the dashboard saves here, keeping only what differs from the shipped file.
preferences:
  design: [sol, opus5]
  code: [terra, sonnet5]
  explore: [spark, luna, haiku]
  review: [sonnet5, opus5, terra]
  sensitive: [opus5, sol]
  critical: [opus5, sol]
  long: [sol, sonnet5]
  chat:
  - haiku
  - luna
  default:
  - opus5
  - sol
default_model: terra
usage_guard:
  accounts:
    anthropic:
      soft_percent: 80
    openai-codex:
      soft_percent: 80
workflow: claude_delegation
claude_delegation:
  enabled: true
callable:
  qwen: false
"""


def _files(directory, local):
    """The real shipped router_config.yaml, with ``local`` beside it (None: no local file)."""
    shipped = Path(directory) / "router_config.yaml"
    shipped.write_text((ROOT / "router_config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    if local is not None:
        (Path(directory) / "router_config.local.yaml").write_text(local, encoding="utf-8")
    return shipped


def _load(local):
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(model_router, "_CONFIG_PATH", _files(directory, local)):
        return model_router._load_config()


def _claude(cfg):
    return {name: cfg["callable"].get(name) for name in CLAUDE}


class ShippedDefaultsTests(unittest.TestCase):
    """An install without a Claude subscription works out of the box."""

    def setUp(self):
        self.shipped = yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))

    def test_claude_ships_switched_off_like_grok(self):
        self.assertEqual(_claude(self.shipped), dict.fromkeys(CLAUDE, False))
        self.assertIs(self.shipped["callable"]["grok"], False)

    def test_the_retired_keys_are_gone_from_the_shipped_file(self):
        self.assertNotIn("workflow", self.shipped)
        self.assertNotIn("enabled", self.shipped["claude_delegation"])
        text = (ROOT / "router_config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("workflow", text)

    def test_the_built_in_defaults_match(self):
        defaults = model_router._DEFAULT_CONFIG
        self.assertNotIn("workflow", defaults)
        self.assertEqual({name: defaults["callable"].get(name) for name in CLAUDE}, dict.fromkeys(CLAUDE, False))

    def test_a_fresh_install_offers_no_claude(self):
        cfg = _load(None)
        self.assertEqual(_claude(cfg), dict.fromkeys(CLAUDE, False))
        self.assertNotEqual(claude_delegation.availability_block(cfg), "")

    def test_the_workflow_helpers_are_retired(self):
        self.assertFalse(hasattr(model_router, "workflow_name"))
        self.assertFalse(hasattr(model_router, "_apply_workflow"))
        self.assertFalse(hasattr(model_router, "WORKFLOWS"))


class LegacyLocalFileTests(unittest.TestCase):
    """R3: the legacy keys in router_config.local.yaml become Claude switches."""

    def test_the_owners_file_keeps_his_claude_models_on(self):
        cfg = _load(OWNER_LOCAL)
        self.assertEqual(_claude(cfg), dict.fromkeys(CLAUDE, True))
        self.assertIs(cfg["callable"]["qwen"], False)
        self.assertEqual(cfg["preferences"]["review"], ["sonnet5", "opus5", "terra"])
        self.assertEqual(claude_delegation.availability_block(cfg), "")

    def test_the_legacy_keys_are_dropped_from_the_loaded_config(self):
        cfg = _load(OWNER_LOCAL)
        self.assertNotIn("workflow", cfg)
        self.assertNotIn("enabled", cfg["claude_delegation"])
        self.assertEqual(cfg["claude_delegation"]["default_tier"], "sonnet", "other keys of the block stay")

    def test_codex_switches_every_claude_model_off_whatever_the_local_file_says(self):
        cfg = _load("workflow: codex\ncallable:\n  opus5: true\n  sonnet5: true\n")
        self.assertEqual(_claude(cfg), dict.fromkeys(CLAUDE, False))
        self.assertNotIn("workflow", cfg)

    def test_codex_no_longer_clears_the_preference_chains(self):
        cfg = _load("workflow: codex\npreferences:\n  review: [sonnet5, terra]\n")
        self.assertEqual(cfg["preferences"], {"review": ["sonnet5", "terra"]})

    def test_the_workflow_is_read_case_and_space_insensitively(self):
        self.assertEqual(_claude(_load("workflow: ' Codex '\ncallable: {sonnet5: true}\n")),
                         dict.fromkeys(CLAUDE, False))
        self.assertEqual(_claude(_load("workflow: ' Claude_Delegation '\n")), dict.fromkeys(CLAUDE, True))

    def test_on_keeps_a_claude_switch_the_local_file_sets(self):
        cfg = _load("workflow: claude_delegation\ncallable:\n  haiku: false\n")
        self.assertEqual(_claude(cfg), {"opus5": True, "sonnet5": True, "haiku": False})

    def test_the_workflow_wins_over_the_delegation_flag(self):
        cfg = _load("workflow: codex\nclaude_delegation:\n  enabled: true\n")
        self.assertEqual(_claude(cfg), dict.fromkeys(CLAUDE, False))

    def test_without_a_workflow_the_delegation_flag_decides(self):
        self.assertEqual(_claude(_load("claude_delegation:\n  enabled: true\n")), dict.fromkeys(CLAUDE, True))
        cfg = _load("claude_delegation:\n  enabled: false\ncallable:\n  sonnet5: true\n")
        self.assertEqual(_claude(cfg), dict.fromkeys(CLAUDE, False))
        self.assertNotIn("enabled", cfg["claude_delegation"])

    def test_an_unknown_workflow_or_a_non_bool_flag_gives_no_verdict(self):
        for local in ("workflow: gemini\ncallable: {sonnet5: true}\n",
                      "claude_delegation:\n  enabled: 'yes'\ncallable: {sonnet5: true}\n"):
            with self.subTest(local=local):
                cfg = _load(local)
                self.assertEqual(_claude(cfg), {"opus5": False, "sonnet5": True, "haiku": False})
                self.assertNotIn("workflow", cfg)
                self.assertNotIn("enabled", cfg["claude_delegation"])

    def test_a_local_file_without_legacy_keys_is_taken_as_merged(self):
        self.assertEqual(_claude(_load("callable:\n  opus5: true\n")),
                         {"opus5": True, "sonnet5": False, "haiku": False})

    def test_only_the_local_file_is_read_for_legacy_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped = _files(directory, None)
            shipped.write_text(shipped.read_text(encoding="utf-8") + "\nworkflow: claude_delegation\n",
                               encoding="utf-8")
            with patch.object(model_router, "_CONFIG_PATH", shipped):
                cfg = model_router._load_config()
        self.assertEqual(_claude(cfg), dict.fromkeys(CLAUDE, False))
        self.assertNotIn("workflow", cfg)

    def test_the_loader_never_writes_the_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped = _files(directory, OWNER_LOCAL)
            with patch.object(model_router, "_CONFIG_PATH", shipped):
                model_router._load_config()
            self.assertEqual((Path(directory) / "router_config.local.yaml").read_text(encoding="utf-8"), OWNER_LOCAL)


def _switches(on=True, **extra):
    return {"callable": dict.fromkeys(CLAUDE, on), **extra}


class AvailabilityTests(unittest.TestCase):
    def test_a_callable_claude_model_makes_delegate_claude_available(self):
        self.assertEqual(claude_delegation.availability_block({"callable": {"haiku": True}}), "")

    def test_every_claude_target_switched_off_is_unavailable(self):
        self.assertIn("switched off", claude_delegation.availability_block(_switches(False)))

    def test_a_stale_delegation_flag_is_ignored(self):
        cfg = _switches(True, claude_delegation={"enabled": False}, workflow="codex")
        self.assertEqual(claude_delegation.availability_block(cfg), "")

    def test_the_retired_flag_is_not_a_default_any_more(self):
        self.assertNotIn("enabled", claude_delegation.delegation_config({}))

    def test_the_check_fn_reads_the_live_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped = _files(directory, "callable:\n  sonnet5: true\n")
            with patch.object(model_router, "_CONFIG_PATH", shipped):
                self.assertTrue(claude_delegation.tool_available())
                (Path(directory) / "router_config.local.yaml").write_text("callable:\n  sonnet5: false\n",
                                                                          encoding="utf-8")
                self.assertFalse(claude_delegation.tool_available())


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, claude_delegation, "_ACTIVE", False)

    def _register(self, cfg, host=(True, "")):
        ctx = MagicMock()
        with patch.object(claude_delegation, "host_check", return_value=host), \
             patch.object(claude_delegation, "_independent_completions", return_value=False), \
             patch.object(claude_delegation, "_exempt_from_sequential_deadline", return_value=True):
            registered = claude_delegation.register(ctx, cfg)
        return registered, ctx

    def test_claude_switched_off_still_registers_so_switching_on_needs_no_restart(self):
        registered, ctx = self._register(_switches(False))
        self.assertTrue(registered)
        self.assertIs(ctx.register_tool.call_args.kwargs["check_fn"], claude_delegation.tool_available)

    def test_a_host_without_the_api_still_does_not_register(self):
        registered, ctx = self._register(_switches(), host=(False, "delegate_task lacks credentials_cfg"))
        self.assertFalse(registered)
        ctx.register_tool.assert_not_called()

    def test_the_check_fn_is_never_ttl_cached(self):
        registry = types.ModuleType("tools.registry")
        marked = []
        registry.no_cache_check_fn = lambda fn: marked.append(fn) or fn
        tools = types.ModuleType("tools")
        tools.registry = registry
        with patch.dict(sys.modules, {"tools": tools, "tools.registry": registry}):
            self._register(_switches())
        self.assertEqual(marked, [claude_delegation.tool_available])


class RequestScopeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, claude_delegation, "_ACTIVE", False)
        claude_delegation._ACTIVE = True

    def test_outside_a_request_it_is_the_registration(self):
        self.assertTrue(claude_delegation.is_active())

    def test_inside_a_request_the_scope_decides_and_is_restored(self):
        with claude_delegation.request_scope(False):
            self.assertFalse(claude_delegation.is_active())
        self.assertTrue(claude_delegation.is_active())

    def test_offered_accepts_the_mcp_prefix(self):
        self.assertTrue(claude_delegation.offered(["read_file", "mcp__delegate_claude"]))
        self.assertTrue(claude_delegation.offered(["delegate_claude"]))
        self.assertFalse(claude_delegation.offered(["delegate_task"]))

    def _active_during_routing(self, local, tools):
        seen = []

        def inner(**kwargs):
            seen.append(claude_delegation.is_active())
            return None

        request = {"messages": [{"role": "user", "content": "hi"}],
                   "tools": [{"type": "function", "function": {"name": name}} for name in tools]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_CONFIG_PATH", _files(directory, local)), \
             patch.object(model_router, "_route_llm_request", side_effect=inner):
            model_router.route_llm_request(model="gpt-terra", request=request)
        return seen[0]

    def test_a_request_that_offers_the_tool_with_claude_on_is_active(self):
        self.assertTrue(self._active_during_routing("callable: {sonnet5: true}\n",
                                                    ["delegate_task", "delegate_claude"]))

    def test_claude_off_is_inactive_even_where_the_session_still_has_the_tool(self):
        self.assertFalse(self._active_during_routing(None, ["delegate_task", "delegate_claude"]))

    def test_a_session_built_without_the_tool_is_never_told_to_call_it(self):
        self.assertFalse(self._active_during_routing("callable: {sonnet5: true}\n", ["delegate_task"]))


class HostToolCacheTests(unittest.TestCase):
    """Availability still flips -- now when the Claude switches flip."""

    def setUp(self):
        self.addCleanup(setattr, claude_delegation, "_LAST_AVAILABLE", None)
        claude_delegation._LAST_AVAILABLE = None

    def _note(self, *values):
        fake = types.ModuleType("model_tools")
        fake._clear_tool_defs_cache = MagicMock()
        with patch.dict(sys.modules, {"model_tools": fake}):
            for value in values:
                claude_delegation.note_availability(value)
        return fake._clear_tool_defs_cache.call_count

    def test_the_first_reading_clears_nothing(self):
        self.assertEqual(self._note(True), 0)

    def test_an_unchanged_reading_clears_nothing(self):
        self.assertEqual(self._note(True, True, True), 0)

    def test_every_flip_clears_the_hosts_tool_list_memo(self):
        self.assertEqual(self._note(True, False, False, True), 2)

    def test_a_host_without_the_memo_is_not_an_error(self):
        with patch.dict(sys.modules, {"model_tools": types.ModuleType("model_tools")}):
            claude_delegation.note_availability(True)
            claude_delegation.note_availability(False)

    def test_the_gateway_hook_notes_the_claude_switches_before_dispatch(self):
        for local, expected in ((None, False), ("callable: {haiku: true}\n", True)):
            with self.subTest(local=local), tempfile.TemporaryDirectory() as directory, \
                 patch.object(model_router, "_CONFIG_PATH", _files(directory, local)), \
                 patch.object(claude_delegation, "note_availability") as note:
                self.assertIsNone(model_router.on_pre_gateway_dispatch(event=None))
            note.assert_called_once_with(expected)

    def test_register_wires_the_gateway_hook(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "register"):
            model_router.register(ctx)
        hooks = [call.args[0] for call in ctx.register_hook.call_args_list]
        self.assertIn("pre_gateway_dispatch", hooks)


class HandlerTests(unittest.TestCase):
    def test_a_call_with_claude_switched_off_is_refused_and_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_CONFIG_PATH", _files(directory, "workflow: codex\n")), \
             patch.object(claude_delegation, "_host") as host:
            payload = json.loads(claude_delegation.handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        host.assert_not_called()
        self.assertIn("switched off", payload["error"])
        self.assertIn("delegate_task", payload["error"])
        self.assertNotIn("workflow", payload["error"])


class TargetAdviceTests(unittest.TestCase):
    def test_switched_off_claude_targets_are_not_advised(self):
        cfg = _load(None)
        for tier in CLAUDE:
            self.assertFalse(model_router._target_is_offered(tier, cfg))
        self.assertTrue(model_router._target_is_offered("terra", cfg))

    def test_switched_on_claude_targets_are_advised_whatever_a_stale_workflow_says(self):
        cfg = _switches(True, workflow="codex")
        for tier in CLAUDE:
            self.assertTrue(model_router._target_is_offered(tier, cfg))


class ExecutionGuardTests(unittest.TestCase):
    MESSAGE = "Claude workers are switched off in Settings. The parent model is unchanged."

    def test_a_claude_worker_whose_model_is_switched_off_is_refused(self):
        cfg = {"callable": {"opus5": True, "sonnet5": False, "haiku": True}}
        self.assertEqual(worker_admission.refusal("anthropic", "claude-sonnet-5", cfg), self.MESSAGE)
        self.assertEqual(worker_admission.refusal("anthropic", "claude-opus-5-5", cfg), "")

    def test_a_claude_model_on_another_provider_is_refused_too(self):
        cfg = {"callable": dict.fromkeys(CLAUDE, False)}
        self.assertEqual(worker_admission.refusal("openrouter", "claude-sonnet-5", cfg), self.MESSAGE)

    def test_an_unmapped_claude_model_needs_some_claude_switch_on(self):
        self.assertEqual(worker_admission.refusal("anthropic", "claude-x", _switches(False)), self.MESSAGE)
        self.assertEqual(worker_admission.refusal("anthropic", "claude-x", {"callable": {"haiku": True}}), "")

    def test_a_stale_workflow_no_longer_refuses(self):
        self.assertEqual(worker_admission.refusal("anthropic", "claude-sonnet-5", _switches(True, workflow="codex")), "")

    def test_other_accounts_are_untouched(self):
        self.assertEqual(worker_admission.refusal("openai-codex", "gpt-terra", _switches(False)), "")

    def test_a_running_claude_child_stops_once_its_model_is_switched_off(self):
        cfg = {"enabled": True, "callable": {"sonnet5": True}}
        downstream = Mock(return_value="Claude continued")
        request = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "Continue"}]}
        with patch.object(model_router, "_load_config", side_effect=lambda: json.loads(json.dumps(cfg))):
            kwargs = dict(request=request, original_request=request, next_call=downstream,
                          provider="anthropic", platform="subagent", turn_id="root:sa-1")
            self.assertEqual(model_router.run_llm_with_transient_failover(**kwargs), "Claude continued")
            cfg["callable"]["sonnet5"] = False
            stopped = model_router.run_llm_with_transient_failover(**kwargs)
        self.assertIn("ROUTER WORKER STOPPED", stopped.output_text)
        self.assertIn("switched off in Settings", stopped.output_text)
        downstream.assert_called_once()


class CliReviewBridgeTests(unittest.TestCase):
    REQUEST = {"messages": [{"role": "user", "content": "[sonnet-review] Review parser"}]}

    def _run(self, cfg):
        with patch.object(model_router, "_verified_delegated_claude_review",
                          return_value=(Path("/tmp/repo"), "sonnet")), \
             patch.object(model_router, "_run_opus5_bridge", return_value={"result": "ok"}) as bridge, \
             patch.object(model_router, "_opus5_response", return_value="verdict"), \
             patch.object(model_router.claude_delegation, "_log"):
            result = model_router._maybe_run_opus5(self.REQUEST, cfg, platform="subagent",
                                                   api_mode="codex_responses")
        return result, bridge

    def _cfg(self, on, **extra):
        return {"callable": {"sonnet5": on, "opus5": on}, "cooldown": {"enabled": False},
                "usage_guard": {}, "coding_agent": {"enabled": True, "delegated_review": {"enabled": True}},
                **extra}

    def test_a_switched_off_claude_is_skipped(self):
        result, bridge = self._run(self._cfg(False))
        self.assertIsNone(result)
        bridge.assert_not_called()

    def test_a_switched_on_claude_runs_whatever_a_stale_workflow_says(self):
        result, bridge = self._run(self._cfg(True, workflow="codex"))
        self.assertEqual(result, "verdict")
        bridge.assert_called_once()


PREF_CFG = {
    "models": {"luna": "gpt-luna", "spark": "gpt-spark", "terra": "gpt-terra", "sol": "gpt-sol"},
    "callable": {"luna": True, "spark": True, "terra": True, "sol": True,
                 "opus5": False, "sonnet5": False, "haiku": False},
    "fallbacks": {"sol": "terra"},
    "thresholds": {"sol_min_chars": 3500, "luna_max_chars": 700},
    "effort": {},
    "default_model": "terra",
    "cooldown": {"enabled": False},
}


def _pref_cfg(**overrides):
    cfg = json.loads(json.dumps(PREF_CFG))
    cfg.update(overrides)
    return cfg


class PreferenceChainTests(unittest.TestCase):
    """R4: a chain whose Claude entries are switched off skips them."""

    REVIEW = {"messages": [{"role": "user", "content": "nezd at a kodot es reviewold"}]}

    def test_a_review_chain_with_claude_off_routes_to_terra_and_advises_no_claude(self):
        cfg = _pref_cfg(preferences={"review": ["sonnet5", "opus5", "terra"]})
        decision = _require_callable(classify_request(self.REVIEW, config=cfg), cfg)
        self.assertEqual((decision.kind, decision.tier, decision.prefer_target), ("review", "terra", ""))

    def test_a_claude_only_chain_falls_back_to_the_built_in_route(self):
        cfg = _pref_cfg(preferences={"review": ["sonnet5", "opus5"]})
        builtin = classify_request(self.REVIEW, config=_pref_cfg())
        decision = _require_callable(classify_request(self.REVIEW, config=cfg), cfg)
        self.assertEqual((decision.tier, decision.prefer_target), (builtin.tier, ""))

    def test_an_uncallable_route_with_a_claude_only_chain_follows_the_built_in_fallbacks(self):
        cfg = _pref_cfg(preferences={"long": ["sonnet5"]})
        cfg["callable"]["sol"] = False
        decision = RouteDecision(tier="sol", model="gpt-sol", reason="long", kind="long")
        self.assertIs(_apply_preferences(decision, cfg), decision)
        self.assertEqual(_require_callable(decision, cfg).tier, "terra")

    def test_a_mandatory_route_still_refuses_the_fallback_chain(self):
        cfg = _pref_cfg(preferences={"design": ["opus5"]})
        cfg["callable"]["sol"] = False
        decision = RouteDecision(tier="sol", model="gpt-sol", reason="design", kind="design", mandatory=True)
        with self.assertRaises(RuntimeError):
            _require_callable(decision, cfg)


if __name__ == "__main__":
    unittest.main()
