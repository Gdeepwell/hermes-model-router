"""The workflow switch: Claude delegation, or the original Codex workflow.

``workflow: codex`` is master's behaviour -- Codex does the work, the parent stays
where Hermes put it, no delegate_claude, and the built-in routes instead of the
Claude-tuned preference chains. The switch is read live: routing follows it on
the next request, and delegate_claude is always registered but offered only while
the workflow allows it.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import model_router
from model_router import claude_delegation

CONFIG = """\
enabled: true
workflow: {workflow}
callable:
  terra: true
  opus5: true
  sonnet5: true
  haiku: true
preferences:
  review: [sonnet5, opus5, terra]
claude_delegation:
  enabled: true
  default_tier: sonnet
"""


def _config_file(directory, workflow):
    path = Path(directory) / "router_config.yaml"
    path.write_text(CONFIG.format(workflow=workflow), encoding="utf-8")
    return path


def _load(workflow):
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(model_router, "_CONFIG_PATH", _config_file(directory, workflow)):
        return model_router._load_config()


class LoadConfigTests(unittest.TestCase):
    def test_the_codex_workflow_is_masters_behaviour(self):
        cfg = _load("codex")
        self.assertEqual(cfg["workflow"], "codex")
        self.assertEqual(cfg["preferences"], {})
        self.assertFalse(cfg["claude_delegation"]["enabled"])

    def test_the_claude_delegation_workflow_keeps_the_file_as_written(self):
        cfg = _load("claude_delegation")
        self.assertEqual(cfg["preferences"], {"review": ["sonnet5", "opus5", "terra"]})
        self.assertTrue(cfg["claude_delegation"]["enabled"])

    def test_the_value_is_read_case_and_space_insensitively(self):
        self.assertEqual(_load("' Codex '")["preferences"], {})

    def test_an_absent_or_unknown_workflow_changes_nothing(self):
        for value in ("''", "somethingelse"):
            cfg = _load(value)
            self.assertEqual(model_router.workflow_name(cfg), "claude_delegation")
            self.assertTrue(cfg["claude_delegation"]["enabled"])
            self.assertIn("review", cfg["preferences"])


def _delegation_cfg(workflow="claude_delegation", enabled=True):
    return {
        "workflow": workflow,
        "callable": {"opus5": True, "sonnet5": True, "haiku": True},
        "claude_delegation": {"enabled": enabled},
    }


class AvailabilityTests(unittest.TestCase):
    def test_a_disabled_block_is_unavailable(self):
        self.assertIn("enabled", claude_delegation.availability_block(_delegation_cfg(enabled=False)))

    def test_every_claude_target_switched_off_is_unavailable(self):
        cfg = _delegation_cfg()
        cfg["callable"] = {"opus5": False, "sonnet5": False, "haiku": False}
        self.assertIn("switched off", claude_delegation.availability_block(cfg))

    def test_an_enabled_block_with_a_callable_target_is_available(self):
        self.assertEqual(claude_delegation.availability_block(_delegation_cfg()), "")

    def test_the_check_fn_reads_the_live_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _config_file(directory, "claude_delegation")
            with patch.object(model_router, "_CONFIG_PATH", path):
                self.assertTrue(claude_delegation.tool_available())
                path.write_text(CONFIG.format(workflow="codex"), encoding="utf-8")
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

    def test_the_codex_workflow_still_registers_so_switching_back_needs_no_restart(self):
        registered, ctx = self._register(_delegation_cfg(workflow="codex", enabled=False))
        self.assertTrue(registered)
        self.assertIs(ctx.register_tool.call_args.kwargs["check_fn"], claude_delegation.tool_available)

    def test_a_host_without_the_api_still_does_not_register(self):
        registered, ctx = self._register(_delegation_cfg(), host=(False, "delegate_task lacks credentials_cfg"))
        self.assertFalse(registered)
        ctx.register_tool.assert_not_called()

    def test_the_check_fn_is_never_ttl_cached(self):
        registry = types.ModuleType("tools.registry")
        marked = []
        registry.no_cache_check_fn = lambda fn: marked.append(fn) or fn
        tools = types.ModuleType("tools")
        tools.registry = registry
        with patch.dict(sys.modules, {"tools": tools, "tools.registry": registry}):
            self._register(_delegation_cfg())
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

    def _active_during_routing(self, workflow, tools):
        seen = []

        def inner(**kwargs):
            seen.append(claude_delegation.is_active())
            return None

        request = {"messages": [{"role": "user", "content": "hi"}],
                   "tools": [{"type": "function", "function": {"name": name}} for name in tools]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_CONFIG_PATH", _config_file(directory, workflow)), \
             patch.object(model_router, "_route_llm_request", side_effect=inner):
            model_router.route_llm_request(model="gpt-terra", request=request)
        return seen[0]

    def test_a_request_that_offers_the_tool_under_claude_delegation_is_active(self):
        self.assertTrue(self._active_during_routing("claude_delegation", ["delegate_task", "delegate_claude"]))

    def test_the_codex_workflow_is_inactive_even_where_the_session_still_has_the_tool(self):
        self.assertFalse(self._active_during_routing("codex", ["delegate_task", "delegate_claude"]))

    def test_a_session_built_without_the_tool_is_never_told_to_call_it(self):
        self.assertFalse(self._active_during_routing("claude_delegation", ["delegate_task"]))


class HostToolCacheTests(unittest.TestCase):
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

    def test_the_gateway_hook_notes_the_live_config_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_CONFIG_PATH", _config_file(directory, "codex")), \
             patch.object(claude_delegation, "note_availability") as note:
            self.assertIsNone(model_router.on_pre_gateway_dispatch(event=None))
        note.assert_called_once_with(False)

    def test_register_wires_the_gateway_hook(self):
        ctx = MagicMock()
        with patch.object(claude_delegation, "register"):
            model_router.register(ctx)
        hooks = [call.args[0] for call in ctx.register_hook.call_args_list]
        self.assertIn("pre_gateway_dispatch", hooks)


class HandlerTests(unittest.TestCase):
    def test_a_call_in_the_codex_workflow_is_refused_and_names_the_route(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_CONFIG_PATH", _config_file(directory, "codex")):
            payload = json.loads(claude_delegation.handle_delegate_claude({"tasks": [{"goal": "g"}]}))
        self.assertIn("workflow: codex", payload["error"])
        self.assertIn("delegate_task", payload["error"])


if __name__ == "__main__":
    unittest.main()


class AllClaudeTransportsTests(unittest.TestCase):
    def test_codex_workflow_excludes_native_claude_targets_from_advice(self):
        cfg = _load("codex")
        for tier in ("opus5", "sonnet5", "haiku"):
            self.assertFalse(model_router._target_is_offered(tier, cfg))
        self.assertTrue(model_router._target_is_offered("terra", cfg))

    def test_codex_workflow_refuses_claude_worker_execution(self):
        from model_router.worker_admission import refusal
        self.assertIn("workflow: codex", refusal("anthropic", "claude-sonnet-5", _load("codex")))
        self.assertEqual(refusal("openai-codex", "gpt-terra", {"workflow": "codex"}), "")

    def test_codex_workflow_cannot_launch_cli_reviews(self):
        cfg = {"workflow": "codex", "callable": {"sonnet5": True, "opus5": True},
               "coding_agent": {"enabled": True, "delegated_review": {"enabled": True}}}
        with patch.object(model_router, "_run_opus5_bridge") as bridge:
            self.assertIsNone(model_router._maybe_run_opus5(
                {"messages": [{"role": "user", "content": "[sonnet-review] Review parser"}]},
                cfg, platform="subagent", api_mode="codex_responses"))
        bridge.assert_not_called()
