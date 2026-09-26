import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import model_router as router
from model_router import _maybe_run_opus5, route_llm_request, usage_guard


class BridgePolicyTests(unittest.TestCase):
    def run_bridge(self, label, switches, weekly=10):
        cfg = {"callable": switches, "coding_agent": {"delegated_review": {"enabled": True}},
               "usage_guard": {"accounts": {"anthropic": {
                   "soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}}}}}
        request = {"messages": [{"role": "user", "content": f"[{label}-review] Review parser"}]}
        with patch("model_router._verified_delegated_claude_review", return_value=(Path("/tmp"), label)), \
             patch("model_router.usage_guard.read", return_value=usage_guard.Reading(weekly, 10, None, None, time.time())), \
             patch("model_router._run_opus5_bridge", return_value={"result": "reviewed", "model": "claude-sonnet-5"}) as bridge:
            _maybe_run_opus5(request, cfg, platform="subagent", api_mode="codex_responses")
        return bridge

    def test_sonnet_obeys_its_own_switch(self):
        self.run_bridge("sonnet", {"sonnet5": False, "opus5": True}).assert_not_called()
        self.run_bridge("sonnet", {"sonnet5": True, "opus5": False}).assert_called_once()

    def test_hard_limit_blocks_both_cli_tiers(self):
        for label in ("opus", "sonnet"):
            self.run_bridge(label, {"sonnet5": True, "opus5": True}, weekly=95).assert_not_called()

    def test_soft_limit_selects_only_an_enabled_lighter_tier(self):
        bridge = self.run_bridge("opus", {"sonnet5": True, "opus5": True}, weekly=75)
        self.assertEqual(bridge.call_args.kwargs["model"], "sonnet")
        self.assertEqual(bridge.call_args.kwargs["requested_alias"], "opus")
        self.assertIn("opus5→sonnet5", bridge.call_args.kwargs["adjustment"])
        self.run_bridge("opus", {"sonnet5": False, "opus5": True}, weekly=75).assert_not_called()


class DelegatedReviewRepositoryTests(unittest.TestCase):
    def _config(self, **coding_overrides):
        coding = {"delegated_review": {"enabled": True, "models": ["sonnet", "opus"]}}
        coding.update(coding_overrides)
        return {"callable": {"sonnet5": True, "opus5": True}, "coding_agent": coding}

    def _git_repo(self, directory):
        repo = Path(directory) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
        return repo

    def test_goal_absolute_path_uses_its_git_top_level(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            child = repo / "nested"
            child.mkdir()
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    f"[sonnet-review] Review repository {child}.", self._config())
        self.assertEqual(routed, (repo.resolve(), "sonnet"))

    def test_git_probe_timeout_is_not_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("model_router.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)) as run:
                resolved = router._repo_directory(directory, git_top_level=True)
        self.assertIsNone(resolved)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_goal_repository_skips_a_non_git_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch-notes"
            scratch.mkdir()
            resolved = router._goal_repository(f"[sonnet-review] Review {scratch}")
        self.assertIsNone(resolved)

    def test_goal_repository_skips_a_non_git_directory_before_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch-notes"
            scratch.mkdir()
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review {scratch} then {repo}")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_resolves_a_backticked_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review `{repo}`")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_resolves_a_parenthesised_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review ({repo})")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_still_resolves_a_plain_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review {repo}")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_does_not_take_a_url_as_a_path(self):
        self.assertIsNone(router._goal_repository("[sonnet-review] Review https://example.com/a"))

    def test_workspace_path_in_child_request_resolves_a_review_without_goal_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            request = {
                "instructions": f"WORKSPACE PATH:\n{repo}\nUse this exact path.",
                "messages": [{"role": "user", "content": "[sonnet-review] Review parser"}],
            }
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    "[sonnet-review] Review parser", self._config(), request=request)
        self.assertEqual(routed, (repo.resolve(), "sonnet"))

    def test_workspace_path_in_a_chat_system_message_resolves_a_review(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            request = {
                "messages": [
                    {"role": "system", "content": f"WORKSPACE PATH:\n{repo}\nUse this exact path."},
                    {"role": "user", "content": "[sonnet-review] Review parser"},
                ],
            }
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    "[sonnet-review] Review parser", self._config(), request=request)
        self.assertEqual(routed, (repo.resolve(), "sonnet"))

    def test_nonexistent_goal_and_shipped_aliases_are_skipped(self):
        cfg = self._config(repo_aliases={"router": "/home/deepwell/hermes-model-router"})
        with patch("model_router.shutil.which", return_value="/claude"):
            routed = router._verified_delegated_claude_review(
                "[sonnet-review] Review /not/a/repository, router", cfg)
        self.assertIsNone(routed)

    def test_resolved_sonnet_review_calls_the_cli_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            request = {
                "instructions": f"WORKSPACE PATH:\n{repo}\nUse this exact path.",
                "messages": [{"role": "user", "content": "[sonnet-review] Review parser"}],
            }
            with patch("model_router.shutil.which", return_value="/claude"), \
                 patch("model_router.usage_guard.read", return_value=usage_guard.Reading(10, 0, None, None, time.time())), \
                 patch("model_router._run_opus5_bridge", return_value={
                     "result": "reviewed", "effective_model": "claude-sonnet-5"}) as bridge:
                result = _maybe_run_opus5(request, self._config(), platform="subagent",
                                          api_mode="codex_responses")
        self.assertEqual(result.model, "claude-sonnet-5")
        bridge.assert_called_once()
        self.assertEqual(bridge.call_args.kwargs["repo"], str(repo.resolve()))

    def test_route_reason_names_a_missing_claude_cli_for_a_review_leaf(self):
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": {"terra": "gpt-terra"},
            "callable": {"terra": True, "sonnet5": True},
            "tier_providers": {"terra": "openai-codex", "sonnet5": "openai-codex"},
            "coding_agent": {"delegated_review": {"enabled": True, "models": ["sonnet"]}},
        }
        request = {"model": "gpt-terra", "messages": [{"role": "user", "content":
                   "[sonnet-review] Review parser"}]}
        with patch("model_router._load_config", return_value=cfg), \
             patch("model_router.shutil.which", return_value=None), \
             patch("model_router._log_decision"):
            routed = route_llm_request(request=request, provider="openai-codex", model="gpt-terra",
                                       platform="subagent", turn_id="root:sa-1", api_call_count=1)
        self.assertIn("Claude review not taken (Claude CLI is unavailable)", routed["reason"])


class AccountOfExecutionTests(unittest.TestCase):
    def _route(self, *, codex, claude, bridge_error=None, label='sonnet', worker=True, eligible=True):
        from unittest.mock import Mock
        from model_router import run_llm_with_transient_failover
        cfg = {
            'enabled': True, 'provider': 'openai-codex',
            'callable': {'opus5': True, 'sonnet5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
            'usage_guard': {'accounts': {
                'anthropic': {'soft_percent': 70, 'hard_percent': 90},
                'openai-codex': {'soft_percent': 70, 'hard_percent': 90},
            }},
        }
        reading = lambda weekly: usage_guard.Reading(weekly, 10, None, None, time.time())
        downstream = Mock(return_value='Codex ran')
        bridge = Mock(return_value={'result': 'Claude reviewed', 'effective_model':
                                    'claude-sonnet-5' if label == 'sonnet' else 'claude-opus-5-5'})
        if bridge_error:
            bridge.side_effect = bridge_error
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   f'[{label}-review] Review parser'}]}
        with patch('model_router._load_config', return_value=cfg), \
             patch('model_router._verified_delegated_claude_review',
                   return_value=(Path('/tmp'), label) if eligible else None), \
             patch('model_router.usage_guard.read', return_value=reading(claude)) as claude_read, \
             patch('model_router.usage_guard.peek', return_value=reading(codex)), \
             patch('model_router._run_opus5_bridge', bridge):
            result = run_llm_with_transient_failover(
                request=request, original_request=request, next_call=downstream,
                provider='openai-codex', api_mode='codex_responses', api_call_count=1,
                platform='subagent' if worker else 'cli', turn_id='s:sa-1' if worker else 'root')
        return result, downstream, bridge, claude_read

    def test_closed_codex_does_not_block_healthy_claude(self):
        result, codex, bridge, _ = self._route(codex=95, claude=10)
        self.assertEqual(result.model, 'claude-sonnet-5')
        bridge.assert_called_once()
        codex.assert_not_called()

    def test_closed_claude_can_fall_back_only_to_open_codex(self):
        result, codex, bridge, _ = self._route(codex=10, claude=95)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_not_called()

    def test_both_closed_or_failed_bridge_never_reach_codex(self):
        for claude, error in ((95, None), (10, RuntimeError('CLI failed'))):
            result, codex, bridge, _ = self._route(codex=95, claude=claude, bridge_error=error)
            self.assertIn('ROUTER WORKER STOPPED', result.output_text)
            codex.assert_not_called()
            self.assertEqual(bridge.call_count, 0 if claude == 95 else 1)

    def test_no_eligible_bridge_does_not_probe_claude(self):
        result, codex, bridge, read = self._route(codex=10, claude=10, eligible=False)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_not_called()
        read.assert_not_called()

    def test_root_is_not_stopped_by_worker_account_limit(self):
        result, codex, _, _ = self._route(codex=95, claude=95, worker=False)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
