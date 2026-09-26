import subprocess
import tempfile
import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import model_router as router
from model_router import worker_admission, usage_guard
from model_router.test_usage_guard import ROUTER_CFG

class AdmissionTests(unittest.TestCase):
    def test_live_switch_stops_existing_claude_child_on_its_next_call(self):
        switches = {"sonnet5": True}
        downstream = Mock(return_value="Claude continued")
        request = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "Continue"}]}
        with patch.object(router, "_load_config", side_effect=lambda: {"enabled": True, "callable": dict(switches)}):
            kwargs = dict(request=request, original_request=request, next_call=downstream,
                          provider="anthropic", platform="subagent", turn_id="root:sa-1")
            self.assertEqual(router.run_llm_with_transient_failover(**kwargs), "Claude continued")
            switches["sonnet5"] = False
            stopped = router.run_llm_with_transient_failover(**kwargs)
        self.assertIn("ROUTER WORKER STOPPED", stopped.output_text)
        self.assertIn("switched off in Settings", stopped.output_text)
        downstream.assert_called_once()

    def test_anthropic_fallback_child_obeys_the_current_claude_switch(self):
        request = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "Continue"}]}
        downstream = Mock(return_value="Fallback child ran")
        for on, should_run in ((True, True), (False, False)):
            with self.subTest(sonnet5=on), patch.object(router, "_load_config", return_value={
                    "enabled": True, "callable": {"sonnet5": on}}):
                result = router.run_llm_with_transient_failover(
                    request=request, original_request=request, next_call=downstream,
                    provider="anthropic", platform="subagent", turn_id="root:sa-2")
                self.assertEqual(result == "Fallback child ran", should_run)
        self.assertEqual(downstream.call_count, 1)

    def test_both_accounts_stop_at_execution_without_calling_provider(self):
        from hermes_cli.middleware import run_llm_execution_middleware
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        for provider in ('openai-codex', 'anthropic'):
            call = Mock()
            with patch.object(router, '_load_config', return_value=ROUTER_CFG), \
                 patch.object(usage_guard, 'peek', return_value=usage_guard.Reading(95, 0, None, None, time.time())), \
                 patch('hermes_cli.plugins._delivery_manager', return_value=manager):
                result = run_llm_execution_middleware({'model': 'worker'}, call, provider=provider,
                                                      platform='subagent', turn_id='s:sa-1')
            call.assert_not_called()
            manager._report_hook_failure.assert_not_called()
            self.assertIn('ROUTER WORKER STOPPED', result.output_text)
            self.assertEqual(result.usage.total_tokens, 0)

    def test_closed_codex_refuses_spawn_but_allows_control(self):
        call = Mock(return_value='controlled')
        with patch.object(router, '_load_config', return_value=ROUTER_CFG), \
             patch.object(worker_admission, 'delegate_task_route', return_value=('openai-codex', 'gpt-terra')), \
             patch.object(usage_guard, 'read', return_value=usage_guard.Reading(95, 0, None, None, time.time())):
            result = worker_admission.guard_tool_execution(tool_name='delegate_task',
                        args={'tasks': [{'goal': 'Implement parser'}]}, next_call=call)
            call.assert_not_called()
            self.assertIn('closed', result)
            self.assertEqual(worker_admission.guard_tool_execution(tool_name='delegate_task',
                             args={'action': 'stop'}, next_call=call), 'controlled')

    def test_resolvable_review_uses_the_claude_account_at_dispatch(self):
        call = Mock(return_value='admitted')
        cfg = {
            'enabled': True, 'callable': {'sonnet5': True},
            'coding_agent': {'delegated_review': {'enabled': True, 'models': ['sonnet']}},
            'usage_guard': {'accounts': {
                'openai-codex': {'soft_percent': 70, 'hard_percent': 90},
                'anthropic': {'soft_percent': 70, 'hard_percent': 90},
            }},
        }
        with patch.object(router, '_load_config', return_value=cfg), \
             patch.object(worker_admission, 'delegate_task_route', return_value=('openai-codex', 'gpt-terra')), \
             patch.object(router, '_delegated_claude_review_status',
                          return_value=((object(), 'sonnet'), '')) as status, \
             patch.object(worker_admission, 'refusal', return_value='') as refusal:
            result = worker_admission.guard_tool_execution(
                tool_name='delegate_task', args={'tasks': [{'goal': '[sonnet-review] Review parser'}]}, next_call=call)
        self.assertEqual(result, 'admitted')
        status.assert_called_once()
        refusal.assert_called_once_with('anthropic', 'claude-sonnet-5', cfg, blocking=True)

    def test_resolvable_review_refuses_with_the_claude_account_message(self):
        call = Mock(return_value='admitted')
        cfg = {'enabled': True, 'callable': {'sonnet5': True},
               'coding_agent': {'delegated_review': {'enabled': True}}}
        with patch.object(router, '_load_config', return_value=cfg), \
             patch.object(worker_admission, 'delegate_task_route', return_value=('openai-codex', 'gpt-terra')), \
             patch.object(router, '_delegated_claude_review_status',
                          return_value=((object(), 'sonnet'), '')), \
             patch.object(worker_admission, 'refusal', return_value='Claude account closed') as refusal:
            result = worker_admission.guard_tool_execution(
                tool_name='delegate_task', args={'goal': '[sonnet-review] Review parser'}, next_call=call)
        call.assert_not_called()
        refusal.assert_called_once_with('anthropic', 'claude-sonnet-5', cfg, blocking=True)
        self.assertIn('Claude account closed', result)

    def test_review_without_a_resolvable_repository_keeps_the_codex_check(self):
        call = Mock(return_value='admitted')
        cfg = {'enabled': True, 'callable': {'sonnet5': True},
               'coding_agent': {'delegated_review': {'enabled': True}}}
        with patch.object(router, '_load_config', return_value=cfg), \
             patch.object(worker_admission, 'delegate_task_route', return_value=('openai-codex', 'gpt-terra')), \
             patch.object(router, '_delegated_claude_review_status', return_value=(None, 'no repository')), \
             patch.object(worker_admission, 'refusal', return_value='Codex account closed') as refusal:
            result = worker_admission.guard_tool_execution(
                tool_name='delegate_task', args={'goal': '[sonnet-review] Review parser'}, next_call=call)
        call.assert_not_called()
        refusal.assert_called_once_with('openai-codex', 'gpt-terra', cfg, blocking=True)
        self.assertIn('Codex account closed', result)

    def test_resolvable_review_bypasses_a_closed_codex_account_for_an_open_claude_account(self):
        call = Mock(return_value='admitted')
        cfg = {
            'enabled': True, 'callable': {'sonnet5': True},
            'coding_agent': {'delegated_review': {'enabled': True, 'models': ['sonnet']}},
            'usage_guard': {'accounts': {
                'openai-codex': {'soft_percent': 70, 'hard_percent': 90},
                'anthropic': {'soft_percent': 70, 'hard_percent': 90},
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            repo = f'{directory}/repo'
            subprocess.run(['git', 'init', repo], check=True, capture_output=True)
            def reading(account, _cfg):
                weekly = 10 if account == 'anthropic' else 95
                return usage_guard.Reading(weekly, 0, None, None, time.time())
            with patch.object(router, '_load_config', return_value=cfg), \
                 patch.object(worker_admission, 'delegate_task_route', return_value=('openai-codex', 'gpt-terra')), \
                 patch('model_router.shutil.which', return_value='/claude'), \
                 patch.object(usage_guard, 'read', side_effect=reading) as read:
                result = worker_admission.guard_tool_execution(
                    tool_name='delegate_task',
                    args={'goal': f'[sonnet-review] Review repository {repo}'}, next_call=call)
        self.assertEqual(result, 'admitted')
        call.assert_called_once()
        self.assertEqual([entry.args[0] for entry in read.call_args_list], ['anthropic'])

    def test_resolvable_review_refuses_against_a_closed_claude_account(self):
        call = Mock(return_value='admitted')
        cfg = {
            'enabled': True, 'callable': {'sonnet5': True},
            'coding_agent': {'delegated_review': {'enabled': True, 'models': ['sonnet']}},
            'usage_guard': {'accounts': {
                'openai-codex': {'soft_percent': 70, 'hard_percent': 90},
                'anthropic': {'soft_percent': 70, 'hard_percent': 90},
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            repo = f'{directory}/repo'
            subprocess.run(['git', 'init', repo], check=True, capture_output=True)
            with patch.object(router, '_load_config', return_value=cfg), \
                 patch.object(worker_admission, 'delegate_task_route', return_value=('openai-codex', 'gpt-terra')), \
                 patch('model_router.shutil.which', return_value='/claude'), \
                 patch.object(usage_guard, 'read', return_value=usage_guard.Reading(95, 0, None, None, time.time())) as read:
                result = worker_admission.guard_tool_execution(
                    tool_name='delegate_task',
                    args={'goal': f'[sonnet-review] Review repository {repo}'}, next_call=call)
        call.assert_not_called()
        self.assertIn('Claude delegation closed: weekly usage 95% (hard limit 90%)', result)
        self.assertEqual([entry.args[0] for entry in read.call_args_list], ['anthropic'])
