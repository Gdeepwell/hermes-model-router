import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import model_router as router
from model_router import worker_admission, usage_guard
from model_router.test_usage_guard import ROUTER_CFG

class AdmissionTests(unittest.TestCase):
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
