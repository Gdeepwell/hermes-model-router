import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import web_viewer


class SettingsTransactionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.shipped = self.root / 'router_config.yaml'
        self.hermes = self.root / 'config.yaml'
        self.shipped.write_text('''default_model: terra
models: {terra: gpt-terra, sol: gpt-sol}
callable: {terra: true, sol: true}
preferences: {}
''')
        self.hermes.write_text('''model: {default: gpt-terra, provider: openai-codex}
fallback_providers: []
''')
        for key, value in [('CONFIG_PATH', self.shipped), ('HERMES_CONFIG_PATH', self.hermes)]:
            mock = patch.object(web_viewer, key, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.local = web_viewer._local_config_path()

    def test_invalid_later_field_cannot_partially_write_hermes(self):
        before = self.hermes.read_bytes()
        status, _ = web_viewer._save_config_payload({
            'hermes_fallback': {'orchestrator': [{'provider': 'openai-codex', 'model': 'gpt-sol'}]},
            'preferences': 'invalid',
        })
        self.assertEqual(status, 400)
        self.assertEqual(self.hermes.read_bytes(), before)
        self.assertFalse(self.local.exists())

    def test_unparseable_hermes_config_refusal_names_the_yaml_error(self):
        self.hermes.write_text('plugins:\n  enabled:\n    - a\n  - b\n')
        before = self.hermes.read_bytes()
        message = web_viewer._save_hermes_fallback({'orchestrator': []}, {})
        self.assertIn('refusing to overwrite', message)
        self.assertIn('not valid YAML', message)
        self.assertIn('line 4', message)
        self.assertEqual(self.hermes.read_bytes(), before)

    def test_local_write_failure_restores_exact_hermes_content(self):
        before = self.hermes.read_bytes()
        real = web_viewer._atomic_write
        def fail_local(path, content):
            if path == self.local:
                raise OSError('disk full')
            return real(path, content)
        with patch.object(web_viewer, '_atomic_write', side_effect=fail_local):
            with self.assertRaisesRegex(OSError, 'disk full'):
                web_viewer._save_config_payload({'default_model': 'sol'})
        self.assertEqual(self.hermes.read_bytes(), before)
        self.assertFalse(self.local.exists())

    def test_concurrent_stale_revision_is_rejected_without_losing_first_save(self):
        revision = web_viewer._config_revision()
        barrier = threading.Barrier(2)
        def save(tier):
            barrier.wait()
            return web_viewer._save_config_payload({'revision': revision, 'callable': {tier: False}})
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(save, ['terra', 'sol']))
        self.assertEqual(sorted(status for status, _ in results), [200, 409])
        saved = web_viewer._read_router_config()['callable']
        self.assertEqual(sum(value is False for value in saved.values()), 1)

    def test_both_host_changes_commit_together(self):
        status, result = web_viewer._save_config_payload({
            'default_model': 'sol',
            'hermes_fallback': {'orchestrator': [{'provider': 'openai-codex', 'model': 'gpt-terra'}]},
        })
        self.assertEqual(status, 200)
        self.assertEqual(result['revision'], web_viewer._config_revision())
        hermes = web_viewer.yaml.safe_load(self.hermes.read_text())
        self.assertEqual(hermes['model']['default'], 'gpt-sol')
        self.assertEqual(hermes['fallback_providers'][0]['model'], 'gpt-terra')
        self.assertEqual(web_viewer._read_router_config()['default_model'], 'sol')
