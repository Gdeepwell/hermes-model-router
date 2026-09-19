"""router_config.yaml ships master's behaviour; an operator's settings live in
router_config.local.yaml, git-ignored and layered on top.

The shipped file used to double as one operator's live config, so a pull request
carried that operator's workflow, chains and limits into everyone else's router.
Now the router and the dashboard read the shipped file with the local one merged
over it, and the dashboard writes only what differs from the shipped file into
the local one.
"""

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml

import model_router
import web_viewer

ROOT = Path(model_router.__file__).resolve().parent

SHIPPED = """\
enabled: true
workflow: codex
models:
  terra: gpt-5.6-terra
  qwen: qwen3.7-plus
callable:
  terra: true
  qwen: true
  sonnet5: true
preferences: {}
default_model: qwen
tier_providers:
  terra: openai-codex
  qwen: qwen-token
  sonnet5: anthropic
claude_delegation:
  enabled: false
  default_tier: sonnet
usage_guard:
  accounts:
    anthropic:
      soft_percent: 70
      hard_percent: 90
"""

LOCAL = """\
# My own settings.
workflow: claude_delegation
default_model: terra
claude_delegation:
  enabled: true
usage_guard:
  accounts:
    anthropic:
      soft_percent: 80   # the parent needs headroom
"""


def _files(directory, local=LOCAL):
    shipped = Path(directory) / "router_config.yaml"
    shipped.write_text(SHIPPED, encoding="utf-8")
    if local is not None:
        (Path(directory) / "router_config.local.yaml").write_text(local, encoding="utf-8")
    return shipped


class RouterLayeringTests(unittest.TestCase):
    def _load(self, local=LOCAL):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(model_router, "_CONFIG_PATH", _files(directory, local)):
            return model_router._load_config()

    def test_the_local_file_overrides_the_shipped_one_key_by_key(self):
        cfg = self._load()
        self.assertEqual(cfg["workflow"], "claude_delegation")
        self.assertEqual(cfg["default_model"], "terra")
        self.assertTrue(cfg["claude_delegation"]["enabled"])
        self.assertEqual(cfg["claude_delegation"]["default_tier"], "sonnet", "shipped keys under it survive")
        self.assertEqual(cfg["usage_guard"]["accounts"]["anthropic"], {"soft_percent": 80, "hard_percent": 90})

    def test_without_a_local_file_the_shipped_behaviour_applies(self):
        cfg = self._load(local=None)
        self.assertEqual(cfg["workflow"], "codex")
        self.assertEqual(cfg["default_model"], "qwen")
        self.assertFalse(cfg["claude_delegation"]["enabled"])

    def test_a_broken_local_file_falls_back_to_the_shipped_one_not_to_nothing(self):
        cfg = self._load(local="workflow: [unclosed\n")
        self.assertEqual((cfg["workflow"], cfg["default_model"]), ("codex", "qwen"))

    def test_the_local_file_sits_beside_whichever_config_is_loaded(self):
        with patch.object(model_router, "_CONFIG_PATH", Path("/somewhere/router_config.yaml")):
            self.assertEqual(model_router._local_config_path(), Path("/somewhere/router_config.local.yaml"))


class DashboardLayeringTests(unittest.TestCase):
    def _serve(self, directory, calls, local=LOCAL):
        shipped = _files(directory, local)
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_viewer.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        results = []
        try:
            with patch.object(web_viewer, "CONFIG_PATH", shipped):
                for method, body in calls:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/config",
                        data=None if body is None else json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"}, method=method)
                    with urllib.request.urlopen(request) as response:
                        results.append(json.load(response))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        local_path = shipped.with_name("router_config.local.yaml")
        return (shipped.read_text(encoding="utf-8"),
                local_path.read_text(encoding="utf-8") if local_path.exists() else None, results)

    def test_the_dashboard_reads_the_layered_config(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, results = self._serve(directory, [("GET", None)])
        self.assertEqual((results[0]["workflow"], results[0]["default_model"]), ("claude_delegation", "terra"))

    def test_a_save_writes_the_local_file_only(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped, local, _ = self._serve(directory, [("POST", {"workflow": "codex"})])
        self.assertEqual(shipped, SHIPPED)
        self.assertNotIn("workflow:", local, "codex equals the shipped value, so it needs no override")
        self.assertIn("default_model: terra", local)

    def test_the_local_file_holds_only_what_differs_and_keeps_its_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, _ = self._serve(directory, [("POST", {"callable": {"terra": True, "qwen": False,
                                                                           "sonnet5": True}})])
        loaded = yaml.safe_load(local)
        self.assertEqual(loaded["callable"], {"qwen": False})
        self.assertIn("# My own settings.", local)
        self.assertIn("# the parent needs headroom", local)

    def test_a_save_with_nothing_to_override_creates_no_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, _ = self._serve(directory, [("POST", {"workflow": "codex"})], local=None)
        self.assertIsNone(local)

    def test_a_first_override_creates_the_local_file_with_a_header(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped, local, _ = self._serve(directory, [("POST", {"workflow": "claude_delegation"})], local=None)
        self.assertEqual(shipped, SHIPPED)
        self.assertTrue(local.startswith("#"))
        self.assertEqual(yaml.safe_load(local)["workflow"], "claude_delegation")
        self.assertTrue(yaml.safe_load(local)["claude_delegation"]["enabled"])


class ShippedDefaultsTests(unittest.TestCase):
    """The committed file is master's behaviour: nothing changes until an operator opts in."""

    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))

    def test_the_shipped_config_is_masters_behaviour(self):
        self.assertEqual(self.cfg["workflow"], "codex")
        self.assertEqual(self.cfg["preferences"], {})
        self.assertEqual(self.cfg["default_model"], "qwen")
        self.assertTrue(self.cfg["callable"]["qwen"])
        self.assertFalse(self.cfg["claude_delegation"]["enabled"])

    def test_the_local_file_is_ignored_by_git(self):
        self.assertIn("router_config.local.yaml", (ROOT / ".gitignore").read_text(encoding="utf-8").split())


if __name__ == "__main__":
    unittest.main()
