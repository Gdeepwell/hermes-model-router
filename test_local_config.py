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
import urllib.error
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
        self.assertEqual(cfg["default_model"], "terra")
        # The legacy workflow in the local file turns Claude on and is then dropped.
        self.assertIs(cfg["callable"]["opus5"], True)
        self.assertNotIn("workflow", cfg)
        self.assertNotIn("enabled", cfg["claude_delegation"])
        self.assertEqual(cfg["claude_delegation"]["default_tier"], "sonnet", "shipped keys under it survive")
        self.assertEqual(cfg["usage_guard"]["accounts"]["anthropic"], {"soft_percent": 80, "hard_percent": 90})

    def test_without_a_local_file_the_shipped_behaviour_applies(self):
        cfg = self._load(local=None)
        self.assertEqual(cfg["default_model"], "qwen")
        self.assertNotIn("workflow", cfg, "a legacy key in the shipped file is ignored and dropped")
        self.assertIs(cfg["callable"]["opus5"], False)

    def test_a_broken_local_file_falls_back_to_the_shipped_one_not_to_nothing(self):
        cfg = self._load(local="workflow: [unclosed\n")
        self.assertEqual(cfg["default_model"], "qwen")
        self.assertNotIn("workflow", cfg)

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
        self.assertEqual(results[0]["default_model"], "terra")
        self.assertNotIn("workflow", results[0], "the workflow switch is retired")
        # The legacy workflow in the local file reads as Claude switched on.
        self.assertIs(results[0]["callable"]["sonnet5"], True)

    def test_a_save_writes_the_local_file_only(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped, local, _ = self._serve(directory, [("POST", {"default_model": "terra"})])
        self.assertEqual(shipped, SHIPPED)
        self.assertNotIn("workflow:", local, "a save drops the retired key")
        self.assertIn("default_model: terra", local)

    def test_the_local_file_holds_only_what_differs_and_keeps_its_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, _ = self._serve(directory, [("POST", {"callable": {"terra": True, "qwen": False,
                                                                           "sonnet5": True}})])
        loaded = yaml.safe_load(local)
        # The local file's legacy workflow is materialised as the Claude switch it stood for.
        self.assertEqual(loaded["callable"], {"qwen": False, "sonnet5": True})
        self.assertNotIn("workflow", loaded)
        self.assertIn("# My own settings.", local)
        self.assertIn("# the parent needs headroom", local)

    def test_a_save_with_nothing_to_override_creates_no_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, _ = self._serve(directory, [("POST", {"default_model": "qwen"})], local=None)
        self.assertIsNone(local)

    def test_a_post_still_carrying_workflow_is_ignored_not_written(self):
        """An old open tab still posts ``workflow``: accepted, and never written."""
        with tempfile.TemporaryDirectory() as directory:
            shipped, local, results = self._serve(
                directory, [("POST", {"workflow": "claude_delegation", "default_model": "terra"})], local=None)
        self.assertTrue(results[0]["success"])
        self.assertEqual(shipped, SHIPPED)
        self.assertTrue(local.startswith("#"))
        self.assertEqual(yaml.safe_load(local), {"default_model": "terra"})

    def test_a_post_still_carrying_the_delegation_flag_never_writes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, results = self._serve(
                directory, [("POST", {"claude_delegation": {"enabled": False, "default_tier": "sonnet"}})],
                local=None)
        self.assertTrue(results[0]["success"])
        self.assertIsNone(local, "enabled is ignored and default_tier equals the shipped value")


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


class LegacyClaudeSwitchMigrationTests(unittest.TestCase):
    """The owner's router_config.local.yaml, against the real shipped file.

    The router reads its legacy ``workflow: claude_delegation`` as Claude on; the
    dashboard must show the same, and its first save must write those switches
    into the local file and drop both retired keys.
    """

    def _serve(self, directory, calls, local=OWNER_LOCAL):
        shipped = Path(directory) / "router_config.yaml"
        shipped.write_text((ROOT / "router_config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        local_path = shipped.with_name("router_config.local.yaml")
        local_path.write_text(local, encoding="utf-8")
        hermes = Path(directory) / "hermes-config.yaml"
        hermes.write_text("model:\n  default: gpt-5.6-terra\n  provider: openai-codex\n", encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_viewer.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        results = []
        try:
            with patch.object(web_viewer, "CONFIG_PATH", shipped), \
                 patch.object(web_viewer, "HERMES_CONFIG_PATH", hermes):
                for method, body in calls:
                    if callable(body):
                        body = body(results)
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/config",
                        data=None if body is None else json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"}, method=method)
                    try:
                        with urllib.request.urlopen(request) as response:
                            results.append((response.status, json.load(response)))
                    except urllib.error.HTTPError as error:
                        results.append((error.code, json.load(error)))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        return shipped.read_text(encoding="utf-8"), local_path.read_text(encoding="utf-8"), results

    @staticmethod
    def _page_save(results, change):
        """What saveSettings() posts: the whole page state from the last GET, plus one change."""
        page = results[-1][1]
        payload = {
            "callable": dict(page["callable"]),
            "workflow": "claude_delegation",  # an old open tab still sends it
            "balance": {k: page["balance"][k] for k in ("enabled", "busy_percent", "margin_percent")},
            "default_model": page["default_model"],
            "effort": {k: v for k, v in page["effort"].items() if v},
            "claude_reasoning_effort": page["claude_reasoning_effort"]["levels"],
            "preferences": page["preferences"],
            "hermes_fallback": {},
            "usage_limits": {a: {"soft_percent": i["soft_percent"], "hard_percent": i["hard_percent"]}
                             for a, i in page["accounts"].items() if i["guard"]},
            "claude_delegation": {"default_tier": page["accounts"]["anthropic"]["delegation"]["default_tier"]},
            "revision": page["revision"],
        }
        payload.update(change)
        return payload

    def test_the_dashboard_shows_the_claude_switches_the_router_uses(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, results = self._serve(directory, [("GET", None)])
            with patch.object(model_router, "_CONFIG_PATH", Path(directory) / "router_config.yaml"):
                router_view = model_router._load_config()["callable"]
        page = results[0][1]
        for model in ("opus5", "sonnet5", "haiku"):
            self.assertIs(page["callable"][model], True, model)
            self.assertIs(router_view[model], True, model)
        self.assertIs(page["callable"]["qwen"], False)
        self.assertNotIn("workflow", page)
        delegation = page["accounts"]["anthropic"]["delegation"]
        self.assertIs(delegation["enabled"], True)
        self.assertNotIn("workflow", delegation)
        self.assertEqual(local, OWNER_LOCAL, "a GET never writes")

    def test_the_first_full_page_save_materialises_the_switches_and_drops_the_legacy_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            shipped, local, results = self._serve(directory, [
                ("GET", None),
                ("POST", lambda r: self._page_save(r, {"balance": {"enabled": True, "busy_percent": 30,
                                                                   "margin_percent": 10}})),
                ("GET", None),
            ])
        self.assertEqual(results[1], (200, results[1][1]))
        self.assertTrue(results[1][1]["success"], results[1])
        self.assertEqual(shipped, (ROOT / "router_config.yaml").read_text(encoding="utf-8"))
        written = yaml.safe_load(local)
        self.assertEqual(written["callable"], {"opus5": True, "sonnet5": True, "haiku": True, "qwen": False})
        self.assertNotIn("workflow", written)
        self.assertNotIn("claude_delegation", written, "only enabled was there, so the block goes")
        self.assertEqual(written["usage_guard"]["balance"], {"busy_percent": 30}, "the unrelated change landed")
        # Everything else the owner had is unchanged, and nothing else was pinned.
        self.assertEqual(written["preferences"], yaml.safe_load(OWNER_LOCAL)["preferences"])
        self.assertEqual(written["default_model"], "terra")
        self.assertEqual(written["usage_guard"]["accounts"],
                         {"anthropic": {"soft_percent": 80}, "openai-codex": {"soft_percent": 80}})
        self.assertEqual(set(written), {"preferences", "default_model", "usage_guard", "callable"})
        self.assertIn("# Your own router settings", local)
        self.assertIn("design: [sol, opus5]", local, "flow lists survive the in-place update")
        for model in ("opus5", "sonnet5", "haiku"):
            self.assertIs(results[2][1]["callable"][model], True, "still on after the migration")

    def test_a_legacy_codex_workflow_materialises_claude_off(self):
        local = OWNER_LOCAL.replace("workflow: claude_delegation", "workflow: codex").replace(
            "callable:\n  qwen: false\n", "callable:\n  qwen: false\n  sonnet5: true\n")
        with tempfile.TemporaryDirectory() as directory:
            _, written_text, results = self._serve(directory, [
                ("GET", None), ("POST", lambda r: self._page_save(r, {"default_model": "terra"}))], local=local)
        self.assertEqual([results[0][1]["callable"][m] for m in ("opus5", "sonnet5", "haiku")], [False] * 3)
        written = yaml.safe_load(written_text)
        self.assertNotIn("workflow", written)
        self.assertEqual(written["callable"], {"qwen": False, "sonnet5": False, "opus5": False, "haiku": False},
                         "the verdict is written explicitly, even where it equals today's shipped default")

    def test_a_switch_turned_off_on_the_page_is_what_gets_written(self):
        with tempfile.TemporaryDirectory() as directory:
            _, local, _ = self._serve(directory, [
                ("GET", None),
                ("POST", lambda r: self._page_save(r, {"callable": {**r[-1][1]["callable"], "opus5": False}})),
            ])
        self.assertEqual(yaml.safe_load(local)["callable"],
                         {"qwen": False, "opus5": False, "sonnet5": True, "haiku": True})

    def test_after_the_migration_an_ordinary_save_writes_only_the_delta(self):
        migrated = OWNER_LOCAL.replace("workflow: claude_delegation\nclaude_delegation:\n  enabled: true\n", "")
        migrated = migrated.replace("callable:\n  qwen: false\n",
                                    "callable:\n  qwen: false\n  opus5: true\n  sonnet5: true\n  haiku: true\n")
        with tempfile.TemporaryDirectory() as directory:
            _, local, _ = self._serve(directory, [
                ("GET", None),
                ("POST", lambda r: self._page_save(r, {"callable": {**r[-1][1]["callable"], "haiku": False}})),
            ], local=migrated)
        self.assertEqual(yaml.safe_load(local)["callable"], {"qwen": False, "opus5": True, "sonnet5": True},
                         "haiku back at the shipped default leaves the file")

    def test_a_dashboard_without_the_router_refuses_to_guess_the_switches(self):
        """Fail closed: without the router's translation the GET says why instead of showing Claude off."""
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(web_viewer, "_router_module", return_value=None):
            _, local, results = self._serve(directory, [("GET", None), ("POST", {"default_model": "terra"})])
        self.assertEqual(results[0][0], 500)
        self.assertIn("router", results[0][1]["error"].lower())
        self.assertEqual(results[1][0], 500)
        self.assertEqual(local, OWNER_LOCAL, "nothing written")


class ShippedDefaultsTests(unittest.TestCase):
    """The committed file is master's behaviour: nothing changes until an operator opts in."""

    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))

    def test_the_shipped_config_is_masters_behaviour(self):
        self.assertNotIn("workflow", self.cfg)
        self.assertEqual(self.cfg["preferences"], {})
        self.assertEqual(self.cfg["default_model"], "qwen")
        self.assertTrue(self.cfg["callable"]["qwen"])
        # Claude needs a subscription, so it ships switched off like Grok.
        self.assertEqual([self.cfg["callable"][m] for m in ("opus5", "sonnet5", "haiku")], [False] * 3)
        self.assertNotIn("enabled", self.cfg["claude_delegation"])

    def test_the_local_file_is_ignored_by_git(self):
        self.assertIn("router_config.local.yaml", (ROOT / ".gitignore").read_text(encoding="utf-8").split())


if __name__ == "__main__":
    unittest.main()
