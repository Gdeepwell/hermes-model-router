import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router.hermes_paths import hermes_home, hermes_path


class HermesHomeTests(unittest.TestCase):
    """The home is Hermes's own answer, not a hardcoded ~/.hermes.

    On Windows Hermes lives in %LOCALAPPDATA%\\hermes and ~/.hermes does not
    exist, so the router read no delegation targets and logged beside a home
    nothing else reads.
    """

    def test_asks_hermes_when_hermes_is_importable(self):
        fake = types.ModuleType("hermes_constants")
        fake.get_hermes_home = lambda: Path("/profiles/work")
        with patch.dict(sys.modules, {"hermes_constants": fake}), \
             patch.dict(os.environ, {"HERMES_HOME": "/elsewhere"}):
            self.assertEqual(hermes_home(), Path("/profiles/work"))

    def test_standalone_reads_hermes_home_from_the_environment(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"hermes_constants": None}), \
             patch.dict(os.environ, {"HERMES_HOME": directory}):
            self.assertEqual(hermes_home(), Path(directory))

    def test_standalone_windows_default_is_local_appdata(self):
        env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
        env["LOCALAPPDATA"] = "C:\\Users\\someone\\AppData\\Local"
        with patch.dict(sys.modules, {"hermes_constants": None}), \
             patch.dict(os.environ, env, clear=True), \
             patch.object(sys, "platform", "win32"):
            self.assertEqual(hermes_home(), Path(env["LOCALAPPDATA"]) / "hermes")

    def test_standalone_posix_default_is_dot_hermes(self):
        env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
        with patch.dict(sys.modules, {"hermes_constants": None}), \
             patch.dict(os.environ, env, clear=True), \
             patch.object(sys, "platform", "linux"):
            self.assertEqual(hermes_home(), Path.home() / ".hermes")


class HermesPathTests(unittest.TestCase):

    def setUp(self):
        self._home = tempfile.TemporaryDirectory()
        self.home = Path(self._home.name)
        patcher = patch("model_router.hermes_paths.hermes_home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._home.cleanup)

    def test_rewrites_a_leading_dot_hermes_onto_the_real_home(self):
        self.assertEqual(hermes_path("~/.hermes/logs/model-router.jsonl"),
                         self.home / "logs" / "model-router.jsonl")

    def test_accepts_a_backslash_after_the_prefix(self):
        self.assertEqual(hermes_path("~\\.hermes\\config.yaml"), self.home / "config.yaml")

    def test_the_bare_prefix_is_the_home_itself(self):
        self.assertEqual(hermes_path("~/.hermes"), self.home)

    def test_a_lookalike_directory_is_not_rewritten(self):
        self.assertEqual(hermes_path("~/.hermes-backup/x"),
                         Path(os.path.expanduser("~/.hermes-backup/x")))

    def test_other_paths_only_get_the_user_expanded(self):
        self.assertEqual(hermes_path("~/elsewhere/log.jsonl"),
                         Path(os.path.expanduser("~/elsewhere/log.jsonl")))
        self.assertEqual(hermes_path("/var/log/x.jsonl"), Path("/var/log/x.jsonl"))


if __name__ == "__main__":
    unittest.main()
