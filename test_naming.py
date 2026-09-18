"""The working slang "wing" never reaches code, config or text the operator or a model reads.

The feature is Claude delegation. Tests are exempt: they may describe history.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLANG = re.compile(r"\bwings?\b", re.IGNORECASE)


def _shipped_files():
    for path in sorted(ROOT.glob("*.py")):
        if not path.name.startswith("test_"):
            yield path
    yield ROOT / "router_config.yaml"


class NoSlangTests(unittest.TestCase):
    def test_no_shipped_file_says_wing(self):
        hits = []
        for path in _shipped_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if SLANG.search(line):
                    hits.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(hits, [], "\n".join(hits))

    def test_the_old_module_is_gone(self):
        self.assertFalse((ROOT / "claude_wing.py").exists())
        self.assertTrue((ROOT / "claude_delegation.py").exists())


if __name__ == "__main__":
    unittest.main()
