import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from view_log import _annotate_internal_prompts, _attach_lifecycle_provenance


class ViewLogProvenanceTests(unittest.TestCase):
    def test_completion_gets_human_label_and_redacted_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT PRIMARY KEY, parent_session_id TEXT, dispatched_at REAL, task_json TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, content TEXT)")
                conn.execute("INSERT INTO messages VALUES (7, 'parent', 'user', 10, 'Eredeti feladat: secret=hide-me')")
                conn.execute(
                    "INSERT INTO async_delegations VALUES (?, ?, ?, ?)",
                    ("deleg_test", "parent", 20, json.dumps({"goal": "Részfeladat token=hide-me"})),
                )
            entries = [{"turn_id": "completion:turn", "prompt_preview": "[ASYNC DELEGATION COMPLETE — deleg_test] raw result"}]
            _attach_lifecycle_provenance(entries, db)
            entry = entries[0]
            self.assertEqual(entry["event_kind"], "async_delegation_completion")
            self.assertEqual(entry["origin_message_id"], 7)
            self.assertIn("Delegált feladat befejezési eseménye", entry["lifecycle_prompt"])
            self.assertNotIn("hide-me", entry["origin_preview"])

    def test_typed_completion_stays_an_internal_lifecycle_node_after_raw_preview_is_removed(self):
        entries = [
            {"turn_id": "root:one", "prompt_preview": "Valódi fő feladat"},
            {
                "turn_id": "root:two",
                "event_kind": "async_delegation_completion",
                "prompt_preview": "Delegált feladat befejezési eseménye",
            },
        ]
        _annotate_internal_prompts(entries)
        self.assertTrue(entries[1]["is_internal_prompt"])
        self.assertEqual(entries[1]["parent_turn_id"], "root:one")


if __name__ == "__main__":
    unittest.main()