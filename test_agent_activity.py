import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_activity import load_agent_activity


class AgentActivityTests(unittest.TestCase):
    def test_started_external_bridge_with_matching_process_identity_is_running(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT, origin_session TEXT, parent_session_id TEXT, state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL, task_json TEXT, result_json TEXT)")
                conn.execute("CREATE TABLE sessions (id TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
            lifecycle = Path(directory) / "bridge.jsonl"
            lifecycle.write_text(json.dumps({"bridge_run_id":"live","event":"started","state":"running","timestamp":20,"pid":os.getpid(),"process_started_at":123})+"\n")
            with patch("agent_activity._process_start_identity", return_value=123):
                activity = load_agent_activity(db_path, now=30, bridge_lifecycle_path=lifecycle)
        self.assertEqual(activity["parents"][0]["children"][0]["state"], "running")

    def test_external_bridge_is_projected_once_under_matching_parent_and_suppresses_route_root(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT, origin_session TEXT, parent_session_id TEXT, state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL, task_json TEXT, result_json TEXT)")
                conn.execute("CREATE TABLE sessions (id TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.execute("INSERT INTO messages VALUES (1,'parent','user','Review current diff',NULL,10)")
            lifecycle = Path(directory) / "bridge.jsonl"
            lifecycle.write_text("\n".join(map(json.dumps, [
                {"bridge_run_id":"run-1","event":"started","state":"running","timestamp":20,"parent_session_id":"parent","parent_turn_id":"parent:turn","pid":999999,"review":True,"requested_read_only":True},
                {"bridge_run_id":"run-1","event":"terminal","state":"success","timestamp":25,"parent_session_id":"parent","parent_turn_id":"parent:turn","canonical_model":"claude-opus-5","num_turns":2,"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":3,"total_cost_usd":0.1,"duration_seconds":5,"review":True,"requested_read_only":True},
                {"bridge_run_id":"run-1","event":"terminal","state":"success","timestamp":24,"parent_session_id":"parent"},
            ]))+"\n")
            router = Path(directory) / "router.jsonl"
            router.write_text(json.dumps({"turn_id":"run-1:2","tier":"opus5","model":"claude-opus-5","effort":"external"})+"\n")
            activity = load_agent_activity(db_path, now=30, router_log_path=router, bridge_lifecycle_path=lifecycle)
        self.assertEqual(len(activity["parents"]), 1)
        child = activity["parents"][0]["children"][0]
        self.assertEqual((child["id"], child["goal"], child["state"], child["model"]), ("run-1", "Opus review", "success", "claude-opus-5"))
        self.assertEqual(child["api_calls"], 2)
        self.assertEqual(child["metrics"]["input_tokens"], 10)
        self.assertEqual(child["access_mode"], "read_only")
        self.assertEqual(activity["external_bridge_run_ids"], ["run-1"])

    def test_unmatched_external_bridge_uses_one_labelled_orphan_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT, origin_session TEXT, parent_session_id TEXT, state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL, task_json TEXT, result_json TEXT)")
                conn.execute("CREATE TABLE sessions (id TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
            lifecycle = Path(directory) / "bridge.jsonl"
            lifecycle.write_text(json.dumps({"bridge_run_id":"orphan","event":"started","state":"running","timestamp":20,"pid":999999})+"\n")
            activity = load_agent_activity(db_path, now=30, bridge_lifecycle_path=lifecycle)
        self.assertEqual(activity["parents"][0]["prompt"], "Külső Claude Code futások")
        self.assertEqual(activity["parents"][0]["children"][0]["state"], "error")

    def test_groups_running_and_completed_delegations_by_parent_session(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""CREATE TABLE async_delegations (
                    delegation_id TEXT PRIMARY KEY, origin_session TEXT, parent_session_id TEXT,
                    state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL,
                    task_json TEXT, result_json TEXT
                )""")
                conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", ("child-session", "session-a", 101.0, None, "gpt-5.6-sol"))
                conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (1, "session-a", "user", "Kérlek nézd át a változtatást.", None, 80.0))
                conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (2, "child-session", "tool", json.dumps({"output": "tests are running"}), "terminal", 103.0))
                conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                    "deleg-1", "session-a", "session-a", "running", 100.0, None, 120.0,
                    json.dumps({"goal": "Build dashboard", "toolsets": ["file", "terminal"], "model": "gpt-5.6-sol"}), None,
                ))
                conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                    "deleg-2", "session-a", "session-a", "completed", 90.0, 110.0, 110.0,
                    json.dumps({"goal": "Review tests", "context": "Read-only code review requested", "toolsets": ["file"]}), json.dumps({"api_calls": 4}),
                ))

            log_path = Path(directory) / "agent.log"
            log_path.write_text("2026-07-17 20:00:01,000 INFO [child-session] agent.turn_context: conversation turn: session=child-session model=x platform=subagent history=0 msg='Build dashboard'\n2026-07-17 20:00:03,000 INFO [child-session] agent.tool_executor: Tool terminal completed (0.2s)\n", encoding="utf-8")
            router_log_path = Path(directory) / "router.jsonl"
            router_log_path.write_text(
                "\n".join([
                    json.dumps({"turn_id": "child-session:turn-1", "tier": "spark", "model": "gpt-5.3-codex-spark", "effort": "medium"}),
                    json.dumps({"turn_id": "child-session:turn-2", "tier": "sol", "model": "gpt-5.6-sol", "effort": "medium"}),
                ]) + "\n",
                encoding="utf-8",
            )
            activity = load_agent_activity(db_path, now=130.0, log_path=log_path, router_log_path=router_log_path)

        self.assertEqual(activity["summary"], {"running": 1, "completed": 1, "failed": 0})
        self.assertEqual(activity["parents"][0]["session_id"], "session-a")
        self.assertEqual(activity["parents"][0]["started_at"], 80.0)
        self.assertEqual(activity["parents"][0]["children"][0]["goal"], "Build dashboard")
        self.assertEqual(activity["parents"][0]["children"][0]["age_seconds"], 30)
        self.assertEqual(activity["parents"][0]["children"][0]["activity"], "Terminal parancs fut")
        self.assertEqual(activity["parents"][0]["children"][0]["routed_calls"], [
            {"tier": "spark", "model": "gpt-5.3-codex-spark", "effort": "medium"},
            {"tier": "sol", "model": "gpt-5.6-sol", "effort": "medium"},
        ])
        self.assertEqual(activity["parents"][0]["children"][0]["model"], "gpt-5.6-sol")
        self.assertEqual(activity["parents"][0]["children"][0]["console"][0]["tool"], "terminal")
        self.assertNotIn("output", activity["parents"][0]["children"][0]["console"][0])
        # "Review tests" has no session row of its own in this fixture — only
        # "Build dashboard" does. It used to be handed that child's two router
        # calls, because both delegations resolved to the same first session in
        # the window; now an unmatched delegation reports the count it recorded
        # for itself.
        self.assertEqual(activity["parents"][0]["children"][1]["api_calls"], 4)
        self.assertIsNone(activity["parents"][0]["children"][1]["agent_session_id"])
        self.assertEqual(activity["parents"][0]["children"][1]["access_mode"], "requested_read_only")
        self.assertEqual(activity["parents"][0]["children"][1]["reason"], "Review / ellenőrzés")

    def test_batch_delegation_expands_every_child_with_its_own_goal_result_and_route(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""CREATE TABLE async_delegations (
                    delegation_id TEXT PRIMARY KEY, origin_session TEXT, parent_session_id TEXT,
                    state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL,
                    task_json TEXT, result_json TEXT
                )""")
                conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (1, "parent", "user", "Három párhuzamos feltárás", None, 80.0))
                for index, session_id in enumerate(("child-0", "child-1", "child-2")):
                    conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (session_id, "parent", 100.0 + index, None, "gpt-5.3-codex-spark"))
                conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                    "deleg-batch", "parent", "parent", "completed", 100.0, 130.0, 130.0,
                    json.dumps({"is_batch": True, "goals": ["Schema feltárás", "Read-path feltárás", "UX feltárás"], "model": "gpt-5.3-codex-spark"}),
                    json.dumps({"results": [
                        {"task_index": 0, "status": "completed", "api_calls": 3, "duration_seconds": 11, "model": "gpt-5.3-codex-spark"},
                        {"task_index": 1, "status": "completed", "api_calls": 4, "duration_seconds": 12, "model": "gpt-5.3-codex-spark"},
                        {"task_index": 2, "status": "completed", "api_calls": 5, "duration_seconds": 13, "model": "gpt-5.3-codex-spark"},
                    ]}),
                ))
            router_log_path = Path(directory) / "router.jsonl"
            router_log_path.write_text("\n".join(
                json.dumps({"turn_id": f"child-{index}:sa-{index}:turn", "tier": "spark", "model": "gpt-5.3-codex-spark"})
                for index in range(3)
            ) + "\n", encoding="utf-8")

            activity = load_agent_activity(db_path, now=140.0, router_log_path=router_log_path)

        children = activity["parents"][0]["children"]
        self.assertEqual(activity["summary"], {"running": 0, "completed": 3, "failed": 0})
        self.assertEqual([child["goal"] for child in children], ["Schema feltárás", "Read-path feltárás", "UX feltárás"])
        self.assertEqual([child["agent_session_id"] for child in children], ["child-0", "child-1", "child-2"])
        self.assertEqual([child["api_calls"] for child in children], [1, 1, 1])
        self.assertTrue(all(child["model_source"] == "router" for child in children))

    def test_separates_different_parent_turns_inside_one_session(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""CREATE TABLE async_delegations (
                    delegation_id TEXT PRIMARY KEY, origin_session TEXT, parent_session_id TEXT,
                    state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL,
                    task_json TEXT, result_json TEXT
                )""")
                conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (1, "session-a", "user", "Első fő feladat", None, 80.0))
                conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (2, "session-a", "user", "Második fő feladat", None, 180.0))
                conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                    "deleg-1", "session-a", "session-a", "completed", 100.0, 120.0, 120.0,
                    json.dumps({"goal": "Review first"}), json.dumps({"api_calls": 2}),
                ))
                conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                    "deleg-2", "session-a", "session-a", "completed", 200.0, 220.0, 220.0,
                    json.dumps({"goal": "Review second"}), json.dumps({"api_calls": 3}),
                ))

            activity = load_agent_activity(db_path, now=230.0)

        self.assertEqual(len(activity["parents"]), 2)
        self.assertEqual([parent["prompt"] for parent in activity["parents"]], ["Második fő feladat", "Első fő feladat"])
        self.assertEqual([len(parent["children"]) for parent in activity["parents"]], [1, 1])

    def test_builds_a_recursive_tree_without_accounting_projection(self):
        """A child supervisor and its worker are one expanded thread, not two roots."""
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""CREATE TABLE async_delegations (
                    delegation_id TEXT PRIMARY KEY, origin_session TEXT, parent_session_id TEXT,
                    state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL,
                    task_json TEXT, result_json TEXT
                )""")
                conn.execute("""CREATE TABLE sessions (
                    id TEXT PRIMARY KEY, parent_session_id TEXT, started_at REAL, ended_at REAL,
                    model TEXT, api_call_count INTEGER
                )""")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?)", [
                    ("root", None, 1.0, 100.0, "gpt-5.6-terra", 2),
                    ("terra", "root", 10.0, 90.0, "gpt-5.6-terra", 3),
                    ("spark", "terra", 20.0, 80.0, "gpt-5.6-spark", 4),
                ])
                conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", [
                    (1, "root", "user", "Root stored task", None, 1.0),
                    (2, "terra", "user", "Terra stored task", None, 10.0),
                    (3, "spark", "user", "Spark bounded read-only task", None, 20.0),
                ])
                conn.executemany("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", [
                    ("root-to-terra", "root", "root", "completed", 9.0, 90.0, 90.0,
                     json.dumps({"goal": "dispatch fallback must not win"}), json.dumps({"api_calls": 99})),
                    ("terra-to-spark", "terra", "terra", "completed", 19.0, 80.0, 80.0,
                     json.dumps({"goal": "nested fallback must not win"}), json.dumps({"api_calls": 99})),
                ])

            activity = load_agent_activity(db_path, now=100.0)

        self.assertEqual(len(activity["parents"]), 1)
        root = activity["parents"][0]
        terra = root["children"][0]
        spark = terra["children"][0]
        self.assertEqual(root["session_id"], "root")
        self.assertEqual(terra["task_description"], "Terra stored task")
        self.assertEqual(terra["task_description_source"], "child_user_message")
        self.assertEqual(spark["task_description"], "Spark bounded read-only task")
        self.assertEqual(spark["task_description_source"], "child_user_message")
        self.assertNotIn("accounting", root)
        self.assertNotIn("accounting", terra)
        self.assertNotIn("accounting", spark)


if __name__ == "__main__":
    unittest.main()


class TwoDispatchesInsideOneWindowTests(unittest.TestCase):
    """A parent that dispatches twice within the matching window.

    Position inside a ±90s window used to decide which session ran which child,
    so both delegations selected the first one: the dashboard showed one worker
    twice, under the wrong model, while the other ran with no row at all. Rows
    are read newest-first, so the newest delegation took the oldest session.
    """

    def _fixture(self, directory):
        db_path = Path(directory) / "state.db"
        stopped = "[opus5] Implement the safe, tenant-scoped customer profile merge feature in /home/x"
        retried = "Implement the safe, tenant-scoped customer profile merge feature in /home/x"
        with sqlite3.connect(db_path) as conn:
            conn.execute("""CREATE TABLE async_delegations (
                delegation_id TEXT PRIMARY KEY, origin_session TEXT, parent_session_id TEXT,
                state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL,
                task_json TEXT, result_json TEXT
            )""")
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
            conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (1, "parent", "user", "inplementald", None, 90.0))
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", ("child-stopped", "parent", 100.0, None, "gpt-5.6-terra"))
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", ("child-opus", "parent", 118.0, None, "claude-opus-5"))
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (2, "child-stopped", "user", stopped, None, 100.0))
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (3, "child-opus", "user", retried, None, 118.0))
            conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                "deleg-stopped", "parent", "parent", "completed", 100.0, 110.0, 110.0,
                json.dumps({"goal": stopped}),
                json.dumps({"results": [{"task_index": 0, "status": "completed", "api_calls": 1}]}),
            ))
            conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)", (
                "deleg-opus", "parent", "parent", "completed", 118.0, 200.0, 200.0,
                json.dumps({"goal": retried}),
                json.dumps({"results": [{"task_index": 0, "status": "completed", "api_calls": 16}]}),
            ))
        router_log_path = Path(directory) / "router.jsonl"
        router_log_path.write_text("\n".join([
            json.dumps({"turn_id": "child-stopped:sa-0:t", "tier": "sol", "model": "gpt-5.6-sol"}),
            *(json.dumps({"turn_id": "child-opus:sa-0:t", "tier": "opus5", "model": "claude-opus-5"})
              for _ in range(16)),
        ]) + "\n", encoding="utf-8")
        return db_path, router_log_path

    def _children(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path, router_log_path = self._fixture(directory)
            activity = load_agent_activity(db_path, now=210.0, router_log_path=router_log_path)
        return activity["parents"][0]["children"]

    def test_each_delegation_gets_its_own_session(self):
        sessions = {child["agent_session_id"] for child in self._children()}
        self.assertEqual(sessions, {"child-stopped", "child-opus"})

    def test_the_worker_on_the_other_account_is_listed(self):
        """It ran sixteen calls on a separate subscription and had no row at all."""
        models = {child["model"] for child in self._children()}
        self.assertIn("claude-opus-5", models)

    def test_no_session_is_shown_twice(self):
        sessions = [child["agent_session_id"] for child in self._children()]
        self.assertEqual(len(sessions), len(set(sessions)))

    def test_the_call_counts_follow_the_right_session(self):
        by_model = {child["model"]: child["api_calls"] for child in self._children()}
        self.assertEqual(by_model["claude-opus-5"], 16)
        self.assertEqual(by_model["gpt-5.6-sol"], 1)

    def test_a_relabelled_retry_still_matches_its_own_session(self):
        """The retry carries the same work with the [opus5] prefix stripped —
        exactly the pair that has to be recognised as two distinct tasks."""
        goals = {child["agent_session_id"]: child["goal"] for child in self._children()}
        self.assertTrue(goals["child-stopped"].startswith("[opus5]"))
        self.assertFalse(goals["child-opus"].startswith("[opus5]"))
