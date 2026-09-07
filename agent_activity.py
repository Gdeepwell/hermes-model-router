"""Read-only activity projection for the local Model Router dashboard."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _as_object(value: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _redact_sensitive(text: str) -> str:
    """Keep the local dashboard useful without exposing credential values."""
    text = re.sub(
        r"(?i)\b(password|passwd|api[ _-]?key|secret|token)\s*([=:])\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bauthorization\s*:\s*bearer\s+[^\s,;]+", "Authorization: Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-[REDACTED]", text)
    return re.sub(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", "[REDACTED KEY MATERIAL]", text, flags=re.DOTALL)


def _preview(value: Any, limit: int = 180) -> str:
    raw = str(value or "Háttérfeladat")
    if raw.startswith("\x00json:"):
        try:
            blocks = json.loads(raw[len("\x00json:"):])
            if isinstance(blocks, list):
                raw = " ".join(
                    str(block.get("text", ""))
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ) or "Háttérfeladat"
        except (TypeError, ValueError):
            pass
    text = _redact_sensitive(" ".join(raw.split()))
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _recent_subagent_activity(log_path: Path | None) -> str:
    if not log_path or not log_path.exists():
        return "Várakozik a következő lépésre"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-1000:]
    except OSError:
        return "Várakozik a következő lépésre"
    for line in reversed(lines):
        if "platform=subagent" in line:
            return "Feladat elemzése"
        if "agent.tool_executor:" in line and "Tool terminal" in line:
            return "Terminal parancs fut"
        if "agent.tool_executor:" in line and "Tool search_files" in line:
            return "Kódbázis keresése"
        if "agent.conversation_loop:" in line and "API call" in line:
            return "Agent válaszra vár"
    return "Várakozik a következő lépésre"



def _live_turns(rows: List[tuple]) -> List[Dict[str, Any]]:
    """Return only lifecycle records whose owning process still matches.

    This is a process-liveness guard over the producer's explicit `running`
    state, never an inference from a router-call timestamp.
    """
    try:
        from gateway.status import get_process_start_time
    except Exception:
        get_process_start_time = None
    live: List[Dict[str, Any]] = []
    for turn_id, session_id, pid, process_started_at, started_at, model in rows:
        try:
            os.kill(int(pid), 0)
            same_process = (
                process_started_at is None
                or get_process_start_time is None
                or get_process_start_time(int(pid)) == int(process_started_at)
            )
        except (OSError, TypeError, ValueError):
            same_process = False
        if same_process:
            live.append({
                "turn_id": str(turn_id), "session_id": str(session_id),
                "pid": int(pid), "started_at": float(started_at),
                "model": str(model or ""),
            })
    return live


def _delegation_reason(goal: Any, context: Any) -> str:
    """A compact, user-facing purpose label inferred from the child contract."""
    text = " ".join((str(goal or ""), str(context or ""))).casefold()
    if any(word in text for word in ("review", "ellenor", "audit")):
        return "Review / ellenőrzés"
    if any(word in text for word in ("teszt", "test", "assert")):
        return "Tesztelési részfeladat"
    if any(word in text for word in ("css", "html", "ui", "dashboard", "fejleszt", "kod", "code")):
        return "Kisebb fejlesztési részfeladat"
    if any(word in text for word in ("kutat", "research", "keres", "search", "compare", "hasonlit")):
        return "Feltárás / összehasonlítás"
    return "Önálló részfeladat"


def _requested_read_only(goal: Any, context: Any) -> bool:
    """Show a read-only badge only when that scope was explicitly requested."""
    text = " ".join((str(goal or ""), str(context or ""))).casefold()
    return any(marker in text for marker in ("read-only", "read only", "csak olvas", "csak kiolvas"))


def _router_calls_by_session(router_log_path: Path | None) -> Dict[str, List[Dict[str, Any]]]:
    """Read retained router records once to associate child sessions with routes."""
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not router_log_path or not router_log_path.exists():
        return by_session
    try:
        lines = router_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return by_session
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        turn_id = str(entry.get("turn_id", ""))
        session_id = turn_id.split(":", 1)[0] if ":" in turn_id else ""
        model = str(entry.get("model", ""))
        if not session_id or not model:
            continue
        by_session[session_id].append({
            "tier": str(entry.get("tier", "")),
            "model": model,
            "effort": str(entry.get("effort", "")),
            "timestamp": str(entry.get("timestamp", "")),
        })
    return by_session


def _recent_routed_calls(router_log_path: Path | None, child_session_id: str | None) -> List[Dict[str, str]]:
    """Read every actual router record for one child session.

    The dashboard is an audit view: truncating to a byte tail silently changes
    retained router records for older children, so the per-session filter must scan the
    retained JSONL rather than only its newest chunk.
    """
    return [
        {"tier": call["tier"], "model": call["model"], "effort": call.get("effort", "")}
        for call in _router_calls_by_session(router_log_path).get(str(child_session_id or ""), [])
    ]


def _stored_description(value: Any) -> str:
    """Redact but do not truncate a persisted user task description."""
    raw = str(value or "")
    if raw.startswith("\x00json:"):
        try:
            blocks = json.loads(raw[len("\x00json:"):])
            if isinstance(blocks, list):
                raw = " ".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict) and block.get("type") == "text")
        except (TypeError, ValueError):
            pass
    return _redact_sensitive(" ".join(raw.split()))


_BRIDGE_PRECEDENCE = {"success": 0, "error": 4, "budget": 3, "max-turn": 2, "timeout": 5}


def _process_start_identity(pid: int) -> int | None:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        try:
            return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
        except (OSError, ValueError, IndexError):
            return None


def _bridge_runs(path: Path | None) -> List[Dict[str, Any]]:
    """Fold append-only bridge events defensively, choosing one terminal outcome."""
    runs: Dict[str, Dict[str, Any]] = {}
    if not path or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        run_id = str(event.get("bridge_run_id") or "")
        if not run_id:
            continue
        run = runs.setdefault(run_id, {"bridge_run_id": run_id})
        if event.get("event") == "started":
            run.update(event)
            run["started_event"] = event
        elif event.get("event") == "terminal":
            previous = run.get("terminal_event") or {}
            if _BRIDGE_PRECEDENCE.get(str(event.get("state")), 1) > _BRIDGE_PRECEDENCE.get(str(previous.get("state")), -1):
                run["terminal_event"] = event
    return list(runs.values())


def load_agent_activity(
    db_path: Path,
    now: float | None = None,
    log_path: Path | None = None,
    router_log_path: Path | None = None,
    bridge_lifecycle_path: Path | None = None,
) -> Dict[str, Any]:
    """Project durable async delegations into parent/child dashboard rows."""
    now = time.time() if now is None else now
    empty = {"summary": {"running": 0, "completed": 0, "failed": 0}, "parents": [], "active_turns": []}
    if not db_path.exists():
        return empty
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            has_session_call_count = "api_call_count" in session_columns
            rows = conn.execute(
                "SELECT delegation_id, origin_session, parent_session_id, state, dispatched_at, "
                "completed_at, updated_at, task_json, result_json FROM async_delegations "
                "ORDER BY dispatched_at DESC LIMIT 100"
            ).fetchall()
            try:
                calls = "api_call_count" if has_session_call_count else "0"
                sessions = conn.execute(
                    f"SELECT id, parent_session_id, started_at, ended_at, model, {calls} FROM sessions "
                    "ORDER BY started_at DESC LIMIT 500"
                ).fetchall()
                messages = conn.execute(
                    "SELECT session_id, role, tool_name, content, timestamp FROM messages "
                    "WHERE role IN ('user', 'assistant', 'tool') ORDER BY id DESC LIMIT 3000"
                ).fetchall()
                try:
                    active_turns = conn.execute(
                        "SELECT turn_id, session_id, pid, process_started_at, started_at, model "
                        "FROM turn_lifecycle WHERE state='running' ORDER BY started_at DESC LIMIT 50"
                    ).fetchall()
                except sqlite3.Error:
                    active_turns = []
            except sqlite3.Error:
                sessions, messages, active_turns = [], [], []
    except sqlite3.Error:
        return empty

    summary = {"running": 0, "completed": 0, "failed": 0}
    raw_calls_by_session = _router_calls_by_session(router_log_path)
    sessions_by_parent: Dict[str, List[tuple]] = defaultdict(list)
    sessions_by_id: Dict[str, tuple] = {}
    for session in sessions:
        sessions_by_id[str(session[0])] = session
        if session[1] is not None:
            sessions_by_parent[str(session[1])].append(session)
    console_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    user_messages: Dict[str, List[tuple]] = defaultdict(list)
    assistant_messages: Dict[str, int] = defaultdict(int)
    for session_id, role, tool_name, content, timestamp in messages:
        if role == "assistant":
            assistant_messages[str(session_id)] += 1
            continue
        if role == "user":
            description = _stored_description(content)
            if description.startswith("[Your active task list was preserved") or description.startswith("[ASYNC DELEGATION"):
                continue
            user_messages[str(session_id)].append((float(timestamp or 0), description))
            continue
        if len(console_by_session[str(session_id)]) >= 10:
            continue
        console_by_session[str(session_id)].append({
            "tool": str(tool_name),
            "timestamp": float(timestamp or 0),
        })
    for events in console_by_session.values():
        events.reverse()
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for delegation_id, origin, parent, state, dispatched, completed, updated, task_json, result_json in rows:
        task, result = _as_object(task_json), _as_object(result_json)
        raw_results = result.get("results") if isinstance(result.get("results"), list) else []
        child_results = [item for item in raw_results if isinstance(item, dict)]
        batch_goals = task.get("goals") if isinstance(task.get("goals"), list) else []
        is_batch = bool(task.get("is_batch")) or len(batch_goals) > 1 or len(child_results) > 1
        state = str(state or "unknown")
        started = float(dispatched or 0)
        ended = float(completed or now) if state not in {"running", "finalizing"} else now
        parent_key = str(parent or origin or "ismeretlen-session")
        prompts = [entry for entry in user_messages.get(parent_key, []) if entry[0] <= started + 2]
        parent_prompt_at, parent_prompt = max(prompts, default=(started, "Korábbi fő feladat"), key=lambda entry: entry[0])
        candidates = sessions_by_parent.get(parent_key, [])
        matching = sorted(
            (session for session in candidates if abs(float(session[2] or 0) - started) <= 90),
            key=lambda session: float(session[2] or 0),
        )

        if is_batch:
            indexed_results = {int(item.get("task_index", index)): item for index, item in enumerate(child_results)}
            child_count = max(len(batch_goals), len(indexed_results), 1)
            children = [
                (index, batch_goals[index] if index < len(batch_goals) else task.get("goal"), indexed_results.get(index, {}))
                for index in range(child_count)
            ]
        else:
            children = [(0, task.get("goal"), child_results[0] if child_results else {})]

        for index, child_goal, child_result in children:
            child_state = str(child_result.get("status") or state)
            if child_state in {"running", "finalizing"}:
                summary["running"] += 1
            elif child_state in {"error", "failed", "unknown"}:
                summary["failed"] += 1
            else:
                summary["completed"] += 1
            selected_session = matching[index] if index < len(matching) else None
            child_session_id = str(selected_session[0]) if selected_session else None
            routed_calls = [
                {"tier": call["tier"], "model": call["model"], "effort": call.get("effort", "")}
                for call in raw_calls_by_session.get(child_session_id or "", [])
            ]
            actual_model = routed_calls[-1]["model"] if routed_calls else None
            received_prompts = user_messages.get(child_session_id or "", [])
            # The child session's first real user message is the authoritative
            # task payload.  The delegation record is a fallback only: a router
            # or runtime wrapper may augment the stored dispatch contract.
            received_goal = min(received_prompts, default=(0.0, ""), key=lambda entry: entry[0])[1]
            task_description = received_goal or _preview(child_goal, 20_000)
            displayed_goal = _preview(task_description, 900)
            api_calls = len(routed_calls) if routed_calls else int(child_result.get("api_calls") or result.get("api_calls") or 0)
            grouped[(parent_key, parent_prompt_at, parent_prompt)].append({
                "id": f"{delegation_id}:{index}" if is_batch else delegation_id,
                "goal": displayed_goal,
                "task_description": task_description,
                "task_description_source": "child_user_message" if received_goal else "delegation_task_json",
                "reason": _delegation_reason(displayed_goal, task.get("context")),
                "access_mode": "requested_read_only" if _requested_read_only(child_goal, task.get("context")) else "standard",
                "parent_prompt": parent_prompt,
                "parent_started_at": parent_prompt_at,
                "state": "running" if child_state == "finalizing" else child_state,
                "model": actual_model or child_result.get("model") or task.get("model") or result.get("model"),
                "model_source": "router" if actual_model else "delegation-config",
                "routed_calls": routed_calls,
                "toolsets": task.get("toolsets") or [],
                "started_at": started,
                "updated_at": float(updated or started),
                "age_seconds": max(0, round(float(child_result.get("duration_seconds") or (ended - started)))),
                "api_calls": api_calls,
                "activity": _recent_subagent_activity(log_path) if child_state in {"running", "finalizing"} else "Befejezett feladat",
                "agent_session_id": child_session_id,
                "console": console_by_session.get(child_session_id or "", []),
                "children": [],
            })
    parents = [
        {
            "session_id": session_id,
            "started_at": parent_prompt_at,
            "prompt": parent_prompt,
            "children": sorted(children, key=lambda child: (child["state"] != "running", -child["started_at"])),
        }
        for (session_id, parent_prompt_at, parent_prompt), children in grouped.items()
    ]

    # `sessions.parent_session_id` is the durable hierarchy.  Fold a
    # supervisor's own group into its child node, so the dashboard receives one
    # recursive tree rather than a collection of artificial roots.
    parent_groups = {parent["session_id"]: parent for parent in parents}
    attached = set()
    for parent in parents:
        for child in parent["children"]:
            nested = parent_groups.get(child.get("agent_session_id"))
            if nested and nested is not parent:
                child["children"] = nested["children"]
                attached.add(id(nested))

    # Also follow stored session edges that were not accompanied by another
    # async_delegations row (the state DB remains the evidence of the branch).
    represented = {child.get("agent_session_id") for parent in parents for child in parent["children"]}
    for parent in parents:
        stack = list(parent["children"])
        while stack:
            node = stack.pop()
            for nested in sorted(sessions_by_parent.get(node.get("agent_session_id") or "", []), key=lambda row: float(row[2] or 0)):
                nested_id = str(nested[0])
                if nested_id in represented:
                    continue
                represented.add(nested_id)
                description = min(user_messages.get(nested_id, []), default=(0.0, "Korábbi mellékfeladat"), key=lambda entry: entry[0])[1]
                # Preserve the complete raw route record.  The execution tree
                # uses it as the primary audit source for both model and effort;
                # session metadata is a fallback only when no raw record exists.
                nested_routed_calls = [
                    {"tier": call["tier"], "model": call["model"], "effort": call.get("effort", "")}
                    for call in raw_calls_by_session.get(nested_id, [])
                ]
                inferred = {
                    "id": f"session:{nested_id}", "goal": _preview(description, 900), "task_description": description,
                    "task_description_source": "child_user_message", "reason": _delegation_reason(description, None),
                    "access_mode": "requested_read_only" if _requested_read_only(description, None) else "standard",
                    "state": "completed" if nested[3] else "running", "model": nested[4], "model_source": "session",
                    "routed_calls": nested_routed_calls, "toolsets": [],
                    "started_at": float(nested[2] or now), "updated_at": float(nested[3] or now),
                    "age_seconds": max(0, round(float(nested[3] or now) - float(nested[2] or now))), "api_calls": 0,
                    "activity": "Befejezett feladat" if nested[3] else "Futó mellékfeladat", "agent_session_id": nested_id,
                    "console": console_by_session.get(nested_id, []), "children": [],
                }
                node["children"].append(inferred)
                stack.append(inferred)

    # A supervisor attached below its delegating worker must not also render as a root.
    attached_sessions = {
        str(child.get("agent_session_id"))
        for parent in parents
        for child in parent.get("children", [])
        if child.get("agent_session_id")
    }
    parents = [
        parent for parent in parents
        if id(parent) not in attached and parent["session_id"] not in attached_sessions
    ]
    bridge_run_ids = []
    orphan_children = []
    parents_by_session = {str(parent["session_id"]): parent for parent in parents}
    for run in _bridge_runs(bridge_lifecycle_path):
        started_event = run.get("started_event") or {}
        terminal = run.get("terminal_event")
        state = str((terminal or {}).get("state") or "running")
        # A started record alone is running only with reusable-process-safe
        # identity evidence; PID existence by itself can create ghost runs.
        if terminal is None:
            pid, identity = started_event.get("pid"), started_event.get("process_started_at")
            try:
                os.kill(int(pid), 0)
                live_identity = identity is not None and _process_start_identity(int(pid)) == int(identity)
            except (OSError, TypeError, ValueError):
                live_identity = False
            state = "running" if live_identity else "error"
        evidence = terminal or started_event
        run_id = str(run["bridge_run_id"])
        bridge_run_ids.append(run_id)
        calls = [{"tier": "opus5", "model": evidence.get("canonical_model"), "effort": "external"}] if evidence.get("canonical_model") else []
        child = {
            "id": run_id, "bridge_run_id": run_id, "goal": "Opus review",
            "task_description": "Claude Opus 5 · külső Claude Code reviewer",
            "reason": "Külső Claude Code review", "external": True,
            "access_mode": "read_only" if evidence.get("review") else ("requested_read_only" if evidence.get("requested_read_only") else "standard"),
            "state": state, "model": evidence.get("canonical_model") or "",
            "model_source": "modelUsage" if evidence.get("canonical_model") else "unverified",
            "routed_calls": calls, "toolsets": ["Read"],
            "started_at": float(started_event.get("timestamp") or 0),
            "updated_at": float(evidence.get("timestamp") or started_event.get("timestamp") or 0),
            "age_seconds": round(float(evidence.get("duration_seconds") or max(0, now - float(started_event.get("timestamp") or now)))),
            "api_calls": int(evidence.get("num_turns") or 0),
            "metrics": {key: evidence.get(key) for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "total_cost_usd", "duration_seconds")},
            "activity": "Külső Claude Code reviewer", "agent_session_id": run_id,
            "console": [], "children": [],
        }
        if state == "running": summary["running"] += 1
        elif state == "success": summary["completed"] += 1
        else: summary["failed"] += 1
        parent_session = str(evidence.get("parent_session_id") or started_event.get("parent_session_id") or "")
        target = parents_by_session.get(parent_session)
        if target is None and parent_session:
            prompts = user_messages.get(parent_session, [])
            prompt_at, prompt = max(prompts, default=(child["started_at"], "Korábbi fő feladat"), key=lambda item: item[0])
            target = {"session_id": parent_session, "started_at": prompt_at, "prompt": prompt, "children": []}
            parents.append(target); parents_by_session[parent_session] = target
        (target["children"] if target else orphan_children).append(child)
    if orphan_children:
        parents.append({"session_id": "external-claude-code-orphans", "started_at": min(child["started_at"] for child in orphan_children), "prompt": "Külső Claude Code futások", "children": orphan_children, "external_orphan_group": True})
    parents.sort(key=lambda parent: max((child["started_at"] for child in parent["children"]), default=parent["started_at"]), reverse=True)
    return {"summary": summary, "parents": parents, "active_turns": _live_turns(active_turns), "external_bridge_run_ids": bridge_run_ids}
