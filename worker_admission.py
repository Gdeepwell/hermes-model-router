"""Account admission at execution boundaries, shared by worker transports."""
from __future__ import annotations

import json
from types import SimpleNamespace

from . import usage_guard


def refusal(account, tier, cfg, *, blocking=False):
    if str(cfg.get("workflow") or "").strip().lower() == "codex" and (
        account == "anthropic" or str(tier).startswith("claude-")
    ):
        return "Claude workers are disabled by workflow: codex. The parent model is unchanged."
    if not usage_guard.guarded(account, cfg):
        return ""
    reading = (usage_guard.read if blocking else usage_guard.peek)(account, cfg)
    return usage_guard.apply(account, tier, cfg, reading).refused


def delegate_task_route():
    from tools.delegate_tool_config import _load_config
    from agent.subagent_lifecycle import get_active_subagent_parent

    route = _load_config()
    parent = get_active_subagent_parent()
    return (route.get("provider") or getattr(parent, "provider", ""),
            route.get("model") or getattr(parent, "model", ""))


def guard_tool_execution(**kwargs):
    from . import _load_config, _delegation_targets_detail

    args, next_call = kwargs.get("args") or {}, kwargs["next_call"]
    name = str(kwargs.get("tool_name") or "").removeprefix("mcp__")
    if name != "delegate_task" or args.get("action", "spawn") in ("list", "steer", "stop"):
        return next_call(args)
    cfg = _load_config()
    if not cfg.get("enabled", True):
        return next_call(args)
    account, model = delegate_task_route()
    targets = _delegation_targets_detail()
    tasks = args.get("tasks") or [args]
    for task in tasks if isinstance(tasks, list) else [args]:
        target = targets.get(str(task.get("model") or args.get("model") or ""), {}) if isinstance(task, dict) else {}
        message = refusal(target.get("provider") or account, target.get("model") or model, cfg, blocking=True)
        if message:
            return json.dumps({"error": message + " Re-dispatch on an available account; do not retry this account."})
    return next_call(args)


def stopped_response(message, model):
    """A zero-token router stop, consumable by Hermes's supported response shapes.

    Raising from execution middleware before next_call fails open in Hermes.
    Returning a final message instead stops the worker without contacting the
    provider or inventing task results.
    """
    text = "[ROUTER WORKER STOPPED] " + message + " Re-dispatch this task on an available account. No work was performed by this call."
    usage = SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0,
                            prompt_tokens=0, completion_tokens=0)
    return SimpleNamespace(
        model=model, id="router-worker-stopped", status="completed", usage=usage,
        output_text=text, output=[SimpleNamespace(type="message", status="completed",
            content=[SimpleNamespace(type="output_text", text=text)])],
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
        choices=[SimpleNamespace(index=0, finish_reason="stop", message=SimpleNamespace(
            role="assistant", content=text, tool_calls=None, reasoning_content=None))],
    )
