"""The Claude wing: Claude workers as real Hermes children, next to the Codex workforce.

Hermes's ``delegate_task`` has one delegation route per process, and on this host
it is pinned to Codex. ``delegate_claude`` reaches Claude by calling the same
``delegate_task`` with a per-call route (``credentials_cfg``) pinned to the
``anthropic`` provider -- the mechanism Hermes's own /review uses. Pinning the
provider, rather than inheriting the parent's, is what keeps the wing working
while the parent itself is on a Codex fallback.

Nothing here is imported from the router at module level: the router imports
this module, so router helpers are pulled in function-locally.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_logger = logging.getLogger("model_router.claude_wing")

TIERS: Tuple[str, ...] = ("haiku", "sonnet", "opus")
# Router target names stay what the config, cooldowns and dashboard already use;
# the tool speaks in short tier names. This table is the only place they meet.
TARGET_FOR_TIER: Dict[str, str] = {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"}
TIER_FOR_TARGET: Dict[str, str] = {target: tier for tier, target in TARGET_FOR_TIER.items()}

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"},
    "default_tier": "sonnet",
    "usage_guard": {"soft_percent": 70, "hard_percent": 90, "cache_seconds": 300, "state_path": ""},
    "log_path": "",
}

_ACTIVE = False


def is_active() -> bool:
    """True once ``delegate_claude`` is registered in this process."""
    return _ACTIVE


def wing_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``claude_wing`` block with defaults filled in; off unless configured on."""
    raw = (cfg or {}).get("claude_wing")
    raw = raw if isinstance(raw, dict) else {}
    merged = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in ("tiers", "usage_guard"):
            if isinstance(value, dict):
                merged[key] = {**DEFAULTS[key], **value}
        else:
            merged[key] = value
    return merged


def tier_model(tier: str, cfg: Dict[str, Any]) -> str:
    return str(wing_config(cfg)["tiers"].get(tier) or "").strip()


def target_names(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Router target names for every tier that has a model configured."""
    return tuple(sorted(TARGET_FOR_TIER[tier] for tier in TIERS if tier_model(tier, cfg)))


def target_for_model(model: str, cfg: Dict[str, Any]) -> Optional[str]:
    if not model:
        return None
    for tier in TIERS:
        if tier_model(tier, cfg) == model:
            return TARGET_FOR_TIER[tier]
    return None


def host_check() -> Tuple[bool, str]:
    """Whether this Hermes still has the two internals the wing stands on.

    ``credentials_cfg`` is commented "internal callers only" upstream, and the
    active-parent lookup is not on the plugin context. If either moves, the wing
    must not register rather than fail at call time.
    """
    try:
        from tools.delegate_tool import delegate_task
        from agent.subagent_lifecycle import get_active_subagent_parent  # noqa: F401
    except Exception as exc:
        return False, f"Hermes delegation API not importable ({type(exc).__name__}: {exc})"
    parameters = inspect.signature(delegate_task).parameters
    missing = [name for name in ("tasks", "parent_agent", "credentials_cfg") if name not in parameters]
    if missing:
        return False, "delegate_task lacks " + ", ".join(missing)
    return True, ""


def registration_block(cfg: Dict[str, Any]) -> str:
    """Why ``delegate_claude`` must not be registered, or "" when it may be."""
    if not wing_config(cfg).get("enabled"):
        return "claude_wing.enabled is false"
    switches = cfg.get("callable") or {}
    if not any(switches.get(target) is True for target in TARGET_FOR_TIER.values()):
        return "every Claude target is switched off in `callable`"
    ok, why = host_check()
    return "" if ok else why


# ---------------------------------------------------------------------------
# Usage guard


@dataclass(frozen=True)
class UsageReading:
    weekly: Optional[float]
    session: Optional[float]
    fetched_at: float


@dataclass(frozen=True)
class GuardOutcome:
    tier: str
    refused: str = ""
    adjusted: str = ""
    usage: str = "unknown"


_USAGE_LOCK = threading.Lock()
_USAGE: Dict[str, Any] = {"reading": None, "loaded": False, "failed_at": 0.0, "refreshing": False}


def _reset_usage_cache() -> None:
    with _USAGE_LOCK:
        _USAGE.update(reading=None, loaded=False, failed_at=0.0, refreshing=False)


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def guard_limits(cfg: Dict[str, Any]) -> Tuple[float, float]:
    guard = wing_config(cfg)["usage_guard"]
    return float(guard.get("soft_percent", 70)), float(guard.get("hard_percent", 90))


def _ttl(cfg: Dict[str, Any]) -> float:
    return max(1.0, float(wing_config(cfg)["usage_guard"].get("cache_seconds", 300) or 300))


def _state_path(cfg: Dict[str, Any]) -> Optional[Path]:
    configured = str(wing_config(cfg)["usage_guard"].get("state_path") or "").strip()
    return Path(os.path.expanduser(configured)) if configured else None


def _load_persisted(cfg: Dict[str, Any]) -> Optional[UsageReading]:
    path = _state_path(cfg)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return UsageReading(_num(data.get("weekly")), _num(data.get("session")), float(data["fetched_at"]))
    except Exception:
        return None


def _persist(cfg: Dict[str, Any], reading: UsageReading) -> None:
    path = _state_path(cfg)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(
            {"weekly": reading.weekly, "session": reading.session, "fetched_at": reading.fetched_at}
        ), encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        pass


def _cached(cfg: Dict[str, Any]) -> Optional[UsageReading]:
    """The in-memory reading, seeded once per process from the state file. Call under the lock."""
    if not _USAGE["loaded"]:
        _USAGE["reading"] = _load_persisted(cfg)
        _USAGE["loaded"] = True
    return _USAGE["reading"]


def _fetch_reading() -> Optional[UsageReading]:
    """Weekly and 5-hour utilisation from Anthropic's OAuth usage endpoint."""
    try:
        from agent.account_usage import fetch_account_usage
    except Exception:
        return None
    snapshot = fetch_account_usage("anthropic")
    if snapshot is None or not getattr(snapshot, "available", False):
        return None
    weekly = session = None
    for window in getattr(snapshot, "windows", ()) or ():
        label = str(getattr(window, "label", ""))
        used = _num(getattr(window, "used_percent", None))
        if label == "Current week":
            weekly = used
        elif label == "Current session":
            session = used
    if weekly is None and session is None:
        return None
    return UsageReading(weekly, session, time.time())


def read_usage(cfg: Dict[str, Any], *, now: Optional[float] = None) -> Optional[UsageReading]:
    """The current reading, fetching at most once per ``cache_seconds``.

    A failed fetch is not retried within the period either: the endpoint being
    down must not add a 15-second wait to every delegation.
    """
    now = time.time() if now is None else now
    ttl = _ttl(cfg)
    with _USAGE_LOCK:
        reading = _cached(cfg)
        if reading is not None and now - reading.fetched_at < ttl:
            return reading
        if now - _USAGE["failed_at"] < ttl:
            return None
    try:
        fresh = _fetch_reading()
    except Exception:
        fresh = None
    with _USAGE_LOCK:
        if fresh is None:
            _USAGE["failed_at"] = now
        else:
            fresh = replace(fresh, fetched_at=now)
            _USAGE["reading"], _USAGE["failed_at"] = fresh, 0.0
    if fresh is None:
        _logger.warning("claude_wing: Anthropic usage unavailable; the guard fails open for %ds", int(ttl))
        return None
    _persist(cfg, fresh)
    return fresh


def _start_refresh(cfg: Dict[str, Any]) -> None:
    def run() -> None:
        try:
            read_usage(cfg)
        finally:
            with _USAGE_LOCK:
                _USAGE["refreshing"] = False

    threading.Thread(target=run, name="claude-wing-usage", daemon=True).start()


def peek_usage(cfg: Dict[str, Any]) -> Optional[UsageReading]:
    """The cached reading, never waiting on the network.

    For the routing note, which runs inside the parent's request middleware: a
    stale or missing reading starts one background refresh and is returned as is.
    """
    now = time.time()
    ttl = _ttl(cfg)
    with _USAGE_LOCK:
        reading = _cached(cfg)
        stale = reading is None or now - reading.fetched_at >= ttl
        refresh = stale and not _USAGE["refreshing"] and now - _USAGE["failed_at"] >= ttl
        if refresh:
            _USAGE["refreshing"] = True
    if refresh:
        _start_refresh(cfg)
    return reading


def wing_state(cfg: Dict[str, Any], reading: Optional[UsageReading]) -> str:
    if reading is None:
        return "unknown"
    soft, hard = guard_limits(cfg)
    weekly, session = reading.weekly or 0.0, reading.session or 0.0
    if weekly >= hard or session >= hard:
        return "closed"
    return "soft" if weekly >= soft else "open"


def apply_guard(tier: str, cfg: Dict[str, Any], reading: Optional[UsageReading]) -> GuardOutcome:
    """Workers only: the soft limit exists to leave the Opus parent room."""
    if reading is None:
        return GuardOutcome(tier)
    soft, hard = guard_limits(cfg)
    weekly, session = reading.weekly or 0.0, reading.session or 0.0
    usage = f"{weekly:.0f}%"
    if weekly >= hard:
        return GuardOutcome(tier, refused=f"Claude wing closed: weekly usage {weekly:.0f}% "
                                          f"(hard limit {hard:.0f}%).", usage=usage)
    if session >= hard:
        return GuardOutcome(tier, refused=f"Claude wing closed: 5-hour session usage {session:.0f}% "
                                          f"(hard limit {hard:.0f}%).", usage=usage)
    if weekly >= soft and tier == "opus":
        return GuardOutcome("sonnet", adjusted=f"opus→sonnet (weekly usage {weekly:.0f}%)", usage=usage)
    return GuardOutcome(tier, usage=usage)


# ---------------------------------------------------------------------------
# The delegate_claude tool

TOOL_NAME = "delegate_claude"
_DESCRIPTION = (
    "Spawn subagents on the Claude subscription -- a separate quota from delegate_task, which runs its "
    "workers on Codex. Same tasks shape as delegate_task, plus one tier for the whole call: \"haiku\" for "
    "quick lookups and exploration, \"sonnet\" (the default) as the everyday worker, \"opus\" for hard or "
    "consequential work. Use it when the router's note recommends a Claude target or the work needs Claude. "
    "Top-level calls run in the background and report back like delegate_task; list, steer or stop Claude "
    "children with delegate_task(action=...)."
)
_FALLBACK_TASKS: Dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What this subagent should accomplish. Be specific "
                                                      "and self-contained -- it knows nothing of your conversation."},
            "context": {"type": "string", "description": "Background this child needs: file paths, error "
                                                         "messages, constraints."},
        },
        "required": ["goal"],
    },
}
_MODEL_HIDDEN_TASK_FIELDS = ("acp_command", "acp_args")
_AUDIT_LOCK = threading.Lock()


def _host_delegate_task() -> Callable[..., str]:
    from tools.delegate_tool import delegate_task
    return delegate_task


def _host() -> Tuple[Callable[..., str], Callable[[], Any]]:
    from agent.subagent_lifecycle import get_active_subagent_parent
    return _host_delegate_task(), get_active_subagent_parent


def _independent_completions() -> bool:
    try:
        from tools.delegate_tool_config import _get_independent_completions
        return bool(_get_independent_completions())
    except Exception:
        return False


def _tasks_schema() -> Dict[str, Any]:
    """delegate_task's own ``tasks`` item shape, so the two tools cannot drift apart."""
    try:
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        tasks = deepcopy(DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"])
    except Exception:
        tasks = deepcopy(_FALLBACK_TASKS)
    if not _independent_completions():
        ((tasks.get("items") or {}).get("properties") or {}).pop("group", None)
    tasks["description"] = (
        "One entry per Claude worker. Entries run in parallel, all on the tier chosen for this call."
    )
    return tasks


def build_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_tier = wing_config(cfg).get("default_tier") or "sonnet"
    return {
        "name": TOOL_NAME,
        "description": _DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": _tasks_schema(),
                "tier": {
                    "type": "string",
                    "enum": list(TIERS),
                    "description": f"Claude tier for every task in this call. Default \"{default_tier}\".",
                },
            },
            "required": ["tasks"],
        },
    }


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def next_codex_route(target: str, cfg: Dict[str, Any]) -> str:
    """The Codex tier to name when the Claude wing cannot take this target's work."""
    from . import _is_callable_tier, _is_routable_tier, _peers_for

    def usable(name: str) -> bool:
        return bool(name) and _is_routable_tier(name, cfg) and _is_callable_tier(name, cfg)

    chain = cfg.get("fallbacks") or {}
    seen, current = {target}, target
    for _ in range(4):
        following = str(chain.get(current) or "")
        if not following or following in seen:
            break
        if usable(following):
            return following
        seen.add(following)
        current = following
    for peer in _peers_for(target, cfg):
        if usable(peer):
            return peer
    default = str(cfg.get("default_model") or "")
    return default if usable(default) else ""


def _pointer(target: str, cfg: Dict[str, Any], *, claude_ok: bool) -> str:
    from . import _is_callable_tier, _peers_for

    options = []
    if claude_ok:
        for peer in _peers_for(target, cfg):
            if peer in TIER_FOR_TARGET and _is_callable_tier(peer, cfg):
                options.append(f'delegate_claude with tier "{TIER_FOR_TARGET[peer]}"')
                break
    codex = next_codex_route(target, cfg)
    options.append(f"delegate_task with a goal prefixed [{codex}]" if codex else "delegate_task")
    return "Use " + " or ".join(options) + " instead."


def _unavailable(target: str, cfg: Dict[str, Any]) -> str:
    from . import _tier_cooldown_remaining

    if (cfg.get("callable") or {}).get(target) is not True:
        return "switched off in the dashboard"
    remaining = _tier_cooldown_remaining(target, cfg)
    return f"cooling down for another {int(remaining // 60) + 1} min" if remaining > 0 else ""


def _strip_hidden(tasks: Any) -> Any:
    if not isinstance(tasks, list):
        return tasks
    return [
        {k: v for k, v in task.items() if k not in _MODEL_HIDDEN_TASK_FIELDS} if isinstance(task, dict) else task
        for task in tasks
    ]


def _audit(cfg: Dict[str, Any], requested: str, used: str, outcome: GuardOutcome, result: str,
           message: str = "") -> None:
    configured = str(wing_config(cfg).get("log_path") or "").strip()
    if not configured:
        return
    # No "tier" key on purpose: the router's per-account load counts lines by tier,
    # and the children's own calls are already counted through the middleware.
    entry = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "event": TOOL_NAME,
        "tier_requested": requested,
        "tier_used": used,
        "target": TARGET_FOR_TIER.get(used, ""),
        "model": tier_model(used, cfg),
        "usage": outcome.usage,
        "outcome": result,
    }
    if outcome.adjusted:
        entry["adjusted"] = outcome.adjusted
    if message:
        entry["message"] = message
    try:
        path = Path(os.path.expanduser(configured))
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _raw_error(raw: Any) -> Optional[str]:
    """The "error" text when ``raw`` parses as a JSON object carrying one, else None."""
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return str(error) if error else None


def _annotate(raw: Any, tier: str, outcome: GuardOutcome) -> Any:
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(payload, dict):
        return raw
    payload["claude_tier"] = tier
    if outcome.adjusted:
        payload["tier_adjusted"] = outcome.adjusted
    if outcome.usage == "unknown":
        payload["usage"] = "unknown"
    return json.dumps(payload, ensure_ascii=False)


def _dispatch(args: Dict[str, Any]) -> str:
    from . import _load_config

    cfg = _load_config()
    wing = wing_config(cfg)
    requested = str(args.get("tier") or wing.get("default_tier") or "sonnet").strip().casefold()
    if requested not in TIERS:
        return _error(f"Unknown tier {requested!r}; use one of: {', '.join(TIERS)}.")
    delegate_task, active_parent = _host()
    parent = active_parent()
    if parent is None:
        return _error("delegate_claude must be called from an agent turn; no active Hermes parent was found.")

    outcome = apply_guard(requested, cfg, read_usage(cfg))
    if outcome.refused:
        message = f"{outcome.refused} {_pointer(TARGET_FOR_TIER[requested], cfg, claude_ok=False)}"
        _audit(cfg, requested, requested, outcome, "refused", message)
        return _error(message)
    tier = outcome.tier
    target = TARGET_FOR_TIER[tier]
    unavailable = _unavailable(target, cfg)
    if unavailable:
        message = f"Claude tier \"{tier}\" ({target}) is {unavailable}. {_pointer(target, cfg, claude_ok=True)}"
        _audit(cfg, requested, tier, outcome, "refused", message)
        return _error(message)
    model = tier_model(tier, cfg)
    if not model:
        return _error(f"Claude tier \"{tier}\" has no model under claude_wing.tiers.")

    raw = delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        tasks=_strip_hidden(args.get("tasks")),
        parent_agent=parent,
        # Hermes's own rule (run_agent._dispatch_delegate_task): background at the
        # top level, synchronous for an orchestrator child that needs its results.
        background=not getattr(parent, "_delegate_depth", 0) > 0,
        credentials_cfg={"provider": "anthropic", "model": model, "fallback_providers": []},
    )
    error_message = _raw_error(raw)
    if error_message is not None:
        _audit(cfg, requested, tier, outcome, "error", error_message[:300])
    else:
        _audit(cfg, requested, tier, outcome, "lowered" if outcome.adjusted else "ran")
    return _annotate(raw, tier, outcome)


def handle_delegate_claude(args: Dict[str, Any], **_kwargs: Any) -> str:
    """Tool handler. Never raises into the turn: every failure is a tool error."""
    try:
        return _dispatch(args if isinstance(args, dict) else {})
    except Exception as exc:
        return _error(f"delegate_claude failed: {type(exc).__name__}: {exc}")


def _exempt_from_sequential_deadline() -> bool:
    """Exempt delegate_claude from Hermes's 420s sequential tool deadline.

    A background delegate_task batch runs synchronously when the async pool is
    full or the session can't take async completions, and such a batch can run
    long. delegate_task and manage_connections are already exempt
    (``agent.tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS``); without the same
    exemption a real delegate_claude batch times out and orphans its workers.
    Never raises: a host that no longer exposes this set must not block
    registration.
    """
    try:
        import importlib
        tool_executor = importlib.import_module("agent.tool_executor")
    except Exception as exc:
        _logger.warning(
            "claude_wing: could not exempt delegate_claude from the sequential tool deadline: %s", exc
        )
        return False
    existing = getattr(tool_executor, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", None)
    if not isinstance(existing, frozenset):
        _logger.warning(
            "claude_wing: could not exempt delegate_claude from the sequential tool deadline: "
            "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS is missing or not a frozenset"
        )
        return False
    tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset(existing | {TOOL_NAME})
    return True


def register(ctx: Any, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Register delegate_claude when the wing is on and the host can carry it."""
    global _ACTIVE
    if cfg is None:
        from . import _load_config
        cfg = _load_config()
    reason = registration_block(cfg)
    if reason:
        _ACTIVE = False
        _logger.info("claude_wing: delegate_claude not registered: %s", reason)
        return False
    try:
        handle = ctx.register_tool(name=TOOL_NAME, toolset="delegation", schema=build_schema(cfg),
                                   handler=handle_delegate_claude, description=_DESCRIPTION, emoji="🪶")
    except Exception as exc:
        _ACTIVE = False
        _logger.warning("claude_wing: registering delegate_claude failed: %s", exc)
        return False
    if handle is None:
        _ACTIVE = False
        _logger.info("claude_wing: delegate_claude not registered: ctx.register_tool returned None")
        return False
    _ACTIVE = True
    _exempt_from_sequential_deadline()
    return True
