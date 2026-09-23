"""Claude delegation: Claude workers as real Hermes children, next to the Codex workforce.

Hermes's ``delegate_task`` has one delegation route per process, and on this host
it is pinned to Codex. ``delegate_claude`` reaches Claude by calling the same
``delegate_task`` with a per-call route (``credentials_cfg``) pinned to the
``anthropic`` provider -- the mechanism Hermes's own /review uses. Pinning the
provider, rather than inheriting the parent's, is what keeps Claude delegation working
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
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from . import usage_guard

_logger = logging.getLogger("model_router.claude_delegation")

TIERS: Tuple[str, ...] = ("haiku", "sonnet", "opus")
# Router target names stay what the config, cooldowns and dashboard already use;
# the tool speaks in short tier names. This table is the only place they meet.
TARGET_FOR_TIER: Dict[str, str] = {"haiku": "haiku", "sonnet": "sonnet5", "opus": "opus5"}
TIER_FOR_TARGET: Dict[str, str] = {target: tier for tier, target in TARGET_FOR_TIER.items()}

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "tiers": {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5-5"},
    "default_tier": "sonnet",
    "log_path": "",
}

_ACTIVE = False
# Set by the router for the request it is routing: whether *this* request can use
# delegate_claude. A session's tool list is fixed when its agent is built, so the
# live workflow and the tools the request actually carries can disagree; the
# request is what the conductor sees, so it decides.
_REQUEST_ACTIVE: ContextVar[Optional[bool]] = ContextVar("claude_delegation_request_active", default=None)
# The last availability the router saw, so a flip can drop Hermes's tool-list memo.
_LAST_AVAILABLE: Optional[bool] = None


def is_active() -> bool:
    """Whether Claude delegation may be offered right now.

    Inside a routed request: the request's own answer (workflow allows it AND the
    request offers the tool). Outside one: whether the tool is registered.
    """
    scoped = _REQUEST_ACTIVE.get()
    return _ACTIVE if scoped is None else scoped


@contextmanager
def request_scope(active: bool) -> Iterator[None]:
    token = _REQUEST_ACTIVE.set(bool(active))
    try:
        yield
    finally:
        _REQUEST_ACTIVE.reset(token)


# Hermes's Tool Search defers plugin tools: a live parent request carries only the
# tool_search/tool_describe/tool_call bridge, and delegate_claude is reached through
# tool_call. Whether it is in that session's deferred scope follows tool_available().
_BRIDGE_CALL_NAMES = frozenset({"tool_call", "mcp__tool_call"})


def offered(tool_names: Iterable[str]) -> bool:
    """Whether a request can reach delegate_claude: listed itself, or through tool_call."""
    names = set(tool_names)
    return bool(names & ({TOOL_NAME, f"mcp__{TOOL_NAME}"} | _BRIDGE_CALL_NAMES))


def delegation_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``claude_delegation`` block with defaults filled in; off unless configured on."""
    raw = (cfg or {}).get("claude_delegation")
    raw = raw if isinstance(raw, dict) else {}
    merged = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in ("tiers",):
            if isinstance(value, dict):
                merged[key] = {**DEFAULTS[key], **value}
        else:
            merged[key] = value
    return merged


def tier_model(tier: str, cfg: Dict[str, Any]) -> str:
    return str(delegation_config(cfg)["tiers"].get(tier) or "").strip()


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
    """Whether this Hermes still has the two internals Claude delegation stands on.

    ``credentials_cfg`` is commented "internal callers only" upstream, and the
    active-parent lookup is not on the plugin context. If either moves,
    delegate_claude must not register rather than fail at call time.
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


def availability_block(cfg: Dict[str, Any]) -> str:
    """Why ``delegate_claude`` must not be offered now, or "" when it may be.

    Config only, so it is cheap enough to run on every tool-list build. The
    router's ``_load_config`` has already applied ``workflow: codex``, which turns
    ``claude_delegation.enabled`` off.
    """
    if not delegation_config(cfg).get("enabled"):
        return "claude_delegation.enabled is false"
    switches = cfg.get("callable") or {}
    if not any(switches.get(target) is True for target in TARGET_FOR_TIER.values()):
        return "every Claude target is switched off in `callable`"
    return ""


def tool_available() -> bool:
    """delegate_claude's check_fn: Hermes offers the tool only while this is True."""
    from . import _load_config

    return availability_block(_load_config()) == ""


def note_availability(available: bool) -> None:
    """Drop Hermes's memoized tool list when availability flips.

    ``model_tools`` memoizes whole tool lists without re-running check_fns, so a
    flipped workflow would otherwise reach new sessions only after a restart.
    ``_clear_tool_defs_cache`` is private upstream; without it, new sessions
    still follow the switch once the memo is rebuilt for another reason.
    """
    global _LAST_AVAILABLE
    previous, _LAST_AVAILABLE = _LAST_AVAILABLE, bool(available)
    if previous is None or previous == _LAST_AVAILABLE:
        return
    try:
        import model_tools

        clear = getattr(model_tools, "_clear_tool_defs_cache", None)
        if callable(clear):
            clear()
    except Exception as exc:
        _logger.debug("claude_delegation: could not clear the host tool-list memo: %s", exc)


def _uncached(fn: Callable[[], bool]) -> Callable[[], bool]:
    """Exempt a check_fn from Hermes's 30-second TTL cache, where the host supports it."""
    try:
        from tools.registry import no_cache_check_fn
    except Exception:
        return fn
    return no_cache_check_fn(fn)


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
    default_tier = delegation_config(cfg).get("default_tier") or "sonnet"
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
    """The Codex tier to name when Claude delegation cannot take this target's work."""
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


ACCOUNT = "anthropic"


def _log(cfg: Dict[str, Any], entry: Dict[str, Any]) -> None:
    configured = str(delegation_config(cfg).get("log_path") or "").strip()
    if not configured:
        return
    line = {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **entry}
    try:
        path = Path(os.path.expanduser(configured))
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _audit(cfg: Dict[str, Any], parent: Any, requested: str, used: str,
           outcome: "usage_guard.GuardOutcome", result: str, message: str = "") -> None:
    # No "tier" key on purpose: the router's per-account load counts lines by tier,
    # and the children's own calls are already counted through the middleware.
    entry = {
        "event": TOOL_NAME,
        "tier_requested": requested,
        "tier_used": used,
        "target": TARGET_FOR_TIER.get(used, ""),
        "model": tier_model(used, cfg),
        "usage": outcome.usage,
        "outcome": result,
    }
    session_id = str(getattr(parent, "session_id", "") or "")
    turn_id = str(getattr(parent, "_current_turn_id", "") or "")
    if session_id:
        entry["session_id"] = session_id
    if turn_id:
        entry["turn_id"] = turn_id
    if outcome.adjusted:
        entry["adjusted"] = outcome.adjusted
    if message:
        entry["message"] = message
    _log(cfg, entry)


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


def _annotate(raw: Any, tier: str, outcome: "usage_guard.GuardOutcome") -> Any:
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
    settings = delegation_config(cfg)
    if not settings.get("enabled"):
        if str(cfg.get("workflow") or "").strip().casefold() == "codex":
            return _error("Claude delegation is off: router_config.yaml is on workflow: codex. "
                          "Use delegate_task, which runs on the Codex route.")
        return _error("Claude delegation is switched off in router_config.yaml.")
    requested = str(args.get("tier") or settings.get("default_tier") or "sonnet").strip().casefold()
    if requested not in TIERS:
        return _error(f"Unknown tier {requested!r}; use one of: {', '.join(TIERS)}.")
    delegate_task, active_parent = _host()
    parent = active_parent()
    if parent is None:
        return _error("delegate_claude must be called from an agent turn; no active Hermes parent was found.")

    guarded = usage_guard.apply(ACCOUNT, TARGET_FOR_TIER[requested], cfg, usage_guard.read(ACCOUNT, cfg))
    tier = TIER_FOR_TARGET.get(guarded.tier, requested)
    outcome = replace(guarded, tier=tier,
                      adjusted=f"{requested}→{tier} (weekly usage {guarded.usage})" if guarded.adjusted else "")
    if outcome.refused:
        message = f"{outcome.refused} {_pointer(TARGET_FOR_TIER[requested], cfg, claude_ok=False)}"
        _audit(cfg, parent, requested, requested, outcome, "refused", message)
        return _error(message)
    target = TARGET_FOR_TIER[tier]
    unavailable = _unavailable(target, cfg)
    if unavailable:
        message = f"Claude tier \"{tier}\" ({target}) is {unavailable}. {_pointer(target, cfg, claude_ok=True)}"
        _audit(cfg, parent, requested, tier, outcome, "refused", message)
        return _error(message)
    model = tier_model(tier, cfg)
    if not model:
        return _error(f"Claude tier \"{tier}\" has no model under claude_delegation.tiers.")

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
        _audit(cfg, parent, requested, tier, outcome, "error", error_message[:300])
    else:
        _audit(cfg, parent, requested, tier, outcome, "lowered" if outcome.adjusted else "ran")
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
            "claude_delegation: could not exempt delegate_claude from the sequential tool deadline: %s", exc
        )
        return False
    existing = getattr(tool_executor, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", None)
    if not isinstance(existing, frozenset):
        _logger.warning(
            "claude_delegation: could not exempt delegate_claude from the sequential tool deadline: "
            "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS is missing or not a frozenset"
        )
        return False
    tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset(existing | {TOOL_NAME})
    return True


def register(ctx: Any, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Register delegate_claude whenever the host can carry it.

    Registered even while the workflow keeps it off: its check_fn decides, per
    tool-list build, whether Hermes offers it, so the switch needs no restart.
    """
    global _ACTIVE
    if cfg is None:
        from . import _load_config
        cfg = _load_config()
    ok, reason = host_check()
    if not ok:
        _ACTIVE = False
        _logger.info("claude_delegation: delegate_claude not registered: %s", reason)
        _log(cfg, {"event": "registration", "registered": False, "reason": reason})
        return False
    try:
        handle = ctx.register_tool(name=TOOL_NAME, toolset="delegation", schema=build_schema(cfg),
                                   handler=handle_delegate_claude, check_fn=_uncached(tool_available),
                                   description=_DESCRIPTION, emoji="🪶")
    except Exception as exc:
        _ACTIVE = False
        reason = f"registering delegate_claude failed: {exc}"
        _logger.warning("claude_delegation: %s", reason)
        _log(cfg, {"event": "registration", "registered": False, "reason": reason})
        return False
    if handle is None:
        _ACTIVE = False
        reason = "ctx.register_tool returned None"
        _logger.info("claude_delegation: delegate_claude not registered: %s", reason)
        _log(cfg, {"event": "registration", "registered": False, "reason": reason})
        return False
    _ACTIVE = True
    _exempt_from_sequential_deadline()
    _log(cfg, {"event": "registration", "registered": True, "reason": "",
               "available": availability_block(cfg) == ""})
    return True
