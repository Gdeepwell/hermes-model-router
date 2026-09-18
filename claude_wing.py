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
