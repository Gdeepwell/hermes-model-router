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
