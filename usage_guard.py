"""Account usage and the soft/hard usage guard, the same for every delegation account.

An account is a ``tier_providers`` value (``anthropic``, ``openai-codex``, ...).
Each configured account gets the same rules: from ``soft_percent`` weekly usage
its heaviest tier steps down (``step_down``), and from ``hard_percent`` weekly or
5-hour usage delegation to it is closed. An account that is not configured under
``usage_guard.accounts`` is never touched.

``usage_guard.balance`` compares two accounts instead of one against its limits:
the router uses ``load``/``fresh``/``balance_config`` to put the freer account
first when the preferred one is busy (see ``_advised_chain`` in the router).

Nothing here imports the router, so both the router and the standalone
dashboard can use it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

_logger = logging.getLogger("model_router.usage_guard")

ACCOUNT_LABELS: Dict[str, str] = {"openai-codex": "Codex", "anthropic": "Claude", "qwen-token": "Qwen"}
DEFAULTS: Dict[str, Any] = {"cache_seconds": 300, "state_path": "", "accounts": {}}
BALANCE_DEFAULTS: Dict[str, Any] = {"enabled": False, "busy_percent": 20.0, "margin_percent": 10.0,
                                    "window": "5-hour"}
# 5-hour: compare the 5-hour windows only, and leave the weekly one to the soft/hard
# limits (a Claude parent keeps Claude's week ahead anyway). tighter: max of both.
BALANCE_WINDOWS = ("5-hour", "tighter")


def account_label(account: str) -> str:
    return ACCOUNT_LABELS.get(account, account)


def guard_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = (cfg or {}).get("usage_guard")
    merged = deepcopy(DEFAULTS)
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if k != "accounts"})
        if isinstance(raw.get("accounts"), dict):
            merged["accounts"] = raw["accounts"]
    return merged


def account_limits(account: str, cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    entry = guard_config(cfg)["accounts"].get(account)
    if not isinstance(entry, dict):
        return None
    step_down = entry.get("step_down")
    return {
        "soft_percent": float(entry.get("soft_percent", 70)),
        "hard_percent": float(entry.get("hard_percent", 90)),
        "step_down": dict(step_down) if isinstance(step_down, dict) else {},
    }


def guarded(account: str, cfg: Optional[Dict[str, Any]]) -> bool:
    return account_limits(account, cfg) is not None


def balance_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``usage_guard.balance`` with defaults; off unless configured on."""
    raw = guard_config(cfg).get("balance")
    merged = dict(BALANCE_DEFAULTS)
    if isinstance(raw, dict):
        merged["enabled"] = raw.get("enabled", False) is True
        for key in ("busy_percent", "margin_percent"):
            try:
                merged[key] = float(raw.get(key, merged[key]))
            except (TypeError, ValueError):
                pass
        if raw.get("window") in BALANCE_WINDOWS:
            merged["window"] = raw["window"]
    return merged


@dataclass(frozen=True)
class Reading:
    weekly: Optional[float]
    session: Optional[float]
    weekly_resets_at: Optional[str]
    session_resets_at: Optional[str]
    fetched_at: float


@dataclass(frozen=True)
class GuardOutcome:
    tier: str
    refused: str = ""
    adjusted: str = ""
    usage: str = "unknown"


def load(reading: Optional["Reading"], window: str = "tighter") -> Optional[tuple]:
    """(percent, window name) an account is compared by, or None when it is unknown.

    ``tighter`` is the higher of the weekly and 5-hour windows; ``5-hour`` is that
    window alone.
    """
    if reading is None:
        return None
    candidates = ((reading.session, "5-hour"),) if window == "5-hour" else (
        (reading.weekly, "weekly"), (reading.session, "5-hour"))
    windows = [(value, name) for value, name in candidates if value is not None]
    return max(windows, key=lambda window: window[0]) if windows else None


def fresh(reading: Optional["Reading"], cfg: Dict[str, Any], now: Optional[float] = None) -> bool:
    """Whether a reading is recent enough to steer by: under twice ``cache_seconds``,
    the same line the dashboard greys a card out at."""
    if reading is None:
        return False
    return (time.time() if now is None else now) - reading.fetched_at < 2 * _ttl(cfg)


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Fetchers

_ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"


def _status_code(exc: BaseException) -> Optional[int]:
    """The HTTP status an exception carries, without importing the http client."""
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _anthropic_tokens() -> Iterator[str]:
    """Tokens to try against the usage endpoint, best first.

    ``resolve_anthropic_token`` reads the credential pool with ``refresh=False``
    on purpose, so that diagnostic callers never mutate auth.json or hit the
    network. A pool row that is not the Claude Code one -- a ``manual:hermes_pkce``
    entry, say -- therefore shadows the refreshable Claude Code credentials with a
    token that may already have expired, and the endpoint answers 401. The API
    call path recovers by refreshing; this reader is not on that path, so it falls
    back explicitly rather than reporting the account unreadable for as long as
    the stale row sits in the pool (measured on this host 2026-09-21: the resolver
    gave a 401 token while the Claude Code credentials were valid for hours).
    """
    from agent import anthropic_credentials as credentials

    seen = set()
    resolvers = (
        getattr(credentials, "resolve_anthropic_token", None),
        getattr(credentials, "_resolve_claude_code_token_from_credentials", None),
    )
    for resolve in resolvers:
        if not callable(resolve):
            continue
        try:
            token = (resolve() or "").strip()
        except Exception:
            continue
        if token and token not in seen:
            seen.add(token)
            yield token


def _fetch_anthropic() -> Optional[Reading]:
    """Read raw: the endpoint reports utilization as a percentage (live 2026-09-18:
    5.0 / 13.0), and Hermes's fetch_account_usage scales any value <= 1 by 100."""
    try:
        from agent.account_usage import _get_json
        import agent.anthropic_credentials  # noqa: F401  -- _anthropic_tokens needs it
    except Exception:
        return None
    payload = None
    for token in _anthropic_tokens():
        try:
            payload = _get_json(_ANTHROPIC_USAGE_URL, {
                "Authorization": f"Bearer {token}", "Accept": "application/json",
                "Content-Type": "application/json",
                "anthropic-beta": "oauth-2025-04-20", "User-Agent": "claude-code/2.1.0",
            }, timeout=15.0)
            break
        except Exception as exc:
            # Only a rejected token is worth another identity; anything else
            # (network, 5xx, a changed endpoint) fails the read as before.
            if _status_code(exc) not in (401, 403):
                return None
            _logger.debug("usage_guard: the Anthropic usage endpoint rejected a token; trying the next one")
    if not isinstance(payload, dict):
        return None
    week, session = payload.get("seven_day") or {}, payload.get("five_hour") or {}
    weekly, used = _num(week.get("utilization")), _num(session.get("utilization"))
    if weekly is None and used is None:
        return None
    return Reading(weekly, used, _iso(week.get("resets_at")), _iso(session.get("resets_at")), time.time())


def _fetch_codex() -> Optional[Reading]:
    """Hermes's reader: Codex reports used_percent already as a percentage."""
    try:
        from agent.account_usage import fetch_account_usage
    except Exception:
        return None
    snapshot = fetch_account_usage("openai-codex")
    if snapshot is None or not getattr(snapshot, "available", False):
        return None
    weekly = session = None
    weekly_reset = session_reset = None
    for window in getattr(snapshot, "windows", ()) or ():
        label = str(getattr(window, "label", ""))
        if label == "Weekly":
            weekly, weekly_reset = _num(getattr(window, "used_percent", None)), _iso(getattr(window, "reset_at", None))
        elif label == "Session":
            session, session_reset = _num(getattr(window, "used_percent", None)), _iso(getattr(window, "reset_at", None))
    if weekly is None and session is None:
        return None
    return Reading(weekly, session, weekly_reset, session_reset, time.time())


FETCHERS: Dict[str, Callable[[], Optional[Reading]]] = {
    "anthropic": _fetch_anthropic,
    "openai-codex": _fetch_codex,
}


def has_fetcher(account: str) -> bool:
    return account in FETCHERS


# ---------------------------------------------------------------------------
# Cache: memory per process, plus one shared state file when configured

_LOCK = threading.Lock()
_FILE_LOCK = threading.Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}


def _reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _slot(account: str) -> Dict[str, Any]:
    return _CACHE.setdefault(account, {"reading": None, "failed_at": 0.0, "refreshing": False})


def _ttl(cfg: Dict[str, Any]) -> float:
    return max(1.0, float(guard_config(cfg).get("cache_seconds", 300) or 300))


def _state_path(cfg: Dict[str, Any]) -> Optional[Path]:
    configured = str(guard_config(cfg).get("state_path") or "").strip()
    return Path(os.path.expanduser(configured)) if configured else None


def _load_file(cfg: Dict[str, Any]) -> Dict[str, Reading]:
    path = _state_path(cfg)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    readings = {}
    for account, entry in (data.items() if isinstance(data, dict) else ()):
        try:
            readings[account] = Reading(_num(entry.get("weekly")), _num(entry.get("session")),
                                        entry.get("weekly_resets_at"), entry.get("session_resets_at"),
                                        float(entry["fetched_at"]))
        except Exception:
            continue
    return readings


def _persist(cfg: Dict[str, Any], account: str, reading: Reading) -> None:
    path = _state_path(cfg)
    if path is None:
        return
    try:
        with _FILE_LOCK:
            data = {a: r.__dict__ for a, r in _load_file(cfg).items()}
            data[account] = reading.__dict__
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(data), encoding="utf-8")
            os.replace(temporary, path)
    except Exception:
        pass


def _newest(account: str, cfg: Dict[str, Any]) -> Optional[Reading]:
    """The newer of this process's reading and the shared file's. Call under the lock."""
    memory = _slot(account)["reading"]
    shared = _load_file(cfg).get(account)
    candidates = [r for r in (memory, shared) if r is not None]
    return max(candidates, key=lambda r: r.fetched_at) if candidates else None


def cached(account: str, cfg: Dict[str, Any]) -> Optional[Reading]:
    """The latest known reading, without fetching or starting a refresh (for the dashboard)."""
    with _LOCK:
        return _newest(account, cfg)


def read(account: str, cfg: Dict[str, Any], *, now: Optional[float] = None,
         force: bool = False) -> Optional[Reading]:
    """The current reading, fetching at most once per ``cache_seconds``; failures back off too.

    ``force`` skips both gates. They throttle *automatic* reads, so applying them
    to someone pressing Refresh made the button a no-op for a whole period: it
    answered from the cache and the bars never moved. An explicit refresh fetches.
    """
    fetcher = FETCHERS.get(account)
    if fetcher is None:
        return None
    now = time.time() if now is None else now
    ttl = _ttl(cfg)
    if not force:
        with _LOCK:
            reading = _newest(account, cfg)
            if reading is not None and now - reading.fetched_at < ttl:
                return reading
            if now - _slot(account)["failed_at"] < ttl:
                return None
    try:
        fresh = fetcher()
    except Exception:
        fresh = None
    with _LOCK:
        slot = _slot(account)
        if fresh is None:
            slot["failed_at"] = now
        else:
            fresh = replace(fresh, fetched_at=now)
            slot["reading"], slot["failed_at"] = fresh, 0.0
    if fresh is None:
        _logger.warning("usage_guard: %s usage unavailable; the guard fails open for %ds",
                        account_label(account), int(ttl))
        return None
    _persist(cfg, account, fresh)
    return fresh


def _start_refresh(account: str, cfg: Dict[str, Any]) -> None:
    def run() -> None:
        try:
            read(account, cfg)
        finally:
            with _LOCK:
                _slot(account)["refreshing"] = False

    threading.Thread(target=run, name=f"usage-guard-{account}", daemon=True).start()


def peek(account: str, cfg: Dict[str, Any]) -> Optional[Reading]:
    """The latest reading, never waiting on the network: a stale one starts one refresh."""
    if account not in FETCHERS:
        return None
    now = time.time()
    ttl = _ttl(cfg)
    with _LOCK:
        reading = _newest(account, cfg)
        slot = _slot(account)
        stale = reading is None or now - reading.fetched_at >= ttl
        refresh = stale and not slot["refreshing"] and now - slot["failed_at"] >= ttl
        if refresh:
            slot["refreshing"] = True
    if refresh:
        try:
            _start_refresh(account, cfg)
        except Exception:
            # Never leave `refreshing` stuck at True: that would permanently
            # block every future refresh for this account.
            with _LOCK:
                _slot(account)["refreshing"] = False
    return reading


def state(account: str, cfg: Dict[str, Any], reading: Optional[Reading]) -> str:
    limits = account_limits(account, cfg)
    if limits is None or reading is None:
        return "unknown"
    weekly, session = reading.weekly or 0.0, reading.session or 0.0
    if weekly >= limits["hard_percent"] or session >= limits["hard_percent"]:
        return "closed"
    return "soft" if weekly >= limits["soft_percent"] else "open"


def apply(account: str, tier: str, cfg: Dict[str, Any], reading: Optional[Reading]) -> GuardOutcome:
    """The guard for one worker on ``account``; ``tier`` is a router target name."""
    limits = account_limits(account, cfg)
    if limits is None or reading is None:
        return GuardOutcome(tier)
    weekly, session = reading.weekly or 0.0, reading.session or 0.0
    usage, label, hard = f"{weekly:.0f}%", account_label(account), limits["hard_percent"]
    if weekly >= hard:
        return GuardOutcome(tier, refused=f"{label} delegation closed: weekly usage {weekly:.0f}% "
                                          f"(hard limit {hard:.0f}%).", usage=usage)
    if session >= hard:
        return GuardOutcome(tier, refused=f"{label} delegation closed: 5-hour session usage {session:.0f}% "
                                          f"(hard limit {hard:.0f}%).", usage=usage)
    target = limits["step_down"].get(tier)
    if weekly >= limits["soft_percent"] and target:
        return GuardOutcome(target, adjusted=f"{tier}→{target} (weekly usage {weekly:.0f}%)", usage=usage)
    return GuardOutcome(tier, usage=usage)
