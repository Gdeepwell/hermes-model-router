"""Conservative GPT-5.6 Luna/Terra/Sol + Codex-Spark request router."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import unicodedata
from copy import deepcopy
from types import SimpleNamespace

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - Hermes includes PyYAML
    yaml = None


_PLUGIN_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _PLUGIN_DIR / "router_config.yaml"
_LOG_LOCK = threading.Lock()
_SHADOW_LOCK = threading.RLock()
_QUOTA_LOCK = threading.Lock()
_SPARK_QUOTA_EXHAUSTED_TURNS: set[str] = set()
_MAX_REMEMBERED_QUOTA_TURNS = 2048

_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "provider": "openai-codex",
    "models": {
        "luna": "gpt-5.6-luna",
        "spark": "gpt-5.3-codex-spark",
        "terra": "gpt-5.6-terra",
        "sol": "gpt-5.6-sol",
        "qwen": "qwen3.7-plus",
    },
    # Which tiers are callable — togglable from the web dashboard.
    # When a tier is disabled, any route that selected it falls back
    # according to the `fallbacks` map below.
    "callable": {
        "luna": True,
        "spark": True,
        "terra": True,
        "sol": True,
        "opus5": True,
        "qwen": True,
    },
    # When a callable tier is disabled, routes that selected it fall back here.
    "fallbacks": {
        "spark": "luna",
        "luna": "terra",
        "sol": "terra",
        "opus5": "sol",
        "qwen": "terra",
    },
    # The default model is both the general-purpose route destination and the
    # orchestration owner. Changing it rewrites this file and the Hermes config.
    "default_model": "terra",
    # spark_max_chars / spark_dev_max_chars removed: no automatic Spark route
    # exists for them to bound, so they only advertised a control that was never
    # consulted. Spark is reached via an explicit [spark] label or delegation.
    "thresholds": {
        "luna_max_chars": 700,
        "sol_min_chars": 3500,
    },
    "effort": {"luna": "low", "spark": "low", "terra": "medium", "sol": "high", "qwen": "medium", "explicit_sol": "xhigh"},
    # Provider mapping per tier: which Hermes provider handles each tier.
    "tier_providers": {
        "luna": "openai-codex",
        "spark": "openai-codex",
        "terra": "openai-codex",
        "sol": "openai-codex",
        "opus5": "openai-codex",
        "qwen": "qwen-token",
    },
    "quota_fallbacks": {"spark": {"model": "luna", "effort": "medium"}},
    "logging": {
        "enabled": True,
        "path": "~/.hermes/logs/model-router.jsonl",
    },
    "delegation": {"preserve_spark_subagents": True},
    "coding_agent": {
        "enabled": False,
        "tier": "opus5",
        "model": "claude-opus-5",
        "default_repo": "",
        "max_turns": 8,
        "max_budget_usd": 5.0,
        "timeout_seconds": 300,
        "lifecycle_path": "~/.hermes/logs/claude-code-bridge.jsonl",
        "reviewer": {"enabled": False, "max_chars": 8000},
        # Delegated read-only Claude review, off by default and independent of
        # ``enabled`` above, which also arms the label-free coding classifier.
        "delegated_review": {"enabled": False, "max_chars": 8000, "models": ["opus", "sonnet"]},
    },
    # A user-facing session has one durable parent.  The router may still
    # classify specialist *workers*, but it must not turn each user message into
    # a cold planner/model handoff.
    "session_policy": {
        "pin_root_parent": True,
        "handoff_capsule_version": "v1",
    },
    "orchestration": {
        # Automatic planner fan-out is opt-in.  Parent agents delegate only
        # when they identify a genuinely independent bounded worker task.
        "enabled": False,
        "min_chars": 180,
        "max_tasks": 1,
        # Recovery gate for an active Terra tool loop whose initial preflight
        # was missed (for example, a process that loaded an older plugin).
        "rescue_min_calls": 6,
        "path": "~/.hermes/logs/terra-spark-orchestration.jsonl",
    },
    # A tier that just rejected a call for quota is not a candidate for the next
    # one. Held on disk because the interactive TUI and the gateway are separate
    # processes: an in-memory note would not be seen by the other one.
    "cooldown": {
        "enabled": True,
        "path": "~/.hermes/state/model-router-cooldowns.json",
        "quota_seconds": 900,
        "allowed_fails": 3,
        "failure_window_seconds": 60,
        "failure_seconds": 60,
    },
    "usage_report": {"enabled": True, "window_seconds": 3600},
    # Targets of comparable strength, on deliberately different accounts. Used
    # to move work off a loaded or cooling target rather than queueing on it.
    "peer_groups": {
        "heavy": ["terra", "opus5", "qwen"],
        "light": ["luna", "sonnet5", "spark"],
    },
    "shadow": {
        "enabled": False,
        "limit": 10,
        "path": "~/.hermes/logs/spark-shadow-benchmark.jsonl",
    },
}


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    model: str
    reason: str
    effort: str = "medium"
    # Tiers this request was independently eligible for but did not get. The
    # route log records only the winning reason, which hides why an alternative
    # never fires: an eligible-but-never-chosen tier looks identical to one whose
    # preconditions are never met. These signals separate the two.
    vetoed_by: Tuple[str, ...] = ()
    # True when the tier was chosen by policy or a hard capability limit rather
    # than preference. Such a route must not be satisfied by the fallback chain:
    # falling back from it grants exactly the access the decision denied.
    mandatory: bool = False
    # The work kind the gate recognised ("design", "code", "explore", ...). It is
    # what a user preference list is keyed on, and it is recorded in the route log
    # so a surprising route can be traced back to the category that produced it.
    kind: str = ""
    # The external delegation target preferred for this kind, when the preference
    # list names one. The router cannot route across providers, so this travels
    # as advice to the conductor rather than as the route itself.
    prefer_target: str = ""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Work kinds a preference list may be keyed on. Kept as an explicit tuple so the
# dashboard, the config validator and the router cannot drift apart on the names.
WORK_KINDS: Tuple[str, ...] = (
    "design", "code", "explore", "review", "sensitive", "critical", "long", "chat", "default",
)


def _preference_list(kind: str, cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """The configured order of preferred tiers for one work kind, or () when unset.

    Unset is meaningful: it means "keep the built-in route", which is why a missing
    or malformed entry never silently becomes an empty preference.
    """
    if not kind:
        return ()
    raw = (cfg.get("preferences") or {}).get(kind)
    if not isinstance(raw, list):
        return ()
    seen: list[str] = []
    for item in raw:
        name = str(item or "").strip().casefold()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def _is_routable_tier(tier: str, cfg: Dict[str, Any]) -> bool:
    """Whether the router itself can serve this tier by rewriting the model name.

    ``route_llm_request`` runs after the provider is chosen, so it can only swap
    models inside its own provider. Anything else (Claude, Qwen) reaches work
    through delegation, never through a route.
    """
    return tier in (cfg.get("models") or {})


def _preferred_route(kind: str, cfg: Dict[str, Any]) -> Optional[str]:
    """First routable+callable tier of the kind's preference list, else None."""
    for tier in _preference_list(kind, cfg):
        if _is_routable_tier(tier, cfg) and _is_callable_tier(tier, cfg):
            return tier
    return None


def _preferred_target(kind: str, cfg: Dict[str, Any]) -> str:
    """First callable EXTERNAL entry of the kind's preference list, else "".

    Ranked above the routable tiers on purpose: if the user put ``opus5`` first for
    design work, the router cannot honour that as a route, but it can tell the
    conductor that design leaves belong on Opus.
    """
    for tier in _preference_list(kind, cfg):
        if _is_routable_tier(tier, cfg):
            continue
        if _is_callable_tier(tier, cfg):
            return tier
    return ""


def _apply_preferences(decision: RouteDecision, cfg: Dict[str, Any]) -> RouteDecision:
    """Overlay the user's per-kind preference onto a built-in decision.

    A configured list is authoritative: it replaces the tier AND the fallback
    chain, and it clears ``mandatory`` because the built-in policy it would have
    enforced is exactly what the user chose to override. With no list configured
    the decision is returned untouched, so the shipped defaults still apply.
    """
    prefs = _preference_list(decision.kind, cfg)
    if not prefs:
        return decision
    target = _preferred_target(decision.kind, cfg)
    tier = _preferred_route(decision.kind, cfg)
    if tier is None or tier == decision.tier:
        # Nothing routable in the list (or it already agrees): keep the route and
        # carry only the delegation advice.
        return replace(decision, prefer_target=target) if target else decision
    try:
        preferred = _decision(tier, f"preferred {decision.kind} route", cfg, kind=decision.kind)
    except (KeyError, ValueError):
        return decision
    return replace(preferred, vetoed_by=decision.vetoed_by, prefer_target=target)


def _resolve_callable_fallback(
    decision: RouteDecision, cfg: Dict[str, Any]
) -> RouteDecision:
    """If the chosen tier is not callable, follow the fallback chain once."""
    # Fail closed: a tier is routable only when the live config explicitly says
    # callable: true and it is not cooling down. Going through
    # ``_is_callable_tier`` rather than reading the flag directly is what makes
    # a cooling tier follow the same path as a disabled one.
    chosen_tier = decision.tier
    if _is_callable_tier(chosen_tier, cfg):
        return decision

    # A configured preference list IS the fallback chain for its kind: the user
    # wrote the order, so walk it before anything built-in and never leave it.
    prefs = _preference_list(decision.kind, cfg)
    if prefs:
        for tier in prefs:
            if tier != chosen_tier and _is_routable_tier(tier, cfg) and _is_callable_tier(tier, cfg):
                try:
                    return _decision(
                        tier, f"preferred {decision.kind} fallback from {chosen_tier}",
                        cfg, kind=decision.kind,
                    )
                except (KeyError, ValueError):
                    continue
        return decision

    # A policy route is not a preference. Design work reaches Sol because only
    # Sol may do it, so answering "Sol is unavailable" with Terra performs the
    # work on the tier the rule exists to keep it away from -- and it does so
    # exactly when Sol has run out of quota, which is when the rule matters
    # most. Decline the chain and let the caller fail loudly instead.
    if decision.mandatory:
        return decision

    # Tier is disabled — follow fallback chain (max 3 hops to prevent cycles)
    fallbacks = cfg.get("fallbacks") or {}
    visited = {chosen_tier}
    current = chosen_tier
    for _ in range(3):
        next_tier = fallbacks.get(current)
        if next_tier and next_tier not in visited:
            if _is_callable_tier(next_tier, cfg):
                try:
                    return _decision(next_tier, f"fallback from disabled {chosen_tier}", cfg)
                except (KeyError, ValueError):
                    break
            visited.add(next_tier)
            current = next_tier

    # No valid fallback found — return original decision unchanged
    return decision


# The route log reaches tens of megabytes; the recent window lives in its tail.
_USAGE_TAIL_BYTES = 1_000_000

_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_CACHE: Dict[str, Any] = {"key": None, "state": {}}


def _cooldown_path(cfg: Dict[str, Any]) -> Optional[Path]:
    """The shared state file, or None when this config did not name one.

    Deliberately not defaulted to the production path. A component that writes
    to a shared location must take that location from the config it was handed;
    inventing one means any caller with a partial config -- a test, a probe --
    silently writes to the real file and its state leaks into unrelated runs.
    """
    configured = str((cfg.get("cooldown") or {}).get("path") or "").strip()
    return Path(os.path.expanduser(configured)) if configured else None


def _read_cooldown_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load the shared cooldown file, cached on its own mtime and size.

    Read on every routing decision, so it must not cost a parse per call; it
    must also not go stale, because the process that recorded the cooldown is
    usually not the process that needs to honour it.
    """
    path = _cooldown_path(cfg)
    if path is None:
        return {}
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    if _COOLDOWN_CACHE.get("key") == key:
        return _COOLDOWN_CACHE["state"]
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        state = state if isinstance(state, dict) else {}
    except Exception:
        state = {}
    _COOLDOWN_CACHE["key"] = key
    _COOLDOWN_CACHE["state"] = state
    return state


def _write_cooldown_state(cfg: Dict[str, Any], state: Dict[str, Any]) -> None:
    path = _cooldown_path(cfg)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        # A cooldown that cannot be persisted must never break routing.
        pass


def _tier_cooldown_remaining(tier: str, cfg: Dict[str, Any]) -> float:
    """Seconds left on this tier's cooldown, or 0.0 when it is available."""
    if not (cfg.get("cooldown") or {}).get("enabled", True):
        return 0.0
    entry = (_read_cooldown_state(cfg).get("tiers") or {}).get(tier)
    if not isinstance(entry, dict):
        return 0.0
    remaining = float(entry.get("until", 0) or 0) - datetime.now(timezone.utc).timestamp()
    return remaining if remaining > 0 else 0.0


def _enter_cooldown(tier: str, cfg: Dict[str, Any], *, seconds: float, reason: str) -> None:
    if not tier or not (cfg.get("cooldown") or {}).get("enabled", True):
        return
    now = datetime.now(timezone.utc).timestamp()
    with _COOLDOWN_LOCK:
        state = dict(_read_cooldown_state(cfg))
        tiers = dict(state.get("tiers") or {})
        current = tiers.get(tier) or {}
        # Never shorten a cooldown already in force: a transient blip arriving
        # during a quota cooldown must not release the tier early.
        until = max(float(current.get("until", 0) or 0), now + float(seconds))
        tiers[tier] = {"until": until, "reason": reason, "recorded_at": now}
        state["tiers"] = tiers
        _write_cooldown_state(cfg, state)


def _reset_hint_seconds(error: BaseException) -> Optional[float]:
    """Seconds until the provider says the quota returns, from the error body.

    Codex answers a usage-limit 429 with ``resets_in_seconds`` and ``resets_at``.
    Benching for a fixed 15 minutes against a three-hour reset is what turns one
    refusal into a loop: the cooldown lapses, the tier is offered again, and the
    next leaf spends its retries rediscovering the same wall.
    """
    text = str(error)
    match = re.search(r"'?\"?resets_in_seconds\"?'?\s*:\s*([0-9]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"'?\"?resets_at\"?'?\s*:\s*([0-9]{9,13})", text)
    if match:
        value = float(match.group(1))
        if value > 1e11:  # milliseconds
            value /= 1000.0
        remaining = value - datetime.now(timezone.utc).timestamp()
        return remaining if remaining > 0 else None
    return None


def _account_siblings(tier: str, cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Other tiers billed to the same account as ``tier``.

    A session/usage quota belongs to the account, not the model, so benching only
    the tier that happened to ask leaves its siblings looking available — and the
    next leaf burns another call learning what this one already established.
    """
    providers = cfg.get("tier_providers") or {}
    account = providers.get(tier)
    if not account:
        return ()
    # Not restricted to routable models: a delegation-only target such as sonnet5
    # shares Opus's account, and benching it is what stops the conductor being
    # advised to send the next leaf into the same exhausted subscription.
    known = set(cfg.get("models") or {}) | set(cfg.get("callable") or {})
    return tuple(
        name for name, owner in providers.items()
        if owner == account and name != tier and name in known
    )


def _record_tier_failure(
    tier: str, cfg: Dict[str, Any], *, quota: bool, error: Optional[BaseException] = None
) -> None:
    """Cool a tier down: at once for quota, or after repeated recent failures."""
    policy = cfg.get("cooldown") or {}
    if not tier or not policy.get("enabled", True):
        return
    if quota:
        # Prefer what the provider actually said over the configured guess, capped so
        # a malformed or absurd hint cannot bench a tier for a day.
        hint = _reset_hint_seconds(error) if error is not None else None
        configured = float(policy.get("quota_seconds", 900) or 900)
        cap = float(policy.get("quota_max_seconds", 21600) or 21600)
        seconds = min(hint, cap) if hint else configured
        reason = "quota exhausted (provider reset)" if hint else "quota exhausted"
        _enter_cooldown(tier, cfg, seconds=seconds, reason=reason)
        # A usage quota is the account's, not the model's.
        for sibling in _account_siblings(tier, cfg):
            if _tier_cooldown_remaining(sibling, cfg) < seconds:
                _enter_cooldown(
                    sibling, cfg, seconds=seconds,
                    reason=f"{reason}; shares an account with {tier}",
                )
        return
    now = datetime.now(timezone.utc).timestamp()
    window = float(policy.get("failure_window_seconds", 60) or 60)
    allowed = max(1, int(policy.get("allowed_fails", 3) or 3))
    with _COOLDOWN_LOCK:
        state = dict(_read_cooldown_state(cfg))
        failures = dict(state.get("failures") or {})
        recent = [float(ts) for ts in (failures.get(tier) or []) if now - float(ts) < window]
        recent.append(now)
        failures[tier] = recent[-allowed:]
        state["failures"] = failures
        _write_cooldown_state(cfg, state)
    if len(recent) >= allowed:
        _enter_cooldown(
            tier, cfg,
            seconds=float(policy.get("failure_seconds", 60) or 60),
            reason=f"{len(recent)} failures within {int(window)}s",
        )


def _is_callable_tier(tier: str, cfg: Dict[str, Any]) -> bool:
    # Live router policy is explicit: missing or malformed entries are disabled.
    if (cfg.get("callable") or {}).get(tier) is not True:
        return False
    # A tier serving 429s is not available, whatever the dashboard says. Routing
    # this through callability means the existing fallback chain and the
    # mandatory-route rule both apply with no further wiring.
    return _tier_cooldown_remaining(tier, cfg) <= 0


def _require_callable(decision: RouteDecision, cfg: Dict[str, Any]) -> RouteDecision:
    """Never emit a route for a tier disabled in the dashboard."""
    resolved = _resolve_callable_fallback(decision, cfg)
    if not _is_callable_tier(resolved.tier, cfg):
        if decision.mandatory:
            cooling = _tier_cooldown_remaining(decision.tier, cfg)
            unavailable = (
                f"cooling down for another {int(cooling)}s" if cooling else "disabled"
            )
            raise RuntimeError(
                f"'{decision.tier}' is required for this request ({decision.reason}) but is "
                f"{unavailable}; no fallback may take its place."
            )
        raise RuntimeError(
            f"No enabled ModelRouter tier is available for requested '{decision.tier}'"
        )
    return resolved


def _load_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists() or yaml is None:
        return _deep_merge({}, _DEFAULT_CONFIG)
    try:
        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            return _deep_merge({}, _DEFAULT_CONFIG)
        return _deep_merge(_DEFAULT_CONFIG, loaded)
    except Exception:
        return _deep_merge({}, _DEFAULT_CONFIG)


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents.casefold()).strip()


_DESIGN_WORK = re.compile(
    r"\b(visual\s+design|product\s+design|ui\s*(?:/|and)?\s*ux|ux\s*(?:/|and)?\s*ui|"
    r"ui|ux|css|stylesheet|styling|layout|elrendezes|tipograf|typography|"
    r"wireframe|mockup|figma|design\s+system|brand(?:ing)?|responsive\s+(?:ui|layout|card|component)|"
    r"frontend\s+design|visualis\s+terv(?:ezes)?|felulet(?:et|i)?\s+terv(?:ezes)?|"
    r"look\s+and\s+feel|visual\s+appearance|vizualis\s+megjelenes|arculat|"
    r"(?:color|colour)\s+palette|szinpaletta|font|betutipus)\b"
)
_ACKNOWLEDGEMENT_ONLY = re.compile(
    r"^(?:(?:a|az|this|that)\s+)?(?:(?:ui|ux|design|layout|css|frontend)\s+){0,3}"
    r"(?:(?:nagyon\s+)?(?:jo|szuper|remek|kivalo|nagyszeru|tokeletes)\s+lett|"
    r"(?:koszonom|koszi|thanks|thank\s+you|rendben\s+van|oke|ok|"
    r"jovahagyom|elfogadom|approved|accepted))"
    r"(?:\s*[,!.]\s*(?:(?:nagyon\s+)?(?:jo|szuper|remek|kivalo|nagyszeru|tokeletes)\s+lett|"
    r"koszonom|koszi|thanks|thank\s+you|rendben\s+van|oke|ok|"
    r"jovahagyom|elfogadom|approved|accepted))*[!. ]*$"
)
_SPARK_MUTATING_WORK = re.compile(
    r"\b(add|create|implement|modify|change|edit|write|patch|delete|remove|"
    r"deploy|publish|send|restart|configure|install|fix|refactor|javitsd|"
    r"modositsd|hozd\s+letre|torold|telepitsd|allitsd\s+be)\b"
)
_SPARK_READ_ONLY_WORK = re.compile(
    r"\b(inspect|read|review|audit|report|analy[sz]e|compare|search|find|"
    r"identify|list|check|investigate|research|explore|trace|map|survey|"
    r"enumerate|discover|determine|locate|gather|document|test[- ]case\s+design|"
    r"nezd\s+meg|nezd\s+at|olvasd|ellenorizd|elemezd|jelentsd|keresd|azonositsd|"
    r"hasonlitsd\s+ossze|kutass|tard\s+fel|deritsd\s+ki|vizsgald|tekintsd\s+at|"
    r"gyujtsd\s+ossze|terkepezd\s+fel|merd\s+fel|listazd|allapitsd\s+meg)\b"
)
_SPARK_CONSEQUENTIAL_WORK = re.compile(
    r"\b(production|prod|security|biztonsag|auth(?:entication|orization)?|"
    r"credential|jelszo|password|payment|fizetes|migration|migrate|deploy|"
    r"szerver|server|database|adatbazis)\b"
)


_PLAN_LABEL = re.compile(r"^\s*\[(luna|spark|terra|sol)(?::xhigh)?\](?:\s|$)")
_CLAUDE_REVIEW_LABEL = re.compile(r"^\s*\[(opus|sonnet)5?-review\](?:\s|$)")


def _is_plan_labelled_worker(text: str) -> bool:
    """True when this text is a delegated worker goal labelled by this router.

    The keyword test predates the planner. It exists because the router once had
    to guess a tier from raw prompt text; a labelled worker goal is instead the
    output of a planner that saw the screenshot, the objective and the repo, and
    that routes design to a [sol] leaf under a schema-enforced contract. Re-deciding
    that with forty keywords overrides a better-informed decision with a worse one
    -- and it cannot even tell the cases apart: `_is_design_request` is true both
    for "identify the layout branches" and for "implement a responsive CSS card".
    """
    return bool(_PLAN_LABEL.match(text))


def _is_design_request(text: str) -> bool:
    return bool(_DESIGN_WORK.search(_normalise(text)))


_EXPLICIT_OPUS_REQUEST = re.compile(
    r"(?:\[(?:opus|opus5)\]|\b(?:let|have)\s+opus\s+(?:work|handle|do)|"
    r"\bopus\s+(?:work|handle|do)|\bopus\s+dolgozzon|\bopussal\b)"
)
_DIRECT_OPUS_UI_RISK = re.compile(
    r"\b(production|prod(?:ra|on|ban|ba|ot)?|deploy|security|biztonsag|auth|oauth|"
    r"credential|jelszo|password|payment|fizetes|billing|database|adatbazis|migration|migracio)\b"
)


def _is_explicit_bounded_opus_ui_request(text: str, cfg: Dict[str, Any]) -> bool:
    """Recognise only a user's explicit, small, low-risk Opus UI request.

    Classification ownership remains Sol; this predicate merely authorises one
    external Opus execution bridge instead of a planner/preflight fan-out.
    """
    policy = ((cfg.get("coding_agent") or {}).get("explicit_ui") or {})
    normalised = _normalise(text)
    return bool(
        policy.get("enabled")
        and _is_design_request(text)
        and _EXPLICIT_OPUS_REQUEST.search(normalised)
        and len(text or "") <= int(policy.get("max_chars", 1200) or 1200)
        and not _DIRECT_OPUS_UI_RISK.search(_normalise(_without_negated_safety_constraints(text)))
    )


def _is_acknowledgement_only(text: str) -> bool:
    """Recognise a closed praise/approval follow-up with no requested action."""
    return bool(_ACKNOWLEDGEMENT_ONLY.fullmatch(_normalise(text)))


def _is_spark_read_only_work(text: str) -> bool:
    """A plan-labelled leaf is read-only unless it says otherwise.

    Separated from the design test because mixing them made the question
    unanswerable: "identify the layout branches" and "implement a CSS card"
    both mention design, so a combined predicate rejected both. The verbs
    separate them cleanly -- one reads, the other writes.

    This side looks for *contradiction*, not corroboration. The conductor has
    already declared the leaf read-only by labelling it, so demanding a second
    positive signal means the router overrules that claim whenever the phrasing
    falls outside a hand-written verb list -- which a Hungarian goal did on its
    first outing ("Tárd fel..." reads nothing but says so with a verb the list
    never had). Write verbs are the small, stable set worth enumerating;
    read-only phrasings are open-ended.
    """
    affirmative = _normalise(_without_negated_safety_constraints(text))
    return bool(affirmative and not _SPARK_MUTATING_WORK.search(affirmative))


def _is_spark_read_only_request(text: str) -> bool:
    """Spark may receive only affirmative, bounded non-design evidence work.

    The stricter form, for a claim no conductor vouched for: a bare ``[spark]``
    on a root turn is a label someone typed, so here a positive read-only signal
    is still required.
    """
    affirmative = _normalise(_without_negated_safety_constraints(text))
    return bool(
        _is_spark_read_only_work(text)
        and _SPARK_READ_ONLY_WORK.search(affirmative)
        and not _is_design_request(affirmative)
    )


def _is_consequential_spark_request(text: str) -> bool:
    return bool(_SPARK_CONSEQUENTIAL_WORK.search(_normalise(_without_negated_safety_constraints(text))))


def _without_negated_safety_constraints(text: str) -> str:
    """Remove standalone prohibition clauses before judging a child as risky.

    Delegation goals routinely say things like ``Do not restart services``.
    Those exclusions must not combine with an earlier harmless ``config`` noun
    and become a false consequential-system action. Affirmative clauses are
    retained, so a real production/deploy request still escalates to Sol.
    """
    clauses = re.split(r"(?<=[.!?;])\s+|[\r\n]+", text or "")
    prohibition = re.compile(
        r"\b(do not|don't|must not|never|without|prohibit(?:ed)?|"
        r"ne\s+|tilos|nem szabad)\b",
        re.IGNORECASE,
    )
    return " ".join(clause for clause in clauses if clause.strip() and not prohibition.search(clause)).strip()


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                for key in ("text", "input_text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
                        break
        return "\n".join(chunks)
    if isinstance(content, dict):
        for key in ("text", "input_text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def _request_items(request: Dict[str, Any]) -> list:
    for key in ("messages", "input"):
        value = request.get(key)
        if isinstance(value, list):
            return value
    return []


_IMAGE_TYPES = {"image", "image_url", "input_image", "image_file", "input_image_file"}
_IMAGE_TEXT_MARKERS = ("[image attached", "[screenshot]", "data:image/")


def _contains_image_attachment(value: Any) -> bool:
    """Recognise textual and structured image parts in one prompt item."""
    if isinstance(value, str):
        return any(marker in value.casefold() for marker in _IMAGE_TEXT_MARKERS)
    if isinstance(value, list):
        return any(_contains_image_attachment(item) for item in value)
    if not isinstance(value, dict):
        return False
    if str(value.get("type") or "").casefold() in _IMAGE_TYPES:
        return True
    if any(value.get(key) for key in ("image_url", "input_image", "image_file")):
        return True
    return any(_contains_image_attachment(item) for item in value.values())


def _request_has_image_attachment(request: Dict[str, Any]) -> bool:
    """Return true only when the latest user turn itself contains an image.

    A historical image is context, not a permanent vision requirement. The
    router must classify the current objective, otherwise one screenshot would
    pin every later text-only prompt and delegated child to Terra forever.
    Before Luna/Spark dispatch, historical image parts are removed from the
    outgoing request by ``_strip_historical_image_attachments`` so text-only
    models never receive unsupported media.
    """
    items = _request_items(request)
    _, user_index = _last_user_text_and_index(items)
    if user_index < 0:
        return False
    item = items[user_index]
    return _contains_image_attachment(item.get("content", "")) if isinstance(item, dict) else False


def _strip_historical_image_attachments(request: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a request without visual media from turns preceding the current user.

    This preserves the current turn exactly. It only runs for Luna/Spark after
    routing has established that the current objective is text-only, preventing
    stale session screenshots from leaking to text-only providers.
    """
    cleaned = deepcopy(request)
    items = _request_items(cleaned)
    _, user_index = _last_user_text_and_index(items)
    if user_index <= 0:
        return cleaned

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for marker in _IMAGE_TEXT_MARKERS:
                result = re.sub(re.escape(marker) + r"[^\]\n]*\]", "[Earlier image omitted]", result, flags=re.IGNORECASE)
            return result
        if isinstance(value, list):
            return [next_value for item in value if (next_value := clean(item)) is not None]
        if not isinstance(value, dict):
            return value
        if str(value.get("type") or "").casefold() in _IMAGE_TYPES:
            return None
        if any(value.get(key) for key in ("image_url", "input_image", "image_file")):
            return None
        return {key: next_value for key, item in value.items() if (next_value := clean(item)) is not None}

    for item in items[:user_index]:
        if isinstance(item, dict) and "content" in item:
            item["content"] = clean(item["content"])
    return cleaned


def _last_user_text_and_index(items: Iterable[Any]) -> tuple[str, int]:
    sequence = list(items)
    for index in range(len(sequence) - 1, -1, -1):
        item = sequence[index]
        if isinstance(item, dict) and item.get("role") == "user":
            return _text_from_content(item.get("content", "")), index
    return "", -1


def _prompt_preview(request: Any) -> str:
    """Return the complete latest user prompt on a single line."""
    if not isinstance(request, dict):
        return ""
    text, _ = _last_user_text_and_index(_request_items(request))
    return re.sub(r"\s+", " ", text).strip()


def _redacted_preview(value: Any, limit: int = 280) -> str:
    """Bound a local observability description without retaining secrets."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(
        r"(?i)\b(password|passwd|api[ _-]?key|secret|token)\s*([=:])\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bauthorization\s*:\s*bearer\s+[^\s,;]+", "Authorization: Bearer ***", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-[REDACTED]", text)
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _logged_prompt_preview(request: Any, log_cfg: Dict[str, Any]) -> str:
    """Return the local audit preview without retaining full user content."""
    preview = _prompt_preview(request)
    if bool(log_cfg.get("redact_prompt_preview", True)):
        preview = _redacted_preview(preview, limit=max(1, int(log_cfg.get("prompt_preview_chars", 240) or 240)))
    limit = max(1, int(log_cfg.get("prompt_preview_chars", 240) or 240))
    return preview[:limit]


def _lifecycle_event_kind(request: Any) -> Optional[str]:
    """Classify synthetic internal turns without exposing their raw envelope."""
    text = _prompt_preview(request).casefold()
    if text.startswith("[async delegation batch complete") or text.startswith("[async delegation complete"):
        return "async_delegation_completion"
    if text.startswith("[context compaction"):
        return "context_compaction"
    return None


def _completion_delegation_id(request: Any) -> str:
    match = re.match(
        r"^\[ASYNC DELEGATION(?: BATCH)? COMPLETE\s+[—-]\s*([^\]\s]+)",
        _prompt_preview(request),
        re.I,
    )
    return match.group(1) if match else ""


def _is_tool_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("role") in {"tool", "function"}:
        return True
    if item.get("type") in {"function_call", "function_call_output", "tool_result"}:
        return True
    if item.get("tool_calls"):
        return True
    content = item.get("content")
    if isinstance(content, list):
        return any(_is_tool_item(part) for part in content)
    return False


def _current_turn_has_tool_activity(items: list, last_user_index: int) -> bool:
    if last_user_index < 0:
        return False
    return any(_is_tool_item(item) for item in items[last_user_index + 1 :])


# Default effort per tier. ``.get`` rather than ``[]``: a preference list may name a
# tier this map never anticipated, and an unknown tier must not raise inside routing.
_DEFAULT_EFFORT = {"luna": "low", "spark": "low", "terra": "medium", "sol": "high", "qwen": "medium"}


def _decision(
    tier: str,
    reason: str,
    cfg: Dict[str, Any],
    *,
    explicit: bool = False,
    effort_key: Optional[str] = None,
    mandatory: bool = False,
    kind: str = "",
) -> RouteDecision:
    if effort_key is None:
        effort_key = "explicit_sol" if explicit and tier == "sol" else tier
    fallback = _DEFAULT_EFFORT.get(tier, "medium")
    effort = str((cfg.get("effort") or {}).get(effort_key) or fallback).casefold()
    return RouteDecision(
        tier=tier, model=cfg["models"][tier], reason=reason, effort=effort,
        mandatory=mandatory, kind=kind,
    )


def _eligible_tiers(
    user_text: str,
    *,
    has_image_attachment: bool,
    api_call_count: int,
    is_tool_loop: bool,
) -> Tuple[str, ...]:
    """Tiers this request independently qualifies for, ignoring gate precedence.

    Deliberately position-independent: it answers "was this tier ever a candidate"
    rather than "which gate won". Comparing it against the chosen tier is what
    turns a silent never-fires route into a visible preempted one.
    """
    text = _normalise(user_text)
    # Terra is deliberately absent: it is the unconditional default, so listing it
    # would mark every non-Terra decision as preempting it and carry no signal.
    eligible = []

    if _is_design_request(user_text):
        eligible.append("sol")

    # Spark is text-only, first-call-only, and restricted to non-design read-only
    # work. Anything else can only reach it through an explicit label.
    if (
        not has_image_attachment
        and api_call_count == 1
        and not is_tool_loop
        and _is_spark_read_only_request(user_text)
        and not _is_consequential_spark_request(user_text)
    ):
        eligible.append("spark")

    if _is_acknowledgement_only(user_text) or re.match(
        r"^(szia|hello|hi|hey|jo reggelt|jo estet|koszonom|koszi|thanks|thank you)[!. ]*$", text
    ):
        eligible.append("luna")

    return tuple(dict.fromkeys(eligible))


def classify_request(
    request: Dict[str, Any],
    api_call_count: int = 1,
    config: Optional[Dict[str, Any]] = None,
    *,
    allow_plan_label_over_design: bool = False,
) -> RouteDecision:
    """Classify one provider request, annotated with the tiers it lost out on."""
    decision = _classify_request(
        request,
        api_call_count,
        config,
        allow_plan_label_over_design=allow_plan_label_over_design,
    )
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    eligible = _eligible_tiers(
        user_text,
        has_image_attachment=_request_has_image_attachment(request),
        api_call_count=api_call_count,
        is_tool_loop=_current_turn_has_tool_activity(items, user_index),
    )
    vetoed = tuple(tier for tier in eligible if tier != decision.tier)
    if vetoed:
        decision = replace(decision, vetoed_by=vetoed)
    # Last, so a preference is applied to whatever the gates decided rather than
    # competing with them, and so the veto list still records the built-in view.
    return _apply_preferences(decision, config or _load_config())


def _classify_request(
    request: Dict[str, Any],
    api_call_count: int = 1,
    config: Optional[Dict[str, Any]] = None,
    *,
    allow_plan_label_over_design: bool = False,
) -> RouteDecision:
    """Classify one provider request. Terra is the normal durable default."""
    cfg = config or _load_config()
    has_image_attachment = _request_has_image_attachment(request)
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    text = _normalise(user_text)

    # A closed praise/approval follow-up has no implementation objective.  Check
    # it before the design boundary so a mention of UI/CSS does not by itself
    # create an unnecessary Sol delegation.
    if _is_acknowledgement_only(user_text):
        return _decision("luna", "acknowledgement-only follow-up", cfg, kind="chat")

    # Role separation is a hard policy boundary: only Sol performs visual/product
    # design analysis or design implementation. Terra may coordinate and approve
    # the resulting evidence, while Spark may inspect only non-design read-only
    # facts. This check intentionally precedes manual/benchmark overrides so a
    # label cannot route design work to another tier.
    # The gate precedes the manual override so a *user* label cannot route design
    # work off Sol. For a delegated conductor the keyword test misfires: it cannot
    # tell "coordinate work that includes design" from "do design", so any
    # UI-adjacent objective pinned the planner to Sol, which then owned both the
    # conducting and the [sol] leaf it was meant to delegate -- 15 of 16 routing
    # decisions on one turn. Only the conductor label is exempt; a [spark] or
    # [sol] leaf still faces the gate, and Sol still owns every [sol] leaf.
    if _is_design_request(user_text) and not (
        allow_plan_label_over_design and _is_plan_labelled_worker(text)
    ):
        return _decision("sol", "design analysis or implementation is Sol-only", cfg, mandatory=True, kind="design")

    benchmark_force = _normalise(os.getenv("MODEL_ROUTER_BENCHMARK_FORCE_MODEL", ""))
    if benchmark_force in ("luna", "spark", "terra", "sol"):
        if benchmark_force == "spark":
            if has_image_attachment:
                return _decision("terra", "image attachment requires a vision-capable route", cfg, mandatory=True)
            if not _is_spark_read_only_request(user_text):
                return _decision("terra", "Spark is restricted to non-design read-only subtasks", cfg)
        return _decision(benchmark_force, "benchmark environment force override", cfg, explicit=True)

    override = re.match(r"^\s*\[(luna|spark|terra|sol)(?::(xhigh))?\](?:\s|$)", text)
    if override:
        tier = override.group(1)
        if tier == "spark" and has_image_attachment:
            return _decision("terra", "image attachment requires a vision-capable route", cfg, mandatory=True)
        # A delegated leaf carries a conductor's declaration, so the router
        # looks only for contradiction. A root [spark] is a label someone typed
        # with nothing behind it, and still has to show its read-only intent.
        spark_read_only = (
            _is_spark_read_only_work(user_text)
            if allow_plan_label_over_design
            else _is_spark_read_only_request(user_text)
        )
        if tier == "spark" and not spark_read_only:
            if _is_consequential_spark_request(user_text):
                return _decision("sol", "consequential Spark task requires Sol", cfg, mandatory=True)
            if _is_design_request(user_text):
                return _decision("sol", "design analysis or implementation is Sol-only", cfg, mandatory=True, kind="design")
            return _decision("terra", "Spark is restricted to non-design read-only subtasks", cfg)
        requested_effort = override.group(2)
        effort_key = "explicit_sol_xhigh" if tier == "sol" and requested_effort == "xhigh" else None
        label = f"{tier}:{requested_effort}" if requested_effort else tier
        return _decision(tier, f"explicit [{label}] override", cfg, explicit=True, effort_key=effort_key)

    # The durable completion of a [terra] orchestrator includes the complete
    # reviewed evidence and can be much longer than the normal Sol threshold.
    # Keep the final hand-back with Terra so the supervisor's acceptance gate,
    # integration ownership, and final answer are not silently reassigned.
    if "[async delegation batch complete" in text and "role: orchestrator" in text and "[terra]" in text:
        return _decision("terra", "completed Terra supervisor review", cfg)

    sol_min_chars = int(cfg.get("thresholds", {}).get("sol_min_chars", 3500))
    if len(user_text) >= sol_min_chars:
        return _decision("sol", f"long request ({len(user_text)} characters)", cfg, effort_key="sol_long", kind="long")

    # Consequential domains are biased toward Sol even when the prompt is short.
    sensitive = re.compile(
        r"\b(security|biztonsag|vulnerability|sebezhetoseg|malware|"
        r"auth|oauth|authentication|authorization|jogosultsag|"
        r"credential|credentials|belepesi\s+adat\w*|jelsz\w*|password|passwd|"
        r"payment|fizetes|billing|szamlazas|webhook|jogi|legal|"
        r"orvosi|medical|diagnos|gyogyszer|befektetes|investment|adozas|tax)\b"
    )
    if sensitive.search(text):
        return _decision("sol", "sensitive or consequential domain", cfg, mandatory=True, kind="sensitive")

    # High-consequence engineering stays on Sol. Ordinary repository debugging,
    # refactoring and test execution stay on Terra: the implementation benchmark
    # showed that Terra is the safer integration owner, while bounded Spark work
    # remains available through explicit/delegated workers.
    critical_work = re.compile(
        r"\b(migrate|migr(?:al|ald)|deploy(?:ol|old)?|telepitsd|install|production|prod(?:ra|on|ban|ba|ot)?|"
        r"deep research|mely kutatas|kutass reszletesen)\b"
    )
    if critical_work.search(text):
        return _decision("sol", "consequential engineering or research task", cfg, mandatory=True, kind="critical")

    action = re.compile(
        r"\b(modositsd|konfigurald|configure|restart|ujraindit|torol(?:d|j)|delete|remove|"
        r"upload|toltsd fel|publish|send|kuldd|execute|futtasd|javitsd|fix|connect|"
        r"lepj be|allitsd be|create|hozd letre)\b"
    )
    consequential_system = re.compile(
        r"\b(ssh|sudo|server|szerver|firewall|dns|database|adatbazis|"
        r"hosting|gateway|docker|kubernetes|systemd|config|konfiguracio)\b"
    )
    if action.search(text) and consequential_system.search(text):
        return _decision("sol", "consequential system action", cfg, mandatory=True, kind="critical")

    # Spark is text-only. This hard route sits after the higher-priority Sol
    # safety routes and before every Spark classifier, so an image cannot be
    # routed to Spark by a primary, override, benchmark, or tool-loop branch.
    if has_image_attachment:
        return _decision("terra", "image attachment requires a vision-capable route", cfg, mandatory=True)

    # Unlabelled work is never sent directly to Spark. Terra plans and owns the
    # task first; only its explicit [spark] leaf goals may use Spark.  Repeated
    # calls merely preserve Terra ownership rather than reclassifying from
    # prompt keywords.
    if api_call_count > 1:
        if _current_turn_has_tool_activity(items, user_index):
            return _decision("terra", "ordinary current-turn tool loop", cfg)
        return _decision("terra", "repeated current-turn call safety promotion", cfg)

    # Review and exploration are recognised so a preference list can reach them.
    # Both deliberately keep Terra as their built-in route: naming the category
    # must not change any behaviour on its own, only make it addressable.
    review_request = re.compile(
        r"\b(review|reviewold|nezd at|nezd meg a kodot|code review|atnezes|"
        r"velemenyezd|critique|audit|ellenorizd a kodot)\b"
    )
    if review_request.search(text):
        return _decision("terra", "code review or critique", cfg, kind="review")

    if _is_spark_read_only_request(user_text) and not _is_consequential_spark_request(user_text):
        return _decision("terra", "read-only inspection", cfg, kind="explore")

    repo_implementation = re.compile(
        r"\b(debug|debugold|hibakeres|traceback|stack trace|root cause|"
        r"refactor|teszteld|run the tests|futtasd a teszt|javitsd|fix|"
        r"implement|repo|kod|code)\b"
    )
    if repo_implementation.search(text):
        return _decision("terra", "normal repository implementation owner", cfg, kind="code")

    if re.search(r"\b(csinald meg|hajtsd vegre|do it|make the changes|folytasd)\b", text):
        return _decision("terra", "context-dependent action owned by Terra", cfg, kind="code")



    luna_max_chars = int(cfg.get("thresholds", {}).get("luna_max_chars", 700))
    if len(user_text) <= luna_max_chars:
        greeting = re.compile(
            r"^(szia|hello|hi|hey|jo reggelt|jo estet|koszonom|koszi|thanks|thank you)[!. ]*$"
        )
        simple_transform = re.compile(
            r"^(forditsd|fordits|translate|ird at|fogalmazd at|rewrite|javitsd a helyesirast|"
            r"helyesiras|roviditsd|shorten)\b"
        )
        simple_definition = re.compile(
            r"^(mi az|mit jelent|what is|what does|ki az|who is)\b[^?\n]{0,160}\??$"
        )
        brief_chat = re.compile(
            r"^(ez|az|hat ez|oke|ok|rendben|ertem|furcsa|szomoru|kar|igazad van)\b[^\n]{0,180}$"
        )
        short_explanation = re.compile(
            r"^(miert|hogyhogy|why)\b[^\n]{0,240}\??$"
        )
        technical = re.compile(
            r"\b(implement|kod|code|api|ssh|server|szerver|database|adatbazis|deploy|"
            r"production|config|konfiguracio|debug|teszt|test|security|biztonsag)\b"
        )
        if greeting.search(text):
            return _decision("luna", "greeting or acknowledgement", cfg, kind="chat")
        if simple_transform.search(text) and "```" not in user_text:
            return _decision("luna", "simple language transformation", cfg, kind="chat")
        if simple_definition.search(text):
            return _decision("luna", "short definition request", cfg, kind="chat")
        if brief_chat.search(text):
            return _decision("luna", "brief non-actionable conversation", cfg, kind="chat")
        if short_explanation.search(text) and not technical.search(text):
            return _decision("luna", "short low-risk explanation", cfg, kind="chat")

    return _decision(
        str(cfg.get("default_model", "terra")),
        "default general-purpose route",
        cfg,
        kind="default",
    )


def _log_decision(decision: RouteDecision, kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("enabled", True):
        return
    # No invented default: a config that does not name the audit log does not
    # get written to it. Defaulting here meant every caller holding a partial
    # config -- the test suite above all -- appended to the real log, and those
    # entries then show up as real traffic to anything that reads it back.
    configured = str(log_cfg.get("path") or "").strip()
    if not configured:
        return
    path = Path(os.path.expanduser(configured))
    request = kwargs.get("request")
    event_kind = _lifecycle_event_kind(request)
    delegation_id = _completion_delegation_id(request) if event_kind == "async_delegation_completion" else ""
    entry = {
        # Routing decisions do not need sub-second precision. Keeping this at
        # whole seconds makes the JSONL easier to scan and group.
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pid": os.getpid(),
        "turn_id": kwargs.get("turn_id", ""),
        "api_call_count": kwargs.get("api_call_count", 1),
        "tier": decision.tier,
        "model": decision.model,
        "effort": decision.effort,
        "reason": decision.reason,
        # Completion envelopes can contain child summaries and must not become
        # a second uncontrolled raw-prompt store.  Persist only a stable
        # correlation id; the viewer resolves a bounded redacted description.
        "prompt_preview": "Delegált feladat befejezési eseménye" if delegation_id else _logged_prompt_preview(request, log_cfg),
    }
    if decision.vetoed_by:
        # Tiers this request qualified for but did not get. Absence over a large
        # sample means the tier's preconditions never hold; presence means the
        # tier is reachable and something ahead of it keeps winning.
        entry["vetoed_by"] = list(decision.vetoed_by)
    if event_kind:
        # The consumer resolves any origin task from durable state.  Do not put
        # the synthetic completion envelope (which can contain a full result)
        # into a second provenance field.
        entry["event_kind"] = event_kind
    if delegation_id:
        entry["delegation_id"] = delegation_id
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _shadow_path(cfg: Dict[str, Any]) -> Path:
    shadow = cfg.get("shadow") or {}
    return Path(os.path.expanduser(str(shadow.get("path", "~/.hermes/logs/spark-shadow-benchmark.jsonl"))))


def _shadow_event(cfg: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Persist local-only benchmark progress without delaying the parent turn."""
    path = _shadow_path(cfg)
    cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
    event = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **({"cycle_id": cycle_id} if cycle_id else {}),
        **event,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SHADOW_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _benchmark_id(parent_turn_id: str) -> str:
    """Stable local identifier for one forced parent shadow attempt."""
    digest = hashlib.sha256(parent_turn_id.encode("utf-8")).hexdigest()[:16]
    return f"shadow-{digest}"


def _read_shadow_events(cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    path = _shadow_path(cfg)
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _shadow_forced_event(cfg: Dict[str, Any], parent_turn_id: str) -> Optional[Dict[str, Any]]:
    cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
    for event in reversed(_read_shadow_events(cfg)):
        if (
            event.get("event") == "delegation_forced"
            and event.get("turn_id") == parent_turn_id
            and (not cycle_id or str(event.get("cycle_id") or "") == cycle_id)
        ):
            return event
    return None


def _completed_shadow_benchmark_count(events: Iterable[Dict[str, Any]]) -> int:
    """Count only benchmark IDs with an auditable, fully closed parent/child lifecycle."""
    required_events = {"delegation_forced", "child_started", "child_completed", "parent_completed"}
    lifecycle_by_benchmark: Dict[str, set[str]] = {}
    for event in events:
        benchmark_id = event.get("benchmark_id")
        if not benchmark_id:
            continue
        lifecycle_by_benchmark.setdefault(str(benchmark_id), set()).add(str(event.get("event", "")))
    return sum(required_events <= lifecycle for lifecycle in lifecycle_by_benchmark.values())


def _completed_actual_spark_benchmark_count(events: Iterable[Dict[str, Any]], cfg: Dict[str, Any]) -> int:
    """Count closed shadow pairs only when their child was actually routed to Spark.

    A completed lifecycle with no correlatable router rows is counted conservatively so
    a broken log cannot create unbounded new shadow work. A child with correlated
    non-Spark rows is explicitly excluded and may be replaced by another sample.
    """
    required_events = {"delegation_forced", "child_started", "child_completed", "parent_completed"}
    cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
    if cycle_id:
        events = [event for event in events if str(event.get("cycle_id") or "") == cycle_id]
    lifecycle_by_benchmark: Dict[str, set[str]] = {}
    child_session_by_benchmark: Dict[str, str] = {}
    for event in events:
        benchmark_id = event.get("benchmark_id")
        if not benchmark_id:
            continue
        benchmark_id = str(benchmark_id)
        lifecycle_by_benchmark.setdefault(benchmark_id, set()).add(str(event.get("event", "")))
        if event.get("event") == "child_completed" and event.get("child_session_id"):
            child_session_by_benchmark[benchmark_id] = str(event["child_session_id"])

    route_path = Path(os.path.expanduser(str((cfg.get("logging") or {}).get("path", _DEFAULT_CONFIG["logging"]["path"]))))
    try:
        route_events = [json.loads(line) for line in route_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return _completed_shadow_benchmark_count(events)

    spark_model = str((cfg.get("models") or {}).get("spark", ""))
    completed = 0
    for benchmark_id, lifecycle in lifecycle_by_benchmark.items():
        if not required_events <= lifecycle:
            continue
        child_session_id = child_session_by_benchmark.get(benchmark_id)
        matching_routes = [
            route for route in route_events
            if child_session_id and child_session_id in str(route.get("turn_id", ""))
        ]
        if not matching_routes or all(str(route.get("model", "")) == spark_model for route in matching_routes):
            completed += 1
    return completed


def _summary_digest(summary: Any) -> str:
    return hashlib.sha256(str(summary or "").encode("utf-8")).hexdigest()


def _prepare_shadow_delegation(request: Dict[str, Any], benchmark_id: str) -> Dict[str, Any]:
    """Force the parent to create one safe Spark worker through Hermes' own tool loop."""
    shadow = deepcopy(request)
    instruction = (
        "\n\n[INTERNAL SPARK MEDIUM SHADOW BENCHMARK]\n"
        f"Benchmark ID: {benchmark_id}. Call delegate_task now with exactly one fully specified child task. "
        "Its goal is to independently analyze the same user objective in read-only mode; its context must prohibit "
        "edits, commands, external messages, deploys, credentials, database/payment operations, and destructive "
        "actions. Request a concise evidence-based plan, risks, test/review checklist, and proposed answer.\n"
    )
    _append_user_instruction(shadow, instruction)
    delegate_tool = _find_delegate_tool(shadow)
    if delegate_tool is None:
        return shadow
    # Codex Responses has historically treated a named function choice as a
    # best-effort hint. A one-tool, required call is deterministic and leaves
    # the normal complete toolset untouched on the following parent iteration.
    shadow["tools"] = [delegate_tool]
    shadow["tool_choice"] = "required"
    shadow["parallel_tool_calls"] = False
    return shadow


def _append_user_instruction(request: Dict[str, Any], instruction: str) -> None:
    """Append an internal instruction to the latest user turn in the request's
    own wire shape.

    ``input_text`` is a Responses-API part type.  Hermes converts to the
    provider wire format in ``build_api_kwargs`` *before* llm_request
    middleware runs, so an ``anthropic_messages`` route (the TokenPlan Qwen
    endpoint) reaches this code already Anthropic-shaped, where ``input_text``
    is not a valid content block.  Emitting it there either 400s the call or
    gets the block dropped — which is how a "forced" preflight can arrive at
    the model with its entire instruction missing.
    """
    items = _request_items(request)
    _, index = _last_user_text_and_index(items)
    if index < 0 or not isinstance(items[index], dict):
        return
    block_type = (
        "input_text"
        if not isinstance(request.get("messages"), list) and isinstance(request.get("input"), list)
        else "text"
    )
    content = items[index].get("content")
    if isinstance(content, str):
        items[index]["content"] = content + instruction
    elif isinstance(content, list):
        items[index]["content"] = [*content, {"type": block_type, "text": instruction}]


def _tool_names(request: Any) -> list:
    """Tool names in a request, in either wire shape; [] when there are none."""
    if not isinstance(request, dict):
        return []
    names = []
    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            name = tool.get("name") or (tool.get("function") or {}).get("name")
            if name:
                names.append(str(name))
    return names


# Anthropic OAuth requests are normalised for Claude Code compatibility, which
# prefixes every tool name with ``mcp__``. Matching the bare name there found
# nothing, so a parent on a Claude account was told it had no delegate_task tool
# and skipped its preflight — the one path where spreading work matters most.
_DELEGATE_TOOL_NAMES = frozenset({"delegate_task", "mcp__delegate_task"})


def _is_delegate_tool_name(name: Any) -> bool:
    return isinstance(name, str) and name in _DELEGATE_TOOL_NAMES


def _find_delegate_tool(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the delegate_task tool definition in any wire shape or naming."""
    for tool in request.get("tools") or []:
        if isinstance(tool, dict) and (
            _is_delegate_tool_name(tool.get("name"))
            or _is_delegate_tool_name((tool.get("function") or {}).get("name"))
        ):
            return tool
    return None


def _tool_schema_slot(tool: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Return the (owner, key) pair holding a tool's JSON schema.

    OpenAI/Codex tools keep it at ``function.parameters``; Anthropic Messages
    tools keep it at ``input_schema``.  Reading only ``parameters`` silently
    skipped every schema-hardening step on the Anthropic path, so the
    ``role="orchestrator"`` constraint never reached a Qwen planner.
    """
    if isinstance(tool.get("function"), dict):
        return tool["function"], "parameters"
    if isinstance(tool.get("input_schema"), dict):
        return tool, "input_schema"
    return tool, "parameters"


def _is_anthropic_shaped(request: Dict[str, Any]) -> bool:
    """True when the request already carries the Anthropic Messages shape."""
    if not isinstance(request.get("messages"), list):
        return False
    return any(
        isinstance(tool, dict) and isinstance(tool.get("input_schema"), dict)
        for tool in request.get("tools") or []
    )


def _supports_forced_tool_choice(kwargs: Dict[str, Any], decision: RouteDecision) -> bool:
    """Whether this route can be made to call a tool by protocol.

    TokenPlan's Anthropic-compatible Qwen endpoint rejects ``tool_choice``
    outright — including ``{"type": "auto"}`` — so Hermes' Anthropic adapter
    omits the field there.  A preflight on that route is therefore a prompt
    contract, never an enforced one.  This matters because the preflight also
    amputates the toolset to a single tool: without the matching
    ``tool_choice`` the parent is left with one optional tool and no way to do
    anything else, which is strictly worse than not preflighting at all.
    """
    if "qwen" in str(decision.model).casefold():
        return False
    if str(kwargs.get("provider") or "").casefold() == "qwen-token":
        return False
    return "token-plan." not in str(kwargs.get("base_url") or "").casefold()


_HERMES_CONFIG_PATH = Path(os.path.expanduser("~/.hermes/config.yaml"))


def _delegation_target_names() -> Tuple[str, ...]:
    """Targets the host will actually accept in ``delegate_task(model=...)``.

    Read from Hermes's own ``delegation.targets`` rather than this plugin's
    config: that map builds the tool's ``model`` enum, and the host silently
    drops any target with an empty model. Naming one here that the host has
    dropped would point the planner at a route that cannot spawn.
    """
    if yaml is None:
        return ()
    try:
        raw = yaml.safe_load(_HERMES_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        targets = (raw.get("delegation") or {}).get("targets") or {}
        return tuple(sorted(
            str(name).strip().casefold()
            for name, spec in targets.items()
            if isinstance(spec, dict) and str(spec.get("model") or "").strip()
        ))
    except Exception:
        return ()


def _recent_account_load(cfg: Dict[str, Any], window_seconds: int) -> Dict[str, int]:
    """Calls per account over the recent window, read from this router's own log.

    Call counts, not quota readings: the runtime does not report tokens or cost
    to the route log, so anything phrased as "83% used" would be invented. A
    relative load figure is what the data supports, and it is enough to tell an
    idle account from a busy one.

    Only the tail of the log is parsed. It reaches tens of megabytes, and this
    runs on the preflight path where a full scan would be felt.
    """
    log_cfg = cfg.get("logging") or {}
    path = Path(os.path.expanduser(str(log_cfg.get("path") or "")))
    tier_providers = cfg.get("tier_providers") or {}
    if not str(path) or not tier_providers:
        return {}
    cutoff = datetime.now(timezone.utc).timestamp() - max(60, int(window_seconds))
    counts: Dict[str, int] = {}
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _USAGE_TAIL_BYTES))
            if size > _USAGE_TAIL_BYTES:
                handle.readline()  # discard the partial line the seek landed in
            for raw in handle:
                try:
                    entry = json.loads(raw)
                    observed = datetime.fromisoformat(
                        str(entry.get("timestamp") or "").replace("Z", "+00:00")
                    )
                except Exception:
                    continue
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                if observed.timestamp() < cutoff:
                    continue
                account = tier_providers.get(str(entry.get("tier") or ""))
                if account:
                    counts[account] = counts.get(account, 0) + 1
    except OSError:
        return {}
    return counts


def _peers_for(name: str, cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Targets of comparable strength that can take this one's work."""
    for members in (cfg.get("peer_groups") or {}).values():
        if isinstance(members, list) and name in members:
            return tuple(peer for peer in members if peer != name)
    return ()


def _target_availability(names: Iterable[str], cfg: Dict[str, Any]) -> Dict[str, str]:
    """Per-target cooldown note, empty when the target is available.

    Cooling targets are annotated rather than dropped. LiteLLM excludes a
    deployment that would exceed its limit, but its deployments are
    interchangeable and ours are not: hiding a cooling Sol would invite the
    planner to send design work somewhere it is not allowed, which the
    classifier then refuses outright. Saying "unavailable, and for how long"
    lets the conductor wait or narrow the objective instead.
    """
    offered = set(names)
    notes = {}
    for name in names:
        remaining = _tier_cooldown_remaining(name, cfg)
        if not remaining:
            notes[name] = ""
            continue
        alive = [
            peer for peer in _peers_for(name, cfg)
            if peer in offered and not _tier_cooldown_remaining(peer, cfg)
        ]
        instead = f"; use {' or '.join(alive)} instead" if alive else ""
        notes[name] = f" [unavailable for another {int(remaining // 60) + 1} min{instead}]"
    return notes


def _delegation_targets_detail() -> Dict[str, Dict[str, str]]:
    """``{name: {provider, model}}`` from Hermes's own ``delegation.targets``."""
    if yaml is None:
        return {}
    try:
        raw = yaml.safe_load(_HERMES_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        targets = (raw.get("delegation") or {}).get("targets") or {}
        return {
            str(name).strip().casefold(): {
                "provider": str(spec.get("provider") or "").strip().casefold(),
                "model": str(spec.get("model") or "").strip(),
            }
            for name, spec in targets.items()
            if isinstance(spec, dict) and str(spec.get("model") or "").strip()
        }
    except Exception:
        return {}


def _external_target_for_model(model: str) -> Optional[str]:
    """Target name for a model this router cannot route but should still record.

    A child on another provider is invisible here by design -- the middleware
    cannot move a call across providers, so it returns None. But invisible to
    the router became invisible to the operator too: a Claude worker produced no
    card, no count and no line in the per-account load, so the one account whose
    usage most needed watching was the one nothing reported on.
    """
    if not model:
        return None
    for name, spec in _delegation_targets_detail().items():
        if spec.get("model") == model:
            return name
    return None


def _target_is_offered(name: str, cfg: Dict[str, Any]) -> bool:
    """Whether a delegation target should be put in front of the conductor.

    A target with a callability switch obeys it even when it is not a routable
    tier: the dashboard toggle otherwise reads as if it governed Claude while
    changing nothing.
    """
    switches = cfg.get("callable") or {}
    # Deliberately not _is_callable_tier: that folds in the cooldown, and a
    # cooling target must stay visible. Hiding it invites the planner to route
    # work somewhere it is not allowed -- a cooling Sol does not make design work
    # someone else's job -- and it hides the fact that waiting is an option.
    return switches.get(name) is True if name in switches else True


def _model_param_contract(orchestrator_tier: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """The sentence that makes route choice expressible instead of implied.

    A ``[sol]``/``[spark]`` goal prefix is only a model rename inside the
    default provider, so it can never reach a target that lives on a separate
    account. Without this, every leaf inherits the default route: the delegation
    registry shows every child ever spawned running on the default model, even
    ones whose goal was explicitly prefixed for another target.
    """
    cfg = cfg if isinstance(cfg, dict) else _load_config()
    # A target whose tier is switched off is not a route: the cross-provider guard
    # raises for it mid-session, so offering it produces a leaf that never runs.
    names = [
        name for name in _delegation_target_names()
        if name != orchestrator_tier and _target_is_offered(name, cfg)
    ]
    notes = _target_availability(names, cfg)
    scope = (
        f" (targets: {', '.join(name + notes.get(name, '') for name in names)})" if names else ""
    )
    return (
        f"Set the delegate_task 'model' parameter on every worker to choose its route{scope}. "
        "A goal-text prefix only renames the model inside the default provider and cannot reach a "
        "target on a separate account, so a leaf intended for one must carry model:<name>. Prefer "
        "spreading genuinely independent leaves across different targets so separate accounts and "
        "quotas absorb the work in parallel; never split work merely to use more targets. "
        f"{_peer_group_sentence(names, cfg)}"
        f"{_claude_target_sentence(names)}"
        f"{_preference_sentence(names, cfg)}"
        f"{_account_load_sentence(cfg)}"
    )


def _preference_sentence(names: Iterable[str], cfg: Dict[str, Any]) -> str:
    """State the user's per-work-kind target preferences to the conductor.

    The router cannot route across providers, so a preference naming an external
    account is only ever realisable here: the conductor is the one that picks a
    delegate_task target. Only offered targets are mentioned — advising a leaf onto
    a switched-off account would produce a child that never runs.
    """
    offered = set(names)
    pairs = []
    for kind in WORK_KINDS:
        target = _preferred_target(kind, cfg)
        if target and target in offered:
            pairs.append(f"{kind} -> model:{target}")
    if not pairs:
        return ""
    return (
        "The operator has set preferred targets per kind of work: "
        + "; ".join(pairs)
        + ". Honour these when a leaf matches the kind and the target is free. "
    )


def _peer_group_sentence(names: Iterable[str], cfg: Dict[str, Any]) -> str:
    """Name the substitutions, so a loaded target is a choice rather than a wait.

    Dropping an unavailable target from the list told the planner only that it
    was gone. Knowing what replaces it is what turns one account's exhaustion
    into work continuing somewhere else.
    """
    offered = set(names)
    groups = [
        [name for name in members if name in offered]
        for members in (cfg.get("peer_groups") or {}).values()
        if isinstance(members, list)
    ]
    usable = [group for group in groups if len(group) > 1]
    if not usable:
        return ""
    listed = "; ".join(" / ".join(group) for group in usable)
    return (
        f"Comparable in strength and on different accounts, so they substitute for each other "
        f"when one is loaded or unavailable: {listed}. Substituting is for capacity only -- the "
        f"rules each label carries still apply, so a Spark leaf must still be read-only and design "
        f"work still belongs to Sol. "
    )


def _claude_target_sentence(names: Iterable[str]) -> str:
    """What the Claude targets are for, once they are offered at all.

    A bare name in a list tells the conductor nothing about when to reach for it,
    and these are the two that draw on a different subscription entirely -- the
    reason the target list exists.
    """
    claude = [name for name in names if name in {"opus5", "sonnet5"}]
    if not claude:
        return ""
    both = len(claude) == 2
    return (
        f"{' and '.join(claude)} run on Claude, a different subscription from every other "
        f"target, so they are the strongest way to keep independent work off a single quota. "
        f"They are ordinary workers with the usual tools: give them implementation or deep "
        f"review, not just reading. "
        + ("Use sonnet5 by default and reserve opus5 for consequential or hard work. "
           if both else "")
    )


def _account_load_sentence(cfg: Dict[str, Any]) -> str:
    """Tell the conductor where the traffic has actually been going.

    Spreading work was previously an instruction with nothing behind it: the
    conductor was told to use separate accounts but had no way to see that one
    of them had taken every call for the last hour and another had taken none.
    """
    policy = cfg.get("usage_report") or {}
    if not policy.get("enabled", True):
        return ""
    window = int(policy.get("window_seconds", 3600) or 3600)
    counts = _recent_account_load(cfg, window)
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary = ", ".join(f"{account} {calls}" for account, calls in ordered)
    idle = [
        account
        for account in sorted({str(v) for v in (cfg.get("tier_providers") or {}).values()})
        if counts.get(account, 0) == 0
    ]
    # "No calls" reads as spare capacity, but it is equally what an exhausted
    # account looks like -- which is exactly what Qwen was when this sentence
    # last recommended it. State the fact and leave the inference alone.
    tail = f" {', '.join(idle)} has taken none in this window." if idle else ""
    return (
        f"Recent load over the last {window // 60} minutes, in calls per account: {summary}. "
        f"These are call counts from this router's own log, not quota readings -- read them as "
        f"relative load.{tail}"
    )


def _conductor_tier(cfg: Optional[Dict[str, Any]]) -> str:
    """The tier the forced conductor child should run on.

    ``default_model`` when it can actually be called, otherwise the first callable
    tier its ``fallbacks`` chain reaches. Pinning the conductor to a configured
    default is what made an exhausted account fail the whole preflight: the parent
    had already moved to a working account, and its planner was still being sent
    to the one that had run out.
    """
    cfg = cfg or {}
    default = str(cfg.get("default_model", "terra"))
    if _is_callable_tier(default, cfg):
        return default
    chain = cfg.get("fallbacks") or {}
    seen, current = {default}, default
    for _ in range(3):
        nxt = str(chain.get(current) or "")
        if not nxt or nxt in seen:
            break
        if _is_callable_tier(nxt, cfg):
            return nxt
        seen.add(nxt)
        current = nxt
    for tier in (cfg.get("models") or {}):
        if _is_callable_tier(tier, cfg):
            return tier
    return default


def _prepare_orchestration_delegation(
    request: Dict[str, Any],
    plan_id: str,
    max_tasks: int,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    force_tools: bool = True,
) -> Dict[str, Any]:
    """Force one real Terra-supervised Spark dispatch before parent execution.

    This is an operational delegation checkpoint, never a benchmark: the
    parent must use returned evidence, explicitly accept/reject it, and retain
    integration ownership.
    """
    orchestrator_tier = _conductor_tier(cfg)
    
    routed = deepcopy(request)
    instruction = (
        f"\n\n[INTERNAL ORCHESTRATOR PREFLIGHT]\n"
        f"Plan ID: {plan_id}. Before any normal tool action, call delegate_task exactly once with role=\"orchestrator\" "
        f"and a goal beginning with [{orchestrator_tier}]. This creates a dedicated {orchestrator_tier} planner and conductor, not a benchmark worker. "
        "Give that conductor the full current objective. It must first inspect any current image itself and Create a structured dispatch plan "
        f"before any implementation. The plan may contain zero to {max_tasks} independent workers; do not invent work merely to fill slots. "
        f"{orchestrator_tier} chooses the decomposition from the actual task: prefix every Spark leaf goal with [spark] and use it only for bounded low-risk read-only source/component discovery, "
        "logs, test-case design, isolated patch proposals, or research. Any visual/product/UI/UX/CSS/layout/design-system analysis or implementation is Sol-only and must be prefixed [sol]; prefix a consequential "
        f"worker goal with [sol] only for security/auth/credentials/payment/migration/production analysis. Spark/Sol workers receive a "
        "self-contained textual scope, never the original image. Spark leaves must be read-only: prohibit edits, commands with side effects, "
        "external messages, deploys, credentials, database/auth/payment operations, and destructive actions. "
        f"{_model_param_contract(orchestrator_tier, cfg)} "
        "A [sonnet-review] or [opus-review] leaf takes no 'model', because its route is its label; it is read-only "
        "and replaces a single call rather than running an agent. For real work prefer a native Claude target via "
        "model:opus5 / model:sonnet5 when one is offered. "
        "Write the goal as objective and acceptance criteria only: what must change, where, and how it is verified. "
        "Do not restate this routing policy inside the goal. The conductor already receives it verbatim as an immutable "
        "contract in the required `context` field, and the goal is re-read as a description of the work -- routing "
        "vocabulary repeated there is classified as the work itself and re-routes the conductor away from "
        f"{orchestrator_tier}. "
        f"{orchestrator_tier} keeps coordination, acceptance/rejection, shared-file integration, and final verification; Sol owns all design analysis and design implementation. "
        "It waits for delegated evidence, explicitly records SUPERVISOR DECISION: ACCEPT or REJECT for every worker, then completes the owned work. "
        f"The current parent must not perform normal implementation; the {orchestrator_tier} conductor owns the reviewed result.\n"
    )
    _append_user_instruction(routed, instruction)
    # Third copy of the same lookup, and the one that raised rather than skipping when
    # the name did not match. All three now go through _find_delegate_tool.
    delegate_tool = _find_delegate_tool(routed)
    if delegate_tool is None:
        return routed
    # Do not merely ask the parent to create an orchestrator: constrain the
    # one permitted tool schema so the runtime receives an actual
    # ``role=orchestrator`` child. Natural-language instructions alone are not
    # a reliable control plane, as models can otherwise emit the default leaf
    # role and skip the planner layer altogether.
    planner_tool = deepcopy(delegate_tool)
    schema_owner, schema_key = _tool_schema_slot(planner_tool)
    schema = schema_owner.get(schema_key)
    if isinstance(schema, dict):
        properties = schema.setdefault("properties", {})
        properties["role"] = {
            "type": "string",
            "enum": ["orchestrator"],
            "description": f"Required fixed role for the {orchestrator_tier} planning child.",
        }
        # This travels in the actual delegated child's system prompt. It avoids
        # relying on the parent model to faithfully copy the planner contract
        # into a free-form goal/context field.
        properties["context"] = {
            "type": "string",
            "enum": [
                f"You are the {orchestrator_tier} planning conductor. Do not perform design analysis or design implementation. Route every visual/product/UI/UX/CSS/layout/design-system task to Sol with a goal beginning [sol] and model:sol. Delegate only bounded, self-contained low-risk non-design read-only evidence loops to Spark with a goal beginning [spark] and model:spark. Read-only does not make a design question non-design: judging visual hierarchy, appearance, spacing or styling is Sol's work even when nothing is written. Spark receives source discovery, tests, logs and research -- questions with a factual answer. {_model_param_contract(orchestrator_tier, cfg)} A purely read-only review leaf may instead be labelled [sonnet-review] or [opus-review], which runs it through the Claude Code CLI on a separate subscription. Use [sonnet-review] for routine checks and [opus-review] for consequential ones. Such a leaf takes no 'model' -- its route is its label -- must name the repository, must carry every fact it needs in the goal, and must never be asked to edit, run commands, or implement. Write every leaf goal as objective and acceptance criteria only: never restate this routing policy inside a leaf goal, because a leaf is re-classified from its own goal text and routing vocabulary repeated there is read as the work itself. The orchestrator retains coordination, evidence acceptance/rejection, integration, and final approval. Use zero leaves only when the objective genuinely has no independently useful non-design text-only investigation, test, source-discovery, or research subtask."
            ],
            "description": f"Required immutable routing contract for the {orchestrator_tier} planner.",
        }
        required = list(schema.get("required") or [])
        for name in ("goal", "role", "context"):
            if name not in required:
                required.append(name)
        schema["required"] = required
    if force_tools:
        # One tool plus a required choice is deterministic, and leaves the
        # normal complete toolset untouched on the parent's next iteration.
        routed["tools"] = [planner_tool]
        if _is_anthropic_shaped(routed):
            routed["tool_choice"] = {"type": "tool", "name": "delegate_task"}
        else:
            routed["tool_choice"] = "required"
            routed["parallel_tool_calls"] = False
    else:
        # No protocol-level forcing on this route (TokenPlan Qwen). Keep the
        # parent's full toolset — a lone optional tool would leave it unable to
        # act — and swap in the hardened delegate_task schema so that *if* it
        # delegates, it can only produce a role="orchestrator" planner child.
        routed["tools"] = [
            planner_tool if tool is delegate_tool else tool
            for tool in (routed.get("tools") or [])
        ]
    return routed


def _sol_opus5_preflight_enabled(cfg: Dict[str, Any]) -> bool:
    """Return true only for the explicitly configured, currently callable bridge."""
    policy = cfg.get("sol_opus5_preflight") or {}
    return (
        _is_callable_tier("sol", cfg)
        and _is_callable_tier("opus5", cfg)
        and bool(policy.get("enabled"))
        and str(policy.get("owner", "")).casefold() == "sol"
        and str(policy.get("bridge_model", "")).casefold() == "claude-opus-5"
        and bool(policy.get("require_successful_auth_probe"))
    )


def _prepare_sol_opus5_preflight(request: Dict[str, Any], plan_id: str) -> Dict[str, Any]:
    """Force a Sol-owned preflight without routing it through Spark or Terra.

    The normal Sol request remains OpenAI-compatible; this records the required
    read-only external Claude Code Opus 5 review contract rather than attempting
    an unsafe provider/model-string substitution.
    """
    routed = deepcopy(request)
    instruction = (
        "\n\n[INTERNAL SOL + CLAUDE OPUS 5 PREFLIGHT]\n"
        f"Plan ID: {plan_id}. Before normal implementation, call delegate_task exactly once with a goal beginning [sol]. "
        "This is a Sol-owned visual/product/UI/UX/CSS/layout/design-system preflight. The delegated Sol reviewer must first inspect any "
        "current image itself, create a structured dispatch/review plan, and perform the configured read-only Claude Code bridge review "
        "with requested alias `opus`. Accept the review only if the bridge reports canonical effective model `claude-opus-5`; otherwise "
        "stop and report the unavailable bridge without falling back to Spark or Terra. Do not expose credentials or make writes, deploys, "
        "payments, or production changes during preflight. Spark and Terra are not preflight targets for this request.\n"
    )
    _append_user_instruction(routed, instruction)
    delegate_tool = _find_delegate_tool(routed)
    if delegate_tool is None:
        return routed
    routed["tools"] = [delegate_tool]
    routed["tool_choice"] = "required"
    routed["parallel_tool_calls"] = False
    return routed


def _orchestration_path(cfg: Dict[str, Any]) -> Path:
    orchestration = cfg.get("orchestration") or {}
    return Path(os.path.expanduser(str(orchestration.get("path", "~/.hermes/logs/terra-spark-orchestration.jsonl"))))


def _orchestration_event(cfg: Dict[str, Any], event: Dict[str, Any]) -> None:
    path = _orchestration_path(cfg)
    payload = {"timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **event}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SHADOW_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _orchestration_forced_event(cfg: Dict[str, Any], parent_turn_id: str) -> Optional[Dict[str, Any]]:
    path = _orchestration_path(cfg)
    if not path.exists():
        return None
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None
    for event in reversed(events):
        if event.get("event") == "preflight_forced" and event.get("turn_id") == parent_turn_id:
            return event
    return None


def _orchestration_skip_reason(
    kwargs: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision
) -> Optional[str]:
    """Name the gate that rejected this turn, or None when it is eligible.

    The route log records the winning route, not the dispatch that never
    happened, so a turn that runs twenty parent calls with no worker looks
    identical whether orchestration was ineligible, deduped, or never reached.
    Returning the reason lets the caller record it.
    """
    policy = cfg.get("orchestration") or {}
    request = kwargs.get("request")
    sol_preflight = decision.tier == "sol" and _sol_opus5_preflight_enabled(cfg)
    # Orchestration is enabled for Sol (design preflight) and the configured
    # default_model (which acts as the general-purpose orchestrator).
    # An external parent (a fallback account) orchestrates exactly like a local one:
    # only the model rewrite is provider-bound, the delegation contract is not.
    orchestration_tiers = (
        {"sol"} | {str(cfg.get("default_model", "terra"))} | set(_delegation_target_names())
    )
    if not policy.get("enabled"):
        return "orchestration_disabled"
    if decision.tier not in orchestration_tiers:
        return f"tier_not_orchestrator:{decision.tier}"
    if not isinstance(request, dict):
        return "request_not_a_dict"
    if decision.tier == "sol" and not sol_preflight:
        return "sol_preflight_disabled"
    api_call_count = int(kwargs.get("api_call_count", 1) or 1)
    # Normal path: dispatch on the first Terra call. Recovery path: if that
    # process missed its initial checkpoint, rescue a genuinely long tool loop
    # once instead of letting it remain 20-30 Terra calls with no Spark work.
    rescue_min_calls = max(2, int(policy.get("rescue_min_calls", 6) or 6))
    is_rescue = api_call_count >= rescue_min_calls
    if api_call_count != 1 and not is_rescue:
        return f"mid_loop_call:{api_call_count}"
    if str(kwargs.get("platform", "")).casefold() == "subagent" or ":sa-" in str(kwargs.get("turn_id", "")):
        return "subagent_turn"
    # One owner for "is delegate_task on offer": this duplicated the check inline and
    # the two copies drifted the moment Anthropic's mcp__ prefix appeared.
    if _find_delegate_tool(request) is None:
        return "no_delegate_task_tool"
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    if not user_text:
        return "no_user_text"
    # A task too short to decompose is not worth a planner round trip plus up to
    # max_tasks bounded workers. Without this gate every actionable Terra turn
    # forced a fan-out dispatch, the dominant source of perceived latency. That
    # is a first-call latency argument, so it must not gate the rescue: a turn
    # already deep in a tool loop has spent far more than a planner round trip,
    # and a short prompt ("csinald meg") routinely opens the longest loops.
    if decision.tier == str(cfg.get("default_model", "terra")) and not sol_preflight and not is_rescue:
        min_chars = max(0, int(policy.get("min_chars", 180) or 0))
        if len(user_text) < min_chars:
            return f"prompt_shorter_than_min_chars:{len(user_text)}<{min_chars}"
    # Explicit bounded UI requests authorised for the verified bridge are a
    # single-hop exception: Sol retains policy ownership, but no Sol/Terra/Spark
    # planner call is created before Opus execution middleware handles the turn.
    if decision.tier == "sol" and _is_explicit_bounded_opus_ui_request(user_text, cfg):
        return "explicit_bounded_opus_ui_request"
    # A normal first-call preflight must be before parent tool work. The rescue
    # path intentionally runs inside an existing tool loop.
    if not is_rescue and _current_turn_has_tool_activity(items, user_index):
        return "tool_activity_before_first_call"
    text = _normalise(user_text)
    # Completion delivery is the Terra supervisor's reviewed hand-back, never a
    # fresh task to fan out again. Without this guard it can recursively create
    # another orchestration cycle merely because the consolidated evidence is long.
    if "[async delegation batch complete" in text:
        return "delegation_completion_delivery"
    # A compaction envelope can precede the real current user request in the
    # same message. Route from the suffix after its end marker; treating the
    # whole envelope as an internal continuation silently suppresses dispatch.
    if text.startswith("[context compaction"):
        end_marker = re.search(r"\[end of context summary[^\]]*\]", text)
        if not end_marker:
            return "compaction_envelope_without_end_marker"
        text = text[end_marker.end():].strip()
        if not text:
            return "compaction_envelope_with_empty_suffix"
    internal_prefixes = (
        "review the conversation above and consider saving to memory",
        "what do you see in this image?",
    )
    if text.startswith(internal_prefixes):
        return "internal_prompt_prefix"
    # Terra is the planner for every actionable Terra turn. Do not try to infer
    # task decomposability from keyword lists: real work is often introduced by
    # terse contextual requests, screenshots, or a tool loop whose prompt has
    # none of the old multi-step marker words. The planner itself decides whether
    # zero, one, or several bounded workers are useful; host guards still
    # constrain their scopes and hard-risk requests route to Sol first.
    if not text:
        return "empty_normalised_text"
    return None


def _orchestration_eligible(kwargs: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision) -> bool:
    return _orchestration_skip_reason(kwargs, cfg, decision) is None


def _force_terra_supervisor_preflight(
    kwargs: Dict[str, Any], cfg: Dict[str, Any], decision: RouteDecision
) -> Optional[Dict[str, Any]]:
    turn_id = str(kwargs.get("turn_id", ""))
    api_call_count = int(kwargs.get("api_call_count", 1) or 1)
    phase = "rescue" if api_call_count > 1 else "preflight"
    skip_reason = _orchestration_skip_reason(kwargs, cfg, decision)
    if skip_reason is not None:
        # Record only the two calls where a dispatch was actually due. Logging
        # every mid-loop call would bury the signal under one line per parent
        # call, which is the noise the api_call_count gate already rejects.
        policy = cfg.get("orchestration") or {}
        rescue_min_calls = max(2, int(policy.get("rescue_min_calls", 6) or 6))
        # An operator who turned orchestration off has nothing to diagnose, and
        # a config that omits `path` falls back to the shared production log --
        # so logging this case makes every unrelated caller write to it.
        if policy.get("enabled") and (api_call_count == 1 or api_call_count == rescue_min_calls):
            with _SHADOW_LOCK:
                _orchestration_event(
                    cfg,
                    {
                        "event": "preflight_skipped",
                        "phase": phase,
                        "api_call_count": api_call_count,
                        "turn_id": turn_id,
                        "parent_model": decision.tier,
                        "skip_reason": skip_reason,
                        # Names only, no schemas: "no delegate_task tool" is otherwise
                        # indistinguishable from "no tools at all" or "a different wire
                        # shape", and those need different fixes.
                        "tools_seen": _tool_names(kwargs.get("request")),
                    },
                )
        return None
    with _SHADOW_LOCK:
        if _orchestration_forced_event(cfg, turn_id):
            return None
        plan_id = f"plan-{hashlib.sha256(turn_id.encode('utf-8')).hexdigest()[:16]}"
        _orchestration_event(
            cfg,
            {
                "event": "preflight_forced",
                "phase": phase,
                "api_call_count": api_call_count,
                "plan_id": plan_id,
                "turn_id": turn_id,
                "parent_model": decision.tier,
                "preflight_owner": "sol" if decision.tier == "sol" else "terra",
                "preflight_bridge_model": "claude-opus-5" if decision.tier == "sol" else None,
                "max_tasks": min(3, max(1, int((cfg.get("orchestration") or {}).get("max_tasks", 3)))),
                "parent_prompt_preview": _prompt_preview(kwargs.get("request") or {}),
            },
        )
    if decision.tier == "sol":
        return _prepare_sol_opus5_preflight(kwargs["request"], plan_id)
    return _prepare_orchestration_delegation(
        kwargs["request"],
        plan_id,
        min(3, max(1, int((cfg.get("orchestration") or {}).get("max_tasks", 3)))),
        cfg=cfg,
        force_tools=_supports_forced_tool_choice(kwargs, decision),
    )


def _shadow_eligible(kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    shadow = cfg.get("shadow") or {}
    request = kwargs.get("request")
    if not shadow.get("enabled") or not isinstance(request, dict):
        return False
    if _request_has_image_attachment(request):
        return False
    if int(kwargs.get("api_call_count", 1) or 1) != 1:
        return False
    if str(kwargs.get("platform", "")).casefold() == "subagent" or ":sa-" in str(kwargs.get("turn_id", "")):
        return False
    if str(request.get("model", "")) != str((cfg.get("models") or {}).get("terra", "")):
        return False
    items = _request_items(request)
    text, user_index = _last_user_text_and_index(items)
    if not text or _current_turn_has_tool_activity(items, user_index):
        return False
    if "[image attached" in _normalise(text) or "[screenshot]" in _normalise(text):
        return False
    has_delegate_tool = _find_delegate_tool(request) is not None
    if not has_delegate_tool:
        return False
    return classify_request(request, api_call_count=1, config=cfg).tier == "terra"


def _force_shadow_delegation_if_eligible(kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _shadow_eligible(kwargs, cfg):
        return None
    with _SHADOW_LOCK:
        path = _shadow_path(cfg)
        used = 0
        if path.exists():
            try:
                prior_events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                # Earlier named-choice attempts were logged as
                # ``delegation_requested`` but never reached the child tool.
                # Only a required-tool attempt suppresses a repeat for this
                # turn, so already-active Terra turns receive one recovery try.
                cycle_id = str((cfg.get("shadow") or {}).get("cycle_id") or "").strip()
                if any(
                    event.get("event") == "delegation_forced"
                    and event.get("turn_id") == str(kwargs.get("turn_id", ""))
                    and (not cycle_id or str(event.get("cycle_id") or "") == cycle_id)
                    for event in prior_events
                ):
                    return None
                used = _completed_actual_spark_benchmark_count(prior_events, cfg)
            except Exception:
                used = 0
        if used >= int((cfg.get("shadow") or {}).get("limit", 10)):
            return None
        turn_id = str(kwargs.get("turn_id", ""))
        benchmark_id = _benchmark_id(turn_id)
        _shadow_event(
            cfg,
            {
                "event": "delegation_forced",
                "turn_id": turn_id,
                "benchmark_id": benchmark_id,
                "parent_model": "terra",
                "shadow_model": "spark",
                "effort": "medium",
                "parent_prompt_preview": _prompt_preview(kwargs.get("request") or {}),
            },
        )
    return _prepare_shadow_delegation(kwargs["request"], benchmark_id)


def route_llm_request(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Hermes llm_request middleware entrypoint."""
    cfg = _load_config()
    disabled = os.environ.get("HERMES_MODEL_ROUTER_DISABLE", "").casefold() in {
        "1", "true", "yes", "on"
    }
    if disabled or not cfg.get("enabled", True):
        return None

    active_model = str(kwargs.get("model", ""))
    supported_models = set(cfg.get("models", {}).values())
    request = kwargs.get("request")
    # Accept requests from any provider that has supported models
    external_parent = ""
    if active_model not in supported_models:
        external = _external_target_for_model(active_model)
        if not external:
            return None
        # Observed, not routed: the model stays the parent's, because this middleware
        # can only rewrite within one provider. Orchestration is not provider-bound
        # though, and returning here meant a parent on a fallback account silently
        # lost its delegation contract and worked alone.
        _log_decision(
            RouteDecision(external, active_model, "external delegation target", "external"),
            kwargs,
            cfg,
        )
        external_parent = external
    if not isinstance(request, dict):
        return None

    subagent_marker = (
        str(kwargs.get("platform", "")).casefold() == "subagent"
        or ":sa-" in str(kwargs.get("turn_id", ""))
    )

    if external_parent:
        # No classification and no rewrite: the route is not ours to choose. The only
        # thing owed here is the preflight that turns a lone parent into a conductor.
        decision = RouteDecision(
            external_parent, active_model, "external delegation target", "external",
        )
        forced = _force_terra_supervisor_preflight(kwargs, cfg, decision)
        if forced is None:
            return None
        # Same envelope as the normal path, minus any model/provider change: the
        # request carries the added contract, the route stays exactly as it arrived.
        return {
            "request": forced,
            "source": "model-router",
            "reason": decision.reason,
            "metadata": {
                "tier": decision.tier,
                "model": active_model,
                "effort": decision.effort,
                "provider": kwargs.get("provider"),
            },
        }

    decision = classify_request(
        request,
        api_call_count=int(kwargs.get("api_call_count", 1) or 1),
        config=cfg,
        # Only a delegated worker carries a label this router itself emitted; a
        # root turn's label is whatever the user typed, so the design gate keeps
        # precedence there.
        allow_plan_label_over_design=subagent_marker,
    )

    decision = _require_callable(decision, cfg)
    # Preserve one durable user-facing owner for the whole root session.  Sol,
    # Spark, Qwen and Opus are available as explicitly requested or delegated
    # workers, not invisible replacements for the person talking to the user.
    active_tier = next(
        (tier for tier, model in (cfg.get("models") or {}).items() if model == active_model),
        None,
    )
    session_policy = cfg.get("session_policy") or {}
    latest_user_text, _ = _last_user_text_and_index(_request_items(request))
    explicit_root_override = bool(
        re.match(r"^\s*\[(?:luna|spark|terra|sol)(?::xhigh)?\](?:\s|$)", _normalise(latest_user_text))
    )
    if (
        not subagent_marker
        and bool(session_policy.get("pin_root_parent", False))
        and not explicit_root_override
        and active_tier
        and _is_callable_tier(active_tier, cfg)
        and decision.tier != active_tier
    ):
        decision = _decision(
            active_tier,
            f"root session parent pinned; {decision.tier} reserved for an explicit or delegated worker",
            cfg,
        )
    # [spark] is an internal leaf label emitted by a Terra plan, not a public
    # root-route escape hatch. A user/root turn must still be assessed by Terra.
    if decision.tier == "spark" and not subagent_marker:
        decision = _decision(
            str(cfg.get("default_model", "terra")),
            "root Spark label deferred to default orchestrator",
            cfg,
        )
    is_spark_subagent = (
        subagent_marker
        and active_model == str((cfg.get("models") or {}).get("spark", ""))
        and bool((cfg.get("delegation") or {}).get("preserve_spark_subagents", True))
        and _is_callable_tier("spark", cfg)
    )
    design_only = decision.reason == "design analysis or implementation is Sol-only"
    if is_spark_subagent and not _request_has_image_attachment(request) and not design_only:
        # The parent chooses whether a task is eligible for delegation. Preserve
        # the explicitly pinned Spark worker across its bounded tool loop so the
        # delegated task can actually complete. Only an explicit Sol request or
        # a hard safety signal overrides the worker; Sol being the normal parent
        # default must not silently promote every safe child.
        user_text, _ = _last_user_text_and_index(_request_items(request))
        if decision.tier != "sol" and not _is_spark_read_only_request(user_text):
            decision = _decision("terra", "Spark is restricted to non-design read-only subtasks", cfg)
        explicit = decision.reason.startswith("explicit [")
        hard_sol_reasons = {
            "sensitive or consequential domain",
            "consequential engineering or research task",
            "consequential system action",
        }
        if decision.reason == "Spark is restricted to non-design read-only subtasks":
            pass
        elif not explicit and decision.reason in hard_sol_reasons:
            affirmative_text = _without_negated_safety_constraints(user_text)
            if affirmative_text != user_text:
                affirmative_decision = classify_request(
                    {"model": active_model, "messages": [{"role": "user", "content": affirmative_text}]},
                    api_call_count=1,
                    config=cfg,
                )
                if affirmative_decision.reason not in hard_sol_reasons:
                    decision = _decision("spark", "eligible delegated Spark subtask", cfg)
        elif not explicit:
            decision = _decision("spark", "eligible delegated Spark subtask", cfg)
    elif (
        subagent_marker
        and active_model == str((cfg.get("models") or {}).get("terra", ""))
        and not design_only
        # Only *unlabelled* child work belongs to the planner tier. A labelled
        # leaf has already been decided -- accepted, or rejected for a stated
        # reason -- and this branch used to overwrite both. It tested the label
        # by string-matching "explicit [" at the front of the reason, which any
        # later rewrite erases: a [spark] leaf becomes "fallback from disabled
        # spark" the moment Spark is not callable, so every legitimate Spark
        # leaf was demoted to the planner tier and its Luna fallback lost.
        # A rejected leaf fared worse -- "consequential Spark task requires Sol"
        # was demoted to Terra, turning a deliberate escalation into the exact
        # tier the escalation existed to avoid.
        and not _PLAN_LABEL.match(_normalise(latest_user_text))
    ):
        # The delegation default is Terra so the forced first child is a real
        # planner. Unlabelled child work remains with the default_model rather
        # than being silently demoted before it can decompose the task.
        default_model_tier = str(cfg.get("default_model", "terra"))
        if not decision.reason.startswith("explicit ["):
            decision = _decision(default_model_tier, f"{default_model_tier.capitalize()} planner or integration subagent", cfg)
    turn_id = str(kwargs.get("turn_id") or "")
    if decision.tier == "spark" and _spark_quota_exhausted_for_turn(turn_id):
        fallback_model = _quota_fallback_model(decision.model, cfg)
        if fallback_model:
            fallback_tier = next(
                (tier for tier, model in (cfg.get("models") or {}).items() if model == fallback_model),
                "luna",
            )
            decision = RouteDecision(
                fallback_tier,
                fallback_model,
                "Spark quota already exhausted for this turn",
                _quota_fallback_effort(cfg, fallback_tier),
            )
    # Rules above may intentionally rewrite the tier (root labels, subagents,
    # quota). Re-validate the final destination immediately before dispatch.
    decision = _require_callable(decision, cfg)
    forced_preflight_request = (
        _force_terra_supervisor_preflight(kwargs, cfg, decision)
        if decision.tier in {str(cfg.get("default_model", "terra")), "sol"}
        else None
    )
    forced_shadow_request = (
        _force_shadow_delegation_if_eligible(kwargs, cfg)
        if decision.tier == str(cfg.get("default_model", "terra")) and forced_preflight_request is None
        else None
    )
    # Hermes middleware cannot switch the underlying provider/transport. If a
    # rule selects a tier owned by another provider, changing only `model`
    # produces invalid calls such as `gpt-5.6-terra` at the Qwen Anthropic
    # endpoint. Preserve the explicitly active tier instead; an orchestrator
    # change across providers must happen through the persisted model config
    # and a fresh session.
    active_tier = next(
        (tier for tier, model in (cfg.get("models") or {}).items() if model == active_model),
        None,
    )
    tier_providers = cfg.get("tier_providers", {})
    if (
        active_tier
        and decision.tier != active_tier
        and tier_providers.get(decision.tier) != tier_providers.get(active_tier)
    ):
        if not _is_callable_tier(active_tier, cfg):
            raise RuntimeError(
                f"Active ModelRouter tier '{active_tier}' is disabled and cannot be "
                f"moved to provider '{tier_providers.get(decision.tier)}' mid-session"
            )
        decision = _decision(
            active_tier,
            "cross-provider route preserved for the active session",
            cfg,
        )

    # A review leaf that reached another provider cannot be handed to the Claude
    # bridge: the execution middleware returns early off-provider, and the bridge
    # answers in the Codex Responses shape. Without saying so it just looks like
    # an ordinary worker on that model, which is how a [sonnet-review] leaf ran
    # to completion on Qwen with the Claude subscription untouched.
    if (
        str(kwargs.get("provider", "")).casefold() != str(cfg.get("provider", "")).casefold()
        and _CLAUDE_REVIEW_LABEL.match(_normalise(latest_user_text))
    ):
        decision = replace(
            decision,
            reason=(
                f"Claude review unavailable on provider "
                f"'{kwargs.get('provider')}'; ran as an ordinary {decision.tier} worker"
            ),
        )

    routed = forced_preflight_request or forced_shadow_request or dict(request)
    # TokenPlan's Anthropic-compatible Qwen endpoint rejects OpenAI/Codex
    # control fields. Sanitize the final request after every orchestration,
    # shadow, fallback, and cross-provider rewrite has run.
    if "qwen" in str(decision.model).casefold():
        routed.pop("tool_choice", None)
        routed.pop("parallel_tool_calls", None)
        routed.pop("reasoning", None)
    if decision.tier in {"luna", "spark"}:
        routed = _strip_historical_image_attachments(routed)

    # Determine the target provider for this tier (for logging only)
    tier_providers = cfg.get("tier_providers", {})
    target_provider = tier_providers.get(decision.tier, cfg.get("provider", "openai-codex"))

    routed["model"] = decision.model
    # NE váltson providert middleware-ben! A Hermes a config.yaml-ból veszi a providert.
    # A middleware csak a modelt és reasoning effort-ot módosítsa.
    # Only set reasoning effort for OpenAI-compatible providers.
    # Anthropic-transport providers (e.g. qwen-token) do not support the
    # `reasoning` keyword and will reject the request with a TypeError.
    # A provider váltás a config.yaml-ban történik, nem middleware-ben.
    # Check if the target model is Qwen (Anthropic Messages API)
    is_qwen_model = "qwen" in decision.model.lower()
    if not is_qwen_model:
        reasoning = dict(routed.get("reasoning") or {})
        reasoning["effort"] = decision.effort
        routed["reasoning"] = reasoning
    _log_decision(decision, kwargs, cfg)

    return {
        "request": routed,
        "source": "model-router",
        "reason": decision.reason,
        "metadata": {
            "tier": decision.tier,
            "model": decision.model,
            "effort": decision.effort,
            "provider": target_provider,
        },
    }


def _is_transient_provider_failure(error: BaseException) -> bool:
    """Return true only for failures that are safe to retry on another model."""
    text = str(error).casefold()
    return any(
        marker in text
        for marker in (
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "internal server error",
            "upstream connect error",
            "connection termination",
            "disconnect/reset before headers",
            "connection reset",
            "gateway timeout",
        )
    )


def _is_quota_exhaustion(error: BaseException) -> bool:
    """Recognize persistent provider-account exhaustion, not ordinary 429 pacing.

    A short provider rate limit should remain an error for the caller to retry;
    switching models for it would waste the fallback.  Weekly/account quotas do
    not recover during the current task, so Spark can safely hand that task to
    Terra instead.
    """
    text = str(error).casefold()
    return "429" in text and any(
        marker in text
        for marker in (
            "weekly limit",
            "weekly quota",
            "quota exhausted",
            "quota has been exhausted",
            "usage limit",
            "usage quota",
            "account quota",
            "credit balance is too low",
            "insufficient quota",
        )
    )


def _spark_quota_exhausted_for_turn(turn_id: str) -> bool:
    if not turn_id:
        return False
    with _QUOTA_LOCK:
        return turn_id in _SPARK_QUOTA_EXHAUSTED_TURNS


def _remember_spark_quota_exhaustion(turn_id: str) -> None:
    if not turn_id:
        return
    with _QUOTA_LOCK:
        if len(_SPARK_QUOTA_EXHAUSTED_TURNS) >= _MAX_REMEMBERED_QUOTA_TURNS:
            _SPARK_QUOTA_EXHAUSTED_TURNS.pop()
        _SPARK_QUOTA_EXHAUSTED_TURNS.add(turn_id)


def _quota_fallback_model(active_model: str, cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve the configured one-time fallback for a depleted Spark quota."""
    models = cfg.get("models") or {}
    if active_model != models.get("spark"):
        return None
    configured = (cfg.get("quota_fallbacks") or {}).get("spark") or "luna"
    fallback_tier = str(configured.get("model") if isinstance(configured, dict) else configured).casefold()
    candidate = models.get(fallback_tier)
    if not _is_callable_tier(fallback_tier, cfg):
        return None
    return str(candidate) if candidate and candidate != active_model else None


def _quota_fallback_effort(cfg: Dict[str, Any], fallback_tier: str) -> str:
    """Use a quota-recovery-specific effort without altering normal tier routes."""
    configured = (cfg.get("quota_fallbacks") or {}).get("spark") or {}
    if isinstance(configured, dict) and configured.get("effort"):
        return str(configured["effort"])
    return str((cfg.get("effort") or {}).get(fallback_tier) or "medium")


def _is_model_unavailable(error: BaseException) -> bool:
    """A model this account cannot use at all, as opposed to one that is busy.

    Distinct from quota (recovers) and from a 5xx (a blip): the provider is saying
    the model does not exist for these credentials, so retrying it later in the
    same session is pointless. Matched on the message rather than the status code
    because the same refusal arrives as 400 and as 404 depending on the endpoint.
    """
    text = str(error).casefold()
    if not any(code in text for code in ("400", "404")):
        return False
    return any(
        marker in text
        for marker in (
            "is not supported when using",
            "model is not supported",
            "model not supported",
            "does not exist or you do not have access",
            "model not found",
            "no access to model",
            "is not available for your",
        )
    )


def _durable_fallback_model(
    active_model: str, cfg: Dict[str, Any], request: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Walk the CONFIGURED ``fallbacks`` chain for a model this account cannot use.

    Deliberately not ``_transient_fallback_model``: that one holds a hardcoded map
    for provider blips, while this case is the operator's own substitution policy —
    if the config says ``spark: luna``, a Spark that does not exist here belongs on
    Luna and nowhere else. Design and image guards still apply, because an
    unavailable model is no reason to violate a routing policy.
    """
    models = cfg.get("models") or {}
    active_tier = next((tier for tier, model in models.items() if model == active_model), "")
    if not active_tier:
        return None
    if (
        active_tier == "sol"
        and isinstance(request, dict)
        and _is_design_request(_last_user_text_and_index(_request_items(request))[0])
    ):
        return None
    chain = cfg.get("fallbacks") or {}
    visited = {active_tier}
    current = active_tier
    for _ in range(3):
        nxt = str(chain.get(current) or "")
        if not nxt or nxt in visited:
            return None
        visited.add(nxt)
        candidate = models.get(nxt)
        if candidate and candidate != active_model and _is_callable_tier(nxt, cfg):
            if (
                nxt == "spark"
                and isinstance(request, dict)
                and _request_has_image_attachment(request)
            ):
                current = nxt
                continue
            return str(candidate)
        current = nxt
    return None


def _transient_fallback_model(active_model: str, cfg: Dict[str, Any], request: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Choose one fallback without violating the Sol-only design boundary."""
    models = cfg.get("models") or {}
    if (
        active_model == models.get("sol")
        and isinstance(request, dict)
        and _is_design_request(_last_user_text_and_index(_request_items(request))[0])
    ):
        # A provider outage must not downgrade design analysis/implementation
        # to Terra or Spark. Preserve the original error for a later Sol retry.
        return None
    fallback_tier = {
        "luna": "spark",
        "spark": "sol",
        "terra": "sol",
        "sol": "spark",
    }.get(next((tier for tier, model in models.items() if model == active_model), ""))
    candidate = models.get(fallback_tier) if fallback_tier else None
    if fallback_tier and not _is_callable_tier(fallback_tier, cfg):
        candidate = None
    if candidate == models.get("spark") and isinstance(request, dict) and _request_has_image_attachment(request):
        image_fallback_tier = "terra" if active_model != models.get("terra") else "sol"
        candidate = models.get(image_fallback_tier) if _is_callable_tier(image_fallback_tier, cfg) else None
    return str(candidate) if candidate and candidate != active_model else None


def _opus5_write_intent(text: str) -> bool:
    """Return whether the coding request explicitly asks for repo mutation."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return bool(re.search(
        r"\b(implement|fix|change|modify|edit|write|add|remove|refactor|migrate|"
        r"javits|javitas|modosits|modositas|implemental|keszits|hozzaad|torol|ird at)\b",
        normalized,
    ))


def _recent_verified_opus5_route(cfg: Dict[str, Any]) -> bool:
    """Use local route evidence as a cheap, credential-free availability gate."""
    coding_cfg = cfg.get("coding_agent") or {}
    policy = coding_cfg.get("explicit_ui") or {}
    ttl = max(1, int(policy.get("require_recent_verified_probe_seconds", 86400) or 86400))
    path = Path(os.path.expanduser(str((cfg.get("logging") or {}).get("path", _DEFAULT_CONFIG["logging"]["path"]))))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    now = datetime.now(timezone.utc)
    for line in reversed(lines[-5000:]):
        try:
            event = json.loads(line)
            if event.get("tier") != "opus5" or event.get("model") != "claude-opus-5":
                continue
            observed = datetime.fromisoformat(str(event.get("timestamp") or "").replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            return (now - observed).total_seconds() <= ttl
        except Exception:
            continue
    return False


def _opus5_repo_for_request(text: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Resolve only configured local roots; never discover or read credentials."""
    coding_cfg = cfg.get("coding_agent") or {}
    normalised = _normalise(text)
    aliases = coding_cfg.get("repo_aliases") or {}
    for alias, value in aliases.items():
        if _normalise(str(alias)) in normalised:
            candidate = Path(str(value)).expanduser()
            return candidate.resolve() if candidate.is_dir() else None
    value = str(coding_cfg.get("default_repo") or "").strip()
    candidate = Path(value).expanduser() if value else None
    return candidate.resolve() if candidate is not None and candidate.is_dir() else None


def _verified_explicit_opus5_ui_repo(text: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Return a repo only after all cheap local bridge eligibility checks pass."""
    coding_cfg = cfg.get("coding_agent") or {}
    canonical = str(coding_cfg.get("canonical_model") or coding_cfg.get("model") or "")
    if (
        not coding_cfg.get("enabled")
        or canonical != "claude-opus-5"
        or not _is_explicit_bounded_opus_ui_request(text, cfg)
        or shutil.which("claude") is None
        or not _recent_verified_opus5_route(cfg)
    ):
        return None
    return _opus5_repo_for_request(text, cfg)


def _verified_explicit_opus5_review_repo(text: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """Permit an explicit, bounded Opus review with no write capability."""
    coding_cfg = cfg.get("coding_agent") or {}
    reviewer_cfg = coding_cfg.get("reviewer") or {}
    canonical = str(coding_cfg.get("canonical_model") or coding_cfg.get("model") or "")
    if (
        not coding_cfg.get("enabled")
        or not reviewer_cfg.get("enabled")
        or canonical != "claude-opus-5"
        or len(text or "") > int(reviewer_cfg.get("max_chars", 8000) or 8000)
        or shutil.which("claude") is None
        or not _recent_verified_opus5_route(cfg)
    ):
        return None
    from .claude_opus_bridge import classify_review_dispatch

    eligible, _reason = classify_review_dispatch(text)
    return _opus5_repo_for_request(text, cfg) if eligible else None


def _verified_delegated_claude_review(text: str, cfg: Dict[str, Any]) -> Optional[Tuple[Path, str]]:
    """Resolve a delegated, read-only Claude review leaf to (repo, Claude tier).

    Deliberately independent of ``coding_agent.enabled``. That switch also arms
    the conservative coding classifier, which fires with no explicit label and
    would capture the first call of a coding turn -- the reason the whole bridge
    is off. This path needs none of that: it requires an explicit review label
    the planner had to write, and it is the caller's job to admit only delegated
    workers, so a root turn can never be diverted into a subprocess.
    """
    coding_cfg = cfg.get("coding_agent") or {}
    policy = coding_cfg.get("delegated_review") or {}
    if (
        not policy.get("enabled")
        or len(text or "") > int(policy.get("max_chars", 8000) or 8000)
        or shutil.which("claude") is None
    ):
        return None
    from .claude_opus_bridge import CLAUDE_REVIEW_MODELS, review_model_alias

    alias = review_model_alias(text)
    if alias is None:
        return None
    allowed = policy.get("models")
    if isinstance(allowed, list) and alias not in [str(name).casefold() for name in allowed]:
        return None
    if alias not in CLAUDE_REVIEW_MODELS:
        return None
    repo = _opus5_repo_for_request(text, cfg)
    return (repo, alias) if repo is not None else None


def _run_opus5_bridge(*, repo: str, task: str, write: bool, review: bool = False, cfg: Dict[str, Any],
                      model: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    """Lazy bridge import keeps the standalone CLI and package imports independent."""
    from .claude_opus_bridge import dispatch

    coding_cfg = cfg.get("coding_agent") or {}
    return dispatch(
        task,
        Path(repo),
        write=write,
        review=review,
        model=model,
        timeout=int(coding_cfg.get("timeout_seconds", 300)),
    )


def _opus5_response(result: Dict[str, Any]) -> Any:
    """Adapt a verified Claude Code result to Hermes' Codex Responses contract."""
    from .claude_opus_bridge import CLAUDE_REVIEW_MODELS

    text = str(result.get("result") or "").strip()
    model = str(result.get("effective_model") or result.get("model") or "")
    if not text or model not in set(CLAUDE_REVIEW_MODELS.values()):
        raise RuntimeError(f"Claude bridge returned no verified result; effective model was {model or 'missing'}")
    raw_usage = result.get("usage") or result.get("model_usage") or {}
    input_tokens = int(raw_usage.get("input_tokens", raw_usage.get("inputTokens", 0)) or 0)
    output_tokens = int(raw_usage.get("output_tokens", raw_usage.get("outputTokens", 0)) or 0)
    return SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=text)],
        )],
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        status="completed",
        model=model,
    )


def _maybe_run_opus5(request: Dict[str, Any], cfg: Dict[str, Any], **kwargs: Any) -> Optional[Any]:
    """Execute the first safe, non-design coding call through Claude Code OAuth."""
    coding_cfg = cfg.get("coding_agent") or {}
    # Hard gate before any bridge import, auth probe, or subprocess launch.
    if not _is_callable_tier("opus5", cfg):
        return None
    if int(kwargs.get("api_call_count") or 1) != 1:
        return None
    if str(kwargs.get("api_mode") or "") != "codex_responses":
        return None

    source_request = kwargs.get("original_request")
    if not isinstance(source_request, dict):
        source_request = request
    text = _last_user_text_and_index(_request_items(source_request))[0]

    # A delegated review leaf runs on Claude and returns its verdict as the
    # leaf's answer. Restricted to delegated workers: the documented hazard of
    # this bridge is that it captures the first call of a turn, which matters
    # only for the parent that still has to plan.
    if (
        str(kwargs.get("platform", "")).casefold() == "subagent"
        or ":sa-" in str(kwargs.get("turn_id", ""))
    ):
        delegated = _verified_delegated_claude_review(text, cfg)
        if delegated is not None:
            repo, alias = delegated
            result = _run_opus5_bridge(
                repo=str(repo),
                task=text,
                write=False,
                review=True,
                model=alias,
                cfg=cfg,
                turn_id=str(kwargs.get("turn_id") or ""),
                provider=str(kwargs.get("provider") or ""),
            )
            return _opus5_response(result)

    if not coding_cfg.get("enabled"):
        return None
    review_repo = _verified_explicit_opus5_review_repo(text, cfg)
    if review_repo is not None:
        result = _run_opus5_bridge(
            repo=str(review_repo),
            task=text,
            write=False,
            review=True,
            cfg=cfg,
            turn_id=str(kwargs.get("turn_id") or ""),
            provider=str(kwargs.get("provider") or ""),
        )
        return _opus5_response(result)
    explicit_ui_repo = _verified_explicit_opus5_ui_repo(text, cfg)
    if explicit_ui_repo is not None:
        result = _run_opus5_bridge(
            repo=str(explicit_ui_repo),
            task=f"[opus5] {text}",
            write=_opus5_write_intent(text),
            cfg=cfg,
            turn_id=str(kwargs.get("turn_id") or ""),
            provider=str(kwargs.get("provider") or ""),
        )
        return _opus5_response(result)

    decision = classify_request(source_request, api_call_count=1)
    if decision.tier != "terra":
        return None

    from .claude_opus_bridge import classify_coding_dispatch

    eligible, _reason = classify_coding_dispatch(text)
    if not eligible:
        return None
    repo = str(coding_cfg.get("default_repo") or "").strip()
    repo_path = Path(repo).expanduser() if repo else None
    if repo_path is None or not repo_path.is_dir():
        return None

    result = _run_opus5_bridge(
        repo=str(repo_path.resolve()),
        task=text,
        write=_opus5_write_intent(text),
        cfg=cfg,
        turn_id=str(kwargs.get("turn_id") or ""),
        api_request_id=str(kwargs.get("api_request_id") or ""),
        provider=str(kwargs.get("provider") or ""),
    )
    return _opus5_response(result)


def run_llm_with_transient_failover(**kwargs: Any) -> Any:
    """Retry one transient provider failure with a different routed model.

    This is execution middleware, so it runs around the actual provider call;
    request middleware alone cannot recover an exception after the request was
    sent. The original exception is preserved for auth, validation and other
    non-transient failures.
    """
    request = kwargs.get("request")
    next_call = kwargs.get("next_call")
    # Current Hermes exposes an LLM-only retry callback so providers can be
    # retried without violating the single-use downstream ``next_call`` contract.
    # Keep the direct callable fallback for unit-level and older-host compatibility.
    retry_call = kwargs.get("retry_call") or next_call
    cfg = _load_config()
    provider = str(kwargs.get("provider", "")).casefold()
    configured_provider = str(cfg.get("provider", "openai-codex")).casefold()
    if not isinstance(request, dict) or not callable(next_call) or provider != configured_provider:
        if not isinstance(request, dict) or not callable(next_call):
            return next_call(request)
        # The guard below exists because this middleware rewrites request["model"]
        # within one provider. Noticing that an account just refused a call needs
        # none of that, and skipping it here is why a Qwen weekly-quota 429 left
        # no cooldown -- the conductor was still being told that account was idle.
        try:
            return next_call(request)
        except Exception as error:
            if _is_quota_exhaustion(error) or _is_transient_provider_failure(error):
                failing_model = str(request.get("model", ""))
                _record_tier_failure(
                    next(
                        (tier for tier, model in (cfg.get("models") or {}).items()
                         if model == failing_model),
                        "",
                    ) or (_external_target_for_model(failing_model) or ""),
                    cfg,
                    quota=_is_quota_exhaustion(error),
                    error=error,
                )
            raise

    opus_context = {key: value for key, value in kwargs.items() if key not in {"request", "next_call", "retry_call"}}
    try:
        opus_response = _maybe_run_opus5(request, cfg, **opus_context)
    except Exception:
        # OAuth, entitlement and bridge failures fall back to the normal route.
        # A failed attempt must never fabricate an Opus viewer/log entry.
        opus_response = None
    if opus_response is not None:
        return opus_response

    try:
        return next_call(request)
    except Exception as error:
        active_model = str(request.get("model", ""))
        quota_exhausted = _is_quota_exhaustion(error)
        # A model this account cannot use at all. Neither a quota (which recovers)
        # nor a blip (which is worth retrying): switched on in the dashboard but
        # refused by the provider, it would otherwise abort every leaf routed to it.
        unavailable = not quota_exhausted and _is_model_unavailable(error)
        active_tier = next(
            (tier for tier, model in (cfg.get("models") or {}).items() if model == active_model), "",
        )
        # Record before deciding what to do about it. The gap this closes is the
        # case with no fallback configured, where the old code re-raised and left
        # nothing behind -- the next call walked into the same exhausted account.
        if quota_exhausted or _is_transient_provider_failure(error):
            _record_tier_failure(cfg=cfg, tier=active_tier, quota=quota_exhausted, error=error)
        elif unavailable and active_tier:
            # Long, and stated plainly: nothing about this recovers by waiting, so
            # the point of the cooldown is to stop offering the tier this session.
            _enter_cooldown(
                active_tier, cfg,
                seconds=float((cfg.get("cooldown") or {}).get("unavailable_seconds", 21600) or 21600),
                reason="model unavailable on this account",
            )
        if quota_exhausted:
            fallback_model = _quota_fallback_model(active_model, cfg)
        elif unavailable:
            fallback_model = _durable_fallback_model(active_model, cfg, request)
        else:
            fallback_model = _transient_fallback_model(active_model, cfg, request)
        if not fallback_model or not (
            quota_exhausted or unavailable or _is_transient_provider_failure(error)
        ):
            raise
        if quota_exhausted:
            _remember_spark_quota_exhaustion(str(kwargs.get("turn_id") or ""))

        fallback_request = dict(request)
        fallback_request["model"] = fallback_model
        fallback_tier = next(
            (tier for tier, model in (cfg.get("models") or {}).items() if model == fallback_model),
            "fallback",
        )
        reasoning = dict(fallback_request.get("reasoning") or {})
        reasoning["effort"] = (
            _quota_fallback_effort(cfg, fallback_tier)
            if quota_exhausted
            else str((cfg.get("effort") or {}).get(fallback_tier) or "medium")
        )
        fallback_request["reasoning"] = reasoning
        _log_decision(
            RouteDecision(
                fallback_tier,
                fallback_model,
                (
                    f"Spark quota exhausted; failover from {active_model}"
                    if quota_exhausted
                    else f"{active_model} unavailable on this account; configured failover"
                    if unavailable
                    else f"transient failover from {active_model}"
                ),
            ),
            {**kwargs, "request": fallback_request},
            cfg,
        )
        return retry_call(fallback_request)


def on_post_llm_call(**kwargs: Any) -> None:
    """Persist supervisor checkpoints and the legacy disabled benchmark lifecycle."""
    cfg = _load_config()
    turn_id = str(kwargs.get("turn_id") or "")
    orchestrated = _orchestration_forced_event(cfg, turn_id)
    if orchestrated:
        response = kwargs.get("assistant_response") or ""
        _orchestration_event(
            cfg,
            {
                "event": "terra_checkpoint_completed",
                "plan_id": orchestrated.get("plan_id"),
                "turn_id": turn_id,
                "parent_model": kwargs.get("model") or orchestrated.get("parent_model", "terra"),
                "supervisor_decision_present": "supervisor decision:" in str(response).casefold(),
                "response_sha256": _summary_digest(response),
            },
        )
    forced = _shadow_forced_event(cfg, turn_id)
    if not forced:
        return
    response = kwargs.get("assistant_response") or ""
    _shadow_event(
        cfg,
        {
            "event": "parent_completed",
            "benchmark_id": forced.get("benchmark_id") or _benchmark_id(turn_id),
            "turn_id": turn_id,
            "parent_model": kwargs.get("model") or forced.get("parent_model", "terra"),
            "parent_response_chars": len(str(response)),
            "parent_response_sha256": _summary_digest(response),
        },
    )


def on_subagent_start(**kwargs: Any) -> None:
    """Attach child lifecycle records to real supervisor work and legacy shadows."""
    cfg = _load_config()
    parent_turn_id = str(kwargs.get("parent_turn_id") or "")
    orchestrated = _orchestration_forced_event(cfg, parent_turn_id)
    if orchestrated:
        _orchestration_event(
            cfg,
            {
                "event": "spark_child_started",
                "plan_id": orchestrated.get("plan_id"),
                "turn_id": parent_turn_id,
                "child_session_id": kwargs.get("child_session_id"),
                "child_subagent_id": kwargs.get("child_subagent_id"),
                "child_goal_preview": _redacted_preview(kwargs.get("child_goal") or "", limit=240),
            },
        )
    forced = _shadow_forced_event(cfg, parent_turn_id)
    if not forced:
        return
    _shadow_event(
        cfg,
        {
            "event": "child_started",
            "benchmark_id": forced.get("benchmark_id") or _benchmark_id(parent_turn_id),
            "turn_id": parent_turn_id,
            "parent_model": forced.get("parent_model", "terra"),
            "shadow_model": forced.get("shadow_model", "spark"),
            "child_session_id": kwargs.get("child_session_id"),
            "child_subagent_id": kwargs.get("child_subagent_id"),
            "child_role": kwargs.get("child_role"),
            "child_goal_preview": _redacted_preview(kwargs.get("child_goal") or "", limit=240),
        },
    )


def on_subagent_stop(**kwargs: Any) -> None:
    """Persist real supervisor child outcomes and legacy benchmark outcomes."""
    cfg = _load_config()
    parent_turn_id = str(kwargs.get("parent_turn_id") or "")
    child_session_id = kwargs.get("child_session_id")
    summary = kwargs.get("child_summary") or ""
    orchestrated = _orchestration_forced_event(cfg, parent_turn_id)
    if orchestrated:
        _orchestration_event(
            cfg,
            {
                "event": "spark_child_completed",
                "plan_id": orchestrated.get("plan_id"),
                "turn_id": parent_turn_id,
                "child_session_id": child_session_id,
                "child_status": kwargs.get("child_status"),
                "child_model": kwargs.get("child_model"),
                "child_api_calls": kwargs.get("child_api_calls"),
                "summary_sha256": _summary_digest(summary),
            },
        )
    forced = _shadow_forced_event(cfg, parent_turn_id)
    if not forced:
        return
    _shadow_event(
        cfg,
        {
            "event": "child_completed",
            "benchmark_id": forced.get("benchmark_id") or _benchmark_id(parent_turn_id),
            "turn_id": parent_turn_id,
            "child_session_id": child_session_id,
            "child_status": kwargs.get("child_status"),
            "duration_ms": kwargs.get("duration_ms"),
            "child_model": kwargs.get("child_model"),
            "child_api_calls": kwargs.get("child_api_calls"),
            "input_tokens": kwargs.get("input_tokens"),
            "output_tokens": kwargs.get("output_tokens"),
            "cost_usd": kwargs.get("cost_usd"),
            "exit_reason": kwargs.get("exit_reason"),
            "summary_chars": len(str(summary)),
            "summary_sha256": _summary_digest(summary),
        },
    )


def register(ctx: Any) -> None:
    ctx.register_middleware("llm_request", route_llm_request)
    ctx.register_middleware("llm_execution", run_llm_with_transient_failover)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
