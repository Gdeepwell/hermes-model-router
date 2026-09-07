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
        "reviewer": {"enabled": False, "max_chars": 8000},
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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_callable_fallback(
    decision: RouteDecision, cfg: Dict[str, Any]
) -> RouteDecision:
    """If the chosen tier is not callable, follow the fallback chain once."""
    callable_tiers = (cfg.get("callable") or {}).copy()
    # Fail closed: a tier is routable only when the live config explicitly says
    # callable: true. Missing entries must never silently re-enable a model.
    for tier in ("luna", "spark", "terra", "sol", "opus5", "qwen"):
        callable_tiers.setdefault(tier, False)

    chosen_tier = decision.tier
    if callable_tiers.get(chosen_tier, False):
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


def _is_callable_tier(tier: str, cfg: Dict[str, Any]) -> bool:
    # Live router policy is explicit: missing or malformed entries are disabled.
    return (cfg.get("callable") or {}).get(tier) is True


def _require_callable(decision: RouteDecision, cfg: Dict[str, Any]) -> RouteDecision:
    """Never emit a route for a tier disabled in the dashboard."""
    resolved = _resolve_callable_fallback(decision, cfg)
    if not _is_callable_tier(resolved.tier, cfg):
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
    r"identify|list|check|investigate|research|test[- ]case\s+design|"
    r"nezd\s+meg|olvasd|ellenorizd|elemezd|jelentsd|keresd|azonositsd|"
    r"hasonlitsd\s+ossze|kutass)\b"
)
_SPARK_CONSEQUENTIAL_WORK = re.compile(
    r"\b(production|prod|security|biztonsag|auth(?:entication|orization)?|"
    r"credential|jelszo|password|payment|fizetes|migration|migrate|deploy|"
    r"szerver|server|database|adatbazis)\b"
)


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


def _is_spark_read_only_request(text: str) -> bool:
    """Spark may receive only affirmative, bounded non-design evidence work."""
    affirmative = _normalise(_without_negated_safety_constraints(text))
    return bool(
        affirmative
        and not _is_design_request(affirmative)
        and not _SPARK_MUTATING_WORK.search(affirmative)
        and _SPARK_READ_ONLY_WORK.search(affirmative)
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


def _decision(
    tier: str,
    reason: str,
    cfg: Dict[str, Any],
    *,
    explicit: bool = False,
    effort_key: Optional[str] = None,
) -> RouteDecision:
    if effort_key is None:
        effort_key = "explicit_sol" if explicit and tier == "sol" else tier
    fallback = {"luna": "low", "spark": "low", "terra": "medium", "sol": "high", "qwen": "medium"}[tier]
    effort = str((cfg.get("effort") or {}).get(effort_key) or fallback).casefold()
    return RouteDecision(tier=tier, model=cfg["models"][tier], reason=reason, effort=effort)


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
) -> RouteDecision:
    """Classify one provider request, annotated with the tiers it lost out on."""
    decision = _classify_request(request, api_call_count, config)
    items = _request_items(request)
    user_text, user_index = _last_user_text_and_index(items)
    eligible = _eligible_tiers(
        user_text,
        has_image_attachment=_request_has_image_attachment(request),
        api_call_count=api_call_count,
        is_tool_loop=_current_turn_has_tool_activity(items, user_index),
    )
    vetoed = tuple(tier for tier in eligible if tier != decision.tier)
    return replace(decision, vetoed_by=vetoed) if vetoed else decision


def _classify_request(
    request: Dict[str, Any],
    api_call_count: int = 1,
    config: Optional[Dict[str, Any]] = None,
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
        return _decision("luna", "acknowledgement-only follow-up", cfg)

    # Role separation is a hard policy boundary: only Sol performs visual/product
    # design analysis or design implementation. Terra may coordinate and approve
    # the resulting evidence, while Spark may inspect only non-design read-only
    # facts. This check intentionally precedes manual/benchmark overrides so a
    # label cannot route design work to another tier.
    if _is_design_request(user_text):
        return _decision("sol", "design analysis or implementation is Sol-only", cfg)

    benchmark_force = _normalise(os.getenv("MODEL_ROUTER_BENCHMARK_FORCE_MODEL", ""))
    if benchmark_force in ("luna", "spark", "terra", "sol"):
        if benchmark_force == "spark":
            if has_image_attachment:
                return _decision("terra", "image attachment requires a vision-capable route", cfg)
            if not _is_spark_read_only_request(user_text):
                return _decision("terra", "Spark is restricted to non-design read-only subtasks", cfg)
        return _decision(benchmark_force, "benchmark environment force override", cfg, explicit=True)

    override = re.match(r"^\s*\[(luna|spark|terra|sol)(?::(xhigh))?\](?:\s|$)", text)
    if override:
        tier = override.group(1)
        if tier == "spark" and has_image_attachment:
            return _decision("terra", "image attachment requires a vision-capable route", cfg)
        if tier == "spark" and not _is_spark_read_only_request(user_text):
            if _is_consequential_spark_request(user_text):
                return _decision("sol", "consequential Spark task requires Sol", cfg)
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
        return _decision("sol", f"long request ({len(user_text)} characters)", cfg, effort_key="sol_long")

    # Consequential domains are biased toward Sol even when the prompt is short.
    sensitive = re.compile(
        r"\b(security|biztonsag|vulnerability|sebezhetoseg|malware|"
        r"auth|oauth|authentication|authorization|jogosultsag|"
        r"credential|credentials|belepesi\s+adat\w*|jelsz\w*|password|passwd|"
        r"payment|fizetes|billing|szamlazas|webhook|jogi|legal|"
        r"orvosi|medical|diagnos|gyogyszer|befektetes|investment|adozas|tax)\b"
    )
    if sensitive.search(text):
        return _decision("sol", "sensitive or consequential domain", cfg)

    # High-consequence engineering stays on Sol. Ordinary repository debugging,
    # refactoring and test execution stay on Terra: the implementation benchmark
    # showed that Terra is the safer integration owner, while bounded Spark work
    # remains available through explicit/delegated workers.
    critical_work = re.compile(
        r"\b(migrate|migr(?:al|ald)|deploy(?:ol|old)?|telepitsd|install|production|prod(?:ra|on|ban|ba|ot)?|"
        r"deep research|mely kutatas|kutass reszletesen)\b"
    )
    if critical_work.search(text):
        return _decision("sol", "consequential engineering or research task", cfg)

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
        return _decision("sol", "consequential system action", cfg)

    # Spark is text-only. This hard route sits after the higher-priority Sol
    # safety routes and before every Spark classifier, so an image cannot be
    # routed to Spark by a primary, override, benchmark, or tool-loop branch.
    if has_image_attachment:
        return _decision("terra", "image attachment requires a vision-capable route", cfg)

    # Unlabelled work is never sent directly to Spark. Terra plans and owns the
    # task first; only its explicit [spark] leaf goals may use Spark.  Repeated
    # calls merely preserve Terra ownership rather than reclassifying from
    # prompt keywords.
    if api_call_count > 1:
        if _current_turn_has_tool_activity(items, user_index):
            return _decision("terra", "ordinary current-turn tool loop", cfg)
        return _decision("terra", "repeated current-turn call safety promotion", cfg)

    repo_implementation = re.compile(
        r"\b(debug|debugold|hibakeres|traceback|stack trace|root cause|"
        r"refactor|teszteld|run the tests|futtasd a teszt|javitsd|fix|"
        r"implement|repo|kod|code)\b"
    )
    if repo_implementation.search(text):
        return _decision("terra", "normal repository implementation owner", cfg)

    if re.search(r"\b(csinald meg|hajtsd vegre|do it|make the changes|folytasd)\b", text):
        return _decision("terra", "context-dependent action owned by Terra", cfg)



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
            return _decision("luna", "greeting or acknowledgement", cfg)
        if simple_transform.search(text) and "```" not in user_text:
            return _decision("luna", "simple language transformation", cfg)
        if simple_definition.search(text):
            return _decision("luna", "short definition request", cfg)
        if brief_chat.search(text):
            return _decision("luna", "brief non-actionable conversation", cfg)
        if short_explanation.search(text) and not technical.search(text):
            return _decision("luna", "short low-risk explanation", cfg)

    return _decision(
        str(cfg.get("default_model", "terra")),
        "default general-purpose route",
        cfg,
    )


def _log_decision(decision: RouteDecision, kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("enabled", True):
        return
    path = Path(os.path.expanduser(str(log_cfg.get("path", "~/.hermes/logs/model-router.jsonl"))))
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
    delegate_tool = next(
        tool
        for tool in shadow.get("tools") or []
        if isinstance(tool, dict)
        and (tool.get("name") == "delegate_task" or (tool.get("function") or {}).get("name") == "delegate_task")
    )
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


def _find_delegate_tool(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the delegate_task tool definition in either wire shape."""
    for tool in request.get("tools") or []:
        if isinstance(tool, dict) and (
            tool.get("name") == "delegate_task"
            or (tool.get("function") or {}).get("name") == "delegate_task"
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


def _model_param_contract(orchestrator_tier: str) -> str:
    """The sentence that makes route choice expressible instead of implied.

    A ``[sol]``/``[spark]`` goal prefix is only a model rename inside the
    default provider, so it can never reach a target that lives on a separate
    account. Without this, every leaf inherits the default route: the delegation
    registry shows every child ever spawned running on the default model, even
    ones whose goal was explicitly prefixed for another target.
    """
    names = [name for name in _delegation_target_names() if name != orchestrator_tier]
    scope = f" (available targets: {', '.join(names)})" if names else ""
    return (
        f"Set the delegate_task 'model' parameter on every worker to choose its route{scope}. "
        "A goal-text prefix only renames the model inside the default provider and cannot reach a "
        "target on a separate account, so a leaf intended for one must carry model:<name>. Prefer "
        "spreading genuinely independent leaves across different targets so separate accounts and "
        "quotas absorb the work in parallel; never split work merely to use more targets."
    )


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
    # Determine orchestrator tier from config (respects default_model set in web UI)
    orchestrator_tier = "qwen"  # default
    if cfg:
        orchestrator_tier = str(cfg.get("default_model", "qwen"))
    
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
        f"{_model_param_contract(orchestrator_tier)} "
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
    delegate_tool = next(
        tool
        for tool in routed.get("tools") or []
        if isinstance(tool, dict)
        and (tool.get("name") == "delegate_task" or (tool.get("function") or {}).get("name") == "delegate_task")
    )
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
                f"You are the {orchestrator_tier} planning conductor. Do not perform design analysis or design implementation. Route every visual/product/UI/UX/CSS/layout/design-system task to Sol with a goal beginning [sol] and model:sol. Delegate only bounded, self-contained low-risk non-design read-only evidence loops to Spark with a goal beginning [spark] and model:spark. {_model_param_contract(orchestrator_tier)} The orchestrator retains coordination, evidence acceptance/rejection, integration, and final approval. Use zero leaves only when the objective genuinely has no independently useful non-design text-only investigation, test, source-discovery, or research subtask."
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
    delegate_tool = next(
        tool for tool in routed.get("tools") or []
        if isinstance(tool, dict)
        and (tool.get("name") == "delegate_task" or (tool.get("function") or {}).get("name") == "delegate_task")
    )
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
    orchestration_tiers = {"sol"} | {str(cfg.get("default_model", "terra"))}
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
    tools = request.get("tools") or []
    if not any(
        isinstance(tool, dict)
        and (tool.get("name") == "delegate_task" or (tool.get("function") or {}).get("name") == "delegate_task")
        for tool in tools
    ):
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
    tools = request.get("tools") or []
    has_delegate_tool = any(
        isinstance(tool, dict)
        and (tool.get("name") == "delegate_task" or (tool.get("function") or {}).get("name") == "delegate_task")
        for tool in tools
    )
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
    if active_model not in supported_models:
        return None
    if not isinstance(request, dict):
        return None

    decision = classify_request(
        request,
        api_call_count=int(kwargs.get("api_call_count", 1) or 1),
        config=cfg,
    )

    decision = _require_callable(decision, cfg)

    subagent_marker = (
        str(kwargs.get("platform", "")).casefold() == "subagent"
        or ":sa-" in str(kwargs.get("turn_id", ""))
    )
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
    ):
        # The delegation default is Terra so the forced first child is a real
        # planner. A planner may deliberately label bounded leaves [spark] or
        # consequential leaves [sol]; unlabelled child work remains with the
        # default_model rather than being silently demoted before it can
        # decompose the task.
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


def _run_opus5_bridge(*, repo: str, task: str, write: bool, review: bool = False, cfg: Dict[str, Any], **context: Any) -> Dict[str, Any]:
    """Lazy bridge import keeps the standalone CLI and package imports independent."""
    from .claude_opus_bridge import dispatch

    coding_cfg = cfg.get("coding_agent") or {}
    return dispatch(
        task,
        Path(repo),
        write=write,
        review=review,
        timeout=int(coding_cfg.get("timeout_seconds", 300)),
    )


def _opus5_response(result: Dict[str, Any]) -> Any:
    """Adapt a verified Claude Code result to Hermes' Codex Responses contract."""
    text = str(result.get("result") or "").strip()
    model = str(result.get("effective_model") or result.get("model") or "")
    if not text or model != "claude-opus-5":
        raise RuntimeError("Claude Opus bridge returned no verified claude-opus-5 result")
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
    if not coding_cfg.get("enabled") or int(kwargs.get("api_call_count") or 1) != 1:
        return None
    if str(kwargs.get("api_mode") or "") != "codex_responses":
        return None

    source_request = kwargs.get("original_request")
    if not isinstance(source_request, dict):
        source_request = request
    text = _last_user_text_and_index(_request_items(source_request))[0]
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
        return next_call(request)

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
        fallback_model = (
            _quota_fallback_model(active_model, cfg)
            if quota_exhausted
            else _transient_fallback_model(active_model, cfg, request)
        )
        if not fallback_model or (not quota_exhausted and not _is_transient_provider_failure(error)):
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
