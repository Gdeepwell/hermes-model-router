from unittest.mock import patch

import pytest

from model_router import RouteDecision, _log_decision, route_llm_request, run_llm_with_transient_failover


MODELS = {
    "luna": "gpt-5.6-luna",
    "spark": "gpt-5.3-codex-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-5.6-sol",
    "qwen": "qwen3.7-plus",
}


def request(text, model="gpt-5.6-terra"):
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def config(**overrides):
    result = {
        "enabled": True,
        "provider": "openai-codex",
        "models": MODELS,
        "default_model": "terra",
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True, "opus5": True, "qwen": True},
        "effort": {"luna": "low", "spark": "low", "terra": "medium", "sol": "medium", "qwen": "medium"},
    }
    result.update(overrides)
    return result


def test_disabled_opus5_skips_bridge_before_import_or_auth():
    cfg = config(
        callable={**config()["callable"], "opus5": False},
        coding_agent={"enabled": True, "default_repo": "/tmp"},
    )
    with patch("model_router._load_config", return_value=cfg), patch("model_router._run_opus5_bridge") as bridge:
        response = run_llm_with_transient_failover(
            request=request("Implement the parser fix and add tests."),
            next_call=lambda value: value,
            provider="openai-codex",
            api_mode="codex_responses",
            api_call_count=1,
        )
    assert response["model"] == "gpt-5.6-terra"
    bridge.assert_not_called()


def test_disabled_spark_is_rerouted_to_a_live_fallback():
    cfg = config(
        callable={**config()["callable"], "spark": False},
        fallbacks={"spark": "terra"},
    )
    with patch("model_router._load_config", return_value=cfg):
        routed = route_llm_request(
            request=request("[spark] Inspect the parser configuration only.", MODELS["spark"]),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=1,
        )
    assert routed["metadata"]["tier"] == "terra"
    assert routed["request"]["model"] == MODELS["terra"]


def test_qwen_final_request_strips_unsupported_tool_controls():
    cfg = config(
        default_model="qwen",
        tier_providers={"qwen": "qwen-token", "terra": "openai-codex"},
    )
    qwen_request = request("Continue this conversation.", MODELS["qwen"])
    qwen_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
    qwen_request["tool_choice"] = "required"
    qwen_request["parallel_tool_calls"] = False
    with patch("model_router._load_config", return_value=cfg):
        routed = route_llm_request(
            request=qwen_request,
            provider="openai-codex",
            model=MODELS["qwen"],
            api_call_count=2,
        )
    assert routed["metadata"]["tier"] == "qwen"
    assert routed["request"]["tools"] == qwen_request["tools"]
    assert "tool_choice" not in routed["request"]
    assert "parallel_tool_calls" not in routed["request"]


def test_disabled_and_missing_callable_tiers_reroute_before_dispatch():
    for tier in ("spark", "luna", "sol"):
        live = {**config()["callable"], tier: False}
        cfg = config(callable=live, fallbacks={tier: "terra"})
        decision = RouteDecision(tier, MODELS[tier], "explicit test route", "medium")
        with (
            patch("model_router._load_config", return_value=cfg),
            patch("model_router.classify_request", return_value=decision),
        ):
            routed = route_llm_request(
                request=request("Test disabled routing.", MODELS[tier]),
                provider="openai-codex",
                model=MODELS[tier],
                api_call_count=1,
            )
        assert routed["metadata"]["tier"] == "terra"
        assert routed["request"]["model"] == MODELS["terra"]

    live = config()["callable"].copy()
    live.pop("luna")
    cfg = config(callable=live, fallbacks={"luna": "terra"})
    decision = RouteDecision("luna", MODELS["luna"], "missing callable entry", "low")
    with (
        patch("model_router._load_config", return_value=cfg),
        patch("model_router.classify_request", return_value=decision),
    ):
        routed = route_llm_request(
            request=request("Test missing routing entry.", MODELS["luna"]),
            provider="openai-codex",
            model=MODELS["luna"],
            api_call_count=1,
        )
    assert routed["metadata"]["tier"] == "terra"


def test_no_callable_fallback_fails_closed_before_dispatch():
    cfg = config(
        default_model="terra",
        callable={tier: False for tier in config()["callable"]},
        fallbacks={"qwen": "terra"},
    )
    decision = RouteDecision("qwen", MODELS["qwen"], "explicit test route", "medium")
    with (
        patch("model_router._load_config", return_value=cfg),
        patch("model_router.classify_request", return_value=decision),
        pytest.raises(RuntimeError, match="No enabled ModelRouter tier"),
    ):
        route_llm_request(
            request=request("Must fail closed.", MODELS["qwen"]),
            provider="qwen-token",
            model=MODELS["qwen"],
            api_call_count=1,
        )


def test_qwen_sanitization_runs_after_forced_preflight_rewrite():
    cfg = config(
        default_model="terra",
        tier_providers={"qwen": "qwen-token", "terra": "openai-codex"},
    )
    qwen_request = request("Implement and verify this parser change.", MODELS["qwen"])
    qwen_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
    forced = {
        **qwen_request,
        "model": MODELS["terra"],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "reasoning": {"effort": "medium"},
    }
    terra = RouteDecision("terra", MODELS["terra"], "forced supervisor", "medium")
    with (
        patch("model_router._load_config", return_value=cfg),
        patch("model_router.classify_request", return_value=terra),
        patch("model_router._force_terra_supervisor_preflight", return_value=forced) as preflight,
    ):
        routed = route_llm_request(
            request=qwen_request,
            provider="qwen-token",
            model=MODELS["qwen"],
            api_call_count=1,
        )

    preflight.assert_called_once()
    assert routed["metadata"]["tier"] == "qwen"
    assert routed["request"]["model"] == MODELS["qwen"]
    assert routed["request"]["tools"] == qwen_request["tools"]
    assert "tool_choice" not in routed["request"]
    assert "parallel_tool_calls" not in routed["request"]
    assert "reasoning" not in routed["request"]


def test_root_parent_is_pinned_when_classifier_wants_sol_worker():
    """A user-facing conversation must not silently become a cold Sol turn."""
    cfg = config(
        session_policy={"pin_root_parent": True},
        orchestration={"enabled": True, "min_chars": 1, "max_tasks": 3},
    )
    root_request = request("Készíts CSS elrendezést a kártyához.")
    root_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
    with patch("model_router._load_config", return_value=cfg):
        routed = route_llm_request(
            request=root_request,
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=1,
            turn_id="stable-parent-turn",
        )
    assert routed["metadata"]["tier"] == "terra"
    assert routed["request"]["model"] == MODELS["terra"]
    assert "pinned" in routed["reason"]
    assert "tool_choice" not in routed["request"]


def test_route_log_uses_bounded_redacted_preview_when_policy_enabled(tmp_path):
    path = tmp_path / "router.jsonl"
    cfg = {
        "logging": {
            "enabled": True,
            "path": str(path),
            "prompt_preview_chars": 24,
            "redact_prompt_preview": True,
        }
    }
    _log_decision(
        RouteDecision("terra", MODELS["terra"], "test"),
        {"turn_id": "safe-log-turn", "request": request("token=super-secret-value " + "x" * 80)},
        cfg,
    )
    record = __import__("json").loads(path.read_text())
    assert "super-secret-value" not in record["prompt_preview"]
    assert len(record["prompt_preview"]) <= 24
