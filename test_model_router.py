import json
import tempfile

import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import (
    RouteDecision,
    _log_decision,
    _lifecycle_event_kind,
    _prompt_preview,
    classify_request,
    on_post_llm_call,
    on_subagent_start,
    on_subagent_stop,
    route_llm_request,
    run_llm_with_transient_failover,
)


MODELS = {
    "luna": "gpt-5.6-luna",
    "spark": "gpt-5.3-codex-spark",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-5.6-sol",
}
CALLABLE = {**{tier: True for tier in MODELS}, "opus5": True, "qwen": True}


def default_test_config():
    return {
        "enabled": True,
        "provider": "openai-codex",
        "models": MODELS,
        "callable": CALLABLE,
        "effort": {"luna": "low", "spark": "low", "terra": "medium", "sol": "medium"},
        "quota_fallbacks": {"spark": {"model": "luna", "effort": "medium"}},
    }

# Deliberately carries none of the old multi-step marker words: the planner must
# be forced by the turn being actionable, not by keyword matching. Long enough to
# clear orchestration.min_chars, which gates fan-out on decomposable work only.
ACTIONABLE_TERRA_PROMPT = (
    "Nézd meg, mi a baj. Tegnap óta más az eredmény, és nem tudom eldönteni, hogy a bemenet "
    "változott-e meg vagy a feldolgozás. Nézd át a vonatkozó részeket, és mondd meg, mit találsz, "
    "mielőtt bármit módosítanánk rajta."
)


def chat_request(text, *, with_tool_result=False):
    messages = [{"role": "user", "content": text}]
    if with_tool_result:
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ])
    return {"model": MODELS["terra"], "messages": messages, "temperature": 0.2}


def responses_request(text):
    return {
        "model": MODELS["terra"],
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }],
    }


def chat_request_with_image(text, *, image_type="image_url"):
    request = chat_request(text)
    request["messages"][0]["content"] = [
        {"type": "text", "text": text},
        {"type": image_type, "image_url": "https://example.test/ui.png"},
    ]
    return request


def chat_request_with_historical_image(current_text):
    """A plain current turn after a previous visual turn in the same session."""
    return {
        "model": MODELS["terra"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Korábbi képes kérés."},
                    {"type": "image_url", "image_url": "https://example.test/old-ui.png"},
                ],
            },
            {"role": "assistant", "content": "A képet már elemeztem."},
            {"role": "user", "content": current_text},
        ],
    }


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        # Router tests construct realistic requests, including preflight fixtures.
        # Never let those fixtures append to the operator's active JSONL route log.
        # Direct _log_decision tests retain their own explicit temporary paths.
        self._route_log_patch = patch("model_router._log_decision")
        self._route_log_patch.start()
        self._config_patch = patch("model_router._load_config", side_effect=default_test_config)
        self._config_patch.start()

    def tearDown(self):
        self._config_patch.stop()
        self._route_log_patch.stop()

    def test_synthetic_completion_log_is_typed_without_copying_its_result_into_provenance(self):
        request = chat_request("[ASYNC DELEGATION COMPLETE — deleg_test] token=not-for-provenance")
        self.assertEqual(_lifecycle_event_kind(request), "async_delegation_completion")

    def test_synthetic_completion_log_keeps_only_a_stable_id_not_raw_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            request = chat_request("[ASYNC DELEGATION COMPLETE — deleg_test] token=not-for-log")
            _log_decision(
                RouteDecision("terra", MODELS["terra"], "test"),
                {"turn_id": "completion-turn", "request": request},
                {"logging": {"enabled": True, "path": str(path)}},
            )
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["event_kind"], "async_delegation_completion")
        self.assertEqual(record["delegation_id"], "deleg_test")
        self.assertEqual(record["prompt_preview"], "Delegált feladat befejezési eseménye")
        self.assertNotIn("not-for-log", json.dumps(record))

    def test_spark_eligible_request_records_the_tier_it_lost_out_on(self):
        """A read-only first call qualifies for Spark; nothing routes it there."""
        decision = classify_request(
            chat_request("Olvasd el a config fájlt és mondd meg, mi van benne. Ne módosíts semmit."),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("spark", decision.vetoed_by)

    def test_chosen_tier_is_never_listed_as_vetoed(self):
        decision = classify_request(chat_request("Készíts CSS elrendezést a kártyához."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertNotIn("sol", decision.vetoed_by)

    def test_terra_is_never_listed_as_vetoed_because_it_is_the_default(self):
        decision = classify_request(chat_request("Szia!"), 1)
        self.assertEqual(decision.tier, "luna")
        self.assertNotIn("terra", decision.vetoed_by)

    def test_vetoed_tiers_reach_the_route_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            _log_decision(
                RouteDecision("terra", MODELS["terra"], "test", "medium", ("spark",)),
                {"turn_id": "veto-turn", "request": chat_request("bármi")},
                {"logging": {"enabled": True, "path": str(path)}},
            )
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["vetoed_by"], ["spark"])

    def test_route_log_omits_vetoed_by_when_nothing_was_preempted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            _log_decision(
                RouteDecision("terra", MODELS["terra"], "test"),
                {"turn_id": "no-veto-turn", "request": chat_request("bármi")},
                {"logging": {"enabled": True, "path": str(path)}},
            )
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("vetoed_by", record)

    def test_terra_is_the_default_for_normal_work(self):
        decision = classify_request(chat_request("Hasonlítsd össze ezt a két megoldást."), api_call_count=1)
        self.assertEqual(decision.tier, "terra")
        self.assertEqual(decision.model, MODELS["terra"])

    def test_benchmark_force_spark_rejects_mutating_work(self):
        with patch.dict("os.environ", {"MODEL_ROUTER_BENCHMARK_FORCE_MODEL": "spark"}):
            decision = classify_request(
                chat_request("Implement a complex regression fix.", with_tool_result=True),
                api_call_count=99,
            )

        self.assertEqual(decision.tier, "terra")
        self.assertIn("read-only", decision.reason)

    def test_luna_handles_clearly_simple_low_risk_requests(self):
        self.assertEqual(classify_request(chat_request("Szia!"), 1).tier, "luna")
        self.assertEqual(
            classify_request(chat_request("Fordítsd angolra: Jó reggelt!"), 1).tier,
            "luna",
        )

    def test_historical_image_does_not_lock_a_new_plain_turn_to_terra(self):
        decision = classify_request(chat_request_with_historical_image("Szia!"), 1)
        self.assertEqual(decision.tier, "luna")

    @patch("model_router._log_decision")
    def test_historical_image_design_request_routes_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request_with_historical_image("Írj egy CSS példát egy reszponzív kártyához."),
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=1,
            turn_id="history-image-sol-turn",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        payload = json.dumps(result["request"])
        self.assertIn("old-ui.png", payload)
        self.assertIn('"image_url"', payload)

    def test_sol_design_preflight_is_sol_owned_and_contractually_opus5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "sol_opus5_preflight": {"enabled": True, "owner": "sol", "bridge_model": "claude-opus-5", "require_successful_auth_probe": True},
                "shadow": {"enabled": False},
            }
            request = chat_request("Készíts UX/UI preflight értékelést egy bejelentkezési oldal vizuális elrendezéséről.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="sol-opus5-preflight-turn",
                )
        self.assertEqual(result["metadata"]["tier"], "sol")
        preflight = json.dumps(result["request"], ensure_ascii=False)
        self.assertIn("INTERNAL SOL + CLAUDE OPUS 5 PREFLIGHT", preflight)
        self.assertIn("claude-opus-5", preflight)
        self.assertIn("goal beginning [sol]", preflight)
        self.assertNotIn("[spark]", preflight)
        self.assertNotIn("[terra]", preflight)

    @patch("model_router._log_decision")
    def test_complex_current_image_task_can_start_terra_supervisor_preflight(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            request = chat_request_with_image(
                "Elemezd a képet, majd bontsd részfeladatokra a javítást, közben ellenőrizd a komponenst és végül tervezz teszteket."
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="current-image-supervisor-turn",
                )
        self.assertEqual(result["metadata"]["tier"], "terra")
        self.assertEqual(result["request"]["tool_choice"], "required")
        preflight = json.dumps(result["request"], ensure_ascii=False)
        self.assertIn("inspect any current image itself", preflight)
        self.assertIn("prefix a consequential worker goal with [sol]", preflight)

    def test_brief_non_actionable_chat_uses_luna_with_low_effort(self):
        decision = classify_request(chat_request("Ez szomorú."), api_call_count=1)
        self.assertEqual(decision.tier, "luna")
        self.assertEqual(decision.effort, "low")

    def test_design_praise_without_a_new_request_uses_luna_with_low_effort(self):
        decision = classify_request(
            chat_request("A UI design nagyon jó lett, köszönöm!"),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "luna")
        self.assertEqual(decision.effort, "low")

    def test_approval_only_follow_up_uses_luna_but_approval_to_execute_remains_actionable(self):
        acknowledgements = (
            "Nagyon jó lett, jóváhagyom.",
            "A UI design rendben van, elfogadom.",
            "Jóváhagyom, köszönöm!",
        )
        for prompt in acknowledgements:
            with self.subTest(prompt=prompt):
                decision = classify_request(chat_request(prompt), api_call_count=1)
                self.assertEqual(decision.tier, "luna")
                self.assertEqual(decision.effort, "low")

        actionable = classify_request(chat_request("Jóváhagyom, mehet prodra."), api_call_count=1)
        self.assertEqual(actionable.tier, "sol")

    def test_screenshot_backed_ui_action_is_sol_only(self):
        decision = classify_request(
            chat_request("Ez kerüljön át a név mögé. [Image attached at: /tmp/ui.png]"),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def test_repeated_simple_turn_call_is_promoted_out_of_luna_without_tool_marker(self):
        decision = classify_request(chat_request("Ez szomorú."), api_call_count=2)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("repeat", decision.reason.lower())

    def test_short_explanation_question_uses_luna_with_low_effort(self):
        decision = classify_request(chat_request("Miért fordítva van a használati sáv?"), api_call_count=1)
        self.assertEqual(decision.tier, "luna")
        self.assertEqual(decision.effort, "low")

    def test_short_css_design_implementation_is_sol_only(self):
        decision = classify_request(chat_request("Írj egy CSS példát egy reszponzív kártyához."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.model, MODELS["sol"])
        self.assertEqual(decision.effort, "medium")

    def test_unlabelled_coding_tool_loop_stays_with_terra(self):
        decision = classify_request(
            chat_request("Írj egy SQL példát a legutóbbi 10 rendeléshez.", with_tool_result=True),
            2,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("tool", decision.reason)

    def test_small_local_router_dashboard_changes_start_with_terra_planner(self):
        decision = classify_request(
            chat_request("A Model Router dashboardhoz add hozzá a Spark számláló kártyát."),
            1,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("default", decision.reason)

    def test_screenshot_backed_router_design_change_is_sol_only(self):
        decision = classify_request(
            chat_request("A Model Router dashboardhoz add hozzá a Spark kártyát. [Image attached at: /tmp/ui.png]"),
            1,
        )
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def test_structured_chat_image_never_uses_spark_even_with_explicit_override(self):
        decision = classify_request(chat_request_with_image("[spark] Add hozzá a Spark kártyát."), 1)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("image", decision.reason)

    def test_responses_input_image_never_uses_spark(self):
        request = responses_request("[spark] Nézd meg a képet és igazítsd a kártyát.")
        request["input"][0]["content"].append({"type": "input_image", "image_url": "https://example.test/ui.png"})
        decision = classify_request(request, 1)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("image", decision.reason)

    def test_benchmark_spark_force_cannot_override_image_safety_guard(self):
        with patch.dict("os.environ", {"MODEL_ROUTER_BENCHMARK_FORCE_MODEL": "spark"}):
            decision = classify_request(chat_request_with_image("Rövid CSS kérés."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    def test_benchmark_spark_force_cannot_override_text_design_boundary(self):
        with patch.dict("os.environ", {"MODEL_ROUTER_BENCHMARK_FORCE_MODEL": "spark"}):
            decision = classify_request(chat_request("Implement a responsive CSS card."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    @patch("model_router._log_decision")
    def test_root_spark_prefix_cannot_bypass_sol_only_design_policy(self, mocked_log):
        result = route_llm_request(
            request=chat_request("[spark] Írj egy CSS példát egy reszponzív kártyához."),
            provider="openai-codex", model=MODELS["terra"], api_call_count=1,
            turn_id="root-manual-spark-override",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("Sol-only", result["reason"])

    def test_explicit_spark_implementation_override_is_promoted_to_terra(self):
        decision = classify_request(
            chat_request("[spark] Add hozzá a Spark kártyát.", with_tool_result=True),
            2,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("read-only", decision.reason)

    @patch("model_router._log_decision")
    def test_delegated_spark_worker_stays_spark_for_a_bounded_tool_loop(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Inspect the local parser configuration and report the affected setting.", with_tool_result=True),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id="spark-child",
        )
        self.assertEqual(result["metadata"]["tier"], "spark")
        self.assertEqual(result["request"]["model"], MODELS["spark"])
        self.assertEqual(result["reason"], "eligible delegated Spark subtask")

    @patch("model_router._log_decision")
    def test_delegated_read_only_worker_is_not_escalated_by_negated_safety_constraints(self, mocked_log):
        result = route_llm_request(
            request=chat_request(
                "Read-only configuration audit: inspect the local router config and report the relevant settings. "
                "Do not edit files, restart services, or access credentials."
            ),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id="spark-child-safe-constraints",
        )
        self.assertEqual(result["metadata"]["tier"], "spark")
        self.assertEqual(result["reason"], "eligible delegated Spark subtask")

    @patch("model_router._log_decision")
    def test_delegated_spark_worker_escalates_hard_safety_signal_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Debugold a production szerver konfigurációját."),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=1,
            turn_id="spark-child-risky",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")

    def test_terra_owns_normal_repo_implementation_while_sol_keeps_consequential_actions(self):
        risky = classify_request(
            chat_request("SSH-n lépj be a production szerverre, módosítsd a konfigurációt és indítsd újra."),
            1,
        )
        debug = classify_request(
            chat_request("Debugold ezt a hibát, javítsd a kódot, majd futtasd a teszteket."),
            1,
        )
        continuation = classify_request(
            chat_request("Csináld meg a repo módosítást a megbeszéltek szerint."),
            1,
        )
        self.assertEqual(risky.tier, "sol")
        self.assertEqual(debug.tier, "terra")
        self.assertEqual(continuation.tier, "terra")
        self.assertEqual(debug.effort, "medium")

    def test_sol_keeps_security_database_migration_and_production_deploy_work(self):
        prompts = [
            "Javítsd a payment webhook hibáját.",
            "Javítsd az auth jogosultsági regressziót.",
            "Migráld az adatbázist az új sémára.",
            "Deployold productionre és indítsd újra a szervert.",
        ]
        self.assertTrue(all(classify_request(chat_request(prompt), 1).tier == "sol" for prompt in prompts))

    def test_generic_implementation_question_stays_on_terra(self):
        decision = classify_request(chat_request("Hogyan implementáljam ezt az űrlapot?"), 1)
        self.assertEqual(decision.tier, "terra")
        self.assertEqual(decision.effort, "medium")

    def test_explicit_sol_override_is_limited_to_medium_effort(self):
        decision = classify_request(chat_request("[sol] Oldd meg ezt a kritikus production hibát"), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.effort, "high")

    def test_xhigh_override_is_still_capped_at_medium(self):
        decision = classify_request(chat_request("[sol:xhigh] Készíts részletes implementációs tervet."), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.effort, "high")

    def test_completed_terra_supervisor_review_stays_with_terra_despite_long_result(self):
        result = "[ASYNC DELEGATION BATCH COMPLETE — deleg_test] Role: orchestrator [terra] " + ("reviewed evidence " * 500)
        decision = classify_request(chat_request(result))
        self.assertEqual(decision.tier, "terra")
        self.assertEqual(decision.reason, "completed Terra supervisor review")

    def test_long_requests_are_routed_to_sol(self):
        decision = classify_request(chat_request("Elemezd részletesen. " + "x" * 4200), 1)
        self.assertEqual(decision.tier, "sol")
        self.assertEqual(decision.effort, "high")

    def test_merely_medium_length_request_stays_on_terra(self):
        decision = classify_request(chat_request("Elemezd ezt. " + "x" * 2200), 1)
        self.assertEqual(decision.tier, "terra")

    def test_normal_tool_loop_follow_up_stays_on_terra(self):
        decision = classify_request(chat_request("Nézd meg ezt.", with_tool_result=True), 2)
        self.assertEqual(decision.tier, "terra")
        self.assertIn("tool", decision.reason.lower())

    def test_normal_repo_tool_loop_follow_up_stays_on_terra(self):
        decision = classify_request(
            chat_request(
                "Debugold ezt a hibát, javítsd a kódot, majd futtasd a teszteket.",
                with_tool_result=True,
            ),
            2,
        )
        self.assertEqual(decision.tier, "terra")
        self.assertIn("tool", decision.reason.lower())

    def test_explicit_prefix_overrides_heuristics(self):
        self.assertEqual(classify_request(chat_request("[luna] Elemezd részletesen a szervert"), 1).tier, "luna")
        self.assertEqual(classify_request(chat_request("[spark] Oldd meg ezt a kritikus production hibát"), 1).tier, "sol")
        self.assertEqual(classify_request(chat_request("[terra] SSH szerver konfigurálása"), 1).tier, "terra")
        self.assertEqual(classify_request(chat_request("[sol] Szia"), 1).tier, "sol")

    def test_codex_responses_input_is_supported(self):
        decision = classify_request(responses_request("Fordítsd magyarra: good morning"), 1)
        self.assertEqual(decision.tier, "luna")

    @patch("model_router._log_decision")
    def test_middleware_rewrites_only_model_and_returns_trace_metadata(self, mocked_log):
        request = chat_request("Szia!")
        result = route_llm_request(
            request=request,
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=1,
            turn_id="turn-1",
        )
        self.assertEqual(result["request"]["model"], MODELS["luna"])
        self.assertEqual(result["request"]["temperature"], 0.2)
        self.assertEqual(result["request"]["reasoning"]["effort"], "low")
        self.assertEqual(result["source"], "model-router")
        self.assertTrue(result["reason"])

    def test_shadow_lifecycle_logs_a_deterministic_parent_child_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shadow.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "path": str(path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="benchmark-turn-1",
                )
                on_subagent_start(
                    parent_turn_id="benchmark-turn-1",
                    child_session_id="spark-child-session",
                    child_subagent_id="sa-1",
                    child_role="leaf",
                    child_goal="Read-only comparison.",
                )
                on_subagent_stop(
                    parent_turn_id="benchmark-turn-1",
                    child_session_id="spark-child-session",
                    child_status="completed",
                    child_model=MODELS["spark"],
                    child_api_calls=3,
                    input_tokens=100,
                    output_tokens=25,
                    cost_usd=0.01,
                    exit_reason="completed",
                    duration_ms=1250,
                    child_summary="Evidence-based Spark result.",
                )
                on_post_llm_call(
                    turn_id="benchmark-turn-1",
                    model=MODELS["terra"],
                    assistant_response="Evidence-based Terra answer.",
                )
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [event["event"] for event in events],
                ["delegation_forced", "child_started", "child_completed", "parent_completed"],
            )
            self.assertEqual(len({event["benchmark_id"] for event in events}), 1)
            self.assertEqual(events[-2]["child_model"], MODELS["spark"])
            self.assertEqual(events[-2]["child_api_calls"], 3)
            self.assertNotIn("summary_preview", events[-2])
            self.assertIn("summary_sha256", events[-2])
            self.assertEqual(events[-1]["parent_model"], MODELS["terra"])
            self.assertNotIn("parent_response_preview", events[-1])
            self.assertIn("parent_response_sha256", events[-1])

    def test_terra_supervisor_preflight_dispatches_real_bounded_spark_work_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orchestration.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(path)},
                "shadow": {"enabled": False},
            }
            request = chat_request(
                "Vizsgáld meg a futási naplókat, utána ellenőrizd a komponens állapotát, "
                "végül tervezz regressziós teszteket."
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                first = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="supervisor-turn-1",
                )
                second = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="supervisor-turn-1",
                )
            self.assertEqual(first["request"]["model"], MODELS["terra"])
            self.assertEqual(first["request"]["tool_choice"], "required")
            self.assertEqual(len(first["request"]["tools"]), 1)
            self.assertIn("INTERNAL ORCHESTRATOR PREFLIGHT", first["request"]["messages"][-1]["content"])
            self.assertIn("SUPERVISOR DECISION: ACCEPT or REJECT", first["request"]["messages"][-1]["content"])
            self.assertNotIn("tool_choice", second["request"])
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["preflight_forced"])


    def test_compaction_envelope_routes_the_actual_current_user_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orchestration.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(path)},
                "shadow": {"enabled": False},
            }
            request = chat_request(
                "[CONTEXT COMPACTION — REFERENCE ONLY] historical context\n"
                "[END OF CONTEXT SUMMARY — respond to the message below]\n"
                "Csináld újra a videót: először válaszd ki a szolgáltatást, aztán zoomolj rá, "
                "közben lassan scrollozz a következő vezérlőre, végül ellenőrizd a regressziót."
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="compacted-complex-turn",
                )
            self.assertEqual(routed["request"]["tool_choice"], "required")
            self.assertTrue(path.exists())

    def test_long_terra_tool_loop_is_rescued_once_with_real_supervisor_dispatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orchestration.jsonl"
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "min_chars": 40, "max_tasks": 3,
                    "rescue_min_calls": 6, "path": str(path),
                },
                "shadow": {"enabled": False},
            }
            request = chat_request(
                "Készítsd el az animációt: majd zoomolj a kártyára, utána lassan scrollozz, "
                "közben ellenőrizd a futási naplókat és komponens állapotát, végül tervezz regressziós teszteket.",
                with_tool_result=True,
            )
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                rescued = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=6, turn_id="terra-loop-without-preflight",
                )
                later = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=7, turn_id="terra-loop-without-preflight",
                )
            self.assertEqual(rescued["request"]["tool_choice"], "required")
            self.assertNotIn("tool_choice", later["request"])
            event = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(event["event"], "preflight_forced")
            self.assertEqual(event["phase"], "rescue")
            self.assertEqual(event["api_call_count"], 6)

    def test_hungarian_aztan_starts_preflight_but_internal_memory_prompt_does_not(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            video = ("Csináld újra a videót úgy, hogy minden látszódjon: először nyisd ki a blokkokat, "
                     "aztán görgess, közben ellenőrizd a zoomot, végül készíts regressziós ellenőrzést.")
            memory = ("Review the conversation above and consider saving to memory if appropriate. "
                      "Focus on the user preferences and work style, then save them.")
            video_request, memory_request = chat_request(video), chat_request(memory)
            video_request["tools"] = memory_request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                video_result = route_llm_request(request=video_request, provider="openai-codex", model=MODELS["terra"], api_call_count=1, turn_id="video-turn")
                memory_result = route_llm_request(request=memory_request, provider="openai-codex", model=MODELS["terra"], api_call_count=1, turn_id="memory-turn")
            self.assertEqual(video_result["request"].get("tool_choice"), "required")
            self.assertNotIn("tool_choice", memory_result["request"])

    def test_completed_supervisor_handback_is_not_delegated_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "min_chars": 40, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            handback = "[ASYNC DELEGATION BATCH COMPLETE — deleg_test] Role: orchestrator [terra] " + ("SUPERVISOR DECISION: ACCEPT reviewed evidence " * 100)
            request = chat_request(handback)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="supervisor-handback",
                )
            self.assertEqual(routed["request"]["model"], MODELS["terra"])
            self.assertNotIn("tool_choice", routed["request"])

    def test_eligible_terra_turn_forces_one_read_only_spark_shadow_delegation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "path": str(Path(temp_dir) / "shadow.jsonl")},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="benchmark-turn-1",
                )
            self.assertEqual(result["request"]["model"], MODELS["terra"])
            self.assertEqual(result["request"]["tool_choice"], "required")
            self.assertEqual(len(result["request"]["tools"]), 1)
            self.assertEqual(result["request"]["tools"][0]["name"], "delegate_task")
            self.assertIn("INTERNAL SPARK MEDIUM SHADOW BENCHMARK", result["request"]["messages"][-1]["content"])

    def test_incomplete_forced_shadows_do_not_consume_the_ten_completed_lifecycle_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shadow.jsonl"
            events = []
            for index in range(10):
                benchmark_id = f"shadow-{index}"
                events.append({"event": "delegation_forced", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id})
                if index < 8:
                    events.extend([
                        {"event": "child_started", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                        {"event": "child_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                        {"event": "parent_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                    ])
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "path": str(path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="replacement-turn",
                )
            self.assertEqual(result["request"]["tool_choice"], "required")
            updated_events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event.get("event") == "delegation_forced" for event in updated_events), 11)

    def test_new_shadow_cycle_ignores_completed_lifecycles_from_an_earlier_cycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shadow.jsonl"
            events = []
            for index in range(10):
                benchmark_id = f"old-shadow-{index}"
                for event_name in ("delegation_forced", "child_started", "child_completed", "parent_completed"):
                    events.append({
                        "event": event_name,
                        "turn_id": f"old-turn-{index}",
                        "benchmark_id": benchmark_id,
                        "cycle_id": "old",
                        "child_session_id": f"old-child-{index}",
                    })
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "shadow": {"enabled": True, "limit": 10, "cycle_id": "new", "path": str(path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="new-cycle-turn",
                )
            self.assertEqual(result["request"]["tool_choice"], "required")
            updated_events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(updated_events[-1]["event"], "delegation_forced")
            self.assertEqual(updated_events[-1]["cycle_id"], "new")

    def test_sol_promoted_children_do_not_consume_the_actual_spark_benchmark_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shadow_path = Path(temp_dir) / "shadow.jsonl"
            route_path = Path(temp_dir) / "router.jsonl"
            events = []
            routes = []
            for index in range(10):
                benchmark_id = f"shadow-{index}"
                child_session_id = f"child-{index}"
                events.extend([
                    {"event": "delegation_forced", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                    {"event": "child_started", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id, "child_session_id": child_session_id},
                    {"event": "child_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id, "child_session_id": child_session_id},
                    {"event": "parent_completed", "turn_id": f"old-turn-{index}", "benchmark_id": benchmark_id},
                ])
                routes.append({
                    "turn_id": f"parent:{child_session_id}:worker",
                    "model": MODELS["spark"] if index < 8 else MODELS["sol"],
                })
            shadow_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            route_path.write_text("\n".join(json.dumps(route) for route in routes) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "logging": {"path": str(route_path)},
                "shadow": {"enabled": True, "limit": 10, "path": str(shadow_path)},
            }
            request = chat_request("Hasonlítsd össze ezt a két megoldást.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="verified-spark-replacement-turn",
                )
            self.assertEqual(result["request"]["tool_choice"], "required")

    @patch("model_router._log_decision")
    def test_turn_id_marked_terra_planner_stays_terra_until_it_labels_a_leaf(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Read only inspect this local file and report its fields."),
            provider="openai-codex",
            model=MODELS["terra"],
            api_call_count=2,
            turn_id="parent-session:sa-0-child-session:worker-turn",
        )
        self.assertEqual(result["request"]["model"], MODELS["terra"])
        self.assertEqual(result["request"]["reasoning"]["effort"], "medium")
        self.assertEqual(result["reason"], "Terra planner or integration subagent")

    @patch("model_router._log_decision")
    def test_spark_subagent_with_design_image_is_rerouted_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request_with_image("Inspect this UI screenshot and report the affected selector."),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id="spark-child-with-image",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertEqual(result["request"]["model"], MODELS["sol"])

    @patch("model_router._log_decision")
    def test_conductor_label_survives_a_design_flavoured_objective(self, mocked_log):
        """The conductor coordinates design work; it does not perform it. Judging
        it by keywords put the planner on Sol whenever the objective touched UI,
        and Sol then owned both the conducting and the [sol] leaf it should have
        delegated -- 15 of 16 routing decisions on one real turn."""
        result = route_llm_request(
            request=chat_request(
                "[terra] Fix the daily-calendar card layout in /home/deepwell/booking-saas; "
                "trace the computed layout/markup before implementing."
            ),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="conductor-design-objective",
        )
        self.assertEqual(result["metadata"]["tier"], "terra")

    @patch("model_router._log_decision")
    def test_root_design_turn_still_ignores_a_typed_label(self, mocked_log):
        """The exemption is for plan labels only. A user typing [terra] on a root
        turn must not be able to route design work away from Sol."""
        decision = classify_request(
            chat_request("[terra] Design a responsive CSS card layout."),
            api_call_count=1,
        )
        self.assertEqual(decision.tier, "sol")
        self.assertIn("Sol-only", decision.reason)

    @patch("model_router._log_decision")
    def test_a_labelled_leaf_survives_its_own_callable_fallback(self, mocked_log):
        """The planner-tier default asked "was this labelled?" by string-matching
        the reason, which the callable fallback rewrites: a [spark] leaf becomes
        "fallback from disabled spark" the moment Spark is not callable, so every
        legitimate Spark leaf was demoted to the planner tier and lost its Luna
        fallback -- the separate model the leaf was meant to run on."""
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": {**CALLABLE, "spark": False, "luna": True},
            "fallbacks": {"spark": "luna"},
            "default_model": "terra",
        }
        with patch("model_router._load_config", return_value=cfg):
            result = route_llm_request(
                request=chat_request("[spark] Read-only inventory. Identify the build commands."),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-leaf:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "luna")

    @patch("model_router._log_decision")
    def test_a_rejected_leaf_keeps_its_escalation(self, mocked_log):
        """Demoting a rejected [spark] leaf to the planner tier turned a
        deliberate escalation into the very tier it existed to avoid."""
        with patch("model_router._load_config", return_value={
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": CALLABLE, "default_model": "terra",
        }):
            result = route_llm_request(
                request=chat_request("[spark] Review the deploy scripts in production."),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-risky:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("consequential", result["reason"])

    @patch("model_router._log_decision")
    def test_unlabelled_child_work_still_defaults_to_the_planner_tier(self, mocked_log):
        with patch("model_router._load_config", return_value={
            "enabled": True, "provider": "openai-codex", "models": MODELS,
            "callable": CALLABLE, "default_model": "terra",
        }):
            result = route_llm_request(
                request=chat_request("Continue the integration work for the calendar fix."),
                provider="openai-codex", model=MODELS["terra"], platform="subagent",
                api_call_count=1, turn_id="session:sa-1-plain:turn",
            )
        self.assertEqual(result["metadata"]["tier"], "terra")
        self.assertIn("planner or integration subagent", result["reason"])

    @patch("model_router._log_decision")
    def test_read_only_discovery_leaf_keeps_its_spark_label(self, mocked_log):
        """"layout" is an ordinary noun in frontend source discovery. Judging the
        leaf by that word sent every such leaf to Sol -- the planner had already
        decided, with the screenshot and the repo in hand, that this was bounded
        read-only evidence work."""
        result = route_llm_request(
            request=chat_request(
                "[spark] Perform read-only source discovery in /home/deepwell/booking-saas. "
                "Identify the exact daily calendar booking-card renderer, duration layout "
                "branches, and the editor overlay state owner."
            ),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="spark-discovery-leaf",
        )
        self.assertEqual(result["metadata"]["tier"], "spark")

    @patch("model_router._log_decision")
    def test_a_spark_label_still_has_to_be_true(self, mocked_log):
        """The label is trusted, not obeyed: a leaf that writes is not read-only
        evidence work whatever it is labelled, and the design test survives to
        place the rejected leaf rather than to preempt it."""
        decision = classify_request(
            chat_request("[spark] Implement a responsive CSS card."),
            api_call_count=1,
            allow_plan_label_over_design=True,
        )
        self.assertEqual(decision.tier, "sol")

    @patch("model_router._log_decision")
    def test_text_only_design_spark_subagent_is_rerouted_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request("[spark] Implement a responsive CSS card."),
            provider="openai-codex", model=MODELS["spark"], platform="subagent",
            api_call_count=1, turn_id="spark-design-child",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("Sol-only", result["reason"])

    @patch("model_router._log_decision")
    def test_text_only_design_terra_subagent_is_rerouted_to_sol(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Review and implement the UI layout CSS."),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="terra-design-child",
        )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("Sol-only", result["reason"])

    @patch("model_router._log_decision")
    def test_spark_subagent_non_read_only_implementation_is_promoted_to_terra(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Modify the parser implementation and write the patch."),
            provider="openai-codex", model=MODELS["spark"], platform="subagent",
            api_call_count=1, turn_id="spark-implementation-child",
        )
        self.assertEqual(result["metadata"]["tier"], "terra")
        self.assertIn("read-only", result["reason"])

    def test_spark_quota_429_uses_retry_call_when_provided(self):
        next_calls = []
        retry_calls = []

        def next_call(request):
            next_calls.append(request["model"])
            raise RuntimeError("HTTP 429: weekly quota exhausted for gpt-5.3-codex-spark")

        def retry_call(request):
            retry_calls.append(request["model"])
            self.assertEqual(request["reasoning"]["effort"], "medium")
            return "recovered by Luna"

        result = run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=next_call,
            retry_call=retry_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "recovered by Luna")
        self.assertEqual(next_calls, [MODELS["spark"]])
        self.assertEqual(retry_calls, [MODELS["luna"]])

    def test_spark_quota_429_routes_to_luna_at_medium_effort(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 429: weekly quota exhausted for gpt-5.3-codex-spark")
            self.assertEqual(request["reasoning"]["effort"], "medium")
            return "recovered by Luna"

        result = run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=next_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "recovered by Luna")
        self.assertEqual(calls, [MODELS["spark"], MODELS["luna"]])

    def test_weekly_spark_quota_exhaustion_sticks_for_later_calls_in_same_turn(self):
        turn_id = "spark-weekly-quota-sticky-turn"

        def exhausted(request):
            raise RuntimeError("HTTP 429: weekly quota exhausted for gpt-5.3-codex-spark")

        run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=exhausted,
            retry_call=lambda request: "recovered",
            provider="openai-codex",
            turn_id=turn_id,
        )

        routed = route_llm_request(
            request=chat_request("[spark] Continue the same bounded read-only inspection."),
            provider="openai-codex",
            model=MODELS["spark"],
            platform="subagent",
            api_call_count=2,
            turn_id=turn_id,
        )

        self.assertEqual(routed["metadata"]["tier"], "luna")
        self.assertEqual(routed["request"]["model"], MODELS["luna"])
        self.assertEqual(routed["request"]["reasoning"]["effort"], "medium")
        self.assertEqual(routed["reason"], "Spark quota already exhausted for this turn")

    def test_generic_429_rate_limit_does_not_change_transient_failover_behavior(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 429: rate limit exceeded; retry after 30 seconds")

        with self.assertRaisesRegex(RuntimeError, "rate limit"):
            run_llm_with_transient_failover(
                request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["spark"]])

    def test_spark_weekly_quota_429_fails_over_once_to_luna(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 429: weekly limit reached for this account")
            self.assertEqual(request["reasoning"]["effort"], "medium")
            return "luna-recovered"

        result = run_llm_with_transient_failover(
            request={**chat_request("[spark] Bounded read-only source inspection."), "model": MODELS["spark"]},
            next_call=next_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "luna-recovered")
        self.assertEqual(calls, [MODELS["spark"], MODELS["luna"]])

    def test_non_spark_quota_429_does_not_trigger_luna_failover(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 429: weekly quota exhausted")

        with self.assertRaisesRegex(RuntimeError, "weekly quota"):
            run_llm_with_transient_failover(
                request={**chat_request("[terra] Continue."), "model": MODELS["terra"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["terra"]])

    def test_transient_fallback_from_sol_with_an_image_skips_spark(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 503: upstream connect error")
            return "recovered"

        result = run_llm_with_transient_failover(
            request={**chat_request_with_image("Inspect this screenshot."), "model": MODELS["sol"]},
            next_call=next_call,
            provider="openai-codex",
        )
        self.assertEqual(result, "recovered")
        self.assertEqual(calls, [MODELS["sol"], MODELS["terra"]])

    def test_middleware_bypasses_other_providers_and_unrelated_models(self):
        request = chat_request("Szia!")
        self.assertIsNone(route_llm_request(
            request=request,
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_call_count=1,
        ))
        self.assertIsNone(route_llm_request(
            request=request,
            provider="openai-codex",
            model="gpt-5.4",
            api_call_count=1,
        ))

    def test_transient_503_immediately_retries_once_with_a_different_model(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            if len(calls) == 1:
                raise RuntimeError("HTTP 503: upstream connect error or disconnect/reset before headers")
            return "recovered"

        result = run_llm_with_transient_failover(
            request={**chat_request("[sol] Folytasd a fejlesztést."), "model": MODELS["sol"]},
            next_call=next_call,
            provider="openai-codex",
        )

        self.assertEqual(result, "recovered")
        self.assertEqual(calls, [MODELS["sol"], MODELS["spark"]])

    def test_sol_design_transient_failure_never_falls_back_to_another_tier(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 503: upstream connect error")

        with self.assertRaisesRegex(RuntimeError, "503"):
            run_llm_with_transient_failover(
                request={**chat_request("Implement the responsive CSS card design."), "model": MODELS["sol"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["sol"]])


    def test_non_transient_error_does_not_retry_with_another_model(self):
        calls = []

        def next_call(request):
            calls.append(request["model"])
            raise RuntimeError("HTTP 401: invalid authentication")

        with self.assertRaisesRegex(RuntimeError, "401"):
            run_llm_with_transient_failover(
                request={**chat_request("[sol] Folytasd a fejlesztést."), "model": MODELS["sol"]},
                next_call=next_call,
                provider="openai-codex",
            )
        self.assertEqual(calls, [MODELS["sol"]])

    def test_credential_and_password_actions_route_to_sol_in_hungarian_inflections(self):
        prompts = (
            "Állítsd be a belépési adatokat és jelszót.",
            "Add meg a demo account jelszavát.",
        )
        self.assertTrue(all(classify_request(chat_request(prompt), 1).tier == "sol" for prompt in prompts))

    @patch("model_router._log_decision")
    def test_forced_planner_schema_requires_an_orchestrator_child(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            request = chat_request(ACTIONABLE_TERRA_PROMPT)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {"type": "object", "properties": {"goal": {"type": "string"}, "role": {"type": "string"}}}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(request=request, provider="openai-codex", model=MODELS["terra"], api_call_count=1, turn_id="planner-schema-turn")
        schema = routed["request"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["role"]["enum"], ["orchestrator"])
        self.assertIn("role", schema["required"])
        self.assertIn("context", schema["required"])
        self.assertEqual(len(schema["properties"]["context"]["enum"]), 1)
        contract = schema["properties"]["context"]["enum"][0]
        self.assertIn("qwen planning conductor", contract)
        self.assertIn("Do not perform design analysis or design implementation", contract)
        self.assertIn("Sol", contract)
        self.assertIn("non-design read-only", contract)

    @patch("model_router._log_decision")
    def test_terra_orchestrator_subagent_is_not_rewritten_to_spark(self, mocked_log):
        result = route_llm_request(
            request=chat_request("Assess the task and delegate only useful bounded workers."),
            provider="openai-codex", model=MODELS["terra"], platform="subagent",
            api_call_count=1, turn_id="terra-planner:sa-0-child:turn",
        )
        self.assertEqual(result["metadata"]["tier"], "terra")

    @patch("model_router._log_decision")
    def test_every_actionable_terra_turn_forces_a_planner_without_regex_keywords(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {"enabled": True, "max_tasks": 3, "path": str(Path(temp_dir) / "orchestration.jsonl")},
                "shadow": {"enabled": False},
            }
            request = chat_request(ACTIONABLE_TERRA_PROMPT)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="short-actionable-turn",
                )
        self.assertEqual(routed["request"]["tool_choice"], "required")
        preflight = routed["request"]["messages"][-1]["content"]
        self.assertIn("Create a structured dispatch plan", preflight)
        self.assertIn("zero to 3", preflight)

    @patch("model_router._log_decision")
    def test_terra_turn_below_min_chars_is_not_fanned_out(self, mocked_log):
        """A turn too small to decompose must not pay for a planner plus workers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "max_tasks": 3, "min_chars": 180,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "shadow": {"enabled": False},
            }
            request = chat_request("Nézd meg, mi a baj.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="below-min-chars-turn",
                )
            orchestration_log = Path(temp_dir) / "orchestration.jsonl"
            events = [
                json.loads(line)
                for line in orchestration_log.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(routed["metadata"]["tier"], "terra")
        self.assertNotIn("tool_choice", routed["request"])
        # The declined turn is now logged with its reason; only a forced
        # dispatch would mean the gate leaked.
        self.assertFalse(
            [event for event in events if event["event"] == "preflight_forced"],
            "a sub-min_chars turn must not emit a preflight dispatch",
        )
        self.assertEqual(
            [event["skip_reason"].split(":")[0] for event in events],
            ["prompt_shorter_than_min_chars"],
        )

    @patch("model_router._log_decision")
    def test_sol_preflight_ignores_min_chars_because_it_is_not_a_fan_out(self, mocked_log):
        """min_chars gates worker fan-out; the Sol design review is not fan-out."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "max_tasks": 3, "min_chars": 180,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "sol_opus5_preflight": {
                    "enabled": True, "owner": "sol", "bridge_model": "claude-opus-5",
                    "require_successful_auth_probe": True,
                },
                "shadow": {"enabled": False},
            }
            prompt = "Készíts UX/UI preflight értékelést egy bejelentkezési oldal vizuális elrendezéséről."
            self.assertLess(len(prompt), 180, "fixture must sit below min_chars to be meaningful")
            request = chat_request(prompt)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=1, turn_id="sol-short-preflight-turn",
                )
        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertIn("claude-opus-5", json.dumps(result["request"], ensure_ascii=False))

    @patch("model_router._log_decision")
    def test_long_terra_loop_is_rescued_even_when_original_prompt_has_no_regex_markers(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True, "provider": "openai-codex", "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "orchestration": {
                    "enabled": True, "max_tasks": 3, "rescue_min_calls": 6,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "shadow": {"enabled": False},
            }
            request = chat_request(ACTIONABLE_TERRA_PROMPT, with_tool_result=True)
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                routed = route_llm_request(
                    request=request, provider="openai-codex", model=MODELS["terra"],
                    api_call_count=6, turn_id="generic-long-loop-turn",
                )
        self.assertEqual(routed["request"]["tool_choice"], "required")

    def test_first_non_design_coding_call_executes_opus5_instead_of_openai(self):
        with tempfile.TemporaryDirectory() as repo:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "high", "luna": "low"},
                "coding_agent": {
                    "enabled": True,
                    "tier": "opus5",
                    "model": "claude-opus-5",
                    "default_repo": repo,
                    "max_turns": 8,
                    "max_budget_usd": 5.0,
                },
            }
            result = {
                "result": "OPUS IMPLEMENTATION COMPLETE",
                "effective_model": "claude-opus-5",
                "usage": {"input_tokens": 17, "output_tokens": 9},
            }
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router._run_opus5_bridge", return_value=result
            ) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request("Implement the parser fix and add tests."),
                    next_call=lambda _request: self.fail("OpenAI downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="coding-turn",
                )

        bridge.assert_called_once()
        self.assertEqual(response.model, "claude-opus-5")
        self.assertEqual(response.output[0].content[0].text, "OPUS IMPLEMENTATION COMPLETE")

    def test_explicit_opus_review_executes_read_only_for_non_coding_review(self):
        with tempfile.TemporaryDirectory() as repo:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "coding_agent": {
                    "enabled": True,
                    "canonical_model": "claude-opus-5",
                    "default_repo": repo,
                    "reviewer": {"enabled": True, "max_chars": 8000},
                },
            }
            result = {"result": "OPUS REVIEW COMPLETE", "effective_model": "claude-opus-5"}
            prompt = "[opus-review] Review the access-control proposal for missing risks. Do not modify files."
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._recent_verified_opus5_route", return_value=True), patch(
                "model_router._run_opus5_bridge", return_value=result
            ) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: self.fail("OpenAI downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="opus-review-turn",
                )

        bridge.assert_called_once()
        self.assertFalse(bridge.call_args.kwargs["write"])
        self.assertTrue(bridge.call_args.kwargs["review"])
        self.assertEqual(response.model, "claude-opus-5")
        self.assertEqual(response.output[0].content[0].text, "OPUS REVIEW COMPLETE")

    def _delegated_review_cfg(self, repo, **overrides):
        policy = {"enabled": True, "max_chars": 8000, "models": ["opus", "sonnet"]}
        policy.update(overrides)
        return {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "coding_agent": {
                # Deliberately off: the delegated path must not depend on the
                # switch that also arms the label-free coding classifier.
                "enabled": False,
                "canonical_model": "claude-opus-5",
                "default_repo": repo,
                "delegated_review": policy,
            },
        }

    def test_delegated_sonnet_review_leaf_runs_on_claude(self):
        """A read-only review leaf is the one job that can leave the Codex
        account entirely: Claude is reachable only through its own CLI, so the
        leaf's single call becomes a bridge subprocess and its verdict becomes
        the leaf's answer."""
        with tempfile.TemporaryDirectory() as repo:
            result = {"result": "SONNET REVIEW COMPLETE", "effective_model": "claude-sonnet-5"}
            prompt = "[sonnet-review] Review the pending calendar diff for regressions. Report only."
            with patch("model_router._load_config", return_value=self._delegated_review_cfg(repo)), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge", return_value=result) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: self.fail("the Codex downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    platform="subagent",
                    turn_id="session:sa-1-child:turn",
                )

        bridge.assert_called_once()
        self.assertEqual(bridge.call_args.kwargs["model"], "sonnet")
        self.assertTrue(bridge.call_args.kwargs["review"])
        self.assertFalse(bridge.call_args.kwargs["write"])
        self.assertEqual(response.model, "claude-sonnet-5")

    def test_a_root_turn_is_never_diverted_into_the_bridge(self):
        """The documented hazard of this bridge is that it captures the first
        call of a turn. That only matters for the parent that still has to plan,
        so the delegated path admits delegated workers and nothing else."""
        with tempfile.TemporaryDirectory() as repo:
            prompt = "[opus-review] Review the access-control proposal. Report only."
            sentinel = object()
            with patch("model_router._load_config", return_value=self._delegated_review_cfg(repo)), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge") as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: sentinel,
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="plain-root-turn",
                )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_a_claude_tier_left_out_of_the_config_is_not_reachable(self):
        with tempfile.TemporaryDirectory() as repo:
            cfg = self._delegated_review_cfg(repo, models=["opus"])
            prompt = "[sonnet-review] Review the pending calendar diff. Report only."
            sentinel = object()
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge") as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request(prompt),
                    original_request=responses_request(prompt),
                    next_call=lambda _request: sentinel,
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    platform="subagent",
                    turn_id="session:sa-1-child:turn",
                )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_runtime_opus_bridge_uses_the_real_dispatch_entrypoint(self):
        cfg = {"coding_agent": {"timeout_seconds": 300}}
        expected = {"result": "ok", "model": "claude-opus-5"}
        with tempfile.TemporaryDirectory() as repo, patch(
            "model_router.claude_opus_bridge.dispatch", return_value=expected
        ) as dispatch:
            from model_router import _run_opus5_bridge

            result = _run_opus5_bridge(
                repo=repo,
                task="Inspect the parser.",
                write=False,
                cfg=cfg,
                turn_id="entrypoint-test",
            )

        dispatch.assert_called_once()
        self.assertEqual(result, expected)

    def test_design_coding_call_stays_on_existing_sol_route(self):
        cfg = {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "coding_agent": {"enabled": True, "tier": "opus5", "model": "claude-opus-5"},
        }
        sentinel = object()
        with patch("model_router._load_config", return_value=cfg), patch(
            "model_router._run_opus5_bridge"
        ) as bridge:
            response = run_llm_with_transient_failover(
                request=responses_request("Implement the CSS layout from this design."),
                next_call=lambda _request: sentinel,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=1,
                turn_id="design-turn",
            )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    @patch("model_router._log_decision")
    def test_explicit_bounded_opus_ui_request_skips_planner_preflight(self, mocked_log):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "effort": {"terra": "medium", "spark": "medium", "sol": "medium", "luna": "low"},
                "coding_agent": {"enabled": True, "explicit_ui": {"enabled": True, "max_chars": 1200}},
                "orchestration": {
                    "enabled": True,
                    "max_tasks": 3,
                    "path": str(Path(temp_dir) / "orchestration.jsonl"),
                },
                "sol_opus5_preflight": {
                    "enabled": True,
                    "owner": "sol",
                    "bridge_model": "claude-opus-5",
                    "require_successful_auth_probe": True,
                },
                "shadow": {"enabled": False},
            }
            request = chat_request("Ha lehet, az Opus dolgozzon ezen a kis UI címke javításon.")
            request["tools"] = [{"type": "function", "name": "delegate_task", "parameters": {}}]
            with patch("model_router._load_config", return_value=cfg):
                result = route_llm_request(
                    request=request,
                    provider="openai-codex",
                    model=MODELS["terra"],
                    api_call_count=1,
                    turn_id="direct-explicit-opus-ui",
                )

        self.assertEqual(result["metadata"]["tier"], "sol")
        self.assertNotIn("tool_choice", result["request"])
        self.assertNotIn("INTERNAL SOL + CLAUDE OPUS 5 PREFLIGHT", json.dumps(result["request"]))

    def test_explicit_bounded_opus_ui_request_executes_one_verified_bridge_call(self):
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as temp_dir:
            route_log = Path(temp_dir) / "router.jsonl"
            route_log.write_text(json.dumps({
                "timestamp": "2099-01-01T00:00:00+00:00",
                "tier": "opus5",
                "model": "claude-opus-5",
                "reason": "verified probe",
            }) + "\n", encoding="utf-8")
            cfg = {
                "enabled": True,
                "provider": "openai-codex",
                "models": MODELS, "callable": CALLABLE,
                "logging": {"enabled": True, "path": str(route_log)},
                "coding_agent": {
                    "enabled": True,
                    "canonical_model": "claude-opus-5",
                    "default_repo": repo,
                    "timeout_seconds": 300,
                    "explicit_ui": {
                        "enabled": True,
                        "max_chars": 1200,
                        "require_recent_verified_probe_seconds": 86400,
                    },
                },
            }
            result = {
                "result": "OPUS UI FIX COMPLETE",
                "effective_model": "claude-opus-5",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
            with patch("model_router._load_config", return_value=cfg), patch(
                "model_router.shutil.which", return_value="/usr/bin/claude"
            ), patch("model_router._run_opus5_bridge", return_value=result) as bridge:
                response = run_llm_with_transient_failover(
                    request=responses_request("Ha lehet, az Opus dolgozzon ezen a kis UI címke javításon."),
                    original_request=responses_request("Ha lehet, az Opus dolgozzon ezen a kis UI címke javításon."),
                    next_call=lambda _request: self.fail("Sol/OpenAI downstream must not run"),
                    provider="openai-codex",
                    api_mode="codex_responses",
                    api_call_count=1,
                    turn_id="direct-explicit-opus-ui",
                )

        bridge.assert_called_once()
        self.assertTrue(bridge.call_args.kwargs["task"].startswith("[opus5]"))
        self.assertEqual(response.model, "claude-opus-5")

    def test_explicit_ui_opus_request_without_verified_bridge_stays_on_sol(self):
        cfg = {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "logging": {"enabled": True, "path": "/missing/router.jsonl"},
            "coding_agent": {
                "enabled": True,
                "default_repo": "/missing/repo",
                "explicit_ui": {"enabled": True, "max_chars": 1200},
            },
        }
        sentinel = object()
        with patch("model_router._load_config", return_value=cfg), patch(
            "model_router._run_opus5_bridge"
        ) as bridge:
            response = run_llm_with_transient_failover(
                request=responses_request("Let Opus work on this small CSS label fix."),
                original_request=responses_request("Let Opus work on this small CSS label fix."),
                next_call=lambda _request: sentinel,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=1,
                turn_id="unavailable-explicit-opus-ui",
            )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_later_agent_loop_call_does_not_launch_another_opus_process(self):
        cfg = {
            "enabled": True,
            "provider": "openai-codex",
            "models": MODELS, "callable": CALLABLE,
            "coding_agent": {"enabled": True, "tier": "opus5", "model": "claude-opus-5"},
        }
        sentinel = object()
        with patch("model_router._load_config", return_value=cfg), patch(
            "model_router._run_opus5_bridge"
        ) as bridge:
            response = run_llm_with_transient_failover(
                request=responses_request("Implement the parser fix and add tests."),
                next_call=lambda _request: sentinel,
                provider="openai-codex",
                api_mode="codex_responses",
                api_call_count=2,
                turn_id="coding-turn",
            )

        bridge.assert_not_called()
        self.assertIs(response, sentinel)

    def test_log_timestamp_has_whole_second_precision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "router.jsonl"
            cfg = {"logging": {"enabled": True, "path": str(path)}}
            with patch("model_router.datetime") as mocked_datetime:
                mocked_datetime.now.return_value = __import__("datetime").datetime(
                    2026,
                    7,
                    14,
                    7,
                    1,
                    2,
                    987654,
                    tzinfo=__import__("datetime").timezone.utc,
                )
                _log_decision(
                    RouteDecision("terra", MODELS["terra"], "test"),
                    {
                        "turn_id": "turn-1",
                        "api_call_count": 1,
                        "request": chat_request("Ez egy teszt prompt."),
                    },
                    cfg,
                )
            entry = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(entry["timestamp"], "2026-07-14T07:01:02+00:00")
            self.assertEqual(entry["prompt_preview"], "Ez egy teszt prompt.")

    def test_prompt_preview_is_single_line_and_not_truncated(self):
        text = "Első sor\n  második sor " + "á" * 60
        preview = _prompt_preview(responses_request(text))
        self.assertEqual(preview, "Első sor második sor " + "á" * 60)
        self.assertNotIn("\n", preview)


if __name__ == "__main__":
    unittest.main()
