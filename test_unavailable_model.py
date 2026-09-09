"""A model the account cannot use at all.

Distinct from a quota (which recovers) and from a 5xx (worth retrying). Observed
2026-09-09: a [spark] leaf died on `HTTP 400 The 'gpt-5.3-codex-spark' model is not
supported when using Codex with a ChatGPT account.` Spark was switched ON, so the
callable chain never ran; the error was neither quota nor transient, so no runtime
failover ran either, and the leaf simply aborted.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_router import (
    _durable_fallback_model,
    _is_model_unavailable,
    _tier_cooldown_remaining,
    run_llm_with_transient_failover,
)
from model_router.test_model_router import CALLABLE, MODELS, chat_request


REAL_ERROR = (
    "HTTP 400: {\"detail\":\"The 'gpt-5.3-codex-spark' model is not supported "
    "when using Codex with a ChatGPT account.\"}"
)


def _cfg(path, **overrides):
    cfg = {
        "enabled": True, "provider": "openai-codex", "models": MODELS,
        "callable": dict(CALLABLE), "fallbacks": {"spark": "luna", "luna": "terra", "sol": "terra"},
        "thresholds": {"sol_min_chars": 3500, "luna_max_chars": 700},
        "effort": {},
        "cooldown": {"enabled": True, "path": str(path), "quota_seconds": 900,
                     "allowed_fails": 3, "failure_window_seconds": 60, "failure_seconds": 60},
    }
    cfg.update(overrides)
    return cfg


class PredicateTests(unittest.TestCase):
    def test_the_observed_refusal_is_recognised(self):
        self.assertTrue(_is_model_unavailable(RuntimeError(REAL_ERROR)))

    def test_a_404_no_access_refusal_is_recognised(self):
        self.assertTrue(_is_model_unavailable(
            RuntimeError("HTTP 404: The model `x` does not exist or you do not have access to it.")
        ))

    def test_a_quota_429_is_not_a_model_problem(self):
        self.assertFalse(_is_model_unavailable(RuntimeError("HTTP 429: usage limit reached")))

    def test_a_server_error_is_not_a_model_problem(self):
        self.assertFalse(_is_model_unavailable(RuntimeError("HTTP 503: upstream connect error")))

    def test_an_ordinary_400_is_not_a_model_problem(self):
        """Only an availability refusal counts; a malformed request must still raise."""
        self.assertFalse(_is_model_unavailable(RuntimeError("HTTP 400: invalid 'messages' field")))


class DurableFallbackTests(unittest.TestCase):
    def test_it_follows_the_configured_chain_not_the_transient_map(self):
        """`fallbacks: spark -> luna` is the operator's policy; the transient map
        would have sent this to Sol instead."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            self.assertEqual(_durable_fallback_model(MODELS["spark"], cfg), MODELS["luna"])

    def test_a_disabled_link_is_skipped_and_the_chain_continues(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            cfg["callable"]["luna"] = False
            self.assertEqual(_durable_fallback_model(MODELS["spark"], cfg), MODELS["terra"])

    def test_no_chain_entry_means_no_substitute(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json", fallbacks={})
            self.assertIsNone(_durable_fallback_model(MODELS["spark"], cfg))

    def test_design_work_is_not_downgraded_off_sol(self):
        """An unavailable model is no reason to break the Sol-only design boundary."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            request = chat_request("Csinald meg a UI-t, a gombok legyenek kerekek es szep a layout")
            self.assertIsNone(_durable_fallback_model(MODELS["sol"], cfg, request))


class MiddlewareTests(unittest.TestCase):
    def test_the_leaf_survives_on_the_configured_substitute(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")
            seen = []

            def call(request):
                seen.append(request["model"])
                if request["model"] == MODELS["spark"]:
                    raise RuntimeError(REAL_ERROR)
                return {"ok": True}

            with patch("model_router._load_config", return_value=cfg):
                result = run_llm_with_transient_failover(
                    request={**chat_request("Produce a read-only evidence report."),
                             "model": MODELS["spark"]},
                    next_call=call, retry_call=call,
                    provider="openai-codex", turn_id="unavailable-turn",
                )

            self.assertEqual(result, {"ok": True})
            self.assertEqual(seen, [MODELS["spark"], MODELS["luna"]])

    def test_the_tier_is_benched_so_the_next_leaf_does_not_repeat_it(self):
        """Nothing here recovers by waiting, so the cooldown is long on purpose."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")

            def call(request):
                if request["model"] == MODELS["spark"]:
                    raise RuntimeError(REAL_ERROR)
                return {"ok": True}

            with patch("model_router._load_config", return_value=cfg):
                run_llm_with_transient_failover(
                    request={**chat_request("Anything."), "model": MODELS["spark"]},
                    next_call=call, retry_call=call,
                    provider="openai-codex", turn_id="bench-turn",
                )
                self.assertGreater(_tier_cooldown_remaining("spark", cfg), 3600)

    def test_an_ordinary_400_still_aborts(self):
        """Only an availability refusal is survivable; a bad request must surface."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d) / "c.json")

            def call(_request):
                raise RuntimeError("HTTP 400: invalid 'messages' field")

            with patch("model_router._load_config", return_value=cfg):
                with self.assertRaises(RuntimeError):
                    run_llm_with_transient_failover(
                        request={**chat_request("Anything."), "model": MODELS["spark"]},
                        next_call=call, retry_call=call,
                        provider="openai-codex", turn_id="bad-request-turn",
                    )


if __name__ == "__main__":
    unittest.main()
