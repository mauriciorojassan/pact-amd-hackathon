"""Tests for PACT protocol types, heuristic triage, and heuristic judge."""

import json

from pact.protocol import (
    PACTSignal, PACTType, Route, triage, exec_signal, result, verdict,
    next_route,
)
from pact.router import _heuristic_triage, _heuristic_judge
from pact.inference import InferenceResult


class TestPACTTypes:
    def test_triage_signal(self):
        s = triage("easy", "math", "local", 0.92, "simple arithmetic")
        assert s.type == PACTType.TRIAGE
        assert s.data["diff"] == "easy"
        assert s.data["domain"] == "math"
        assert s.data["route"] == "local"
        assert s.data["conf"] == 0.92

    def test_compact_roundtrip(self):
        s = triage("hard", "coding", "fireworks-powerful", 0.85)
        compact = s.compact()
        restored = PACTSignal.from_compact(compact)
        assert restored.type == s.type
        assert restored.data["diff"] == s.data["diff"]

    def test_compact_valid_json(self):
        s = exec_signal("local", "qwen2.5-1.5b", 0)
        parsed = json.loads(s.compact())
        assert parsed["t"] == "EXEC"
        assert parsed["d"]["route"] == "local"

    def test_result_signal(self):
        s = result("def hello(): pass", "local", "qwen2.5-1.5b", 0)
        assert s.type == PACTType.RESULT
        assert "hello" in s.data["o"]

    def test_verdict_pass(self):
        s = verdict("pass", 0.95)
        assert s.data["q"] == "pass"
        assert s.data["c"] == 0.95

    def test_verdict_escalate(self):
        s = verdict("fail", 0.4, "fireworks-powerful", "low confidence")
        assert s.data["q"] == "fail"
        assert s.data["e"] == "fireworks-powerful"

    def test_next_route(self):
        assert next_route(Route.LOCAL) == Route.FIREWORKS_CHEAP
        assert next_route(Route.FIREWORKS_CHEAP) == Route.FIREWORKS_MEDIUM
        assert next_route(Route.FIREWORKS_MEDIUM) == Route.FIREWORKS_POWERFUL
        assert next_route(Route.FIREWORKS_POWERFUL) is None


class TestHeuristicTriage:
    def test_easy_qa(self):
        s = _heuristic_triage("What is the capital of France?")
        assert s.data["diff"] == "easy"
        assert s.data["route"] == "fireworks-cheap"
        assert s.data["domain"] in ("qa", "other")

    def test_easy_code_oneliner(self):
        s = _heuristic_triage("Write a Python one-liner to reverse a string")
        assert s.data["diff"] == "easy"
        assert s.data["domain"] == "coding"

    def test_hard_debug(self):
        s = _heuristic_triage(
            "Debug this distributed system deadlock in the consensus "
            "protocol across multiple data centers"
        )
        assert s.data["diff"] == "hard"

    def test_medium_complex(self):
        s = _heuristic_triage("Explain how gradient descent works with examples")
        assert s.data["diff"] == "medium"

    def test_domain_detection(self):
        s = _heuristic_triage("Write a Dockerfile for a Python Flask app")
        assert s.data["domain"] == "coding"

    def test_writing_domain(self):
        s = _heuristic_triage("Write a persuasive email to a client")
        assert s.data["domain"] == "writing"

    def test_math_domain(self):
        s = _heuristic_triage("Solve this equation: 2x + 5 = 15")
        assert s.data["domain"] == "math"

    def test_always_starts_at_cheapest(self):
        s = _heuristic_triage("hard debugging task")
        assert s.data["route"] == "fireworks-cheap"


class TestHeuristicJudge:
    def test_pass_on_good_output(self):
        result = InferenceResult(
            output="The capital of France is Paris.",
            model="gpt-oss-120b",
            tokens=10,
        )
        v = _heuristic_judge("What is the capital of France?", result, Route.FIREWORKS_CHEAP)
        assert v.data["q"] == "pass"

    def test_escalate_on_empty(self):
        result = InferenceResult(output="", model="gpt-oss-120b", tokens=0)
        v = _heuristic_judge("Some task", result, Route.FIREWORKS_CHEAP)
        assert v.data["q"] == "escalate"
        assert v.data["e"] == "fireworks-medium"

    def test_escalate_on_error(self):
        result = InferenceResult(output="", model="gpt-oss-120b",
                                 tokens=0, error="API timeout")
        v = _heuristic_judge("Some task", result, Route.FIREWORKS_CHEAP)
        assert v.data["q"] == "escalate"

    def test_pass_at_max_tier(self):
        result = InferenceResult(output="", model="deepseek-v4-pro", tokens=0)
        v = _heuristic_judge("Some task", result, Route.FIREWORKS_POWERFUL)
        # At max tier, we pass (accept whatever we have)
        assert v.data["q"] == "pass"


class TestRouterIntegration:
    """Integration tests with mock model."""

    def test_process_returns_expected_keys(self):
        import os
        from pact.router import PactRouter

        os.environ["PACT_MOCK"] = "1"
        router = PactRouter()
        result = router.process("What is 2+2?")

        assert "pact" in result
        p = result["pact"]
        assert "output" in p
        assert "route" in p
        assert "fireworks_tokens" in p
        assert "elapsed_ms" in p
        assert "difficulty" in p

    def test_batch(self):
        import os
        from pact.router import PactRouter

        os.environ["PACT_MOCK"] = "1"
        router = PactRouter()
        results = router.process_batch(["Task 1", "Task 2"], show_trace=True)

        assert len(results) == 2
        for r in results:
            assert "pact" in r
