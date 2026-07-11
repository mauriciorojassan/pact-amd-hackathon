"""Tests for PACT protocol serialization and core types."""

import json

from pact.protocol import (
    PACTSignal, PACTType, Route, triage, exec_signal, result, verdict,
    next_route,
)


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
        assert restored.data["domain"] == s.data["domain"]

    def test_compact_valid_json(self):
        s = exec_signal("local", "qwen2.5-1.5b", 0)
        parsed = json.loads(s.compact())
        assert parsed["t"] == "EXEC"
        assert parsed["d"]["route"] == "local"
        assert parsed["d"]["tokens"] == 0

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

    def test_unknown_route(self):
        assert next_route(None) is None  # type: ignore


class TestPactIntegration:
    """Mini integration tests with mock mode."""

    def test_router_process_returns_expected_keys(self):
        from pact.router import PactRouter
        import os
        os.environ["PACT_MOCK"] = "1"

        router = PactRouter()
        result = router.process("What is 2+2?")

        assert "pact" in result
        p = result["pact"]
        assert "output" in p
        assert "route" in p
        assert "fireworks_tokens" in p
        assert "elapsed_ms" in p
        assert "signals" in p
        assert p["fireworks_tokens"] >= 0

    def test_router_batch(self):
        from pact.router import PactRouter
        import os
        os.environ["PACT_MOCK"] = "1"

        router = PactRouter()
        tasks = ["Task 1", "Task 2", "Task 3"]
        results = router.process_batch(tasks, show_trace=False)

        assert len(results) == 3
        for r in results:
            assert "pact" in r
            assert "_trace" not in r  # stripped per arg
