"""Tests for PACT protocol types, heuristic triage, heuristic judge, and router."""

from pact.protocol import (
    PACTSignal, PACTType, Route, triage, exec_signal, result, verdict,
    next_route,
)
from pact.router import (
    _heuristic_triage, _heuristic_judge, _outputs_differ, _domain_validate,
)
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
        parsed = __import__("json").loads(s.compact())
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

    def test_medium_explain(self):
        s = _heuristic_triage("Explain how gradient descent works with examples")
        assert s.data["diff"] == "medium"

    def test_domain_detection_coding(self):
        s = _heuristic_triage("Write a Dockerfile for a Python Flask app")
        assert s.data["domain"] == "coding"

    def test_domain_math(self):
        s = _heuristic_triage("Solve this equation: 2x + 5 = 15")
        assert s.data["domain"] == "math"

    def test_always_starts_at_cheapest(self):
        s = _heuristic_triage("hard debugging task")
        assert s.data["route"] == "fireworks-cheap"


class TestOutputsDiffer:
    def test_same_outputs(self):
        assert not _outputs_differ("hello world", "hello world")

    def test_similar_outputs(self):
        assert not _outputs_differ("The capital is Paris",
                                    "Paris is the capital of France")

    def test_different_outputs(self):
        assert _outputs_differ("The answer is 42",
                                "aardvark zebra monkey")


class TestDomainValidate:
    def test_math_with_numbers_passes(self):
        assert _domain_validate("math", "The answer is 42") is None

    def test_math_without_numbers_fails(self):
        assert _domain_validate("math", "I don't know") is not None

    def test_coding_with_code_passes(self):
        assert _domain_validate("coding", "def hello():\n    pass") is None

    def test_coding_without_code_fails(self):
        assert _domain_validate("coding", "Sure, I can help") is not None

    def test_unknown_domain_skips(self):
        assert _domain_validate("other", "blah blah") is None


class TestHeuristicJudge:
    def test_pass_on_good_output(self):
        result = InferenceResult(
            output="The capital of France is Paris.",
            model="gpt-oss-120b", tokens=10,
        )
        v = _heuristic_judge("What is the capital of France?",
                             result, Route.FIREWORKS_CHEAP, "qa")
        assert v.data["q"] == "pass"

    def test_escalate_on_empty(self):
        result = InferenceResult(output="", model="gpt-oss-120b", tokens=0)
        v = _heuristic_judge("Some task", result,
                             Route.FIREWORKS_CHEAP, "other")
        assert v.data["q"] == "escalate"
        assert v.data["e"] == "fireworks-medium"

    def test_escalate_on_error(self):
        result = InferenceResult(output="", model="gpt-oss-120b",
                                 tokens=0, error="API timeout")
        v = _heuristic_judge("Some task", result,
                             Route.FIREWORKS_CHEAP, "other")
        assert v.data["q"] == "escalate"

    def test_escalate_on_refusal(self):
        result = InferenceResult(
            output="I'm sorry, I cannot answer that question.",
            model="gpt-oss-120b", tokens=10,
        )
        v = _heuristic_judge("Write code for auth",
                             result, Route.FIREWORKS_CHEAP, "coding")
        assert v.data["q"] == "escalate"

    def test_escalate_on_gibberish(self):
        result = InferenceResult(
            output="aaaaaaaaaaaa bbbbbbbbbbbb",
            model="gpt-oss-120b", tokens=5,
        )
        v = _heuristic_judge("Some task", result,
                             Route.FIREWORKS_CHEAP, "other")
        assert v.data["q"] == "escalate"

    def test_domain_validation_catches(self):
        result = InferenceResult(
            output="Sure, I can help with that",
            model="gpt-oss-120b", tokens=6,
        )
        v = _heuristic_judge("Write a Python function",
                             result, Route.FIREWORKS_CHEAP, "coding")
        assert v.data["q"] == "escalate"

    def test_fail_at_max_tier(self):
        """At max tier with bad output, judge returns FAIL, not PASS."""
        result = InferenceResult(output="", model="deepseek-v4-pro", tokens=0)
        v = _heuristic_judge("Some task", result,
                             Route.FIREWORKS_POWERFUL, "other")
        assert v.data["q"] == "fail"
        assert "max_tier" in v.data["r"]


class TestRouterIntegration:
    def test_process_returns_expected_keys(self):
        import os
        os.environ["PACT_MOCK"] = "1"
        from pact.router import PactRouter

        router = PactRouter()
        result = router.process("What is 2+2?")

        assert "pact" in result
        p = result["pact"]
        assert "output" in p
        assert "route" in p
        assert "fireworks_tokens" in p
        assert "elapsed_ms" in p
        assert "difficulty" in p
        assert "domain" in p

    def test_batch(self):
        import os
        os.environ["PACT_MOCK"] = "1"
        from pact.router import PactRouter

        router = PactRouter()
        results = router.process_batch(["Task 1", "Task 2"], show_trace=True)
        assert len(results) == 2
        for r in results:
            assert "pact" in r

    def test_self_consistency_skipped_for_easy_mock(self):
        """Easy tasks skip self-consistency."""
        import os
        os.environ["PACT_MOCK"] = "1"
        from pact.router import PactRouter

        router = PactRouter()
        result = router.process("What is 2+2?")
        assert result["pact"]["signals"] == 3  # triage + exec + verdict


class TestMaxTierFail:
    """TDD: at max tier, judge must return FAIL (not PASS), cascade must not crash."""

    def test_escalate_or_pass_at_max_tier_returns_fail(self):
        """_escalate_or_pass at FIREWORKS_POWERFUL returns FAIL, not PASS."""
        from pact.router import _escalate_or_pass
        from pact.protocol import Route
        s = _escalate_or_pass(Route.FIREWORKS_POWERFUL, 0.3, "test_reason")
        assert s.data["q"] == "fail"
        assert "max_tier" in s.data["r"]

    def test_heuristic_judge_max_tier_empty_returns_fail(self):
        """At max tier with empty output, heuristic judge returns FAIL."""
        from pact.router import _heuristic_judge
        from pact.protocol import Route
        from pact.inference import InferenceResult
        result = InferenceResult(output="", model="deepseek-v4-pro", tokens=0)
        v = _heuristic_judge("Some task", result, Route.FIREWORKS_POWERFUL, "other")
        assert v.data["q"] == "fail"

    def test_heuristic_judge_max_tier_gibberish_returns_fail(self):
        """At max tier with gibberish, heuristic judge returns FAIL."""
        from pact.router import _heuristic_judge
        from pact.protocol import Route
        from pact.inference import InferenceResult
        result = InferenceResult(
            output="aaaaaaaaaaaa bbbbbbbbbbbb",
            model="deepseek-v4-pro", tokens=5,
        )
        v = _heuristic_judge("Some task", result, Route.FIREWORKS_POWERFUL, "other")
        assert v.data["q"] == "fail"


class TestCmdEvalStdin:
    """TDD: cmd_eval must handle non-JSON piped input without seek crash."""

    def test_cmd_eval_non_json_stdin(self):
        """Plain text piped to 'pact eval' must not crash with seek error."""
        import io
        import json
        import sys
        from unittest.mock import MagicMock
        from pact.__main__ import cmd_eval

        # Simulate non-seekable pipe stdin
        pipe_stdin = MagicMock()
        pipe_stdin.read.return_value = "What is the capital of France?"
        pipe_stdin.seek.side_effect = io.UnsupportedOperation("not seekable")

        old_stdin = sys.stdin
        sys.stdin = pipe_stdin

        stdout_mock = MagicMock()
        old_stdout = sys.stdout
        sys.stdout = stdout_mock

        try:
            # Must NOT raise io.UnsupportedOperation from seek(0)
            cmd_eval(type("args", (), {})())
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

        # Should have printed JSON to stdout
        written = "".join(
            call[0][0] for call in stdout_mock.write.call_args_list
        )
        data = json.loads(written)
        assert "pact" in data


class TestAPITimeout:
    """TDD: FireworksInference must set an explicit, configurable API timeout."""

    def test_fireworks_client_has_explicit_timeout(self):
        """OpenAI client must be created with an explicit timeout."""
        import os
        from pact.inference import FireworksInference
        os.environ["PACT_MOCK"] = "0"
        os.environ["FIREWORKS_API_KEY"] = "fake-key-for-test"
        os.environ["PACT_API_TIMEOUT"] = "15"
        try:
            fw = FireworksInference()
            # _client is an OpenAI instance whose _client is an httpx.Client
            # httpx.Client.timeout is a httpx.Timeout object
            timeout = fw._client._client.timeout
            # The connect timeout should be 15s, not the default 600s read
            assert timeout.connect == 15.0, f"Expected 15.0, got {timeout.connect}"
        finally:
            os.environ.pop("PACT_API_TIMEOUT", None)
            os.environ.pop("FIREWORKS_API_KEY", None)
            os.environ["PACT_MOCK"] = "1"


class TestNoSharedMutation:
    """TDD: FireworksInference.generate() must not mutate self.model."""

    def test_generate_does_not_mutate_self_model(self):
        """Calling generate() with a model must not leave it on self.model."""
        import os
        from pact.inference import FireworksInference
        os.environ["PACT_MOCK"] = "1"
        fw = FireworksInference()
        original_model = fw.model

        fw.generate("accounts/fireworks/models/gpt-oss-120b", "test prompt")
        assert fw.model == original_model, (
            f"self.model was mutated to {fw.model!r}, expected {original_model!r}"
        )

        fw.generate("accounts/fireworks/models/kimi-k2p6", "another prompt")
        assert fw.model == original_model, (
            f"self.model was mutated to {fw.model!r}, expected {original_model!r}"
        )
