"""Pact Router — cascade orchestration with PACT protocol.

Architecture:
  Triage (local, 0 tokens) → Executor (chosen route) → Judge (local, 0 tokens)
  ↳ Escalation loop: if Judge says fail, retry with next tier up

Cascade order: local → fireworks-cheap → fireworks-medium → fireworks-powerful
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List

from .protocol import (
    PACTSignal, Route, Verdict, next_route,
    triage as _triage, exec_signal as _exec, verdict as _verdict,
)
from .inference import LocalInference, FireworksInference, InferenceResult

logger = logging.getLogger(__name__)

# Routing table: difficulty → (local_model, fireworks_model)
# Configurable via env vars at container start.
_ROUTING = {
    "easy": {
        "local": lambda _: os.getenv("PACT_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        "fireworks": lambda _: os.getenv("PACT_FW_MEDIUM",
                                         "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    },
    "medium": {
        "local": lambda _: os.getenv("PACT_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        "fireworks": lambda d: os.getenv("PACT_FW_CHEAP" if d.get("diff") == "easy" else "PACT_FW_MEDIUM",
                                         "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    },
    "hard": {
        "local": lambda _: None,  # ponytail: skip local for hard tasks
        "fireworks": lambda _: os.getenv("PACT_FW_POWERFUL",
                                         "accounts/fireworks/models/llama-v3p1-405b-instruct"),
    },
}


class TriageAgent:
    """Classifies task difficulty using local model (0 Fireworks tokens)."""

    PROMPT = """Analyze this task and classify it. Return ONLY valid JSON with no markdown.

Task: {task}

{{
  "difficulty": "easy" | "medium" | "hard",
  "domain": "coding" | "math" | "reasoning" | "qa" | "writing" | "other",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation in <10 words"
}}"""

    def __init__(self, inference: LocalInference):
        self.inf = inference

    def analyze(self, task: str) -> PACTSignal:
        prompt = self.PROMPT.format(task=task[:1500])
        result = self.inf.generate(prompt, temperature=0.1)

        try:
            parsed = self._extract_json(result.output)
            diff = parsed.get("difficulty", "medium")
            domain = parsed.get("domain", "other")
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")

            # Determine initial route based on difficulty + confidence
            if diff == "hard":
                route = Route.FIREWORKS_CHEAP  # try cheapest fireworks first
            elif diff == "medium" and confidence < 0.7:
                route = Route.FIREWORKS_CHEAP
            else:
                route = Route.LOCAL

            return _triage(diff, domain, route.value, confidence, reasoning)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Triage parse error: %s | raw: %s", e, result.output[:150])
            return _triage("medium", "other", Route.FIREWORKS_CHEAP.value, 0.4, "parse_fallback")

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON from model output, handling markdown fences."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())


class ExecutorAgent:
    """Executes a task on the chosen inference route."""

    PROMPT = """Complete this task. Be concise and accurate.

{task}"""

    def __init__(self, local: LocalInference, fireworks: FireworksInference):
        self.local = local
        self.fireworks = fireworks

    def execute(self, task: str, route: Route, difficulty: str = "medium") -> InferenceResult:
        prompt = self.PROMPT.format(task=task[:3000])
        model = ""

        if route == Route.LOCAL:
            model = _ROUTING[difficulty]["local"]({})
            if model is None:
                # ponytail: local not available for this tier, fallback to fireworks
                logger.info("No local model for %s, falling back to fireworks", difficulty)
                return self.execute(task, Route.FIREWORKS_CHEAP, difficulty)
            result = self.local.generate(prompt)
        else:
            # Map route to Fireworks model key
            tier = "fireworks"
            if route == Route.FIREWORKS_CHEAP:
                tier = "fireworks"
            model = _ROUTING[difficulty][tier]({"diff": difficulty})
            result = self.fireworks.generate(model, prompt)

        return result


class JudgeAgent:
    """Validates output quality using local model (0 Fireworks tokens).

    Triggers escalation if output is low quality or incorrect.
    """

    PROMPT = """Evaluate this output for the task. Return ONLY valid JSON with no markdown.

Task: {task}
Output: {output}

{{
  "quality": "pass" | "fail",
  "confidence": 0.0-1.0,
  "issues": ["issue1"] or [],
  "escalate": true | false,
  "reasoning": "brief"
}}"""

    def __init__(self, inference: LocalInference):
        self.inf = inference

    def evaluate(self, task: str, result: InferenceResult, route: Route) -> PACTSignal:
        if not result.output.strip():
            return _verdict(Verdict.FAIL, 0.0, next_route(route).value if next_route(route) else "",
                           "empty_output")

        prompt = self.PROMPT.format(task=task[:800], output=result.output[:1500])
        judge_result = self.inf.generate(prompt, temperature=0.1)

        try:
            parsed = self._extract_json(judge_result.output)
            quality = parsed.get("quality", "pass")
            confidence = float(parsed.get("confidence", 0.5))
            escalate = parsed.get("escalate", False)
            issues = parsed.get("issues", [])

            if quality == "fail" or escalate:
                next_r = next_route(route)
                if next_r:
                    echo = "; ".join(issues) if issues else "quality_check"
                    return _verdict(Verdict.ESCALATE, confidence, next_r.value, echo)
                else:
                    # Already at max tier — accept whatever we have
                    return _verdict(Verdict.PASS, min(confidence, 0.5),
                                   reason="max_tier_accept")

            return _verdict(Verdict.PASS, confidence, reason="ok")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Judge parse error: %s", e)
            # ponytail: if judge can't parse, pass with low confidence
            return _verdict(Verdict.PASS, 0.4, reason="parse_fallback")

    @staticmethod
    def _extract_json(raw: str) -> dict:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Cascade router
# ---------------------------------------------------------------------------

class PactRouter:
    """Main orchestrator. Triage → Execute → Judge loop with escalation."""

    def __init__(self):
        self.local = LocalInference()
        self.fireworks = FireworksInference()
        self.triage = TriageAgent(self.local)
        self.executor = ExecutorAgent(self.local, self.fireworks)
        self.judge = JudgeAgent(self.local)

    def process(self, task: str) -> dict:
        """Process a single task, return structured result with audit trail."""
        start = time.time()
        signals: List[PACTSignal] = []

        # 1. Triage
        sig_triage = self.triage.analyze(task)
        signals.append(sig_triage)

        route = Route(sig_triage.data["route"])
        difficulty = sig_triage.data.get("diff", "medium")
        escalated = 0
        final_output = ""
        fireworks_tokens = 0
        max_attempts = int(os.getenv("PACT_MAX_ESCALATIONS", "3"))

        # 2-4. Cascade execution loop
        for attempt in range(max_attempts + 1):
            # Execute on current route
            exec_result = self.executor.execute(task, route, difficulty)
            signals.append(_exec(route.value, exec_result.model, exec_result.tokens))

            final_output = exec_result.output
            if route != Route.LOCAL:
                fireworks_tokens += exec_result.tokens

            # Judge the output
            sig_verdict = self.judge.evaluate(task, exec_result, route)
            signals.append(sig_verdict)

            if sig_verdict.data["q"] in (Verdict.PASS.value,):
                break  # Good enough

            if sig_verdict.data["q"] == Verdict.ESCALATE.value:
                next_r = next_route(route)
                if next_r:
                    route = next_r
                    escalated += 1
                    logger.info("Escalating → %s (attempt %d/%d)",
                               route.value, attempt + 1, max_attempts)
                    continue
                else:
                    break  # At max tier, accept output

            # Verdict.FAIL without escalation — still try next tier
            next_r = next_route(route)
            if next_r:
                route = next_r
                escalated += 1
            else:
                break

        elapsed = time.time() - start

        return {
            "pact": {
                "output": final_output,
                "route": route.value,
                "escalated": escalated,
                "fireworks_tokens": fireworks_tokens,
                "elapsed_ms": int(elapsed * 1000),
                "difficulty": difficulty,
                "signals": len(signals),
            },
            # ponytail: full signal trail for debugging/audit
            "_trace": [s.compact() for s in signals],
        }

    def process_batch(self, tasks: List[str], show_trace: bool = False) -> List[dict]:
        """Process multiple tasks sequentially."""
        results = []
        for task in tasks:
            r = self.process(task)
            if not show_trace:
                r.pop("_trace", None)
            results.append(r)
        return results
