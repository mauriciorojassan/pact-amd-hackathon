"""Pact Router — cascade orchestration with PACT protocol.

  Triage (heuristic, 0 tokens) → Executor (FW route) → Judge (heuristic, 0 tokens)
  ↳ Escalation: if judge flags bad output, retry with next tier up

Cascade: fireworks-cheap → fireworks-medium → fireworks-powerful
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import List

from .protocol import (
    PACTSignal, Route, Verdict, next_route,
    triage as _triage, exec_signal as _exec, verdict as _verdict,
)
from .inference import LocalInference, FireworksInference, InferenceResult

logger = logging.getLogger(__name__)

# ── Fireworks model IDs (real, verified 2026-07) ──────────────────────
FW_CHEAP = os.getenv("PACT_FW_CHEAP",
                     "accounts/fireworks/models/gpt-oss-120b")
FW_MEDIUM = os.getenv("PACT_FW_MEDIUM",
                      "accounts/fireworks/models/kimi-k2p6")
FW_POWERFUL = os.getenv("PACT_FW_POWERFUL",
                        "accounts/fireworks/models/deepseek-v4-pro")

ROUTE_MODEL = {
    Route.FIREWORKS_CHEAP: FW_CHEAP,
    Route.FIREWORKS_MEDIUM: FW_MEDIUM,
    Route.FIREWORKS_POWERFUL: FW_POWERFUL,
}


# ── Heuristic triage (zero tokens, no LLM call) ────────────────────────

# ponytail: keyword patterns are cheaper than any model call
_EASY = re.compile(
    r"(what (is|are|was|were) |define |explain briefly|"
    r"write a one.liner|simple |basic |hello world|"
    r"capital of |current (time|date|year)|"
    r"convert |translate |summarize)", re.I
)

_HARD = re.compile(
    r"(debug|optimize|refactor|complex|distributed|"
    r"concurrency|deadlock|race condition|"
    r"architectur|design pattern|"
    r"multi.step|end.to.end|production.grade)", re.I
)

_DOMAIN = [
    (re.compile(r"(python|rust|go|java|typescript|javascript|"
                 r"react|vue|django|flask|sql|bash|docker|kubernetes)", re.I),
     "coding"),
    (re.compile(r"(math|equation|derivative|integral|probability|"
                 r"statistics|algebra)", re.I),
     "math"),
    (re.compile(r"(write |compose |draft |essay|paragraph|email)", re.I),
     "writing"),
    (re.compile(r"(explain |describe |what is |how does |why does)", re.I),
     "qa"),
]


def _heuristic_triage(task: str) -> PACTSignal:
    """Classify task difficulty and domain using patterns. 0 tokens."""
    task_clean = task.strip()
    length = len(task_clean)

    # Domain detection
    domain = "other"
    for pattern, label in _DOMAIN:
        if pattern.search(task_clean):
            domain = label
            break

    # Difficulty scoring
    score = 0

    # Length signals
    if length > 300:
        score += 1
    elif length < 40:
        score -= 1

    # Keyword signals
    if _EASY.search(task_clean):
        score -= 1
    if _HARD.search(task_clean):
        score += 2

    # Structural signals
    if task_clean.count("\n") > 5:
        score += 1
    # "Explain X" (not "briefly") is usually medium+
    if re.search(r"\bexplain\b", task_clean[:80], re.I) \
            and "briefly" not in task_clean[:80].lower():
        score += 1

    diff = "easy" if score <= 0 else ("hard" if score >= 2 else "medium")
    # Always start at cheapest Fireworks tier (local model unavailable)
    route = Route.FIREWORKS_CHEAP
    confidence = max(0.5, min(0.95, 0.7 + score * 0.1))

    return _triage(diff, domain, route.value, confidence,
                   f"len={length} score={score}")


# ── Heuristic judge (zero tokens, no LLM call) ─────────────────────────

def _heuristic_judge(task: str, result: InferenceResult, route: Route) -> PACTSignal:
    """Validate output quality without any model call.

    Rules:
      - Empty output → escalate
      - Error from API → escalate
      - Very short output for a complex task → escalate
      - Otherwise → pass
    """
    output = result.output.strip()
    task_clean = task.strip()

    # Empty output or API error
    if not output:
        nxt = next_route(route)
        if nxt:
            return _verdict(Verdict.ESCALATE, 0.0, nxt.value, "empty_output")
        # At max tier — tried everything, accept what we have
        return _verdict(Verdict.PASS, 0.1, reason="empty_but_max_tier")

    if result.error:
        nxt = next_route(route)
        if nxt:
            return _verdict(Verdict.ESCALATE, 0.2, nxt.value, f"api_error:{result.error[:60]}")
        return _verdict(Verdict.PASS, 0.3, reason="error_but_max_tier")

    # Suspiciously short output for a verbose task
    task_words = len(task_clean.split())
    output_words = len(output.split())

    if task_words > 20 and output_words < 3:
        nxt = next_route(route)
        if nxt:
            return _verdict(Verdict.ESCALATE, 0.3, nxt.value, "too_short")
        return _verdict(Verdict.PASS, 0.4, reason="short_but_max_tier")

    # Check for common failure patterns
    fail_patterns = [
        r"(i( am|'m) (sorry|unable|cannot|not able))",
        r"(as an ai|language model)",
        r"(error|exception|traceback)",  # only if output looks like error
    ]
    for pat in fail_patterns:
        if re.search(pat, output[:200], re.I):
            nxt = next_route(route)
            if nxt:
                return _verdict(Verdict.ESCALATE, 0.4, nxt.value, f"pattern:{pat[:20]}")
            return _verdict(Verdict.PASS, 0.5, reason="refusal_but_max")

    # Looks good
    return _verdict(Verdict.PASS, 0.85, reason="heuristic_ok")


# ── Executor ───────────────────────────────────────────────────────────

class ExecutorAgent:
    """Executes a task on the chosen Fireworks route."""

    PROMPT = """Complete this task. Be concise and accurate.

{task}"""

    def __init__(self, local: LocalInference, fireworks: FireworksInference):
        self.local = local
        self.fireworks = fireworks

    def execute(self, task: str, route: Route) -> InferenceResult:
        prompt = self.PROMPT.format(task=task[:3000])
        model = ROUTE_MODEL.get(route)
        if not model:
            logger.warning("No model for route %s, using cheap", route)
            model = FW_CHEAP
        return self.fireworks.generate(model, prompt)


# ── Cascade router ─────────────────────────────────────────────────────

class PactRouter:
    """Zero-token triage + judge. Cascade: cheap → medium → powerful."""

    def __init__(self):
        self.local = LocalInference()
        self.fireworks = FireworksInference()
        self.executor = ExecutorAgent(self.local, self.fireworks)

    def process(self, task: str) -> dict:
        start = time.time()
        signals: List[PACTSignal] = []

        # 1. Heuristic triage (0 tokens)
        sig_triage = _heuristic_triage(task)
        signals.append(sig_triage)

        route = Route(sig_triage.data["route"])
        escalated = 0
        final_output = ""
        fireworks_tokens = 0
        max_attempts = int(os.getenv("PACT_MAX_ESCALATIONS", "3"))

        # 2-4. Cascade execution loop
        for attempt in range(max_attempts + 1):
            # Execute on current route
            exec_result = self.executor.execute(task, route)
            signals.append(_exec(route.value, exec_result.model, exec_result.tokens,
                                error=exec_result.error or ""))

            final_output = exec_result.output
            fireworks_tokens += exec_result.tokens

            # Heuristic judge (0 tokens)
            sig_verdict = _heuristic_judge(task, exec_result, route)
            signals.append(sig_verdict)

            if sig_verdict.data["q"] == Verdict.PASS.value:
                break

            if sig_verdict.data["q"] == Verdict.ESCALATE.value:
                nxt = next_route(route)
                if nxt:
                    route = nxt
                    escalated += 1
                    logger.info("Escalating → %s (attempt %d/%d)",
                               route.value, attempt + 1, max_attempts)
                    continue
                break

            # Verdict.FAIL without escalation path
            nxt = next_route(route)
            if nxt:
                route = nxt
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
                "difficulty": sig_triage.data.get("diff", "?"),
                "signals": len(signals),
            },
            "_trace": [s.compact() for s in signals],
        }

    def process_batch(self, tasks: List[str], show_trace: bool = False) -> List[dict]:
        results = []
        for task in tasks:
            r = self.process(task)
            if not show_trace:
                r.pop("_trace", None)
            results.append(r)
        return results
