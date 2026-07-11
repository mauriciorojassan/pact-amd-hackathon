"""Pact Router — cascade orchestration with PACT protocol.

  Triage (heuristic, 0 tokens) → Executor (FW route) → Judge (heuristic, 0 tokens)
  ↳ Escalation: if judge flags bad output, retry with next tier up
  ↳ Self-consistency: for medium/hard tasks, run cheap model twice

Cascade: fireworks-cheap → fireworks-medium → fireworks-powerful
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import List, Optional

from .protocol import (
    PACTSignal, Route, Verdict, next_route,
    triage as _triage, exec_signal as _exec, verdict as _verdict,
)
from .inference import LocalInference, FireworksInference, InferenceResult

logger = logging.getLogger(__name__)

# ── Fireworks model IDs ────────────────────────────────────────────────
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

# ── Routing config ─────────────────────────────────────────────────────
# ponytail: env-configurable thresholds
_SELF_CONSISTENCY_ENABLED = os.getenv("PACT_SELF_CONSISTENCY", "1") == "1"
_SELF_CONSISTENCY_THRESHOLD = float(os.getenv("PACT_SC_THRESHOLD", "0.7"))


# ═══════════════════════════════════════════════════════════════════════
#  HEURISTIC TRIAGE  (0 tokens)
# ═══════════════════════════════════════════════════════════════════════

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
                r"statistics|algebra|arithmetic|calculate|compute)", re.I),
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

    domain = "other"
    for pattern, label in _DOMAIN:
        if pattern.search(task_clean):
            domain = label
            break

    score = 0
    if length > 300:
        score += 1
    elif length < 40:
        score -= 1

    if _EASY.search(task_clean):
        score -= 1
    if _HARD.search(task_clean):
        score += 2

    if task_clean.count("\n") > 5:
        score += 1
    if re.search(r"\bexplain\b", task_clean[:80], re.I) \
            and "briefly" not in task_clean[:80].lower():
        score += 1

    diff = "easy" if score <= 0 else ("hard" if score >= 2 else "medium")
    route = Route.FIREWORKS_CHEAP
    confidence = max(0.5, min(0.95, 0.7 + score * 0.1))

    return _triage(diff, domain, route.value, confidence,
                   f"len={length} score={score}")


# ═══════════════════════════════════════════════════════════════════════
#  HEURISTIC JUDGE  (0 tokens — domain-aware validation)
# ═══════════════════════════════════════════════════════════════════════

# ponytail: regex-based validation catches obvious failures at 0 token cost
_GIBBERISH = re.compile(
    r"(.)\1{10,}|"                         # repeated char: "aaaaaa..."
    r"(\b\w+\b)( \1){5,}|"                 # repeated word: "foo foo foo foo foo foo"
    r"^[^a-zA-Z0-9]{10,}$",                # only symbols
)

_REFUSAL = re.compile(
    r"(i( |')m (sorry|unable|cannot|not (able|capable)))|"
    r"(as an ai|as a (large )?language model|"
    r"i cannot|cannot (answer|respond|provide|complete)|"
    r"i don't (know|have|understand)|"
    r"it is not (possible|appropriate|ethical))", re.I
)

_STUCK = re.compile(
    r"^.{0,5}$",  # very short answer (1-5 chars) for non-trivial tasks
)

# Domain-specific validation: (pattern, issue, must_match)
#   must_match=True  → pattern SHOULD be found; if NOT found → issue
#   must_match=False → pattern should NOT be found; if FOUND → issue
_DOMAIN_VALIDATORS = {
    "math": [
        (re.compile(r"\d+"), "no_numbers", True),         # math should have numbers
        (re.compile(r"^(yes|no|maybe|idk|i don't know)$", re.I), "non_answer", False),
    ],
    "coding": [
        (re.compile(r"(def |function|class |import |from |```|=>|->)"), "no_code", True),
        (re.compile(r"^(yes|no|maybe|sure)$", re.I), "non_answer", False),
    ],
    "qa": [
        (re.compile(r"[A-Z][a-z]+"), "no_prose", True),  # Q&A should have real words
        (re.compile(r"(i don't know|i'm not sure|unable to answer)", re.I), "refusal", False),
    ],
    "writing": [
        (re.compile(r"\b\w+\b.*\b\w+\b.*\b\w+\b"), "too_short", True),  # at least 3 words
    ],
}


def _domain_validate(domain: str, output: str) -> Optional[str]:
    """Domain-specific output validation. Returns issue key or None."""
    validators = _DOMAIN_VALIDATORS.get(domain)
    if not validators:
        return None
    for pattern, issue, must_match in validators:
        found = bool(pattern.search(output))
        if must_match and not found:
            return issue
        if not must_match and found:
            return issue
    return None


def _escalate_or_pass(route: Route, confidence: float,
                      reason: str) -> PACTSignal:
    """Escalate to next tier, or FAIL if already at max tier.
    
    At max tier there's nowhere to escalate — FAIL tells the cascade
    loop to stop and lets the scoring harness distinguish bad answers
    from PASS (acceptable) ones.
    """
    nxt = next_route(route)
    if nxt is not None:
        return _verdict(Verdict.ESCALATE, confidence, nxt.value, reason)
    return _verdict(Verdict.FAIL, confidence * 0.5, reason=f"max_tier:{reason}")


def _heuristic_judge(task: str, result: InferenceResult,
                     route: Route, domain: str) -> PACTSignal:
    """Validate output quality. 0 tokens. Domain-aware."""
    output = result.output.strip()
    task_clean = task.strip()

    # ── Empty / Error ──────────────────────────────────────────────────
    if not output:
        return _escalate_or_pass(route, 0.0, "empty_output")

    if result.error:
        return _escalate_or_pass(route, 0.2,
                                f"api_error:{result.error[:60]}")

    # ── Gibberish ──────────────────────────────────────────────────────
    if _GIBBERISH.search(output):
        return _escalate_or_pass(route, 0.1, "gibberish_output")

    # ── Refusal patterns ───────────────────────────────────────────────
    if _REFUSAL.search(output[:300]):
        return _escalate_or_pass(route, 0.3, "model_refusal")

    # ── Too short for non-trivial tasks ────────────────────────────────
    task_words = len(task_clean.split())
    output_words = len(output.split())
    if task_words > 15 and output_words <= 2:
        return _escalate_or_pass(route, 0.3, "too_short")

    # ── Domain-specific validation ─────────────────────────────────────
    issue = _domain_validate(domain, output)
    if issue:
        return _escalate_or_pass(route, 0.4, f"domain:{issue}")

    # ── Pass ───────────────────────────────────────────────────────────
    return _verdict(Verdict.PASS, 0.85, reason="heuristic_ok")


# ═══════════════════════════════════════════════════════════════════════
#  SELF-CONSISTENCY  (verify cheap model by running it twice)
# ═══════════════════════════════════════════════════════════════════════

def _outputs_differ(a: str, b: str, threshold: float = 0.3) -> bool:
    """Check if two outputs are meaningfully different.
    Simple ratio-based: if word overlap is below threshold, they differ.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return True
    intersection = words_a & words_b
    overlap = len(intersection) / min(len(words_a), len(words_b))
    return overlap < threshold


# ═══════════════════════════════════════════════════════════════════════
#  EXECUTOR
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  CASCADE ROUTER
# ═══════════════════════════════════════════════════════════════════════

class PactRouter:
    """Cascade: cheap → medium → powerful. Domain-aware judge.
    Medium/hard tasks get self-consistency verification.
    """

    def __init__(self):
        self.local = LocalInference()
        self.fireworks = FireworksInference()
        self.executor = ExecutorAgent(self.local, self.fireworks)

    def process(self, task: str) -> dict:
        start = time.time()
        signals: List[PACTSignal] = []

        # ── 1. Triaje (heuristic, 0 tokens) ────────────────────────────
        sig_triage = _heuristic_triage(task)
        signals.append(sig_triage)
        route = Route(sig_triage.data["route"])
        domain = sig_triage.data.get("domain", "other")
        difficulty = sig_triage.data.get("diff", "easy")
        escalated = 0
        final_output = ""
        fireworks_tokens = 0
        max_attempts = int(os.getenv("PACT_MAX_ESCALATIONS", "3"))

        # ── 2-4. Cascade loop ──────────────────────────────────────────
        for attempt in range(max_attempts + 1):
            exec_result = self.executor.execute(task, route)
            signals.append(_exec(route.value, exec_result.model, exec_result.tokens,
                                error=exec_result.error or ""))
            final_output = exec_result.output
            fireworks_tokens += exec_result.tokens

            # ── Self-consistency (medium/hard, cheap route only) ───────
            if (_SELF_CONSISTENCY_ENABLED
                    and route == Route.FIREWORKS_CHEAP
                    and difficulty in ("medium", "hard")):
                logger.debug("Self-consistency check for %s task", difficulty)
                sc_result = self.executor.execute(task, route)
                fireworks_tokens += sc_result.tokens
                signals.append(_exec("self-consistency", sc_result.model,
                                    sc_result.tokens, error=sc_result.error or ""))

                if _outputs_differ(final_output, sc_result.output,
                                   _SELF_CONSISTENCY_THRESHOLD):
                    logger.info("Self-consistency mismatch → escalating")
                    nxt = next_route(route)
                    if nxt:
                        route = nxt
                        escalated += 1
                        signals.append(_verdict(Verdict.ESCALATE, 0.5,
                                               nxt.value, "self_consistency_mismatch"))
                        continue
                    else:
                        signals.append(_verdict(Verdict.PASS, 0.4,
                                               reason="mismatch_but_max"))
                        break

            # ── Judge (heuristic, 0 tokens) ────────────────────────────
            sig_verdict = _heuristic_judge(task, exec_result, route, domain)
            signals.append(sig_verdict)

            if sig_verdict.data["q"] == Verdict.PASS.value:
                break

            if sig_verdict.data["q"] == Verdict.FAIL.value:
                logger.warning("Max tier verdict: %s",
                              sig_verdict.data.get("r", ""))
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

        elapsed = time.time() - start
        return {
            "pact": {
                "output": final_output,
                "route": route.value,
                "escalated": escalated,
                "fireworks_tokens": fireworks_tokens,
                "elapsed_ms": int(elapsed * 1000),
                "difficulty": difficulty,
                "domain": domain,
                "signals": len(signals),
            },
            "_trace": [s.compact() for s in signals],
        }

    def process_batch(self, tasks: List[str],
                      show_trace: bool = False) -> List[dict]:
        results = []
        for task in tasks:
            r = self.process(task)
            if not show_trace:
                r.pop("_trace", None)
            results.append(r)
        return results
