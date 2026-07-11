"""PACT: Protocol for Agent Compact Transfer — message types and serialization.

PACT is a structured, token-efficient protocol for inter-agent communication.
Signals carry intent, not prose. Each signal type maps to a routing decision:
TRIAGE, EXEC, RESULT, VERDICT.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PACTType(str, Enum):
    """PACT signal types."""
    TRIAGE = "TRIAGE"
    EXEC = "EXEC"
    RESULT = "RESULT"
    VERDICT = "VERDICT"


class Route(str, Enum):
    """Available routing tiers."""
    LOCAL = "local"
    FIREWORKS_CHEAP = "fireworks-cheap"
    FIREWORKS_MEDIUM = "fireworks-medium"
    FIREWORKS_POWERFUL = "fireworks-powerful"


class Verdict(str, Enum):
    """Judge verdicts."""
    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"


# ponytail: cascade escalation order
# Route.LOCAL excluded — not wired in executor yet. Add when GPU support lands.
ESCALATION_CHAIN = [
    Route.FIREWORKS_CHEAP,
    Route.FIREWORKS_MEDIUM,
    Route.FIREWORKS_POWERFUL,
]


def next_route(current: Route) -> Optional[Route]:
    """Return the next route in the escalation chain, or None if at max."""
    try:
        idx = ESCALATION_CHAIN.index(current)
        return ESCALATION_CHAIN[idx + 1] if idx + 1 < len(ESCALATION_CHAIN) else None
    except ValueError:
        return None


@dataclass
class PACTSignal:
    """A single PACT protocol message between agents."""

    type: PACTType
    data: dict
    timestamp: float = field(default_factory=time.time)

    def compact(self) -> str:
        """Serialize to compact JSON (no whitespace, short keys)."""
        return json.dumps({
            "t": self.type.value,
            "ts": self.timestamp,
            "d": self.data,
        }, separators=(",", ":"), default=str)

    @classmethod
    def from_compact(cls, raw: str) -> PACTSignal:
        obj = json.loads(raw)
        return cls(type=PACTType(obj["t"]), timestamp=obj["ts"], data=obj["d"])

    def __str__(self) -> str:
        return f"[{self.type.value}] {json.dumps(self.data, default=str)[:200]}"


# --- Signal builders (fluent helpers) ---

def triage(difficulty: str, domain: str, route: str,
           confidence: float, reasoning: str = "") -> PACTSignal:
    return PACTSignal(
        type=PACTType.TRIAGE,
        data={
            "diff": difficulty,
            "domain": domain,
            "route": route,
            "conf": round(confidence, 2),
            "r": reasoning[:120],
        }
    )


def exec_signal(route: str, model: str, tokens: int = 0,
                status: str = "ok", error: str = "") -> PACTSignal:
    return PACTSignal(
        type=PACTType.EXEC,
        data={
            "route": route,
            "model": model,
            "tokens": tokens,
            "status": status,
            "error": error[:120],
        }
    )


def result(output: str, route: str, model: str,
           tokens: int = 0) -> PACTSignal:
    return PACTSignal(
        type=PACTType.RESULT,
        data={
            "o": output,
            "route": route,
            "model": model,
            "tokens": tokens,
        }
    )


def verdict(quality: str, confidence: float,
            escalate_to: str = "", reason: str = "") -> PACTSignal:
    return PACTSignal(
        type=PACTType.VERDICT,
        data={
            "q": quality,
            "c": round(confidence, 2),
            "e": escalate_to,
            "r": reason[:120],
        }
    )
