"""Inference backends: local (vLLM/ROCm) and Fireworks AI API.

All models are accessed through the same OpenAI-compatible interface.
Local inference uses 0 Fireworks tokens — ideal for cost-sensitive routing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Normalized result from any inference backend."""
    output: str
    model: str
    tokens: int = 0          # Fireworks tokens (0 for local)
    error: Optional[str] = None


class _OpenAICompatible:
    """Base for OpenAI-compatible inference backends."""

    def __init__(self, api_key: str, base_url: str,
                 model: str, available: bool = True):
        self.model = model
        self.base_url = base_url
        self.available = available
        self._client = None

        # ponytail: 30s default, configurable via env
        timeout = float(os.getenv("PACT_API_TIMEOUT", "30"))

        if available and api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key, base_url=base_url, timeout=timeout,
                )
            except ImportError:
                logger.warning("openai package not installed. pip install openai")
                self.available = False
        else:
            self.available = False

    def _call(self, prompt: str, model: str,
              max_tokens: int = 2048, temperature: float = 0.1) -> dict:
        """Make a chat completion call. Returns dict with output + usage."""
        if not self.available or self._client is None:
            return {"output": "", "usage": {}}

        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "output": resp.choices[0].message.content or "",
                "usage": {
                    "prompt": getattr(resp.usage, "prompt_tokens", 0),
                    "completion": getattr(resp.usage, "completion_tokens", 0),
                    "total": getattr(resp.usage, "total_tokens", 0),
                },
            }
        except Exception as e:
            logger.error("Inference error [%s]: %s", model, e)
            return {"output": "", "usage": {}, "error": str(e)}


class LocalInference(_OpenAICompatible):
    """Local model via vLLM (or any OpenAI-compatible local server).

    Runs on AMD GPU via ROCm. Zero Fireworks token cost.
    Falls back to mock mode when unavailable (dev without GPU).
    """

    def __init__(self):
        model = os.getenv("PACT_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        base_url = os.getenv("PACT_LOCAL_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("PACT_LOCAL_API_KEY", "none")
        available = os.getenv("PACT_LOCAL_AVAILABLE", "1") == "1"
        mock = os.getenv("PACT_MOCK", "0") == "1"

        super().__init__(api_key, base_url, model, available and not mock)
        self.mock = mock

        if self.available:
            logger.info("Local inference: %s @ %s", self.model, self.base_url)
        elif mock:
            logger.info("Local inference: MOCK mode (no GPU needed)")
        else:
            logger.info("Local inference: DISABLED (PACT_LOCAL_AVAILABLE=0)")

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.1) -> InferenceResult:
        if self.mock:
            # ponytail: mock returns a plausible natural-language response.
            # Triage is now heuristic (not LLM), judge validates output
            # quality — mock should produce something that looks real.
            mock_answer = (
                "Here is the result for your request. "
                "The answer is 42. Python code: def hello(): pass"
            )
            return InferenceResult(output=mock_answer, model=self.model, tokens=0)

        resp = self._call(prompt, self.model, max_tokens, temperature)
        return InferenceResult(
            output=resp.get("output", ""),
            model=self.model,
            tokens=0,  # Local = zero Fireworks tokens
            error=resp.get("error"),
        )


class FireworksInference(_OpenAICompatible):
    """Fireworks AI API inference.

    Tokens consumed here count toward the cost metric.
    Also supports mock mode for development (PACT_MOCK=1).
    """

    def __init__(self):
        api_key = os.getenv("FIREWORKS_API_KEY", "")
        base_url = os.getenv(
            "FIREWORKS_BASE_URL",
            "https://api.fireworks.ai/inference/v1",
        )
        available = bool(api_key)
        mock = os.getenv("PACT_MOCK", "0") == "1"
        super().__init__(api_key, base_url, "", available or mock)
        self._name = "Fireworks"
        self.mock = mock

        if self.mock:
            logger.info("Fireworks inference: MOCK mode (no API key needed)")
        elif self.available:
            logger.info("Fireworks API ready @ %s", self.base_url)
        else:
            logger.warning("FIREWORKS_API_KEY not set — Fireworks routing disabled")

    def generate(self, model: str, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.1) -> InferenceResult:
        if self.mock:
            # ponytail: return plausible mock output for any prompt
            return InferenceResult(
                output=f"Mock result for: {prompt[:100]}... [simulated]",
                model=model,
                tokens=25,  # simulate minimal token usage
            )

        resp = self._call(prompt, model, max_tokens, temperature)
        usage = resp.get("usage", {})
        return InferenceResult(
            output=resp.get("output", ""),
            model=model,
            tokens=usage.get("total", 0),
            error=resp.get("error"),
        )
