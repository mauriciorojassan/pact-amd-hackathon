"""Pact — Protocol for Agent Compact Transfer.

A token-efficient hybrid routing agent for multi-model inference on AMD GPUs.
Routes tasks to the cheapest sufficient model via cascade routing:
local (0 tokens) → cheap Fireworks → powerful Fireworks.
"""

__version__ = "0.1.0"
