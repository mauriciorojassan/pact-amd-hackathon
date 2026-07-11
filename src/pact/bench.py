"""Pact benchmark — compares cascade routing vs. all-Fireworks baseline.

Run with: pact bench
Or:       python -m pact bench
"""

from __future__ import annotations

import json
import os
import sys
import time

from .router import PactRouter


# Sample benchmark tasks
SAMPLE_TASKS = [
    "What is 2 + 2?",
    "Write a Python function to check if a string is a palindrome",
    "Explain the difference between a list and a tuple in Python",
    "What is the time complexity of quicksort?",
    "Write a SQL query to find duplicate emails in a users table",
    "Explain how gradient descent works in machine learning",
    "Write a bash one-liner to find all files modified in the last 24 hours",
    "What is the capital of France?",
    "Debug this: why is my React useEffect running twice?",
    "Write a recursive Fibonacci function and explain its time complexity",
]


def run():
    """Run benchmark: Pact cascade vs. all-Fireworks baseline."""
    print("═" * 50, file=sys.stderr)
    print("  Pact Benchmark — Cascade vs All-Fireworks", file=sys.stderr)
    print("═" * 50, file=sys.stderr)

    # Ensure mock mode for benchmark (no GPU needed)
    os.environ.setdefault("PACT_MOCK", "1")

    # Pact cascade
    print("\n▶ Pact cascade routing...", file=sys.stderr)
    pact_router = PactRouter()
    pact_results = []
    pact_tokens = 0
    pact_start = time.time()
    for task in SAMPLE_TASKS:
        r = pact_router.process(task)
        pact_results.append(r)
        pact_tokens += r["pact"]["fireworks_tokens"]
        print(f"  [{r['pact']['route'][:6]:>6}] {r['pact']['difficulty']:>6} "
              f"| {r['pact']['fireworks_tokens']:>3} fw tokens | "
              f"{r['pact']['elapsed_ms']:>4}ms",
              file=sys.stderr)
    pact_elapsed = time.time() - pact_start

    local_count = sum(1 for r in pact_results if r["pact"]["route"] == "local")

    print(file=sys.stderr)

    # Baseline: always Fireworks model (mock — counts simulated tokens)
    print("▶ Baseline (all-Fireworks, simulated)...", file=sys.stderr)
    baseline_results = []
    baseline_tokens = 0
    baseline_start = time.time()
    for task in SAMPLE_TASKS:
        # Simulate always using a cheap Fireworks model
        tokens = len(task.split()) * 3 + 50  # ponytail: rough estimate
        baseline_tokens += tokens
        baseline_results.append({
            "pact": {"route": "fireworks-cheap", "fireworks_tokens": tokens,
                     "elapsed_ms": 200},
        })
        print(f"  [fw-chp] {tokens:>3} fw tokens", file=sys.stderr)
    baseline_elapsed = time.time() - baseline_start

    # Summary
    print("\n" + "─" * 50, file=sys.stderr)
    print("  RESULTS", file=sys.stderr)
    print(f"  Tasks:            {len(SAMPLE_TASKS)}", file=sys.stderr)
    print(f"  Local handled:    {local_count}/{len(SAMPLE_TASKS)} "
          f"({100 * local_count // len(SAMPLE_TASKS)}%)", file=sys.stderr)
    print(f"  Pact FW tokens:   {pact_tokens}", file=sys.stderr)
    print(f"  Baseline tokens:  {baseline_tokens}", file=sys.stderr)
    if baseline_tokens > 0:
        pct = (1 - pact_tokens / baseline_tokens) * 100
        print(f"  Token savings:    {pct:.0f}%", file=sys.stderr)
    print(f"  Pact time:        {pact_elapsed*1000:.0f}ms", file=sys.stderr)
    print(f"  Baseline time:    {baseline_elapsed*1000:.0f}ms", file=sys.stderr)
    print("═" * 50, file=sys.stderr)

    # Machine-readable output to stdout
    print(json.dumps({
        "tasks": len(SAMPLE_TASKS),
        "pact_tokens": pact_tokens,
        "baseline_tokens": baseline_tokens,
        "savings_pct": round((1 - pact_tokens / max(baseline_tokens, 1)) * 100, 1),
        "local_count": local_count,
        "fireworks_count": len(SAMPLE_TASKS) - local_count,
        "pact_elapsed_ms": int(pact_elapsed * 1000),
        "baseline_elapsed_ms": int(baseline_elapsed * 1000),
    }))
