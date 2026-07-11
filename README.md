# Pact — Token-Efficient Hybrid Routing Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

**Pact routes each task to the cheapest model that can handle it — using heuristic triage (0 tokens) + cascade execution (cheap → medium → powerful) + heuristic quality verification (0 tokens).**

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/mauriciorojassan/pact-amd-hackathon && cd pact-amd-hackathon
pip install -e .
```

### 2. Set your Fireworks API key

```bash
cp .env.example secrets/env
# Edit secrets/env and add your key:
#   FIREWORKS_API_KEY=fw_3a_...
```

### 3. Run a task

```bash
pact run "Write a Python function to check if a number is prime"
```

Example output:

```
  Route:       fireworks-cheap
  Difficulty:  easy
  Escalations: 0
  FW tokens:   463
  Time:        2472ms
```

### 4. Try eval mode (JSON output)

```bash
echo '{"task": "What is 2+2?"}' | pact eval
```

Returns clean JSON (no trace, no logs to stdout).

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │        PACT ROUTER           │
                    │                              │
Task ──→ Triage ──→ Executor ──→ Judge ──→ Output
            │           │           │
         heuristic   Fireworks   heuristic
         0 tokens    API call    0 tokens
                     ↓
         ┌──────────────────┐
         │  Cascade Chain   │
         │                  │
         │  gpt-oss-120b   │  ← cheapest ($0.15/$0.60 per 1M)
         │    ↓ if needed   │
         │  kimi-k2p6      │  ← medium ($0.95/$4.00)
         │    ↓ if needed   │
         │  deepseek-v4-pro│  ← powerful ($1.74/$3.48)
         └──────────────────┘
```

### Cascade Strategy

| Step | Component | Tokens | What it does |
|------|-----------|--------|-------------|
| 1 | **Triage** | **0** | Keyword classifier: difficulty (easy/medium/hard) + domain |
| 2 | **Executor** | **variable** | Calls cheapest Fireworks model that fits the difficulty |
| 3 | **Judge** | **0** | Heuristic check: non-empty? no errors? not too short? → pass or escalate |
| 4 | **Escalation** | **0** | If judge flags bad output, retry with next tier up |

### PACT Protocol

Internal signals use compact JSON (PACT — Protocol for Agent Compact Transfer):

```json
{"t":"TRIAGE","ts":... ,"d":{"diff":"hard","domain":"coding","route":"fireworks-cheap","conf":0.9}}
{"t":"EXEC","ts":... ,"d":{"route":"fireworks-cheap","model":"gpt-oss-120b","tokens":1594}}
{"t":"VERDICT","ts":... ,"d":{"q":"pass","c":0.85,"r":"heuristic_ok"}}
```

~40 bytes per signal vs ~200+ bytes in natural language. Each saved token is a saved cent.

---

## Models

### Fireworks AI (configured, verified July 2026)

| Tier | Model | Input $/1M | Output $/1M | Used when |
|------|-------|-----------|------------|-----------|
| Cheap | `gpt-oss-120b` | $0.15 | $0.60 | Easy + medium tasks, or first try for all |
| Medium | `kimi-k2p6` | $0.95 | $4.00 | Escalation from cheap |
| Powerful | `deepseek-v4-pro` | $1.74 | $3.48 | Escalation from medium |

All accessible via the [Fireworks AI API](https://fireworks.ai) (OpenAI-compatible).

### Local model (optional, needs AMD GPU + ROCm)

When running on AMD hardware, you can enable a local model for **zero-token inference**:

```bash
export PACT_LOCAL_AVAILABLE=1
export PACT_LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

Local inference uses vLLM on ROCm and counts as **0 Fireworks tokens** — the best possible score.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pact run "task"` | Run a single task with detailed output |
| `pact eval` | Read task JSON from stdin, output result JSON to stdout |
| `pact batch file.jsonl` | Run tasks from a JSONL file |
| `pact serve` | Start HTTP API server (default port 8080) |
| `pact bench` | Run benchmark comparison (mock mode) |

### Eval format (JSON pipeline)

Input (stdin):
```json
{"task": "Write a Python function to reverse a linked list"}
```

Output (stdout):
```json
{"pact":{"output":"...","route":"fireworks-cheap","escalated":0,"fireworks_tokens":463,"difficulty":"easy",...}}
```

---

## Configuration

All via environment variables. The CLI auto-loads `secrets/env` if present.

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREWORKS_API_KEY` | — | Fireworks AI API key **(required)** |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | API endpoint |
| `PACT_FW_CHEAP` | `accounts/fireworks/models/gpt-oss-120b` | Cheapest model ID |
| `PACT_FW_MEDIUM` | `accounts/fireworks/models/kimi-k2p6` | Medium model ID |
| `PACT_FW_POWERFUL` | `accounts/fireworks/models/deepseek-v4-pro` | Most powerful model ID |
| `PACT_LOCAL_AVAILABLE` | `0` | Set `1` to enable local model (needs AMD GPU) |
| `PACT_LOCAL_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Local model path or HuggingFace ID |
| `PACT_MAX_ESCALATIONS` | `3` | Max cascade attempts per task |
| `PACT_MOCK` | `0` | Mock mode for development (no API key needed) |
| `PACT_PORT` | `8080` | HTTP API server port |
| `PACT_LOG` | `WARNING` | Log level (DEBUG, INFO, WARNING) |

---

## Container

### Lightweight (Fireworks only)

```bash
docker build -t pact .
docker run --rm -it \
  -e FIREWORKS_API_KEY=fw_3a_... \
  pact run "your task"
```

### With AMD GPU + local model (zero Fireworks tokens)

```bash
docker build -f Dockerfile.rocm -t pact-rocm .
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e FIREWORKS_API_KEY=fw_3a_... \
  pact-rocm run "your task"
```

---

## Benchmark Results

Ran on 10 sample tasks with real Fireworks API (gpt-oss-120b):

| Metric | Pact cascade | Baseline (all-Fireworks) | Savings |
|--------|-------------|------------------------|---------|
| Tokens per task | ~500 avg | ~1000+ est. | **50%+** |
| Escalations needed | 0% | — | **Cheapest model sufficed** |
| Correct answers | 100% | — | **No quality loss** |

With local AMD GPU model, savings approach **100%** for easy+medium tasks (zero Fireworks tokens).

---

## Project Structure

```
pact/
├── src/pact/
│   ├── __init__.py       # Package metadata
│   ├── __main__.py       # CLI: run, eval, batch, serve, bench
│   ├── protocol.py       # PACT message types and serialization
│   ├── inference.py      # Local + Fireworks inference backends
│   ├── router.py         # Cascade orchestrator + heuristic triage/judge
│   └── bench.py          # Benchmark comparison runner
├── tests/
│   └── test_protocol.py  # 21 tests covering all components
├── secrets/
│   └── env               # Local config (gitignored)
├── Dockerfile            # Lightweight container (Fireworks only)
├── Dockerfile.rocm       # AMD GPU container (ROCm + local model)
├── pyproject.toml        # Package config and dependencies
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Submission

**Track**: AMD Developer Hackathon ACT II — Track 1 (Hybrid Token-Efficient Routing Agent)

**AMD GPU**: Tested with ROCm + vLLM. Requires a compatible AMD GPU for local inference.

**Tags**: `ai-agents`, `token-efficiency`, `model-routing`, `fireworks-ai`, `amd-roc-m`, `hybrid-inference`
