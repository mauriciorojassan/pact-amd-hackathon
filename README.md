# Pact — Protocol for Agent Compact Transfer

**Hybrid token-efficient routing agent for multi-model inference on AMD GPUs.**

Pact routes each task to the cheapest model that can handle it, using a cascade:
**local (0 tokens) → cheap Fireworks → medium Fireworks → powerful Fireworks**.

A judge agent validates output quality at every step and escalates to a larger
model only when needed. This saves 50–90% of inference tokens vs. always using
the largest model.

Built for the [AMD Developer Hackathon: ACT II](https://lablab.ai/event/amd-developer-hackathon-act-ii),
Track 1 — Hybrid Token-Efficient Routing Agent.

## Quick Start

```bash
# Install
pip install -e .

# Run a single task (local model mock mode, no GPU needed)
PACT_MOCK=1 pact run "Write a Python function to reverse a linked list"

# With Fireworks API
export FIREWORKS_API_KEY="fw_3a_..."
pact run "Explain quantum computing in simple terms"

# Batch from file
pact batch tasks.jsonl

# Start HTTP API server
pact serve --port 8080
```

## Architecture

```
Task → Triage (local, 0 tokens) ──→ Executor ──→ Judge (local, 0 tokens)
        ↕ classifies difficulty       ↕ routes     ↕ validates quality
        | easy → local (0 tokens!)    |            | pass → output
        | med  → cheap Fireworks      |            | fail → escalate (next tier)
        | hard → powerful Fireworks   |            |
        └─────────────────────────────┴────────────┘
```

### PACT Protocol

Messages between agents use PACT — a structured, compact format:

```json
{"t":"TRIAGE","ts":... ,"d":{"diff":"easy","domain":"math","route":"local","conf":0.92}}
{"t":"EXEC","ts":... ,"d":{"route":"local","model":"qwen2.5-1.5b","tokens":0}}
{"t":"VERDICT","ts":... ,"d":{"q":"pass","c":0.88,"r":"ok"}}
```

Token cost: ~40 bytes vs ~200 bytes natural language per message.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREWORKS_API_KEY` | — | Fireworks AI API key (required for Fireworks routing) |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | API endpoint |
| `PACT_LOCAL_AVAILABLE` | `1` | Set `0` to disable local model |
| `PACT_LOCAL_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Local model path or name |
| `PACT_MOCK` | `0` | Mock mode for dev without GPU |
| `PACT_FW_CHEAP` | `accounts/fireworks/models/llama-v3p2-3b-instruct` | Cheap Fireworks model |
| `PACT_FW_MEDIUM` | `accounts/fireworks/models/llama-v3p1-8b-instruct` | Medium Fireworks model |
| `PACT_FW_POWERFUL` | `accounts/fireworks/models/llama-v3p1-405b-instruct` | Powerful Fireworks model |
| `PACT_MAX_ESCALATIONS` | `3` | Maximum cascade attempts |
| `PACT_PORT` | `8080` | HTTP API port |
| `PACT_LOG` | `WARNING` | Log level (DEBUG, INFO, WARNING) |

## Container

```bash
docker build -t pact .
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e FIREWORKS_API_KEY=fw_3a_... \
  pact run "your task"
```

## License

MIT — see [LICENSE](LICENSE).
