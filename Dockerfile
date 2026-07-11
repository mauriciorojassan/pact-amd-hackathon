# Pact — Scoring container (Fireworks only, no AMD GPU required)
# Build:   docker build -t pact .
# Run:     docker run --rm -it -e FIREWORKS_API_KEY=fw_3a_... pact eval < task.json
#
# For AMD GPU with local model, use Dockerfile.rocm instead.

# ── Use a slim Python base (no ROCm needed for Fireworks-only routing) ──
FROM python:3.12-slim

WORKDIR /app

# Install pact + dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Scoring interface
#   pact eval reads task JSON from stdin, writes result JSON to stdout
ENV PYTHONUNBUFFERED=1
ENV PACT_LOCAL_AVAILABLE=0

ENTRYPOINT ["python", "-m", "pact"]
CMD ["eval"]
