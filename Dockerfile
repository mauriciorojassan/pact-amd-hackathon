# Pact — Hybrid Token-Efficient Routing Agent
# Build: docker build -t pact .
# Run:   docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
#         -e FIREWORKS_API_KEY=fw_... pact run "your task"

# ── Base: ROCm + PyTorch ────────────────────────────────────────────────
FROM rocm/pytorch:latest AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ────────────────────────────────────────────────────────
COPY pyproject.toml README.md ./
COPY src/ src/

# Install pact package + vLLM for local inference on AMD GPU
# ponytail: --no-build-isolation avoids pip build env issues in ROCm images
RUN pip install --no-cache-dir --no-build-isolation -e . && \
    pip install --no-cache-dir vllm huggingface_hub

# ── Local models (download at build time, cached in image) ──────────────
ARG LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ARG JUDGE_MODEL=Qwen/Qwen2.5-0.5B-Instruct

RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading', '$LOCAL_MODEL'); \
snapshot_download('$LOCAL_MODEL', local_dir='/models/local', local_dir_use_symlinks=False); \
" && \
    python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading', '$JUDGE_MODEL'); \
snapshot_download('$JUDGE_MODEL', local_dir='/models/judge', local_dir_use_symlinks=False); \
"

# ── Runtime ─────────────────────────────────────────────────────────────
ENV PACT_LOCAL_MODEL=/models/local
ENV PACT_JUDGE_MODEL=/models/judge
ENV PACT_LOCAL_BASE_URL=http://localhost:8000/v1
ENV PYTHONUNBUFFERED=1
ENV PACT_LOCAL_AVAILABLE=1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "pact"]
CMD ["serve", "--port", "8080"]
