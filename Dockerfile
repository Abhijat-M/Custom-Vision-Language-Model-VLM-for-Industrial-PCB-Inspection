# ── Stage 1: Base with CUDA + Python ─────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application code ────────────────────────────────────────────
COPY src/ src/

# Checkpoints are mounted or copied at deploy time
# COPY checkpoints/ checkpoints/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Production: run with uvicorn
ENTRYPOINT ["python", "-m", "uvicorn", "src.api:app", \
            "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
