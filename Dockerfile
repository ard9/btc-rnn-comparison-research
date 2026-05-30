# ─────────────────────────────────────────────────────────────────────────────
# BTC Neural Network Comparison
# Base: PyTorch + CUDA 12.1 on Ubuntu 22.04
# GPU requirement: ≥ 4GB VRAM (tested on 12GB RTX 3080)
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="btc-nn-comparison"
LABEL description="BTC price prediction: ANN vs RNN vs LSTM vs GRU vs GAN (WGAN-GP)"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Create result directories ─────────────────────────────────────────────────
RUN mkdir -p results/plots results/checkpoints data

# ── Default command: run full pipeline ────────────────────────────────────────
CMD ["python", "main.py"]
