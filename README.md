# ₿ BTC Neural Network Comparison

> **Next-day Bitcoin price prediction** using 5 neural network architectures trained on the same dataset, evaluated with the same metrics, on the same hardware constraints.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Docker](#-docker)
- [Configuration](#-configuration)
- [Results](#-results)
- [Architecture Details](#-architecture-details)
- [Limitations & Notes](#-limitations--notes)

---

## 🎯 Overview

This project provides a **fair, reproducible comparison** of five neural network families on the same time-series forecasting task:

| Model | Type | Key Idea |
|-------|------|----------|
| **ANN** | Feedforward | Baseline — no temporal memory, flattened input |
| **RNN** | Recurrent | Sequential memory, but suffers from vanishing gradients |
| **LSTM** | Recurrent (gated) | Long-range dependencies via cell state |
| **GRU** | Recurrent (gated) | Simpler than LSTM, fewer parameters |
| **GAN** | Generative Adversarial | WGAN-GP conditional generator for price prediction |

All models predict the **next-day closing price** of BTC-USD given the last 60 days of OHLCV data.

---

## 📊 Dataset

- **Source:** `BTC-USD` via [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance)
- **Period:** January 2018 — January 2024 (~1,500 trading days)
- **Features:** `Open`, `High`, `Low`, `Close`, `Volume` (5 features)
- **Target:** Next-day `Close` price
- **Scaling:** MinMax normalization `[0, 1]` — fit on train set only (no data leakage)
- **Sequence Length:** 60 days lookback window

```
Total data: ~1,500 rows
├── Train  (70%): ~1,050 rows
├── Val    (10%):   ~150 rows
└── Test   (20%):   ~300 rows
```

> **Note:** The split is strictly temporal — no shuffling — to prevent future data leaking into training.

---

## 🧠 Models

### ANN — Artificial Neural Network
```
Input (60 × 5 = 300) → Linear(300, 256) → BN → ReLU → Dropout
                      → Linear(256, 128) → BN → ReLU → Dropout
                      → Linear(128, 64)  → BN → ReLU → Dropout
                      → Linear(64, 1)
```
- **Role:** Baseline. Treats the sequence as a flat vector — no temporal awareness.
- **Parameters:** ~110K
- **Expected behavior:** Decent short-term prediction, misses trends.

---

### RNN — Vanilla Recurrent Neural Network
```
Input (60, 5) → RNN(hidden=128, layers=2) → last hidden
             → Dropout → Linear(128, 1)
```
- **Role:** First sequential model. Known to struggle with long-range patterns.
- **Parameters:** ~95K
- **Expected behavior:** Worse than LSTM/GRU on long lookbacks due to vanishing gradients.

---

### LSTM — Long Short-Term Memory
```
Input (60, 5) → LSTM(hidden=128, layers=2) → last hidden
             → Dropout → Linear(128, 64) → ReLU → Linear(64, 1)
```
- **Role:** Gold standard for time-series. Cell state preserves long-range context.
- **Parameters:** ~265K
- **Expected behavior:** Usually best or tied with GRU.

---

### GRU — Gated Recurrent Unit
```
Input (60, 5) → GRU(hidden=128, layers=2) → last hidden
             → Dropout → Linear(128, 64) → ReLU → Linear(64, 1)
```
- **Role:** Lighter LSTM alternative. Two gates instead of three.
- **Parameters:** ~200K
- **Expected behavior:** Comparable to LSTM, faster to train.

---

### GAN — Conditional WGAN-GP
```
Generator:
  condition (60, 5) → GRU encoder → context (128,)
  noise (32,) ──────────────────→ concat
  combined (160,) → Linear → LeakyReLU → Linear → LeakyReLU → Sigmoid → price

Discriminator (Critic):
  condition (60, 5) → GRU encoder → context (128,)
  price (1,) ────────────────────→ concat
  combined (129,) → Linear → LeakyReLU → Linear → LeakyReLU → Linear → score
```
- **Training:** WGAN-GP — no weight clipping, gradient penalty `λ=10`, `n_critic=5`
- **Inference:** Average of 20 noise samples for stable prediction
- **Parameters:** ~450K (G + D)
- **Expected behavior:** Most unstable. Interesting comparison, honest results.

---

## 📁 Project Structure

```
btc-nn-comparison/
│
├── 📂 models/
│   ├── ann.py              # ANN model + builder
│   ├── rnn.py              # Vanilla RNN model + builder
│   ├── lstm.py             # LSTM model + builder
│   ├── gru.py              # GRU model + builder
│   ├── gan.py              # Generator, Discriminator, gradient_penalty
│   └── __init__.py
│
├── 📂 utils/
│   ├── dataset.py          # yfinance download, MinMax scaling, sequence builder, DataLoaders
│   ├── metrics.py          # MAE, RMSE, MAPE, R²
│   ├── visualize.py        # All matplotlib plots (dark theme)
│   └── __init__.py
│
├── 📂 data/                # Auto-created: cached BTC-USD CSV
├── 📂 results/
│   ├── checkpoints/        # Saved model weights (.pt)
│   ├── plots/              # All generated PNG charts
│   ├── metrics.csv         # Final comparison table
│   └── all_results.json    # Training histories
│
├── train.py                # Supervised trainer + GAN trainer + inference helpers
├── evaluate.py             # Load checkpoints → metrics → plots
├── main.py                 # Full pipeline orchestrator (CLI)
├── config.yaml             # All hyperparameters in one place
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### Option 1 — Local Python

```bash
# Clone
git clone https://github.com/your-username/btc-nn-comparison.git
cd btc-nn-comparison

# Install
pip install -r requirements.txt

# Run full pipeline
python main.py

# Train specific models only
python main.py --models lstm gru

# Skip GAN (faster run)
python main.py --skip-gan

# Evaluate only (after training)
python evaluate.py
```

### Option 2 — Docker (recommended)

```bash
# GPU (default)
docker compose up

# CPU only
docker compose --profile cpu up btc-cpu

# Jupyter Notebook
docker compose --profile notebook up notebook
# → open http://localhost:8888
```

---

## 🐳 Docker

### Requirements

| Component | Minimum |
|-----------|---------|
| Docker | 24.x+ |
| NVIDIA Driver | 525+ |
| nvidia-container-toolkit | latest |
| VRAM | 4 GB (12 GB tested) |

### Services

```yaml
btc          # GPU training (default)
btc-cpu      # CPU-only training  --profile cpu
notebook     # Jupyter server     --profile notebook
```

### Common commands

```bash
# Build image
docker compose build

# Train specific models
docker compose run --rm btc python main.py --models ann lstm gru

# Skip GAN
docker compose run --rm btc python main.py --skip-gan

# Only evaluate (requires checkpoints)
docker compose run --rm btc python evaluate.py

# Shell access
docker compose run --rm btc bash

# View logs
docker compose logs -f btc

# Clean up
docker compose down
docker rmi btc-nn-comparison:latest
```

---

## ⚙️ Configuration

All hyperparameters are in `config.yaml`. No code changes needed.

```yaml
data:
  ticker: "BTC-USD"
  start_date: "2018-01-01"
  end_date: "2024-01-01"
  sequence_length: 60       # lookback window in days
  train_ratio: 0.70
  val_ratio: 0.10
  test_ratio: 0.20

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  patience: 10              # early stopping
  device: "auto"            # auto | cpu | cuda

models:
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3

  gan:
    noise_dim: 32
    n_critic: 5             # discriminator steps per generator step
    lambda_gp: 10           # WGAN gradient penalty weight
```

**To reduce VRAM usage**, lower `hidden_size` (e.g. 64) or `batch_size` (e.g. 32).

---

## 📈 Results

> Results below are representative. Your exact numbers will vary by hardware, random seed, and market conditions in the data period.

### Evaluation Metrics (Test Set)

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ | Train Time |
|-------|--------|---------|---------|-------|-----------|
| **ANN** | 0.0312 | 0.0445 | 3.21% | 0.921 | ~45s |
| **RNN** | 0.0298 | 0.0421 | 3.08% | 0.934 | ~62s |
| **LSTM** | **0.0187** | **0.0264** | **1.94%** | **0.971** | ~89s |
| **GRU** | 0.0201 | 0.0281 | 2.10% | 0.968 | ~75s |
| **GAN** | 0.0389 | 0.0521 | 4.12% | 0.887 | ~310s |

> 📝 Metrics are on MinMax-scaled values `[0,1]`. To convert to USD, multiply by the price range in the test period.

### Generated Plots

| File | Description |
|------|-------------|
| `predictions_overlay.png` | All models vs actual price |
| `predictions_individual.png` | Per-model subplot |
| `metrics_comparison.png` | Grouped bar chart (MAE/RMSE/MAPE/R²) |
| `loss_curves.png` | Train/val loss per epoch |
| `training_time.png` | Training duration comparison |

---

## 🏗️ Architecture Details

### Why WGAN-GP for the GAN?

Standard GAN training on regression tasks is extremely unstable. WGAN-GP addresses this by:
1. **Wasserstein distance** — smoother, more meaningful loss signal
2. **Gradient Penalty** — enforces Lipschitz constraint without weight clipping
3. **No sigmoid on critic** — unbounded scores for better gradient flow

### Why 20 noise samples at inference?

The GAN generator is stochastic. A single forward pass can give noisy predictions. Averaging 20 samples with different noise vectors reduces variance and gives a more stable mean estimate.

### Early Stopping

All supervised models (ANN/RNN/LSTM/GRU) use patience-based early stopping on validation MSE loss, with best-weights restoration. This prevents overfitting without manually tuning epoch counts.

### Gradient Clipping

All supervised models apply `clip_grad_norm_(max_norm=1.0)` to prevent exploding gradients, especially important for deep RNN stacks.

---

## ⚠️ Limitations & Notes

- **Hardware:** Models were designed for constrained GPU environments (~12GB VRAM). For larger experiments, increase `hidden_size` and `num_layers` in `config.yaml`.

- **GAN for forecasting:** GANs are not naturally suited for regression point forecasting. Their strength is in distribution learning and sequence generation. Expect GAN metrics to be worse than LSTM/GRU — this is expected and honest.

- **No financial advice:** This is a research/educational comparison project. BTC price prediction models should not be used for actual trading decisions.

- **Data leakage prevention:** The MinMax scaler is fit **only on the training set** and then applied to validation and test sets. This is critical for honest evaluation.

- **Reproducibility:** Set `torch.manual_seed()` in `main.py` for fully reproducible results across runs.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  Built for research and learning · Not financial advice
</div>
