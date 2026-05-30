"""
main.py — Orchestrates the full pipeline:
  1. Download / prepare BTC-USD data
  2. Train all models sequentially (ANN → RNN → LSTM → GRU → GAN)
  3. Save checkpoints
  4. Run evaluation + generate all plots
  5. Print final comparison table

Usage:
    python main.py
    python main.py --config config.yaml
    python main.py --skip-gan          # skip GAN (faster)
    python main.py --models lstm gru   # train specific models only
"""

import argparse
import json
import os

import torch
import yaml

from models import build_ann, build_rnn, build_lstm, build_gru, build_gan
from train import train_model, train_gan_model
from utils.dataset import prepare_data


def get_device(cfg):
    pref = cfg["training"]["device"]
    if pref == "auto":
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        d = torch.device(pref)
    print(f"\n[Main] Using device: {d}")
    if d.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return d


def save_checkpoint(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    print(f"  [Checkpoint] Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="BTC Neural Network Comparison")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-gan", action="store_true", help="Skip GAN training")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ann", "rnn", "lstm", "gru", "gan"],
        default=["ann", "rnn", "lstm", "gru", "gan"],
        help="Which models to train",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["paths"]["results"]
    ckpt_dir    = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = get_device(cfg)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[Main] Preparing data...")
    loaders, scaler, meta = prepare_data(cfg)
    seq_len    = meta["seq_len"]
    n_features = meta["n_features"]

    all_results = {}

    # ── ANN ──────────────────────────────────────────────────────────────────
    if "ann" in args.models:
        print("\n[Main] ── Training ANN ─────────────────────────────────────")
        model = build_ann(cfg, seq_len, n_features)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        hist = train_model(model, loaders, cfg, device, "ANN")
        all_results["ANN"] = hist
        save_checkpoint(model.state_dict(), os.path.join(ckpt_dir, "ann.pt"))

    # ── RNN ──────────────────────────────────────────────────────────────────
    if "rnn" in args.models:
        print("\n[Main] ── Training RNN ─────────────────────────────────────")
        model = build_rnn(cfg, n_features)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        hist = train_model(model, loaders, cfg, device, "RNN")
        all_results["RNN"] = hist
        save_checkpoint(model.state_dict(), os.path.join(ckpt_dir, "rnn.pt"))

    # ── LSTM ─────────────────────────────────────────────────────────────────
    if "lstm" in args.models:
        print("\n[Main] ── Training LSTM ────────────────────────────────────")
        model = build_lstm(cfg, n_features)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        hist = train_model(model, loaders, cfg, device, "LSTM")
        all_results["LSTM"] = hist
        save_checkpoint(model.state_dict(), os.path.join(ckpt_dir, "lstm.pt"))

    # ── GRU ──────────────────────────────────────────────────────────────────
    if "gru" in args.models:
        print("\n[Main] ── Training GRU ─────────────────────────────────────")
        model = build_gru(cfg, n_features)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        hist = train_model(model, loaders, cfg, device, "GRU")
        all_results["GRU"] = hist
        save_checkpoint(model.state_dict(), os.path.join(ckpt_dir, "gru.pt"))

    # ── GAN ──────────────────────────────────────────────────────────────────
    if "gan" in args.models and not args.skip_gan:
        print("\n[Main] ── Training GAN (WGAN-GP) ───────────────────────────")
        G, D = build_gan(cfg, n_features)
        n_params_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
        n_params_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
        print(f"  Generator params:     {n_params_G:,}")
        print(f"  Discriminator params: {n_params_D:,}")
        hist = train_gan_model(G, D, loaders, cfg, device)
        all_results["GAN"] = hist
        save_checkpoint(G.state_dict(), os.path.join(ckpt_dir, "gan_g.pt"))
        save_checkpoint(D.state_dict(), os.path.join(ckpt_dir, "gan_d.pt"))

    # ── Save training history ────────────────────────────────────────────────
    hist_path = os.path.join(results_dir, "all_results.json")
    # Convert lists for JSON serialization
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()}
    with open(hist_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[Main] Training history saved → {hist_path}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[Main] ── Evaluating all models ─────────────────────────────")
    from evaluate import evaluate
    metrics_df = evaluate(cfg_path=args.config)

    print("\n[Main] ✓ Pipeline complete!")
    print(f"       Results  → {results_dir}")
    print(f"       Plots    → {cfg['paths']['plots']}")


if __name__ == "__main__":
    main()
