"""
evaluate.py — load saved checkpoints, run inference, compute metrics,
generate comparison tables and plots.

Can be run standalone after training:
    python evaluate.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import yaml

from utils.metrics import compute_all
from utils.visualize import (
    plot_predictions,
    plot_individual,
    plot_metrics_bar,
    plot_loss_curves,
    plot_training_time,
)
from utils.dataset import prepare_data
from models import build_ann, build_rnn, build_lstm, build_gru, build_gan
from train import predict, predict_gan


def get_device(cfg):
    pref = cfg["training"]["device"]
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def load_results(results_dir: str):
    path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results not found at {path}. Run main.py first.")
    with open(path) as f:
        return json.load(f)


def print_table(metrics_df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS — BTC-USD Next-Day Close Prediction")
    print("=" * 65)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 65)
    best_mae  = metrics_df["MAE"].idxmin()
    best_rmse = metrics_df["RMSE"].idxmin()
    best_r2   = metrics_df["R2"].idxmax()
    print(f"  Best MAE  → {best_mae}")
    print(f"  Best RMSE → {best_rmse}")
    print(f"  Best R²   → {best_r2}")
    print("=" * 65 + "\n")


def evaluate(cfg_path: str = "config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["paths"]["results"]
    plot_dir    = cfg["paths"]["plots"]
    ckpt_dir    = os.path.join(results_dir, "checkpoints")
    device      = get_device(cfg)

    print(f"\n[Evaluate] Device: {device}")

    # ── Prepare data ────────────────────────────────────────────
    loaders, scaler, meta = prepare_data(cfg)
    seq_len    = meta["seq_len"]
    n_features = meta["n_features"]
    test_dates = meta["test_dates"]

    # ── Load saved history ──────────────────────────────────────
    all_results = load_results(results_dir)

    # ── Re-build models and load weights ────────────────────────
    model_builders = {
        "ANN":  lambda: build_ann(cfg, seq_len, n_features),
        "RNN":  lambda: build_rnn(cfg, n_features),
        "LSTM": lambda: build_lstm(cfg, n_features),
        "GRU":  lambda: build_gru(cfg, n_features),
    }

    predictions = {}
    metrics_records = []

    for name, builder in model_builders.items():
        ckpt_path = os.path.join(ckpt_dir, f"{name.lower()}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[Evaluate] Checkpoint missing for {name}, skipping.")
            continue

        model = builder()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        y_true, y_pred = predict(model, loaders["test"], device)
        metrics = compute_all(y_true, y_pred)
        metrics["TrainTime"] = all_results.get(name, {}).get("train_time", 0.0)

        predictions[name] = {"actual": y_true, "pred": y_pred}
        metrics_records.append({"Model": name, **metrics})
        print(f"  [{name}] MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")

    # ── GAN ─────────────────────────────────────────────────────
    G_path = os.path.join(ckpt_dir, "gan_g.pt")
    if os.path.exists(G_path):
        G, _ = build_gan(cfg, n_features)
        G.load_state_dict(torch.load(G_path, map_location=device))
        G.to(device)

        y_true, y_pred = predict_gan(G, loaders["test"], cfg, device)
        metrics = compute_all(y_true, y_pred)
        metrics["TrainTime"] = all_results.get("GAN", {}).get("train_time", 0.0)

        predictions["GAN"] = {"actual": y_true, "pred": y_pred}
        metrics_records.append({"Model": "GAN", **metrics})
        print(f"  [GAN] MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")

    # ── Build DataFrame ─────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_records).set_index("Model")
    print_table(metrics_df)

    # ── Save CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(csv_path)
    print(f"[Evaluate] Metrics saved → {csv_path}")

    # ── Plots ────────────────────────────────────────────────────
    dates = test_dates[: len(next(iter(predictions.values()))["actual"])]

    plot_predictions(predictions, dates, plot_dir)
    plot_individual(predictions, dates, plot_dir)
    plot_metrics_bar(metrics_df, plot_dir)

    # Loss curves
    loss_histories = {}
    for name in ["ANN", "RNN", "LSTM", "GRU"]:
        if name in all_results and "train" in all_results[name]:
            loss_histories[name] = {
                "train": all_results[name]["train"],
                "val":   all_results[name]["val"],
            }
    if loss_histories:
        plot_loss_curves(loss_histories, plot_dir)

    # Training time
    time_dict = {r["Model"]: r.get("TrainTime", 0) for r in metrics_records}
    plot_training_time(time_dict, plot_dir)

    print("\n[Evaluate] All plots saved to:", plot_dir)
    return metrics_df


if __name__ == "__main__":
    evaluate()
