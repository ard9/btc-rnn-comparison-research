"""
Visualization utilities — all plots saved to disk.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PALETTE = {
    "ANN":  "#FF6B6B",
    "RNN":  "#4ECDC4",
    "LSTM": "#45B7D1",
    "GRU":  "#96CEB4",
    "GAN":  "#FFEAA7",
}

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.linewidth":   0.8,
    "legend.facecolor": "#161B22",
    "legend.edgecolor": "#30363D",
    "font.family":      "monospace",
})


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved → {path}")


def plot_predictions(results: dict, dates, plot_dir: str):
    """
    Overlay all models' predictions vs actual prices.
    results: {model_name: {"actual": np.array, "pred": np.array}}
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual once
    first = next(iter(results.values()))
    ax.plot(dates, first["actual"], color="#F0F6FC", lw=1.5, label="Actual", zorder=5)

    for name, data in results.items():
        ax.plot(
            dates,
            data["pred"],
            color=PALETTE.get(name, "#FFFFFF"),
            lw=1.0,
            alpha=0.85,
            linestyle="--",
            label=name,
        )

    ax.set_title("BTC-USD Price Prediction — All Models", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD, scaled)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="both")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(plot_dir, "predictions_overlay.png"))


def plot_individual(results: dict, dates, plot_dir: str):
    """One subplot per model."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        ax.plot(dates, data["actual"], color="#F0F6FC", lw=1.2, label="Actual")
        ax.plot(
            dates,
            data["pred"],
            color=PALETTE.get(name, "#FFFFFF"),
            lw=1.0,
            linestyle="--",
            label=f"{name} Prediction",
        )
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle("BTC-USD — Individual Model Predictions", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(plot_dir, "predictions_individual.png"))


def plot_metrics_bar(metrics_df: pd.DataFrame, plot_dir: str):
    """Grouped bar chart for MAE / RMSE / MAPE / R2."""
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric in zip(axes, metrics):
        values = metrics_df[metric]
        colors = [PALETTE.get(m, "#AAAAAA") for m in metrics_df.index]
        bars = ax.bar(metrics_df.index, values, color=colors, edgecolor="#30363D", width=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#C9D1D9",
            )

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, axis="y")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Model Comparison — Evaluation Metrics", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(plot_dir, "metrics_comparison.png"))


def plot_loss_curves(loss_histories: dict, plot_dir: str):
    """
    Training/validation loss curves.
    loss_histories: {model_name: {"train": [...], "val": [...]}}
    """
    fig, axes = plt.subplots(1, len(loss_histories), figsize=(5 * len(loss_histories), 4))
    if len(loss_histories) == 1:
        axes = [axes]

    for ax, (name, hist) in zip(axes, loss_histories.items()):
        color = PALETTE.get(name, "#FFFFFF")
        ax.plot(hist["train"], color=color, lw=1.5, label="Train")
        ax.plot(hist["val"], color=color, lw=1.5, linestyle="--", alpha=0.7, label="Val")
        ax.set_title(f"{name} — Loss", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle("Training & Validation Loss Curves", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plot_dir, "loss_curves.png"))


def plot_training_time(time_dict: dict, plot_dir: str):
    """Horizontal bar chart of training times."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(time_dict.keys())
    times = [time_dict[n] for n in names]
    colors = [PALETTE.get(n, "#AAAAAA") for n in names]

    bars = ax.barh(names, times, color=colors, edgecolor="#30363D", height=0.5)
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison", fontsize=12)
    ax.grid(True, axis="x")
    fig.tight_layout()
    _save(fig, os.path.join(plot_dir, "training_time.png"))
