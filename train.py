"""
Unified trainer for ANN / RNN / LSTM / GRU (standard supervised loop)
and GAN (WGAN-GP adversarial loop).

Usage:
    from train import train_model, train_gan_model
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gan import gradient_penalty


# ────────────────────────────────────────────────────────────────────────────────
# Standard supervised trainer (ANN, RNN, LSTM, GRU)
# ────────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    loaders: dict,
    cfg: dict,
    device: torch.device,
    model_name: str = "Model",
) -> dict:
    """
    Train a regression model with MSE loss + Adam optimizer + early stopping.

    Returns:
        history: {"train": [...], "val": [...], "train_time": float}
    """
    train_cfg = cfg["training"]
    epochs   = train_cfg["epochs"]
    lr       = train_cfg["learning_rate"]
    patience = train_cfg["patience"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train": [], "val": []}

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in loaders["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in loaders["val"]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        scheduler.step(val_loss)

        # ── Early stopping ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [{model_name}] Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f}"
            )

        if no_improve >= patience:
            print(f"  [{model_name}] Early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    history["train_time"] = time.time() - t0
    print(f"  [{model_name}] Done — {history['train_time']:.1f}s | Best Val Loss: {best_val_loss:.6f}")
    return history


# ────────────────────────────────────────────────────────────────────────────────
# GAN trainer (WGAN-GP)
# ────────────────────────────────────────────────────────────────────────────────

def train_gan_model(
    G: nn.Module,
    D: nn.Module,
    loaders: dict,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    WGAN-GP training loop for the conditional GAN.

    Returns:
        history: {"g_loss": [...], "d_loss": [...], "train_time": float}
    """
    gan_cfg   = cfg["models"]["gan"]
    train_cfg = cfg["training"]
    epochs    = train_cfg["epochs"]
    n_critic  = gan_cfg["n_critic"]
    lambda_gp = gan_cfg["lambda_gp"]
    noise_dim = gan_cfg["noise_dim"]

    G.to(device)
    D.to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=gan_cfg["lr_g"], betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=gan_cfg["lr_d"], betas=(0.0, 0.9))

    history = {"g_loss": [], "d_loss": [], "train_time": 0.0}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        g_losses, d_losses = [], []
        data_iter = iter(loaders["train"])

        step = 0
        for X_batch, y_batch in loaders["train"]:
            X_batch = X_batch.to(device)
            y_real  = y_batch.to(device)
            batch   = X_batch.size(0)

            # ── Critic (Discriminator) steps ───────────────────
            for _ in range(n_critic):
                noise  = torch.randn(batch, noise_dim, device=device)
                y_fake = G(X_batch, noise).detach()

                d_real = D(X_batch, y_real)
                d_fake = D(X_batch, y_fake)

                gp   = gradient_penalty(D, X_batch, y_real, y_fake, device, lambda_gp)
                d_loss = -(d_real.mean() - d_fake.mean()) + gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()
                d_losses.append(d_loss.item())

            # ── Generator step ─────────────────────────────────
            noise  = torch.randn(batch, noise_dim, device=device)
            y_fake = G(X_batch, noise)
            g_loss = -D(X_batch, y_fake).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())
            step += 1

        history["g_loss"].append(np.mean(g_losses))
        history["d_loss"].append(np.mean(d_losses))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [GAN] Epoch {epoch:3d}/{epochs} | "
                f"G: {history['g_loss'][-1]:.4f} | "
                f"D: {history['d_loss'][-1]:.4f}"
            )

    history["train_time"] = time.time() - t0
    print(f"  [GAN] Done — {history['train_time']:.1f}s")
    return history


# ────────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ────────────────────────────────────────────────────────────────────────────────

def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Standard model inference → (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred    = model(X_batch).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(y_batch.numpy())
    return np.array(all_true), np.array(all_pred)


def predict_gan(G: nn.Module, loader: DataLoader, cfg: dict, device: torch.device):
    """
    GAN inference — average over multiple noise samples for stable predictions.
    """
    noise_dim    = cfg["models"]["gan"]["noise_dim"]
    n_samples    = 20   # average over 20 noise samples
    G.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds   = []
            for _ in range(n_samples):
                noise = torch.randn(X_batch.size(0), noise_dim, device=device)
                preds.append(G(X_batch, noise).cpu().numpy())
            mean_pred = np.mean(preds, axis=0)
            all_pred.extend(mean_pred)
            all_true.extend(y_batch.numpy())
    return np.array(all_true), np.array(all_pred)
