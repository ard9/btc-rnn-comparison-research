"""
Bitcoin dataset loader and preprocessor.
Downloads BTC-USD data via yfinance, applies MinMax scaling,
and builds sliding-window sequences for time-series models.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
import pickle


class BTCDataset(Dataset):
    """PyTorch Dataset for BTC sliding-window sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def download_data(ticker: str, start: str, end: str, cache_path: str) -> pd.DataFrame:
    """Download or load cached OHLCV data."""
    if os.path.exists(cache_path):
        print(f"[Data] Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[Data] Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    print(f"[Data] Saved to {cache_path} — {len(df)} rows")
    return df


def build_sequences(data: np.ndarray, seq_len: int):
    """
    Build (X, y) pairs from a 2D array.
    X shape: (N, seq_len, n_features)
    y shape: (N,)  — next-step Close price (index 3 in OHLCV)
    """
    X, y = [], []
    close_idx = 3  # Close is 4th column: Open, High, Low, Close, Volume
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(cfg: dict):
    """
    Full pipeline: download → scale → split → build sequences → DataLoaders.

    Returns:
        loaders: dict with keys train / val / test
        scaler: fitted MinMaxScaler (for inverse transform)
        meta:   dict with shape info and split indices
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    df = download_data(
        ticker=data_cfg["ticker"],
        start=data_cfg["start_date"],
        end=data_cfg["end_date"],
        cache_path=data_cfg["data_cache"],
    )

    features = data_cfg["features"]
    df = df[features].copy()

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    # ── Temporal split (no shuffle — preserves time order) ─────────────────
    n = len(scaled)
    train_end = int(n * data_cfg["train_ratio"])
    val_end = train_end + int(n * data_cfg["val_ratio"])

    train_data = scaled[:train_end]
    val_data = scaled[train_end : val_end]
    test_data = scaled[val_end:]

    seq_len = data_cfg["sequence_length"]

    X_train, y_train = build_sequences(train_data, seq_len)
    X_val, y_val = build_sequences(val_data, seq_len)
    X_test, y_test = build_sequences(test_data, seq_len)

    print(
        f"[Data] Splits — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    batch = train_cfg["batch_size"]

    loaders = {
        "train": DataLoader(BTCDataset(X_train, y_train), batch_size=batch, shuffle=True),
        "val": DataLoader(BTCDataset(X_val, y_val), batch_size=batch, shuffle=False),
        "test": DataLoader(BTCDataset(X_test, y_test), batch_size=batch, shuffle=False),
    }

    meta = {
        "n_features": len(features),
        "seq_len": seq_len,
        "close_idx": 3,
        "scaler": scaler,
        "df": df,
        "test_dates": df.index[val_end + seq_len :],
    }

    return loaders, scaler, meta
