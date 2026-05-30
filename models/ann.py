"""
ANN — Fully Connected Baseline
Input: flattened sequence (seq_len × n_features)
Architecture: Linear → ReLU → Dropout → ... → Linear(1)
"""

import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        input_dim = seq_len * n_features
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → flatten
        x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)


def build_ann(cfg: dict, seq_len: int, n_features: int) -> ANN:
    c = cfg["models"]["ann"]
    return ANN(
        seq_len=seq_len,
        n_features=n_features,
        hidden_dims=c["hidden_dims"],
        dropout=c["dropout"],
    )
