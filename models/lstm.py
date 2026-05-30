"""
LSTM — Long Short-Term Memory
Input: (batch, seq_len, n_features)
Architecture: LSTM → last hidden → Linear(1)
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def build_lstm(cfg: dict, n_features: int) -> LSTM:
    c = cfg["models"]["lstm"]
    return LSTM(
        n_features=n_features,
        hidden_size=c["hidden_size"],
        num_layers=c["num_layers"],
        dropout=c["dropout"],
    )
