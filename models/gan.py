"""
Conditional GAN with WGAN-GP for BTC price forecasting.

Architecture:
  Generator  : noise + condition (last seq) → GRU → predicted next price
  Discriminator: real/fake sequence pairs → GRU → scalar score (no sigmoid, Wasserstein)

Training loop (called externally via train_gan):
  - n_critic discriminator steps per 1 generator step
  - Gradient Penalty (GP) stabilizes training (no weight clipping)
"""

import torch
import torch.nn as nn
import torch.autograd as autograd


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Inputs:
        condition : (batch, seq_len, n_features)  — historical window
        noise     : (batch, noise_dim)
    Output:
        pred      : (batch,)  — next-step Close (scaled)
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, noise_dim: int):
        super().__init__()
        self.noise_dim = noise_dim

        self.condition_encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # output in [0,1] — matches MinMax scaled target
        )

    def forward(self, condition: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        _, h = self.condition_encoder(condition)
        context = h[-1]                          # last layer hidden: (batch, hidden_size)
        combined = torch.cat([context, noise], dim=1)
        return self.fc(combined).squeeze(-1)


# ── Discriminator ──────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Inputs:
        condition  : (batch, seq_len, n_features)
        price      : (batch, 1)  — real or generated next price
    Output:
        score      : (batch,)   — Wasserstein critic score (unbounded)
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.condition_encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 1, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            # No activation — Wasserstein critic
        )

    def forward(self, condition: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        _, h = self.condition_encoder(condition)
        context = h[-1]
        combined = torch.cat([context, price.unsqueeze(1) if price.dim() == 1 else price], dim=1)
        return self.fc(combined).squeeze(-1)


# ── Gradient Penalty ───────────────────────────────────────────────────────────

def gradient_penalty(discriminator: Discriminator,
                     condition: torch.Tensor,
                     real_price: torch.Tensor,
                     fake_price: torch.Tensor,
                     device: torch.device,
                     lambda_gp: float = 10.0) -> torch.Tensor:
    """WGAN-GP gradient penalty term."""
    batch = real_price.size(0)
    alpha = torch.rand(batch, 1, device=device)
    interpolated = (alpha * real_price.unsqueeze(1) +
                    (1 - alpha) * fake_price.unsqueeze(1).detach()).requires_grad_(True)

    d_interp = discriminator(condition, interpolated.squeeze(1))

    gradients = autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * penalty


# ── Factory ────────────────────────────────────────────────────────────────────

def build_gan(cfg: dict, n_features: int):
    c = cfg["models"]["gan"]
    G = Generator(
        n_features=n_features,
        hidden_size=c["hidden_size"],
        num_layers=c["num_layers"],
        noise_dim=c["noise_dim"],
    )
    D = Discriminator(
        n_features=n_features,
        hidden_size=c["hidden_size"],
        num_layers=c["num_layers"],
    )
    return G, D
