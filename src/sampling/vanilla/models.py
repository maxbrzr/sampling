import torch
from torch import Tensor, nn


class ConditionalDriftNetwork(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: Tensor, x: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        if t.dim() == 1:
            t = t.view(-1, 1)
        # Concatenate z, x, and t
        drift: Tensor = self.net(torch.cat([z, x, t], dim=-1))
        return drift


class ConditionalFreeEnergyNetwork(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        if t.dim() == 1:
            t = t.view(-1, 1)
        F: Tensor = self.net(torch.cat([x, t], dim=-1))
        return F


class Decoder(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        mu: Tensor = self.net(z)
        return mu
