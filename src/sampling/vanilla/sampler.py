from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from sampling.vanilla.models import (
    ConditionalDriftNetwork,
    ConditionalFreeEnergyNetwork,
)


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer: List[Tuple[Tensor, Tensor, Tensor]] = []
        self.ptr = 0

    def push(self, z: Tensor, x: Tensor, t: Tensor) -> None:
        z, x, t = z.detach().cpu(), x.detach().cpu(), t.detach().cpu()
        batch_size = z.shape[0]
        if len(self.buffer) < self.capacity:
            for i in range(batch_size):
                self.buffer.append((z[i], x[i], t[i]))
                if len(self.buffer) >= self.capacity:
                    break
        else:
            # Circular buffer overwriting
            for i in range(batch_size):
                self.buffer[self.ptr] = (z[i], x[i], t[i])
                self.ptr = (self.ptr + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device
    ) -> None | Tuple[Tensor, Tensor, Tensor]:
        # Handle case where buffer is smaller than batch_size initially
        current_len = len(self.buffer)
        if current_len == 0:
            return None  # Should be handled by caller

        indices = np.random.randint(0, current_len, size=batch_size)

        z, x, t = [], [], []
        for idx in indices:
            zi, xi, ti = self.buffer[idx]
            z.append(zi)
            x.append(xi)
            t.append(ti)
        return (
            torch.stack(z).to(device),
            torch.stack(x).to(device),
            torch.stack(t).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class NETSSampler(nn.Module):
    def __init__(
        self,
        decoder_mu: nn.Module,
        z_dim: int,
        x_dim: int,
        integration_steps: int = 50,
        sigma: float = 0.1,
        epsilon: float = 2.0,
        sigma_learned: bool = True,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.integration_steps = integration_steps
        self.epsilon = epsilon

        self.log_var = (
            nn.Parameter(torch.log(torch.tensor(sigma**2)))
            if sigma_learned
            else torch.log(torch.tensor(sigma**2))
        )

        self.decoder_mu = decoder_mu
        self.drift_net = ConditionalDriftNetwork(z_dim, x_dim)
        self.f_net = ConditionalFreeEnergyNetwork(x_dim)
        self.replay_buffer = ReplayBuffer(capacity=20000)

    def _energy_U(
        self, z: Tensor, x: Tensor, t: float | Tensor, sigma_sq: Tensor
    ) -> Tensor:
        # z: (batch_size, z_dim)
        # x: (batch_size, x_dim)

        u0 = 0.5 * torch.sum(z**2, dim=1, keepdim=True)

        mu_z = self.decoder_mu(z)
        u_like = 0.5 * torch.sum((x - mu_z) ** 2, dim=1, keepdim=True) / sigma_sq

        u1 = u_like + u0

        ut = (1 - t) * u0 + t * u1

        return ut

    def _grad_U(
        self, z: Tensor, x: Tensor, t: float | Tensor, sigma_sq: Tensor
    ) -> Tensor:
        with torch.enable_grad():
            z_in = z.requires_grad_(True)  # z_in = z.detach().requires_grad_(True)
            u_val = self._energy_U(z_in, x, t, sigma_sq)
            grad = torch.autograd.grad(u_val.sum(), z_in, create_graph=True)[0]
        return grad

    def _dt_U(self, z: Tensor, x: Tensor, sigma_sq: Tensor) -> Tensor:
        mu_z = self.decoder_mu(z)
        u_like = 0.5 * torch.sum((x - mu_z) ** 2, dim=1, keepdim=True) / sigma_sq
        return u_like

    def compute_pinn_loss(
        self, z: Tensor, x: Tensor, t: Tensor, sigma_sq: Tensor | None = None
    ) -> Tensor:
        sigma_sq = sigma_sq or self.log_var.exp()

        # Freeze decoder for PINN update
        for param in self.decoder_mu.parameters():
            param.requires_grad = False

        z.requires_grad_(True)
        t.requires_grad_(True)

        b_val: Tensor = self.drift_net(z, x, t)
        F_val: Tensor = self.f_net(x, t)

        dt_F = torch.autograd.grad(F_val.sum(), t, create_graph=True)[0]
        term_grad_U = self._grad_U(z, x, t, sigma_sq)
        term_dt_U = self._dt_U(z, x, sigma_sq)

        # Hutchinson Estimator
        noise = torch.randn_like(z)
        delta = 0.01
        b_plus: Tensor = self.drift_net(z + delta * noise, x, t)
        b_minus: Tensor = self.drift_net(z - delta * noise, x, t)
        term_div_b = (noise * (b_plus - b_minus)).sum(1, keepdim=True) / (2 * delta)

        # Residual = div(b) - grad_U . b - dt_U + dt_F
        residual = (
            term_div_b - (term_grad_U * b_val).sum(1, keepdim=True) - term_dt_U + dt_F
        )

        # Unfreeze decoder
        for param in self.decoder_mu.parameters():
            param.requires_grad = True

        return (residual**2).mean()

    def sample_posterior(self, x: Tensor, sigma_sq: Tensor | None = None) -> Tensor:
        if sigma_sq is None:
            sigma_sq = self.log_var.exp()

        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / self.integration_steps

        z = torch.randn(batch_size, self.z_dim, device=device)

        if self.training:
            self.replay_buffer.push(z, x, torch.zeros(batch_size, 1, device=device))

        for step in range(self.integration_steps):
            t_curr = step * dt
            t_tensor = torch.full((batch_size, 1), t_curr, device=device)

            grad_u = self._grad_U(z, x, t_curr, sigma_sq)
            drift_b: Tensor = self.drift_net(z, x, t_tensor)

            # SDE step
            noise = torch.randn_like(z)
            dz: Tensor = (-self.epsilon * grad_u + drift_b) * dt + np.sqrt(
                2 * self.epsilon * dt
            ) * noise
            z = z + dz

            if self.training and np.random.rand() < 0.2:
                self.replay_buffer.push(
                    z, x, torch.full((batch_size, 1), t_curr + dt, device=device)
                )

        return z
