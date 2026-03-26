from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import grad as torch_grad

from sampling.masked.models import ConditionalDriftNetwork, ConditionalFreeEnergyNetwork
from sampling.masked.models_adaptive import (
    AdaptiveConditionalDriftNetwork,
    AdaptiveConditionalFreeEnergyNetwork,
)


class MaskedMetaReplayBuffer:
    def __init__(self, capacity: int = 5000, max_ctx_size: int = 128):
        self.capacity = capacity
        self.max_ctx_size = max_ctx_size
        # Stores: z, x_pad, y_pad, mask_pad, t
        self.buffer: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        self.ptr = 0

    def push(self, z: Tensor, x: Tensor, y: Tensor, mask: Tensor, t: Tensor) -> None:
        z, t = z.detach().cpu(), t.detach().cpu()
        x, y, mask = x.detach().cpu(), y.detach().cpu(), mask.detach().cpu()

        batch_size = z.shape[0]
        curr_n = x.shape[1]

        # PAD TO MAX SIZE
        if curr_n < self.max_ctx_size:
            pad = self.max_ctx_size - curr_n
            # Pad dim 1 (N). PyTorch pads last dim first, so (0,0, 0,pad) pads N.
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            y = torch.nn.functional.pad(y, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
        elif curr_n > self.max_ctx_size:
            # Crop if too large (rare)
            x = x[:, : self.max_ctx_size]
            y = y[:, : self.max_ctx_size]
            mask = mask[:, : self.max_ctx_size]

        if len(self.buffer) < self.capacity:
            for i in range(batch_size):
                self.buffer.append((z[i], x[i], y[i], mask[i], t[i]))
                if len(self.buffer) >= self.capacity:
                    break
        else:
            for i in range(batch_size):
                self.buffer[self.ptr] = (z[i], x[i], y[i], mask[i], t[i])
                self.ptr = (self.ptr + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        if len(self.buffer) == 0:
            return None

        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        z, x, y, m, t = zip(*[self.buffer[i] for i in indices])

        return (
            torch.stack(z).to(device),
            torch.stack(x).to(device),
            torch.stack(y).to(device),
            torch.stack(m).to(device),
            torch.stack(t).to(device),
        )


class MaskedMetaNETSSampler(nn.Module):
    def __init__(
        self,
        set_encoder: nn.Module,
        set_decoder: nn.Module,
        z_dim: int,
        r_dim: int,
        hidden_dim: int,
        sigma: float = 0.5,
        epsilon: float = 1.0,
        integration_steps: int = 50,
        max_ctx_size: int = 128,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.r_dim = r_dim
        self.sigma_sq = sigma**2
        self.epsilon = epsilon
        self.integration_steps = integration_steps

        self.set_encoder = set_encoder
        self.set_decoder = set_decoder
        # self.drift_net = ConditionalDriftNetwork(z_dim, r_dim, hidden_dim)
        # self.f_net = ConditionalFreeEnergyNetwork(r_dim, hidden_dim)
        self.drift_net = AdaptiveConditionalDriftNetwork(z_dim, r_dim, hidden_dim)
        self.f_net = AdaptiveConditionalFreeEnergyNetwork(r_dim, hidden_dim)
        self.replay_buffer = MaskedMetaReplayBuffer(max_ctx_size=max_ctx_size)

    def _energy_U(
        self,
        z: Tensor,
        x_ctx: Tensor,
        y_ctx: Tensor,
        t: float | Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        # 1. Prior Energy
        u0 = 0.5 * torch.sum(z**2, dim=1, keepdim=True)

        # 2. Likelihood Energy
        mu_y = self.set_decoder(z, x_ctx)
        sq_err = (y_ctx - mu_y) ** 2

        # Apply Mask (zero out padding error)
        if mask is not None:
            sq_err = sq_err * mask

        # Sum over N (dim 1) and Features (dim 2)
        sse = torch.sum(sq_err, dim=(1, 2)).unsqueeze(1)

        likelihood = 0.5 * sse / self.sigma_sq
        u1 = likelihood + u0

        if isinstance(t, Tensor):
            t = t.view(-1, 1)

        ut = (1 - t) * u0 + t * u1
        return ut

    def _grad_U(
        self,
        z: Tensor,
        x_ctx: Tensor,
        y_ctx: Tensor,
        t: float | Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        with torch.enable_grad():
            z_in = z.requires_grad_(True)  # z_in = z.detach().requires_grad_(True)
            u_val = self._energy_U(z_in, x_ctx, y_ctx, t, mask)
            grad = torch_grad(u_val.sum(), z_in, create_graph=True)[0]

        # CLAMPING IS CRITICAL FOR STABILITY
        return torch.clamp(grad, -100.0, 100.0)

    def _dt_U(
        self,
        z: Tensor,
        x_ctx: Tensor,
        y_ctx: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        # Analytic derivative dt_U = U1 - U0 = Likelihood
        mu_y = self.set_decoder(z, x_ctx)
        sq_err = (y_ctx - mu_y) ** 2

        if mask is not None:
            sq_err = sq_err * mask

        sse = torch.sum(sq_err, dim=(1, 2)).unsqueeze(1)
        return 0.5 * sse / self.sigma_sq

    def compute_pinn_loss(self, x_ctx: Tensor, y_ctx: Tensor, mask: Tensor) -> Tensor:
        batch_size = x_ctx.shape[0]
        device = x_ctx.device

        # Freeze decoder for PINN update
        for param in self.set_decoder.parameters():
            param.requires_grad = False

        # 1. Pad current batch to match buffer max size (for concatenation)
        curr_n = x_ctx.shape[1]
        pad_len = 128 - curr_n
        if pad_len > 0:
            x_c = torch.nn.functional.pad(x_ctx, (0, 0, 0, pad_len))
            y_c = torch.nn.functional.pad(y_ctx, (0, 0, 0, pad_len))
            m_c = torch.nn.functional.pad(mask, (0, 0, 0, pad_len))
        else:
            x_c, y_c, m_c = x_ctx, y_ctx, mask

        # 2. Mix with Buffer
        sample = self.replay_buffer.sample(batch_size // 2, device)
        if sample is not None:
            z_buf, x_buf, y_buf, m_buf, t_buf = sample

            z_new = torch.randn(batch_size // 2, self.z_dim, device=device)
            t_new = torch.rand(batch_size // 2, 1, device=device)

            z = torch.cat([z_buf, z_new], dim=0)
            t = torch.cat([t_buf, t_new], dim=0)
            x_comb = torch.cat([x_buf, x_c[: batch_size // 2]], dim=0)
            y_comb = torch.cat([y_buf, y_c[: batch_size // 2]], dim=0)
            m_comb = torch.cat([m_buf, m_c[: batch_size // 2]], dim=0)
        else:
            z = torch.randn(batch_size, self.z_dim, device=device)
            t = torch.rand(batch_size, 1, device=device)
            x_comb, y_comb, m_comb = x_c, y_c, m_c

        z.requires_grad_(True)
        t.requires_grad_(True)

        # 3. Get Representation r
        # Important: pass the mask to the encoder!
        r_comb = self.set_encoder(x_comb, y_comb, m_comb)

        # 4. PINN Terms
        b_val = self.drift_net(z, r_comb, t)
        F_val = self.f_net(r_comb, t)

        dt_F = torch_grad(F_val.sum(), t, create_graph=True)[0]
        grad_U = self._grad_U(z, x_comb, y_comb, t, m_comb)
        dt_U = self._dt_U(z, x_comb, y_comb, m_comb)

        # Hutchinson Estimator
        noise = torch.randn_like(z)
        e = 0.01
        b_p = self.drift_net(z + e * noise, r_comb, t)
        b_m = self.drift_net(z - e * noise, r_comb, t)
        div_b = (noise * (b_p - b_m)).sum(1, keepdim=True) / (2 * e)

        residual = div_b - (grad_U * b_val).sum(1, keepdim=True) - dt_U + dt_F
        loss = (residual**2).mean()

        # 5. Buffer Update
        if self.training:
            with torch.no_grad():
                self.replay_buffer.push(z, x_comb, y_comb, m_comb, t)

        # Unfreeze decoder
        for param in self.set_decoder.parameters():
            param.requires_grad = True

        return loss  # type: ignore

    def sample_posterior(self, x_ctx: Tensor, y_ctx: Tensor, mask: Tensor) -> Tensor:
        batch_size = x_ctx.shape[0]
        device = x_ctx.device
        dt = 1.0 / self.integration_steps

        # Get r
        with torch.no_grad():
            r = self.set_encoder(x_ctx, y_ctx, mask)

        z = torch.randn(batch_size, self.z_dim, device=device)

        if self.training:
            self.replay_buffer.push(
                z, x_ctx, y_ctx, mask, torch.zeros(batch_size, 1, device=device)
            )

        for step in range(self.integration_steps):
            # In MetaNETSSampler.sample_posterior loop:

            t_curr = step * dt
            t_tensor = torch.full((batch_size, 1), t_curr, device=device)

            grad_u = self._grad_U(z, x_ctx, y_ctx, t_curr, mask)
            drift_b = self.drift_net(z, r, t_tensor)

            noise = torch.randn_like(z)
            # SDE: dz = (-epsilon*grad_U + b)dt + sqrt(2*epsilon*dt)*dW
            dz = (-self.epsilon * grad_u + drift_b) * dt + np.sqrt(
                2 * self.epsilon * dt
            ) * noise
            z = z + dz

            # force = (-self.epsilon * grad_u + drift_b) * dt
            # diffusion = np.sqrt(2 * self.epsilon * dt) * noise

            # if step % 10 == 0:
            #     print(
            #         f"Force Norm: {force.norm().item():.4f} | Noise Norm: {diffusion.norm().item():.4f}"
            #     )

            if self.training and np.random.rand() < 0.1:
                self.replay_buffer.push(
                    z,
                    x_ctx,
                    y_ctx,
                    mask,
                    torch.full((batch_size, 1), t_curr + dt, device=device),
                )

        return z
