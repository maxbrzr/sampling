from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import nn, Tensor

from sampling.meta.models import ConditionalDriftNetwork, ConditionalFreeEnergyNetwork

# --- Reuse the provided Architectures ---
# (Assuming the class definitions provided in the prompt are available here)
# For completeness in a standalone script, one would include the classes:
# ConditionalDriftNetwork, ConditionalFreeEnergyNetwork, SetEncoder, SetDecoder


class MetaReplayBuffer:
    """
    Replay Buffer for Meta-Learning NETS.
    Stores tuples of (z, x_ctx, y_ctx, r, t).

    We must store x_ctx and y_ctx to compute the exact energy gradient
    grad_U during the PINN loss calculation.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        # Buffer stores: z, x_ctx, y_ctx, r, t
        self.buffer: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        self.ptr = 0

    def push(
        self, z: Tensor, x_ctx: Tensor, y_ctx: Tensor, r: Tensor, t: Tensor
    ) -> None:
        # Detach and move to CPU to save GPU memory
        z = z.detach().cpu()
        x_ctx = x_ctx.detach().cpu()
        y_ctx = y_ctx.detach().cpu()
        r = r.detach().cpu()
        t = t.detach().cpu()

        batch_size = z.shape[0]

        # We process item by item to handle circular buffer logic
        if len(self.buffer) < self.capacity:
            for i in range(batch_size):
                self.buffer.append((z[i], x_ctx[i], y_ctx[i], r[i], t[i]))
                if len(self.buffer) >= self.capacity:
                    break
        else:
            for i in range(batch_size):
                self.buffer[self.ptr] = (z[i], x_ctx[i], y_ctx[i], r[i], t[i])
                self.ptr = (self.ptr + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        current_len = len(self.buffer)
        if current_len == 0:
            return None

        indices = np.random.randint(0, current_len, size=batch_size)

        z_list, x_list, y_list, r_list, t_list = [], [], [], [], []
        for idx in indices:
            z, x, y, r, t = self.buffer[idx]
            z_list.append(z)
            x_list.append(x)
            y_list.append(y)
            r_list.append(r)
            t_list.append(t)

        return (
            torch.stack(z_list).to(device),
            torch.stack(x_list).to(device),
            torch.stack(y_list).to(device),
            torch.stack(r_list).to(device),
            torch.stack(t_list).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MetaNETSSampler(nn.Module):
    def __init__(
        self,
        set_encoder: nn.Module,
        set_decoder: nn.Module,
        z_dim: int,
        r_dim: int,
        sigma: float = 0.1,  # Lower sigma usually better for regression likelihoods
        epsilon: float = 1.0,
        integration_steps: int = 50,
        buffer_capacity: int = 5000,
    ):
        super().__init__()
        self.set_encoder = set_encoder
        self.set_decoder = set_decoder
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.sigma_sq = sigma**2
        self.epsilon = epsilon
        self.integration_steps = integration_steps

        # Initialize specific networks provided
        # Note: Decoder is passed in, but we init Drift/Energy here as they are NETS specific
        # We assume x_dim (for drift net) corresponds to r_dim in this meta-context
        self.drift_net = ConditionalDriftNetwork(z_dim=z_dim, r_dim=r_dim)
        self.f_net = ConditionalFreeEnergyNetwork(x_dim=r_dim)
        # r acts as x (condition)

        self.replay_buffer = MetaReplayBuffer(capacity=buffer_capacity)

    def _energy_U(
        self, z: Tensor, x_ctx: Tensor, y_ctx: Tensor, t: float | Tensor
    ) -> Tensor:
        """
        Computes the time-dependent energy U_t(z) = (1-t)U_prior + t(U_likelihood + U_prior).

        In Meta-Learning:
        Likelihood is p(Y_ctx | X_ctx, z).
        We treat 'z' as the latent task representation passed to the decoder.
        """
        # Prior Energy: Gaussian U0 = 0.5 * ||z||^2
        u0 = 0.5 * torch.sum(z**2, dim=1, keepdim=True)

        # Likelihood Energy: MSE / sigma^2
        # Decoder forward expects (r, x). We pass z in place of r.
        # This assumes z and r have compatible dimensions or the decoder
        # is capable of handling z as the representation.
        # Ideally, z_dim should equal r_dim, or SetDecoder should be defined to take z_dim.
        mu_y = self.set_decoder(z, x_ctx)

        # Sum over the number of context points (dim=1) and output dim (dim=2)
        sse = torch.sum((y_ctx - mu_y) ** 2, dim=(1, 2)).unsqueeze(1)
        likelihood = 0.5 * sse / self.sigma_sq

        u1 = likelihood + u0

        # Linear interpolation for geometric bridge
        if isinstance(t, Tensor):
            # Ensure broadcasting matches (batch, 1)
            t = t.view(-1, 1)

        ut = (1 - t) * u0 + t * u1
        return ut

    def _grad_U(
        self, z: Tensor, x_ctx: Tensor, y_ctx: Tensor, t: float | Tensor
    ) -> Tensor:
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            u_val = self._energy_U(z_in, x_ctx, y_ctx, t)
            grad = torch.autograd.grad(u_val.sum(), z_in, create_graph=True)[0]
        return torch.clamp(grad, -100.0, 100.0)

    def _dt_U(
        self, z: Tensor, x_ctx: Tensor, y_ctx: Tensor, t: float | Tensor
    ) -> Tensor:
        """
        Time derivative of Energy U.
        dt_U = U1 - U0 = Likelihood
        """
        mu_y = self.set_decoder(z, x_ctx)
        sse = torch.sum((y_ctx - mu_y) ** 2, dim=(1, 2)).unsqueeze(1)
        likelihood = 0.5 * sse / self.sigma_sq
        return likelihood

    def compute_pinn_loss(self, x_ctx: Tensor, y_ctx: Tensor) -> Tensor:
        """
        Computes the NETS Physics-Informed loss.
        x_ctx: (Batch, N, Dx)
        y_ctx: (Batch, N, Dy)
        """
        batch_size = x_ctx.shape[0]
        device = x_ctx.device

        # 1. Get Context Representation r
        # We need r for the Drift and Energy networks
        r = self.set_encoder(x_ctx, y_ctx)

        # 2. Sample Data from Replay Buffer or Init
        # We mix new random samples with buffer samples
        sample_from_buffer = self.replay_buffer.sample(batch_size // 2, device)

        if sample_from_buffer is not None:
            z_buf, x_buf, y_buf, r_buf, t_buf = sample_from_buffer

            # Generate new samples for the other half
            z_new = torch.randn(batch_size // 2, self.z_dim, device=device)
            t_new = torch.rand(batch_size // 2, 1, device=device)

            # Combine
            # Note: We duplicate current batch context for new samples
            # (or we could rely strictly on buffer for past tasks)
            # Here we assume we are training on the current batch context task
            z = torch.cat([z_buf, z_new], dim=0)
            t = torch.cat([t_buf, t_new], dim=0)

            # For buffer items, we must use their stored contexts/r
            # For new items, we use the current batch's context/r (sliced to size)
            x_combined = torch.cat([x_buf, x_ctx[: batch_size // 2]], dim=0)
            y_combined = torch.cat([y_buf, y_ctx[: batch_size // 2]], dim=0)
            r_combined = torch.cat([r_buf, r[: batch_size // 2]], dim=0)
        else:
            z = torch.randn(batch_size, self.z_dim, device=device)
            t = torch.rand(batch_size, 1, device=device)
            x_combined = x_ctx
            y_combined = y_ctx
            r_combined = r

        # Enable grads for PINN
        z.requires_grad_(True)
        t.requires_grad_(True)

        # 3. Network Forward Passes
        b_val: Tensor = self.drift_net(z, r_combined, t)
        F_val: Tensor = self.f_net(r_combined, t)

        # 4. Compute Derivatives
        dt_F = torch.autograd.grad(F_val.sum(), t, create_graph=True)[0]

        # Important: Grad U requires passing the full context set (x, y)
        term_grad_U = self._grad_U(z, x_combined, y_combined, t)
        term_dt_U = self._dt_U(z, x_combined, y_combined, t)

        # 5. Hutchinson Estimator for div(b)
        noise = torch.randn_like(z)
        delta = 0.01
        b_plus: Tensor = self.drift_net(z + delta * noise, r_combined, t)
        b_minus: Tensor = self.drift_net(z - delta * noise, r_combined, t)
        term_div_b = (noise * (b_plus - b_minus)).sum(1, keepdim=True) / (2 * delta)

        # 6. Residual Calculation
        # Residual = div(b) - grad_U . b - dt_U + dt_F
        residual = (
            term_div_b - (term_grad_U * b_val).sum(1, keepdim=True) - term_dt_U + dt_F
        )

        loss = (residual**2).mean()

        # 7. Update Replay Buffer with trajectories (optional but recommended for stability)
        # We propagate Z forward a tiny bit or just add current points
        if self.training:
            with torch.no_grad():
                self.replay_buffer.push(z, x_combined, y_combined, r_combined, t)

        return loss

    def sample_posterior(self, x_ctx: Tensor, y_ctx: Tensor) -> Tensor:
        """
        Samples z ~ p(z | x_ctx, y_ctx) using the learned Drift.
        """
        batch_size = x_ctx.shape[0]
        device = x_ctx.device
        dt = 1.0 / self.integration_steps

        # Get context embedding r
        with torch.no_grad():
            r = self.set_encoder(x_ctx, y_ctx)

        # Start from Prior p(z) ~ N(0, I)
        z = torch.randn(batch_size, self.z_dim, device=device)

        # Add initial state to buffer for training coverage
        if self.training:
            self.replay_buffer.push(
                z, x_ctx, y_ctx, r, torch.zeros(batch_size, 1, device=device)
            )

        # Integration Loop (Euler-Maruyama)
        for step in range(self.integration_steps):
            t_curr = step * dt
            t_tensor = torch.full((batch_size, 1), t_curr, device=device)

            # In sampling, we use the learned drift + grad_U correction?
            # Standard NETS uses the learned drift b(z, r, t) which approximates (grad_U + curl).
            # Usually, we also use the Control Variate trick:
            # dz = (b_net - epsilon * grad_U) dt + ...
            # But if drift_net learned the full drift, we just use b.
            # However, the provided original code uses: (-eps * grad_U + b)

            grad_u = self._grad_U(z, x_ctx, y_ctx, t_curr)
            drift_b = self.drift_net(z, r, t_tensor)

            # SDE update
            noise = torch.randn_like(z)

            # The drift term combines the conservative field (grad U) and the learned drift
            dz = (-self.epsilon * grad_u + drift_b) * dt + np.sqrt(
                2 * self.epsilon * dt
            ) * noise

            z = z + dz

            # Sparse buffer update
            if self.training and np.random.rand() < 0.2:
                self.replay_buffer.push(
                    z,
                    x_ctx,
                    y_ctx,
                    r,
                    torch.full((batch_size, 1), t_curr + dt, device=device),
                )

        return z
