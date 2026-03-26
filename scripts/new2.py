import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons  # Added dependency
from typing import List, Tuple, Dict, Optional
import os

# --- [Core Classes: ReplayBuffer, Networks, NETSSampler] ---
# (These remain identical to the previous implementation)

Tensor = torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer: List[Tuple[Tensor, Tensor, Tensor]] = []
        self.ptr = 0

    def push(self, z: Tensor, x: Tensor, t: Tensor):
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
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

    def __len__(self):
        return len(self.buffer)


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
        return self.net(torch.cat([z, x, t], dim=-1))


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
        return self.net(torch.cat([x, t], dim=-1))


class NETSSampler(nn.Module):
    def __init__(
        self,
        decoder_mu: nn.Module,
        z_dim: int,
        x_dim: int,
        sigma: float = 0.5,
        epsilon: float = 1.0,
        integration_steps: int = 50,
    ):
        super().__init__()
        self.decoder_mu = decoder_mu
        self.z_dim = z_dim
        self.sigma_sq = sigma**2
        self.epsilon = epsilon
        self.integration_steps = integration_steps

        self.drift_net = ConditionalDriftNetwork(z_dim, x_dim)
        self.f_net = ConditionalFreeEnergyNetwork(x_dim)
        self.replay_buffer = ReplayBuffer(capacity=20000)

    def _energy_U(self, z: Tensor, x: Tensor, t: float | Tensor) -> Tensor:
        u0 = 0.5 * torch.sum(z**2, dim=1, keepdim=True)
        mu_z = self.decoder_mu(z)
        likelihood = (
            0.5 * torch.sum((x - mu_z) ** 2, dim=1, keepdim=True) / self.sigma_sq
        )
        u1 = likelihood + u0

        # Linear Interpolation
        if isinstance(t, float):
            return (1 - t) * u0 + t * u1
        return (1 - t) * u0 + t * u1

    def _grad_U(self, z: Tensor, x: Tensor, t: float | Tensor) -> Tensor:
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            u_val = self._energy_U(z_in, x, t)
            grad = torch.autograd.grad(u_val.sum(), z_in, create_graph=True)[0]
        return grad

    def _dt_U(self, z: Tensor, x: Tensor, t: float | Tensor) -> Tensor:
        mu_z = self.decoder_mu(z)
        likelihood = (
            0.5 * torch.sum((x - mu_z) ** 2, dim=1, keepdim=True) / self.sigma_sq
        )
        return likelihood

    def compute_pinn_loss(self, z: Tensor, x: Tensor, t: Tensor) -> Tensor:
        # Freeze decoder for PINN update
        for param in self.decoder_mu.parameters():
            param.requires_grad = False

        z.requires_grad_(True)
        t.requires_grad_(True)

        b_val = self.drift_net(z, x, t)
        F_val = self.f_net(x, t)

        dt_F = torch.autograd.grad(F_val.sum(), t, create_graph=True)[0]
        term_grad_U = self._grad_U(z, x, t)
        term_dt_U = self._dt_U(z, x, t)

        # Hutchinson Estimator
        noise = torch.randn_like(z)
        delta = 0.01
        b_plus = self.drift_net(z + delta * noise, x, t)
        b_minus = self.drift_net(z - delta * noise, x, t)
        term_div_b = (noise * (b_plus - b_minus)).sum(1, keepdim=True) / (2 * delta)

        # Residual = div(b) - grad_U . b - dt_U + dt_F
        residual = (
            term_div_b - (term_grad_U * b_val).sum(1, keepdim=True) - term_dt_U + dt_F
        )

        # Unfreeze decoder
        for param in self.decoder_mu.parameters():
            param.requires_grad = True

        return (residual**2).mean()

    def sample_posterior(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / self.integration_steps

        z = torch.randn(batch_size, self.z_dim, device=device)

        if self.training:
            self.replay_buffer.push(z, x, torch.zeros(batch_size, 1, device=device))

        for step in range(self.integration_steps):
            t_curr = step * dt
            t_tensor = torch.full((batch_size, 1), t_curr, device=device)

            grad_u = self._grad_U(z, x, t_curr)
            drift_b = self.drift_net(z, x, t_tensor)

            # SDE step
            noise = torch.randn_like(z)
            dz = (-self.epsilon * grad_u + drift_b) * dt + np.sqrt(
                2 * self.epsilon * dt
            ) * noise
            z = z + dz

            if self.training and np.random.rand() < 0.2:
                self.replay_buffer.push(
                    z, x, torch.full((batch_size, 1), t_curr + dt, device=device)
                )

        return z


# --- [Visualization & Training] ---


def visualize_progress(
    epoch: int,
    losses: Dict[str, List[float]],
    real_x: Tensor,
    recon_x: Tensor,
    latents: Tensor,
    save_path: str = "nets_moons.png",
):
    plt.figure(figsize=(18, 5))

    # 1. Loss Curves
    plt.subplot(1, 3, 1)
    plt.plot(losses["pinn"], label="PINN (Sampler)", alpha=0.7)
    plt.plot(losses["recon"], label="Recon (Decoder)", alpha=0.7)
    plt.title("Training Losses")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Reconstructions (Data Space)
    rx = real_x.detach().cpu().numpy()
    rcx = recon_x.detach().cpu().numpy()
    plt.subplot(1, 3, 2)
    plt.scatter(rx[:, 0], rx[:, 1], c="black", alpha=0.2, label="Real Data", s=20)
    plt.scatter(rcx[:, 0], rcx[:, 1], c="red", alpha=0.4, label="Recon", s=20)
    plt.title(f"Epoch {epoch}: Data Space (X)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Latents (Z Space)
    lz = latents.detach().cpu().numpy()
    plt.subplot(1, 3, 3)
    plt.scatter(lz[:, 0], lz[:, 1], c="blue", alpha=0.3, s=10)
    plt.title(f"Epoch {epoch}: Latent Space (Z)\nPrior is N(0,1)")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path + f"_epoch_{epoch}.png")
    plt.close()


def train_nets_two_moons(
    data_loader: DataLoader,
    z_dim: int = 2,
    x_dim: int = 2,
    epochs: int = 50,
    device: torch.device = torch.device("cpu"),
):
    # Setup simple MLP Decoder
    decoder = nn.Sequential(
        nn.Linear(z_dim, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, x_dim),
    ).to(device)

    # Init Sampler
    # Use sigma=0.1 to enforce tight reconstruction
    nets = NETSSampler(decoder, z_dim, x_dim, sigma=0.1, epsilon=2.0).to(device)

    opt_nets = optim.Adam(
        list(nets.drift_net.parameters()) + list(nets.f_net.parameters()), lr=5e-4
    )
    opt_model = optim.Adam(decoder.parameters(), lr=5e-4)

    history = {"pinn": [], "recon": []}

    print("Starting Training on Two Moons...")

    for epoch in range(epochs):
        nets.train()
        decoder.train()

        last_items = {}  # For viz

        for batch_idx, (x_batch,) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            batch_size = x_batch.shape[0]

            # --- 1. Train Sampler (PINN) ---
            # Populate buffer (no grad)
            with torch.no_grad():
                _ = nets.sample_posterior(x_batch)

            # Sample from buffer for off-policy training
            if len(nets.replay_buffer) > batch_size:
                z_pinn, x_pinn, t_pinn = nets.replay_buffer.sample(batch_size, device)
            else:
                z_pinn = torch.randn(batch_size, z_dim, device=device)
                x_pinn = x_batch
                t_pinn = torch.rand(batch_size, 1, device=device)

            loss_pinn = nets.compute_pinn_loss(z_pinn, x_pinn, t_pinn)

            opt_nets.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            opt_nets.step()

            # --- 2. Train Model (Decoder) ---
            # Sample z ~ p(z|x) using current sampler
            z_post = nets.sample_posterior(x_batch).detach()
            recon_x = decoder(z_post)

            loss_recon = nn.MSELoss()(recon_x, x_batch)

            opt_model.zero_grad()
            loss_recon.backward()
            opt_model.step()

            history["pinn"].append(loss_pinn.item())
            history["recon"].append(loss_recon.item())

            if batch_idx == len(data_loader) - 1:
                last_items = {"x": x_batch, "recon": recon_x, "z": z_post}

        print(
            f"Epoch {epoch + 1}/{epochs} | PINN: {np.mean(history['pinn'][-len(data_loader) :]):.4f} | Recon: {np.mean(history['recon'][-len(data_loader) :]):.4f}"
        )

        visualize_progress(
            epoch + 1, history, last_items["x"], last_items["recon"], last_items["z"]
        )


if __name__ == "__main__":
    # 1. Generate Two Moons
    n_samples = 2000
    # noise=0.05 gives distinct clean moons; noise=0.1 is fuzzier
    raw_data, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=42)

    # 2. Standardize Data (Crucial for SDE/Score stability)
    # Centers the moons around (0,0) with std dev ~1
    data_t = torch.tensor(raw_data, dtype=torch.float32)
    mean = data_t.mean(dim=0)
    std = data_t.std(dim=0)
    data_t = (data_t - mean) / std

    # 3. Create Loader
    loader = DataLoader(
        TensorDataset(data_t), batch_size=1024, shuffle=True, drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_nets_two_moons(loader, z_dim=2, x_dim=2, epochs=50, device=device)
