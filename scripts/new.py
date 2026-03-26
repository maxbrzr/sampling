import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

# Reuse the previous NETSSampler and components
# We paste the refined class definitions here for a self-contained script

Tensor = torch.Tensor

# --- [Previous Classes: ReplayBuffer, Networks, NETSSampler] ---
# (Included here to ensure the code is runnable as one block)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
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
            for i in range(batch_size):
                self.buffer[self.ptr] = (z[i], x[i], t[i])
                self.ptr = (self.ptr + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(self.buffer) < batch_size:
            # Fallback for early training
            indices = np.random.randint(0, len(self.buffer), size=len(self.buffer))
        else:
            indices = np.random.randint(0, len(self.buffer), size=batch_size)

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
        # Prior U0 = 0.5 * ||z||^2
        u0 = 0.5 * torch.sum(z**2, dim=1, keepdim=True)
        # Posterior U1 = U0 + Likelihood
        mu_z = self.decoder_mu(z)
        likelihood = (
            0.5 * torch.sum((x - mu_z) ** 2, dim=1, keepdim=True) / self.sigma_sq
        )
        u1 = likelihood + u0

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
        # Analytic time derivative for linear interp: U_1 - U_0 = Likelihood
        mu_z = self.decoder_mu(z)
        likelihood = (
            0.5 * torch.sum((x - mu_z) ** 2, dim=1, keepdim=True) / self.sigma_sq
        )
        return likelihood

    def compute_pinn_loss(self, z: Tensor, x: Tensor, t: Tensor) -> Tensor:
        # Important: Detach decoder parameters!
        # We do not want PINN loss to shift mu(z) to make transport easier.
        for param in self.decoder_mu.parameters():
            param.requires_grad = False

        z.requires_grad_(True)
        t.requires_grad_(True)

        b_val = self.drift_net(z, x, t)
        F_val = self.f_net(x, t)

        dt_F = torch.autograd.grad(F_val.sum(), t, create_graph=True)[0]
        term_grad_U = self._grad_U(z, x, t)
        term_dt_U = self._dt_U(z, x, t)

        # Hutchinson's trace estimator for div(b) [cite: 867]
        noise = torch.randn_like(z)
        delta = 0.01
        b_plus = self.drift_net(z + delta * noise, x, t)
        b_minus = self.drift_net(z - delta * noise, x, t)
        term_div_b = (noise * (b_plus - b_minus)).sum(1, keepdim=True) / (2 * delta)

        residual = (
            term_div_b - (term_grad_U * b_val).sum(1, keepdim=True) - term_dt_U + dt_F
        )

        # Re-enable decoder grads
        for param in self.decoder_mu.parameters():
            param.requires_grad = True

        return (residual**2).mean()

    def sample_posterior(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / self.integration_steps

        z = torch.randn(batch_size, self.z_dim, device=device)

        # Push initial state to buffer for training
        if self.training:
            self.replay_buffer.push(z, x, torch.zeros(batch_size, 1, device=device))

        for step in range(self.integration_steps):
            t_curr = step * dt
            t_tensor = torch.full((batch_size, 1), t_curr, device=device)

            grad_u = self._grad_U(z, x, t_curr)
            drift_b = self.drift_net(z, x, t_tensor)

            # SDE Step [cite: 20]
            noise = torch.randn_like(z)
            dz = (-self.epsilon * grad_u + drift_b) * dt + np.sqrt(
                2 * self.epsilon * dt
            ) * noise
            z = z + dz

            # Push intermediate states to buffer randomly (e.g. 20% chance)
            if self.training and np.random.rand() < 0.2:
                self.replay_buffer.push(
                    z, x, torch.full((batch_size, 1), t_curr + dt, device=device)
                )

        return z


# --- [Training and Visualization Logic] ---


def visualize_progress(
    epoch: int,
    losses: Dict[str, List[float]],
    real_x: Tensor,
    recon_x: Tensor,
    latents: Tensor,
    save_path: str = "nets_training.png",
):
    """
    Visualizes:
    1. Loss curves (PINN loss and Reconstruction loss)
    2. Real vs Reconstructed Data (Posterior -> Decoder)
    3. Latent Space (Posterior samples)
    """
    plt.figure(figsize=(15, 5))

    # 1. Loss Curves
    plt.subplot(1, 3, 1)
    plt.plot(losses["pinn"], label="Sampler (PINN)")
    plt.plot(losses["recon"], label="Model (Recon)")
    plt.title("Training Losses")
    plt.xlabel("Batch Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Reconstructions
    # Detach and move to CPU
    rx = real_x.detach().cpu().numpy()
    rcx = recon_x.detach().cpu().numpy()

    plt.subplot(1, 3, 2)
    # Assuming 2D data for simple scatter, or standard plot for signals
    if rx.shape[1] == 2:
        plt.scatter(rx[:, 0], rx[:, 1], c="black", alpha=0.5, label="Real Data", s=10)
        plt.scatter(
            rcx[:, 0],
            rcx[:, 1],
            c="red",
            alpha=0.5,
            label="Recon (Posterior mean)",
            s=10,
        )
    else:
        # Fallback for high-dim: plot first 2 dims
        plt.scatter(rx[:, 0], rx[:, 1], c="black", label="Real")
        plt.scatter(rcx[:, 0], rcx[:, 1], c="red", label="Recon")
    plt.title(f"Reconstruction (Epoch {epoch})")
    plt.legend()

    # 3. Latent Posterior Samples
    lz = latents.detach().cpu().numpy()
    plt.subplot(1, 3, 3)
    plt.scatter(lz[:, 0], lz[:, 1], c="blue", alpha=0.5, s=5)
    plt.title(f"Latent Posterior Samples z~p(z|x)")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_nets_ebm(
    data_loader: DataLoader,
    z_dim: int = 2,
    x_dim: int = 2,
    epochs: int = 20,
    device: torch.device = torch.device("cpu"),
):
    # 1. Setup Model
    decoder = nn.Sequential(
        nn.Linear(z_dim, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, x_dim),
    ).to(device)

    # Initialize NETS Sampler
    nets = NETSSampler(decoder, z_dim, x_dim, sigma=0.1, epsilon=2.0).to(device)

    # 2. Optimizers
    # Separate optimizers: one for the Sampler (Drift/F), one for the Model (Decoder)
    opt_nets = optim.Adam(
        list(nets.drift_net.parameters()) + list(nets.f_net.parameters()), lr=1e-3
    )
    opt_model = optim.Adam(decoder.parameters(), lr=1e-3)

    history = {"pinn": [], "recon": []}

    print("Starting Training...")

    for epoch in range(epochs):
        nets.train()
        decoder.train()

        epoch_pinn_loss = 0.0
        epoch_recon_loss = 0.0

        # For visualization at end of epoch
        last_batch_x = None
        last_batch_recon = None
        last_batch_z = None

        for batch_idx, (x_batch,) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            batch_size = x_batch.shape[0]

            # --- A. Sampler Step (Train NETS) ---
            # 1. Generate new trajectories to populate replay buffer
            #    (No grad for decoder here, just collecting data)
            with torch.no_grad():
                _ = nets.sample_posterior(x_batch)

            # 2. Sample from Replay Buffer for "Off-Policy" Training
            #    Paper recommends training on mixed past samples
            if len(nets.replay_buffer) > batch_size:
                z_pinn, x_pinn, t_pinn = nets.replay_buffer.sample(batch_size, device)
            else:
                # Warmup fallback
                z_pinn = torch.randn(batch_size, z_dim, device=device)
                x_pinn = x_batch
                t_pinn = torch.rand(batch_size, 1, device=device)

            loss_pinn = nets.compute_pinn_loss(z_pinn, x_pinn, t_pinn)

            opt_nets.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            opt_nets.step()

            # --- B. Model Step (Train Decoder) ---
            # 1. Sample z ~ p(z|x) using current NETS
            #    We detach z because we don't differentiate through the SDE sampler
            #    parameters to train the decoder (standard Monte Carlo EM).
            z_post = nets.sample_posterior(x_batch).detach()

            # 2. Reconstruction Loss: -log p(x|z)
            recon_x = decoder(z_post)
            # MSE corresponds to Gaussian likelihood
            loss_recon = nn.MSELoss()(recon_x, x_batch)

            opt_model.zero_grad()
            loss_recon.backward()
            opt_model.step()

            # Logging
            history["pinn"].append(loss_pinn.item())
            history["recon"].append(loss_recon.item())
            epoch_pinn_loss += loss_pinn.item()
            epoch_recon_loss += loss_recon.item()

            # Save for viz
            if batch_idx == len(data_loader) - 1:
                last_batch_x = x_batch
                last_batch_recon = recon_x
                last_batch_z = z_post

        # --- End of Epoch Visualization ---
        print(
            f"Epoch {epoch + 1}/{epochs} | PINN Loss: {epoch_pinn_loss / len(data_loader):.4f} | Recon Loss: {epoch_recon_loss / len(data_loader):.4f}"
        )

        visualize_progress(
            epoch + 1, history, last_batch_x, last_batch_recon, last_batch_z
        )


# --- Entry Point ---
if __name__ == "__main__":
    # Create Dummy "8-mode" like data (2D)
    # Similar to GMM experiment in paper [cite: 362]
    centers = (
        np.array(
            [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (0.7, 0.7),
                (-0.7, -0.7),
                (0.7, -0.7),
                (-0.7, 0.7),
            ]
        )
        * 3
    )
    data = []
    for _ in range(1000):
        c = centers[np.random.choice(len(centers))]
        data.append(c + np.random.randn(2) * 0.2)
    data = torch.tensor(np.array(data), dtype=torch.float32)

    loader = DataLoader(TensorDataset(data), batch_size=512, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_nets_ebm(loader, z_dim=2, x_dim=2, epochs=50, device=device)
