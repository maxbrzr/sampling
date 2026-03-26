import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict, Any
from torch import Tensor
from torch.autograd import grad as torch_grad

from sampling.meta.loader import get_meta_loader

# ==========================================
# 1. ROBUST ARCHITECTURES
# ==========================================


class AttentionSetEncoder(nn.Module):
    """
    Robust Set Encoder using Self-Attention + Cross-Attention Pooling.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        r_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Self-Attention (Contextualization)
        self.sa = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Pooling (Summarization)
        # We use a learnable query vector to summarize the set
        self.latent_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.ca = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, r_dim)

    def forward(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x, y: (Batch, N, Dim)
        # mask: (Batch, N, 1). 1.0=Keep, 0.0=Pad/Ignore.
        batch_size = x.shape[0]

        # 1. Features
        # Concat inputs: (Batch, N, x_dim+y_dim)
        pairs = torch.cat([x, y], dim=-1)
        h = self.feature_extractor(pairs)

        # 2. Masking Logic
        # PyTorch MHA expects key_padding_mask where True = Ignore.
        # Our mask is 1=Keep (False for ignore), 0=Ignore (True for ignore).
        key_padding_mask: Optional[Tensor] = None
        if mask is not None:
            # (Batch, N, 1) -> (Batch, N) -> BoolTensor
            key_padding_mask = mask.squeeze(-1) == 0

        # 3. Self-Attention (SAB)
        # h is (Batch, N, Hidden)
        attn_out, _ = self.sa(
            query=h, key=h, value=h, key_padding_mask=key_padding_mask
        )
        h = self.norm1(h + attn_out)

        # 4. Pooling Attention (PMA)
        # Query is (Batch, 1, Hidden)
        q = self.latent_query.expand(batch_size, -1, -1)

        pool_out, _ = self.ca(
            query=q, key=h, value=h, key_padding_mask=key_padding_mask
        )
        h_out = self.norm2(q + pool_out)

        # 5. Output Project
        # (Batch, 1, Hidden) -> (Batch, r_dim)
        return self.out_proj(h_out).squeeze(1)  # type: ignore


class ResidualDecoder(nn.Module):
    """
    Deep Residual Decoder for high-frequency details.
    """

    class ResBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
            self.norm = nn.LayerNorm(dim)

        def forward(self, x: Tensor) -> Tensor:
            return self.norm(x + self.net(x))  # type: ignore

    def __init__(
        self, x_dim: int, y_dim: int, z_dim: int, hidden_dim: int = 128, layers: int = 3
    ):
        super().__init__()
        self.in_proj = nn.Linear(z_dim + x_dim, hidden_dim)
        self.blocks = nn.Sequential(*[self.ResBlock(hidden_dim) for _ in range(layers)])
        self.out_proj = nn.Linear(hidden_dim, y_dim)

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        # z: (Batch, z_dim)
        # x: (Batch, N, x_dim)

        # 1. Expand z: (Batch, 1, z_dim) -> (Batch, N, z_dim)
        z_exp = z.unsqueeze(1).expand(-1, x.size(1), -1)

        # 2. Concat and forward
        inp = torch.cat([z_exp, x], dim=-1)
        h = self.in_proj(inp)
        h = self.blocks(h)
        return self.out_proj(h)  # type: ignore


class ConditionalDriftNetwork(nn.Module):
    def __init__(self, z_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + r_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        # Ensure t is (Batch, 1)
        if t.dim() < 2:
            t = t.view(-1, 1)

        # Concat [z, r, t]
        return self.net(torch.cat([z, r, t], dim=-1))  # type: ignore


# ==========================================
# 2. META-NETS SAMPLER
# ==========================================


class MetaNETS(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
        r_dim: int,
        sigma_sq: float = 1.0,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.drift = ConditionalDriftNetwork(z_dim, r_dim)

        self.z_dim = z_dim
        # Sigma squared: controls energy steepness. Lower = Steeper.
        self.sigma_sq = sigma_sq
        self.epsilon = epsilon

    def _energy_U(
        self,
        z: Tensor,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        t: float | Tensor = 1.0,
    ) -> Tensor:
        """
        Computes Energy:
            U_t(z) = (1 - t) * E_prior(z) + t * E_joint(z)

        where:
            E_prior(z) = 0.5 * ||z||^2
            E_joint(z) = E_prior(z) + sum_i (1 / (2 sigma^2)) ||y_i - f(x_i, z)||^2
        """

        # ----------------------
        # Prior energy
        # ----------------------
        # Shape: [B, 1]
        u_prior = 0.5 * torch.sum(z**2, dim=1, keepdim=True)

        # ----------------------
        # Likelihood energy (SUM over context)
        # ----------------------
        # Decoder prediction
        y_pred = self.decoder(z, x)  # [B, N, D]

        # Squared error
        sq_err = (y - y_pred) ** 2

        # Mask padding (do NOT normalize)
        sq_err = sq_err * mask

        # Sum over context points and output dimensions
        # Shape: [B, 1]
        u_lik = 0.5 * torch.sum(sq_err, dim=(1, 2), keepdim=True) / self.sigma_sq

        # ----------------------
        # Joint energy
        # ----------------------
        u_joint = u_prior + u_lik

        # ----------------------
        # Geometric bridge
        # ----------------------
        if isinstance(t, Tensor):
            t = t.view(-1, 1)

        return (1 - t) * u_prior + t * u_joint

    def compute_pinn_loss(
        self, z: Tensor, x: Tensor, y: Tensor, mask: Tensor, r: Tensor
    ) -> Tensor:
        """
        Computes residual of the Fokker-Planck equation to train Drift.
        """
        # Enable grads for z and t for derivatives
        z.requires_grad_(True)
        t = torch.rand(z.shape[0], 1, device=z.device).requires_grad_(True)

        # 1. Forward Drift (Prediction)
        b = self.drift(z, r, t)

        # 2. Gradients of Energy (Target)
        # We need gradients to flow back to Drift `b`, but usually we treat Energy as fixed target.
        # However, `grad_u` itself doesn't need to be differentiated unless we train energy via PINN (we don't).
        with torch.enable_grad():
            u_val = self._energy_U(z, x, y, mask, t)

            # Gradient of U w.r.t z (Force)
            grad_u = torch_grad(u_val.sum(), z, create_graph=True)[0]
            grad_u = torch.clamp(grad_u, -100.0, 100.0)  # Stability Clip

            # Time derivative of U
            dt_u = torch_grad(u_val.sum(), t, create_graph=True)[0]

        # 3. Divergence of Drift (Hutchinson Estimator)
        # Calculates div(b) efficiently
        noise_eps = torch.randn_like(z)
        e = 0.01
        b_plus = self.drift(z + e * noise_eps, r, t)
        b_minus = self.drift(z - e * noise_eps, r, t)
        div_b = (noise_eps * (b_plus - b_minus)).sum(1, keepdim=True) / (2 * e)

        # 4. Residual
        # Res = div(b) - grad_U * b - dt_U (Simplified form)
        residual = div_b - (grad_u * b).sum(1, keepdim=True) - dt_u

        return (residual**2).mean()  # type: ignore

    def sample(
        self, x_ctx: Tensor, y_ctx: Tensor, mask: Tensor, steps: int = 20
    ) -> Tensor:
        """
        Generates samples z ~ p(z | context) via SDE.
        """
        batch_size = x_ctx.shape[0]
        device = x_ctx.device

        # Get context embedding r
        with torch.no_grad():
            r = self.encoder(x_ctx, y_ctx, mask)

        # Start at Prior z ~ N(0, I)
        z = torch.randn(batch_size, self.z_dim, device=device)

        dt = 1.0 / steps

        # Euler-Maruyama Integration
        for i in range(steps):
            t_curr = i * dt

            # Calculate Gradient Force from Energy (The "Slope")
            with torch.enable_grad():
                z_in = z.detach().requires_grad_(True)
                u = self._energy_U(z_in, x_ctx, y_ctx, mask, t_curr)
                grad = torch_grad(u.sum(), z_in)[0]
                grad = torch.clamp(grad, -100.0, 100.0)

            # Learned Drift Correction
            t_tens = torch.full((batch_size, 1), t_curr, device=device)
            b = self.drift(z, r, t_tens)

            # SDE Update: dz = (-grad_U + b)dt + sqrt(2dt)dW
            # Note: We assume epsilon=1.0 for simplicity
            noise = torch.randn_like(z)
            force = -grad + b

            z = z + force * dt + np.sqrt(2 * dt) * noise

        return z


# ==========================================
# 3. UNIFIED TRAINER
# ==========================================


def train_end_to_end(
    device: torch.device,
    loader: DataLoader,  # type: ignore
    model: MetaNETS,
    epochs: int = 50,
) -> Dict[str, List[float]]:
    # Separate optimizers for stability
    # 1. Generator: Encoder + Decoder (Creates the Landscape)
    opt_gen = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=5e-4
    )
    # 2. Navigator: Drift Network (Learns to move in the Landscape)
    opt_nav = optim.Adam(model.drift.parameters(), lr=5e-4)

    hist: Dict[str, List[float]] = {"recon": [], "pinn": []}

    print("--- Starting Stable End-to-End Training ---")

    for epoch in range(epochs):
        model.train()

        # --- A. SIGMA ANNEALING ---
        # Decay sigma from 2.0 -> 0.2 over 80% of epochs.
        # This makes the energy well "wide" at first, then "narrow/precise" later.
        prog = min(1.0, epoch / (epochs * 0.8))
        current_sigma = 2.0 - (1.8 * prog)
        model.sigma_sq = current_sigma**2

        losses_recon = []
        losses_pinn = []

        for batch_data in loader:
            x_full, y_full, x_tar, y_tar = [b.to(device) for b in batch_data]

            # Dynamic Context Slicing
            # Randomly choose how many context points this batch has (1 to Max)
            max_points = x_full.shape[1]
            n_ctx = np.random.randint(1, max_points + 1)

            x_ctx = x_full[:, :n_ctx, :]
            y_ctx = y_full[:, :n_ctx, :]

            # Create Mask (Batch, n_ctx, 1)
            mask = torch.ones((x_ctx.shape[0], n_ctx, 1), device=device)

            # Full target set for reconstruction
            x_all = torch.cat([x_ctx, x_tar], dim=1)
            y_all = torch.cat([y_ctx, y_tar], dim=1)

            # --- STEP 1: TRAIN LANDSCAPE (Encoder/Decoder) ---
            # We enforce r to be near 0 so the Energy Well is centered for the sampler
            r = model.encoder(x_ctx, y_ctx, mask)

            # Denoising Trick: z = r + noise
            # Mimics a VAE. Forces decoder to accept a "cloud" around r.
            z_proxy = r + 0.1 * torch.randn_like(r)

            y_pred = model.decoder(z_proxy, x_all)

            loss_recon = nn.MSELoss()(y_pred, y_all)
            loss_reg = 0.01 * (r**2).mean()  # L2 Regularization on r (keep near 0)

            loss_gen = loss_recon + loss_reg

            opt_gen.zero_grad()
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt_gen.step()

            # --- STEP 2: TRAIN SAMPLER (PINN) ---
            # Now that the landscape (Decoder) is updated, train Drift to navigate it.

            # Sample random z from Prior (where inference starts)
            z_rand = torch.randn(x_ctx.shape[0], model.z_dim, device=device)

            # We need r again (detached, so sampler doesn't mess up encoder)
            with torch.no_grad():
                r_fixed = model.encoder(x_ctx, y_ctx, mask)

            loss_pinn = model.compute_pinn_loss(z_rand, x_ctx, y_ctx, mask, r_fixed)

            opt_nav.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(model.drift.parameters(), 1.0)
            opt_nav.step()

            losses_recon.append(loss_recon.item())
            losses_pinn.append(loss_pinn.item())

        # Epoch Logging
        avg_recon = np.mean(losses_recon)
        avg_pinn = np.mean(losses_pinn)
        hist["recon"].append(float(avg_recon))
        hist["pinn"].append(float(avg_pinn))

        print(
            f"Epoch {epoch + 1} | Sigma: {current_sigma:.2f} | "
            f"Recon: {avg_recon:.4f} | PINN: {avg_pinn:.4f}"
        )

        # Visual Check (Optional)
        if (epoch + 1) % 5 == 0:
            visualize_check(model, x_ctx, y_ctx, x_all, y_all, epoch, device)  # type: ignore

    return hist


def visualize_check(
    model: MetaNETS,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_tar: Tensor,
    y_tar: Tensor,
    epoch: int,
    device: torch.device,
) -> None:
    """Quick visual check of Task 0"""
    model.eval()

    # Task 0 Data
    xc = x_ctx[0:1]  # (1, N, 1)
    yc = y_ctx[0:1]
    mask = torch.ones_like(xc)
    xt = x_tar[0:1]
    yt = y_tar[0:1]

    # Expand for sampling 20 variants
    N_S = 20
    xc_e = xc.repeat(N_S, 1, 1)
    yc_e = yc.repeat(N_S, 1, 1)
    mask_e = mask.repeat(N_S, 1, 1)
    xt_e = xt.repeat(N_S, 1, 1)

    # Sample and Decode
    z_s = model.sample(xc_e, yc_e, mask_e, steps=20)
    y_p = model.decoder(z_s, xt_e).detach().cpu().numpy().squeeze()

    # Sorting for Plot
    x_plot = xt.cpu().numpy().flatten()
    y_gt = yt.cpu().numpy().flatten()
    idx = np.argsort(x_plot)

    plt.figure(figsize=(6, 4))
    plt.plot(x_plot[idx], y_gt[idx], "k--", lw=2, label="True")
    plt.scatter(xc.cpu().numpy(), yc.cpu().numpy(), c="k", marker="x", s=50, zorder=5)

    # Plot Samples
    for i in range(N_S):
        if y_p.ndim == 1:
            y_line = y_p[idx]  # Handle single sample case
        else:
            y_line = y_p[i][idx]
        plt.plot(x_plot[idx], y_line, "b-", alpha=0.15)

    plt.title(f"Epoch {epoch + 1} Check")
    plt.tight_layout()
    plt.savefig(f"check_epoch_{epoch + 1}.png")
    plt.close()

    model.train()


from metalearning_benchmarks.sinusoid1d_benchmark import Sinusoid1D


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    benchmark = Sinusoid1D(
        n_task=128,  # number of tasks
        n_datapoints_per_task=64,  # number of evaluations per task
        output_noise=0.01,  # Gaussian noise with std-dev 0.01
        seed_task=0,
        seed_x=0,
        seed_noise=0,
    )

    loader = get_meta_loader(benchmark, batch_size=512, context_size=10)

    x_dim = 1
    y_dim = 1
    z_dim = 2
    r_dim = 8

    set_encoder = AttentionSetEncoder(x_dim=x_dim, y_dim=y_dim, r_dim=r_dim).to(device)

    # Decoder: Predicts target (z, X_target) -> Y_target
    # Note: The decoder treats 'z' as the representation.
    # If SetDecoder expects 'r_dim' as input size, we pass z_dim.
    set_decoder = ResidualDecoder(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim).to(device)

    # Sampler: Manages Drift, F-Net, and Replay Buffer
    # We pass the encoder/decoder so the sampler can compute internal energies
    nets = MetaNETS(
        encoder=set_encoder,
        decoder=set_decoder,
        z_dim=z_dim,
        r_dim=r_dim,
    ).to(device)

    _ = train_end_to_end(device, loader, nets, epochs=100)
