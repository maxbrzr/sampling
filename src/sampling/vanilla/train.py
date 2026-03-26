from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from sampling.vanilla.models import Decoder
from sampling.vanilla.sampler import NETSSampler
from sampling.vanilla.vis import visualize_progress


def train_nets(
    device: torch.device,
    data_loader: DataLoader[Tuple[Tensor, ...]],
    nets: NETSSampler,
    decoder: Decoder,
    opt_nets: optim.Optimizer,
    opt_decoder: optim.Optimizer,
    z_dim: int = 2,
    epochs: int = 50,
    sigma_schedule: bool = True,
    sigma_start: float = 0.3,
    sigma_end: float = 0.1,
) -> None:
    history: Dict[str, List[float]] = {"pinn": [], "recon": []}

    print("Starting Training on Two Moons...")

    for epoch in range(epochs):
        nets.train()
        decoder.train()

        if sigma_schedule:
            sigma = sigma_start + (sigma_end - sigma_start) * (epoch / epochs)
            sigma_sq = torch.tensor(sigma**2, device=device)
        else:
            sigma_sq = None

        last_items = {}  # For viz

        for batch_idx, (x_batch,) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            batch_size = x_batch.shape[0]

            # --- 1. Train Sampler (PINN) ---
            # Populate buffer (no grad)
            with torch.no_grad():
                _ = nets.sample_posterior(x_batch, sigma_sq=sigma_sq)

            # Sample from buffer for off-policy training
            sample = nets.replay_buffer.sample(batch_size, device)

            if sample is not None:
                z_pinn, x_pinn, t_pinn = sample

            else:
                z_pinn = torch.randn(batch_size, z_dim, device=device)
                x_pinn = x_batch
                t_pinn = torch.rand(batch_size, 1, device=device)

            loss_pinn = nets.compute_pinn_loss(
                z_pinn, x_pinn, t_pinn, sigma_sq=sigma_sq
            )

            opt_nets.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            opt_nets.step()

            # --- 2. Train Model (Decoder) ---
            # Sample z ~ p(z|x) using current sampler
            z_post = nets.sample_posterior(x_batch, sigma_sq=sigma_sq).detach()
            recon_x = decoder(z_post)

            loss_recon = nn.MSELoss()(recon_x, x_batch)

            opt_decoder.zero_grad()
            loss_recon.backward()
            opt_decoder.step()

            history["pinn"].append(loss_pinn.item())
            history["recon"].append(loss_recon.item())

            if batch_idx == len(data_loader) - 1:
                last_items = {"x": x_batch, "recon": recon_x, "z": z_post}

        print(
            f"Epoch {epoch + 1}/{epochs} | PINN: {np.mean(history['pinn'][-len(data_loader) :]):.4f} | Recon: {np.mean(history['recon'][-len(data_loader) :]):.4f} | SigmaSQ: {sigma_sq.item() if sigma_sq is not None else nets.log_var.exp().mean().item():.4f}"
        )

        visualize_progress(
            epoch + 1, history, last_items["x"], last_items["recon"], last_items["z"]
        )
