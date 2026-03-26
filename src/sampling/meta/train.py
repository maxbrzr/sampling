from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from sampling.meta.models import SASetEncoder, SetDecoder, SetEncoder
from sampling.meta.sampler import MetaNETSSampler
from sampling.meta.vis import visualize_meta_results

# Assuming visualize_meta_results is imported or defined in scope
# from your_module import visualize_meta_results


def train_meta_nets(
    device: torch.device,
    data_loader: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor]],
    nets: MetaNETSSampler,
    set_encoder: SetEncoder | SASetEncoder,
    set_decoder: SetDecoder,
    opt_nets: optim.Optimizer,  # Includes drift, f_net, AND set_encoder parameters
    opt_decoder: optim.Optimizer,  # Includes set_decoder parameters
    epochs: int,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {"pinn": [], "recon": []}

    print("Starting Meta-Training...")

    for epoch in range(epochs):
        nets.train()
        set_encoder.train()
        set_decoder.train()

        batch_pinn_losses = []
        batch_recon_losses = []

        for batch_idx, batch_data in enumerate(data_loader):
            # Shapes: (Batch, N, Dim)
            x_ctx, y_ctx, x_tar, y_tar = [b.to(device) for b in batch_data]
            x_all = torch.cat([x_ctx, x_tar], dim=1)
            y_all = torch.cat([y_ctx, y_tar], dim=1)

            # ==========================
            # 1. Train Sampler & Encoder (PINN)
            # ==========================
            # We want to train the Encoder to produce 'r' such that the Drift
            # matches the gradient of the Energy defined by the Decoder.
            # We MUST freeze the Decoder so the energy landscape is static target.
            for p in set_decoder.parameters():
                p.requires_grad = False

            # A. Populate Buffer (Exploration)
            # Run sampler to fill buffer with visited states. No grad needed here.
            with torch.no_grad():
                _ = nets.sample_posterior(x_ctx, y_ctx)

            # B. Compute PINN Loss
            # This calculates the residual of the Fokker-Planck equation.
            # Gradients must flow back to: DriftNet, FNet, and SetEncoder.
            # (Ensure `compute_pinn_loss` does NOT use torch.no_grad() for the encoder forward pass)
            loss_pinn = nets.compute_pinn_loss(x_ctx, y_ctx)

            opt_nets.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            # Also clip encoder grads if necessary
            torch.nn.utils.clip_grad_norm_(set_encoder.parameters(), 1.0)
            opt_nets.step()

            # Unfreeze Decoder for the next step
            for p in set_decoder.parameters():
                p.requires_grad = True

            # ==========================
            # 2. Train Decoder (Likelihood)
            # ==========================
            # We want the Decoder to maximize the likelihood of the target data
            # given the latent codes z sampled by our (now slightly better) sampler.

            # A. Sample Posterior z ~ p(z | x_ctx, y_ctx)
            # We DETACH z. We treat the sampler as a "fixed" source of noise/latents.
            # We do not backpropagate reconstruction error into the sampler or encoder.
            z_post = nets.sample_posterior(x_ctx, y_ctx).detach()

            # B. Reconstruct Target Set
            # Decoder takes (z, x_target) -> y_pred
            y_pred = set_decoder(z_post, x_all)  # x_tar)

            # C. Reconstruction Loss
            loss_recon = nn.MSELoss()(y_pred, y_all)  # y_tar)

            opt_decoder.zero_grad()
            loss_recon.backward()
            opt_decoder.step()

            # Logging
            batch_pinn_losses.append(loss_pinn.item())
            batch_recon_losses.append(loss_recon.item())

            # Visualization at end of epoch
            if batch_idx == len(data_loader) - 1:
                visualize_meta_results(
                    epoch=epoch + 1,
                    losses=history,
                    x_ctx=x_ctx,
                    y_ctx=y_ctx,
                    x_tar=x_all,  # x_tar,
                    y_tar=y_all,  # y_tar,
                    y_pred=y_pred,
                    latents=z_post,
                    save_path="vis_epoch.png",  # Overwrites to save space, or use f"vis_{epoch}.png"
                )

        # End of Epoch Stats
        avg_pinn = np.mean(batch_pinn_losses)
        avg_recon = np.mean(batch_recon_losses)

        history["pinn"].append(float(avg_pinn))
        history["recon"].append(float(avg_recon))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"PINN Loss: {avg_pinn:.4f} | "
            f"Recon Loss: {avg_recon:.4f}"
        )

    return history


def train_meta_nets_dyn(
    device: torch.device,
    data_loader: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor]],
    nets: MetaNETSSampler,
    set_encoder: SetEncoder | SASetEncoder,
    set_decoder: SetDecoder,
    opt_nets: optim.Optimizer,
    opt_decoder: optim.Optimizer,
    epochs: int = 50,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {"pinn": [], "recon": []}

    print("Starting Meta-Training with Variable Context Sizes...")

    for epoch in range(epochs):
        nets.train()
        set_encoder.train()
        set_decoder.train()

        # [Safety] Clear buffer at start of epoch to prevent shape mismatch errors
        # if the previous epoch ended with a different context size structure.
        # Ideally, a robust buffer handles padding, but clearing prevents torch.stack crashes.
        if hasattr(nets, "replay_buffer"):
            nets.replay_buffer.buffer = []
            nets.replay_buffer.ptr = 0

        batch_pinn_losses = []
        batch_recon_losses = []

        for batch_idx, batch_data in enumerate(data_loader):
            # 1. Unpack original batch
            # Shapes: (Batch, N_max, Dim)
            x_ctx_full, y_ctx_full, x_tar, y_tar = [b.to(device) for b in batch_data]

            # 2. Random Context Sizing Logic
            # We assume dimension 1 is the number of points (Batch, Points, Dim)
            max_context_points = x_ctx_full.shape[1]

            # Sample a random integer k: 1 <= k <= max_context_points
            current_k = np.random.randint(1, max_context_points + 1)

            # Slice the context sets to the new random size
            # We keep the targets (x_tar) as is, usually we evaluate on the full set
            x_ctx = x_ctx_full[:, :current_k, :]
            y_ctx = y_ctx_full[:, :current_k, :]

            # ==========================
            # 3. Train Sampler & Encoder (PINN)
            # ==========================
            for p in set_decoder.parameters():
                p.requires_grad = False

            # A. Populate Buffer
            with torch.no_grad():
                _ = nets.sample_posterior(x_ctx, y_ctx)

            # B. Compute PINN Loss
            # Note: If ReplayBuffer has mixed sizes from previous iterations, this might error.
            # If so, move `nets.replay_buffer.buffer = []` inside this loop (makes training less stable).
            try:
                loss_pinn = nets.compute_pinn_loss(x_ctx, y_ctx)
            except RuntimeError as e:
                if "stack" in str(e):
                    print("Buffer Shape Mismatch detected. Clearing buffer...")
                    nets.replay_buffer.buffer = []
                    nets.replay_buffer.ptr = 0
                    # Retry with empty buffer (using just current batch)
                    loss_pinn = nets.compute_pinn_loss(x_ctx, y_ctx)
                else:
                    raise e

            opt_nets.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(set_encoder.parameters(), 1.0)
            opt_nets.step()

            for p in set_decoder.parameters():
                p.requires_grad = True

            # ==========================
            # 4. Train Decoder (Likelihood)
            # ==========================
            z_post = nets.sample_posterior(x_ctx, y_ctx).detach()
            y_pred_ctx = set_decoder(z_post, x_ctx)  # x_tar)

            # mse but average over batch and context size
            loss_recon = torch.mean((y_pred_ctx - y_ctx) ** 2, dim=(0, 1))
            # loss_recon = nn.MSELoss()(y_pred_ctx, y_ctx)  # y_tar)

            opt_decoder.zero_grad()
            loss_recon.backward()
            opt_decoder.step()

            with torch.no_grad():
                y_pred_tar = set_decoder(z_post, x_tar)

            # Logging
            batch_pinn_losses.append(loss_pinn.item())
            batch_recon_losses.append(loss_recon.item())

            # Visualization
            if batch_idx == len(data_loader) - 1:
                # We visualize using the specific context size chosen for this batch
                visualize_meta_results(
                    epoch=epoch + 1,
                    losses=history,
                    x_ctx=x_ctx,  # Passed the sliced version
                    y_ctx=y_ctx,  # Passed the sliced version
                    x_tar=x_tar,  # torch.cat([x_ctx, x_tar], dim=1),
                    y_tar=y_tar,  # torch.cat([y_ctx, y_tar], dim=1),
                    y_pred=y_pred_tar,
                    latents=z_post,
                    save_path="vis_epoch.png",
                )

        # End of Epoch Stats
        avg_pinn = np.mean(batch_pinn_losses)
        avg_recon = np.mean(batch_recon_losses)

        history["pinn"].append(float(avg_pinn))
        history["recon"].append(float(avg_recon))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Ctx Size: {current_k}/{max_context_points} | "
            f"PINN: {avg_pinn:.4f} | Recon: {avg_recon:.4f}"
        )

    return history
