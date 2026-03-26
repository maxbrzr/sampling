from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor
import matplotlib.pyplot as plt

from sampling.meta.models import SASetEncoder, SetDecoder
from sampling.meta.sampler import MetaNETSSampler


def train_meta_nets(
    device: torch.device,
    data_loader: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor]],
    nets: MetaNETSSampler,
    set_encoder: SASetEncoder,
    set_decoder: SetDecoder,
    opt_nets: optim.Optimizer,
    opt_model: optim.Optimizer,
    epochs: int,
    pinn_steps: int = 3,  # Optimization: Train sampler more often
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

            # Combine context+target for full evaluation during training recon
            x_all = torch.cat([x_ctx, x_tar], dim=1)
            y_all = torch.cat([y_ctx, y_tar], dim=1)

            # --- 1. Train Sampler (PINN) ---
            # Freeze Decoder
            for p in set_decoder.parameters():
                p.requires_grad = False

            pinn_loss_accum = 0.0
            for _ in range(pinn_steps):
                with torch.no_grad():
                    _ = nets.sample_posterior(x_ctx, y_ctx)  # Explore

                loss_pinn = nets.compute_pinn_loss(x_ctx, y_ctx)

                opt_nets.zero_grad()
                loss_pinn.backward()
                torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(set_encoder.parameters(), 1.0)
                opt_nets.step()
                pinn_loss_accum += loss_pinn.item()

            # Unfreeze Decoder
            for p in set_decoder.parameters():
                p.requires_grad = True

            # --- 2. Train Decoder (Likelihood) ---
            z_post = nets.sample_posterior(x_ctx, y_ctx).detach()
            y_pred = set_decoder(z_post, x_all)
            loss_recon = nn.MSELoss()(y_pred, y_all)

            opt_model.zero_grad()
            loss_recon.backward()
            opt_model.step()

            # Logging
            batch_pinn_losses.append(pinn_loss_accum / pinn_steps)
            batch_recon_losses.append(loss_recon.item())

            # --- Visualization ---
            if batch_idx == len(data_loader) - 1:
                visualize_meta_results(
                    epoch=epoch + 1,
                    losses=history,
                    x_ctx_batch=x_ctx,
                    y_ctx_batch=y_ctx,
                    x_tar_batch=x_all,  # Visualize against FULL data (ctx + tar)
                    y_tar_batch=y_all,
                    nets=nets,
                    set_decoder=set_decoder,
                    device=device,
                    save_path="vis_epoch.png",
                )

        avg_pinn = np.mean(batch_pinn_losses)
        avg_recon = np.mean(batch_recon_losses)
        history["pinn"].append(float(avg_pinn))
        history["recon"].append(float(avg_recon))

        print(
            f"Epoch {epoch + 1}/{epochs} | PINN: {avg_pinn:.4f} | Recon: {avg_recon:.4f}"
        )

    return history


def visualize_meta_results(
    epoch: int,
    losses: Dict[str, List[float]],
    x_ctx_batch: Tensor,  # (Batch, N_ctx, 1)
    y_ctx_batch: Tensor,  # (Batch, N_ctx, 1)
    x_tar_batch: Tensor,  # (Batch, N_tar, 1)
    y_tar_batch: Tensor,  # (Batch, N_tar, 1)
    nets: nn.Module,
    set_decoder: nn.Module,
    device: torch.device,
    save_path: str = "nets_meta_progress.png",
    num_vis_samples: int = 256,  # How many z's to sample for the single task
) -> None:
    """
    Visualizes training progress by performing a 'Deep Dive' evaluation on the
    FIRST task in the provided batch.

    1. Losses: Standard training curves.
    2. Function Space: Draws `num_vis_samples` function predictions for Task 0.
    3. Latent Space: Plots the Energy Landscape U(z) for Task 0 and overlays
       the sampled z's to verify if they fall into the energy wells.
    """
    nets.eval()
    set_decoder.eval()

    # --- 1. Select Task 0 and Prepare Multi-Sample Batch ---
    # We take the first task from the batch
    x_ctx = x_ctx_batch[0]  # (N_ctx, 1)
    y_ctx = y_ctx_batch[0]  # (N_ctx, 1)
    x_tar = x_tar_batch[0]  # (N_tar, 1)
    y_tar = y_tar_batch[0]  # (N_tar, 1)

    # Expand to create a batch of `num_vis_samples` for this single task
    # Shape: (Num_Samples, N_ctx, 1)
    x_c_exp = x_ctx.unsqueeze(0).repeat(num_vis_samples, 1, 1)
    y_c_exp = y_ctx.unsqueeze(0).repeat(num_vis_samples, 1, 1)
    x_t_exp = x_tar.unsqueeze(0).repeat(num_vis_samples, 1, 1)

    # --- 2. Run Inference (Sample & Decode) ---
    with torch.no_grad():
        # Sample z ~ p(z | context)
        z_samples = nets.sample_posterior(x_c_exp, y_c_exp)  # (Num_Samples, z_dim)

        # Decode y ~ p(y | x_target, z)
        y_preds = set_decoder(z_samples, x_t_exp)  # (Num_Samples, N_tar, 1)

    # --- 3. Prepare Plotting Data ---
    # Move to Numpy
    lz = z_samples.cpu().numpy()
    xp = x_tar.squeeze().cpu().numpy()
    yp_gt = y_tar.squeeze().cpu().numpy()
    xc = x_ctx.squeeze().cpu().numpy()
    yc = y_ctx.squeeze().cpu().numpy()

    # Sort X for clean line plotting
    sort_idx = np.argsort(xp)
    xp_sorted = xp[sort_idx]
    yp_gt_sorted = yp_gt[sort_idx]

    # y_preds is (Samples, N_tar, 1) -> (Samples, N_tar)
    y_preds_np = y_preds.squeeze(-1).cpu().numpy()
    y_preds_sorted = y_preds_np[:, sort_idx]

    plt.figure(figsize=(18, 5))

    # === PANEL 1: LOSSES ===
    plt.subplot(1, 3, 1)
    if len(losses["pinn"]) > 0:
        # plt.plot(losses["pinn"], label="PINN (Sampler)", alpha=0.7)
        plt.plot(losses["recon"], label="Recon (Decoder)", alpha=0.7)
    plt.title("Training Losses")
    plt.xlabel("Step/Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # === PANEL 2: FUNCTION SPACE ===
    plt.subplot(1, 3, 2)

    # A. Plot Ground Truth
    plt.plot(
        xp_sorted, yp_gt_sorted, "k--", linewidth=2, label="Ground Truth", zorder=10
    )

    # B. Plot Context
    plt.scatter(xc, yc, c="black", s=60, marker="x", label="Context", zorder=11)

    # C. Plot Predictions (Spaghetti)
    # Plot first 50 samples individually to show diversity
    for i in range(min(50, num_vis_samples)):
        plt.plot(xp_sorted, y_preds_sorted[i], "b-", alpha=0.15, linewidth=0.8)

    # D. Plot Mean Prediction
    mean_pred = np.mean(y_preds_sorted, axis=0)
    plt.plot(xp_sorted, mean_pred, "r-", linewidth=2, label="Posterior Mean", zorder=9)

    plt.title(f"Task 0: Functional Posterior ({num_vis_samples} samples)")
    plt.ylim(-5, 5)  # Adjust for LineSine1D
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, alpha=0.3)

    # === PANEL 3: LATENT SPACE & ENERGY ===
    plt.subplot(1, 3, 3)

    # A. Calculate Energy Landscape on Grid (Task 0 Context)
    # Only if z_dim >= 2
    if nets.z_dim >= 2:
        grid_range = 15
        grid_res = 100
        x = np.linspace(-grid_range, grid_range, grid_res)
        y = np.linspace(-grid_range, grid_range, grid_res)
        xx, yy = np.meshgrid(x, y)

        # Prepare Grid Tensor: (Grid_Size^2, 2)
        grid_flat = np.c_[xx.ravel(), yy.ravel()]
        z_grid = torch.from_numpy(grid_flat).float().to(device)

        # Pad if z_dim > 2
        if nets.z_dim > 2:
            padding = torch.zeros(z_grid.shape[0], nets.z_dim - 2).to(device)
            z_grid = torch.cat([z_grid, padding], dim=1)

        # Expand Context to match Grid Batch Size
        n_grid = z_grid.shape[0]
        # x_ctx: (N_ctx, 1) -> (1, N_ctx, 1) -> (Grid_Size^2, N_ctx, 1)
        x_c_grid = x_ctx.unsqueeze(0).expand(n_grid, -1, -1)
        y_c_grid = y_ctx.unsqueeze(0).expand(n_grid, -1, -1)
        t_final = torch.ones(n_grid, 1).to(device)  # Energy at t=1 (Posterior)

        with torch.no_grad():
            # Compute U(z)
            u_vals = nets._energy_U(z_grid, x_c_grid, y_c_grid, t_final)
            u_grid = u_vals.cpu().numpy().reshape(grid_res, grid_res)

        # Plot Contours (Dark Blue = Low Energy/High Prob)
        plt.contourf(xx, yy, u_grid, levels=25, cmap="viridis_r", alpha=0.8)
        plt.colorbar(label="Energy $U(z|D)$")

    # B. Plot Sampled Zs
    plt.scatter(
        lz[:, 0],
        lz[:, 1],
        c="white",
        s=15,
        edgecolors="black",
        alpha=0.8,
        label="Sampled $z$",
        zorder=5,
    )

    plt.title("Latent Posterior & Energy Landscape\n(White dots = Samples)")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Restore training mode
    nets.train()
    set_decoder.train()
