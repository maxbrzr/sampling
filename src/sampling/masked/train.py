from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from sampling.masked.models import MaskedSetEncoder, SetDecoder
from sampling.masked.models_adaptive import MaskedSetEncoder as AdaptiveMaskedSetEncoder
from sampling.masked.sampler import MaskedMetaNETSSampler


def train_meta_nets_final(
    device: torch.device,
    loader: DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor]],
    nets: MaskedMetaNETSSampler,
    encoder: MaskedSetEncoder | AdaptiveMaskedSetEncoder,
    decoder: SetDecoder,
    opt_nets: optim.Optimizer,
    opt_dec: optim.Optimizer,
    epochs: int = 50,
    pinn_update_freq: int = 10,
    train_decoder: bool = True,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {"pinn": [], "recon": []}

    print("Starting Meta-Training...")

    for epoch in range(epochs):
        nets.train()
        encoder.train()
        decoder.train()
        p_loss_ep, r_loss_ep = [], []

        for batch_i, batch_data in enumerate(loader):
            # Shapes: (Batch, Max_Points, 1)
            x_full, y_full, x_tar, y_tar = [b.to(device) for b in batch_data]

            # --- DYNAMIC CONTEXT SLICING ---
            # Randomly select context size k in [1, Max_Points]
            max_n = x_full.shape[1]
            n = np.random.randint(1, max_n + 1)

            print(f"Batch {batch_i + 1}/{len(loader)}: Context Size = {n}")

            x_ctx = x_full[:, :n, :]
            y_ctx = y_full[:, :n, :]

            # Create Mask (Batch, n, 1) of Ones
            mask = torch.ones((x_ctx.shape[0], n, 1), device=device)

            # Combine for FULL reconstruction target (ctx + tar)
            x_all = torch.cat([x_ctx, x_tar], dim=1)
            y_all = torch.cat([y_ctx, y_tar], dim=1)

            # --- 1. PINN Step (Multi-Step) ---
            # Freeze decoder
            for p in decoder.parameters():
                p.requires_grad = False

            # 3 PINN updates per Decoder update for stability
            for _ in range(pinn_update_freq):
                with torch.no_grad():
                    # Explore & Fill Buffer
                    _ = nets.sample_posterior(x_ctx, y_ctx, mask)

                loss_pinn = nets.compute_pinn_loss(x_ctx, y_ctx, mask)

                opt_nets.zero_grad()
                loss_pinn.backward()
                torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt_nets.step()

            # Unfreeze decoder
            for p in decoder.parameters():
                p.requires_grad = True

            # --- 2. Decoder Step ---
            # Detach z so we don't backprop through sampling
            z_post = nets.sample_posterior(x_ctx, y_ctx, mask).detach()

            # Predict on all points
            y_pred = decoder(z_post, x_all)

            loss_recon = torch.nn.MSELoss()(y_pred, y_all)

            if train_decoder:
                opt_dec.zero_grad()
                loss_recon.backward()
                opt_dec.step()

            p_loss_ep.append(loss_pinn.item())
            r_loss_ep.append(loss_recon.item())

            # Visualization
            if batch_i == len(loader) - 1:
                visualize_meta_results(
                    history,
                    x_ctx,
                    y_ctx,
                    x_all,
                    y_all,
                    mask,
                    nets,
                    decoder,
                    device,
                    "epoch_vis.png",
                )

        # Epoch Stats
        avg_p = np.mean(p_loss_ep)
        avg_r = np.mean(r_loss_ep)
        print(f"Epoch {epoch + 1}: PINN={avg_p:.2f}, Recon={avg_r:.4f}")
        history["pinn"].append(float(avg_p))
        history["recon"].append(float(avg_r))

    return history


def visualize_meta_results(
    losses: Dict[str, List[float]],
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_tar: Tensor,
    y_tar: Tensor,
    mask: Tensor,
    nets: MaskedMetaNETSSampler,
    set_decoder: SetDecoder,
    device: torch.device,
    save_path: str,
    num_samples: int = 64,
    v_min: float = 0,
    v_max: float = 500,
) -> None:
    """
    Visualizes Task 0 from the batch.
    Plots Energy Landscape and Uncertainty Bands.
    """
    nets.eval()
    set_decoder.eval()

    # 1. Select Task 0
    # x_c = x_ctx[0]  # (N, 1)
    # y_c = y_ctx[0]
    # x_t = x_tar[0]
    # y_t = y_tar[0]

    # select task with smallest context size
    ctx_sizes = mask.sum(dim=1).squeeze(-1)  # (Batch,)
    task_idx = int(torch.argmin(ctx_sizes).item())
    x_c = x_ctx[task_idx]  # (N, 1)
    y_c = y_ctx[task_idx]
    x_t = x_tar[task_idx]
    y_t = y_tar[task_idx]

    # Create Mask (Task 0 might be padded in batch, but here we treat visible points as valid)
    # Note: If x_c has padding from the loader, you should ideally mask it out,
    # but for visualization usually taking the raw data is fine.
    mask = torch.ones((1, x_c.shape[0], 1), device=device)

    # 2. Expand for sampling
    x_c_exp = x_c.unsqueeze(0).repeat(num_samples, 1, 1)
    y_c_exp = y_c.unsqueeze(0).repeat(num_samples, 1, 1)
    mask_exp = mask.repeat(num_samples, 1, 1)
    x_t_exp = x_t.unsqueeze(0).repeat(num_samples, 1, 1)

    with torch.no_grad():
        z_samples = nets.sample_posterior(x_c_exp, y_c_exp, mask_exp)
        y_preds: Tensor = set_decoder(z_samples, x_t_exp)

    # 3. Plotting Prep
    lz = z_samples.cpu().numpy()
    xp = x_t.cpu().numpy().flatten()
    yp = y_t.cpu().numpy().flatten()
    xc = x_c.cpu().numpy().flatten()
    yc = y_c.cpu().numpy().flatten()
    preds_np = y_preds.squeeze(-1).cpu().numpy()

    # Sort
    idx = np.argsort(xp)
    xp_s, yp_s = xp[idx], yp[idx]
    preds_s = preds_np[:, idx]

    plt.figure(figsize=(18, 5))

    # A. Losses
    plt.subplot(1, 3, 1)
    if len(losses["pinn"]) > 0:
        # plt.plot(losses["pinn"], label="PINN")
        plt.plot(losses["recon"], label="Recon")
    plt.yscale("log")
    plt.title("Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # B. Function
    plt.subplot(1, 3, 2)
    plt.plot(xp_s, yp_s, "k--", lw=2, label="True")
    plt.scatter(xc, yc, c="k", marker="x", s=60, label="Context")
    # Uncertainty
    mean = np.mean(preds_s, axis=0)
    std = np.std(preds_s, axis=0)
    plt.fill_between(xp_s, mean - 2 * std, mean + 2 * std, color="b", alpha=0.2)
    plt.plot(xp_s, mean, "b", lw=2, label="Mean")
    plt.ylim(-5, 5)
    plt.title("Functional Posterior")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # C. Energy
    plt.subplot(1, 3, 3)
    # Only plot if z is at least 2D
    grid_res = 100
    r = 10.0
    x = np.linspace(-r, r, grid_res)
    y = np.linspace(-r, r, grid_res)
    xx, yy = np.meshgrid(x, y)
    z_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    padding = torch.zeros(z_grid.shape[0], nets.z_dim - 2).to(device)
    z_grid = torch.cat([z_grid, padding], dim=1)

    # Context for grid
    n_grid = z_grid.shape[0]
    x_cg = x_c.unsqueeze(0).expand(n_grid, -1, -1)
    y_cg = y_c.unsqueeze(0).expand(n_grid, -1, -1)
    mask_g = torch.ones((n_grid, x_c.shape[0], 1), device=device)
    t_final = torch.ones(n_grid, 1).to(device)

    with torch.no_grad():
        u = nets._energy_U(z_grid, x_cg, y_cg, t_final, mask_g)
        u_np = u.cpu().numpy().reshape(grid_res, grid_res)

    levels = np.linspace(v_min, v_max, 21)
    plt.contourf(xx, yy, u_np, levels=levels, cmap="viridis_r", extend="both")
    plt.colorbar(label="Energy")

    plt.scatter(lz[:, 0], lz[:, 1], c="w", s=10, edgecolor="k", alpha=0.6)
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.title("Latent Energy & Samples")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Assumes other modules are imported
# from .vis import visualize_meta_results
# from .sampler import MaskedMetaNETSSampler
# from .models import MaskedSetEncoder, SetDecoder
