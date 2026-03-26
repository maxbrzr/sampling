from typing import Dict, List
from matplotlib import pyplot as plt
from torch import Tensor
import numpy as np
import torch

from sampling.meta.models import SASetEncoder, SetDecoder, SetEncoder
from sampling.meta.sampler import MetaNETSSampler


def visualize_meta_results(
    epoch: int,
    losses: Dict[str, List[float]],
    x_ctx: Tensor,  # (Batch, N_ctx, 1)
    y_ctx: Tensor,  # (Batch, N_ctx, 1)
    x_tar: Tensor,  # (Batch, N_tar, 1)
    y_tar: Tensor,  # (Batch, N_tar, 1)
    y_pred: Tensor,  # (Batch, N_tar, 1) -> Decoder Output
    latents: Tensor,  # (Batch, z_dim)
    save_path: str = "nets_meta_progress.png",
) -> None:
    """
    Visualizes Meta-Learning progress for 1D regression tasks.
    """
    plt.figure(figsize=(20, 5))

    # --- Panel 1: Loss Curves ---
    plt.subplot(1, 4, 1)
    # Handle case where history might be empty initially
    if len(losses["pinn"]) > 0:
        # plt.plot(losses["pinn"], label="PINN (Sampler)", alpha=0.7)
        plt.plot(losses["recon"], label="Recon (Decoder)", alpha=0.7)
    plt.title("Training Losses")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Panel 2 & 3: Task Reconstructions (Show first 2 tasks in batch) ---
    # We plot the first 2 distinct tasks from the batch to show diversity

    # Move to CPU/Numpy
    xc = x_ctx.detach().cpu().numpy()
    yc = y_ctx.detach().cpu().numpy()
    xt = x_tar.detach().cpu().numpy()
    yt = y_tar.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()

    # Plot up to 2 tasks
    num_tasks_to_plot = min(2, xc.shape[0])

    for i in range(num_tasks_to_plot):
        plt.subplot(1, 4, 2 + i)

        # Sort by x for clean line plotting
        sort_idx_t = np.argsort(xt[i].flatten())
        xt_sorted = xt[i][sort_idx_t]
        yt_sorted = yt[i][sort_idx_t]
        y_pred_sorted = yp[i][sort_idx_t]

        # Ground Truth (Target)
        plt.plot(
            xt_sorted, yt_sorted, "k--", alpha=0.4, label="True Function", linewidth=1.5
        )

        # Context Points (What the encoder saw)
        plt.scatter(xc[i], yc[i], c="black", s=40, label="Context", zorder=5)

        # Model Prediction
        plt.plot(
            xt_sorted,
            y_pred_sorted,
            "r-",
            alpha=0.8,
            label="Posterior Mean",
            linewidth=2,
        )

        plt.title(f"Task {i + 1} (Epoch {epoch})")
        if i == 0:
            plt.legend(loc="upper right", fontsize="small")
        plt.xlim(-4, 4)
        plt.ylim(-10, 10)
        plt.grid(True, alpha=0.3)

    # --- Panel 4: Latent Space ---
    lz = latents.detach().cpu().numpy()
    plt.subplot(1, 4, 4)

    # If z is 2D, simple scatter. If >2D, plot first 2 dims.
    plt.scatter(lz[:, 0], lz[:, 1], c="blue", alpha=0.5, s=15, edgecolors="none")
    plt.title("Latent Space $z$ (Batch)\nPrior $p(z) \sim \mathcal{{N}}(0, I)$")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.grid(True, alpha=0.3)

    # Draw unit circle/box to show prior scale
    # circle = plt.Circle((0, 0), 2, color="k", fill=False, linestyle="--", alpha=0.3)
    # plt.gca().add_patch(circle)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_task_eval(
    nets: MetaNETSSampler,
    set_encoder: SetEncoder | SASetEncoder,
    set_decoder: SetDecoder,
    x_ctx: Tensor,  # Shape: (N_ctx, 1) or (1, N_ctx, 1)
    y_ctx: Tensor,  # Shape: (N_ctx, 1) or (1, N_ctx, 1)
    x_tar: Tensor,  # Shape: (N_tar, 1) or (1, N_tar, 1)
    y_tar: Tensor,  # Shape: (N_tar, 1) or (1, N_tar, 1) (Ground Truth)
    device: torch.device,
    num_samples: int = 512,
    save_path: str = "eval_task_512.png",
) -> None:
    """
    Evaluates a trained Meta-NETS model on a single task by drawing
    `num_samples` from the posterior p(z | x_ctx, y_ctx).

    Visualizes:
    1. The predictive distribution (mean +/- 2 std dev) in data space.
    2. The empirical posterior distribution in latent space.
    """
    nets.eval()
    set_encoder.eval()
    set_decoder.eval()

    # 1. Prepare Data Batching
    # Ensure inputs are (1, N, Dim) initially
    if x_ctx.dim() == 2:
        x_ctx = x_ctx.unsqueeze(0)
    if y_ctx.dim() == 2:
        y_ctx = y_ctx.unsqueeze(0)
    if x_tar.dim() == 2:
        x_tar = x_tar.unsqueeze(0)

    # Expand to batch size of `num_samples` to process in parallel
    # Shape becomes: (512, N_ctx, 1)
    x_ctx_expanded = x_ctx.repeat(num_samples, 1, 1).to(device)
    y_ctx_expanded = y_ctx.repeat(num_samples, 1, 1).to(device)
    x_tar_expanded = x_tar.repeat(num_samples, 1, 1).to(device)

    # 2. Sample Posterior (512 z's for the same context)
    with torch.no_grad():
        # This runs the SDE integration 512 times in parallel
        z_samples = nets.sample_posterior(x_ctx_expanded, y_ctx_expanded)

        # Decode all z samples
        # y_pred_batch: (512, N_tar, 1)
        y_pred_batch = set_decoder(z_samples, x_tar_expanded)

    # 3. Compute Statistics for Plotting
    # y_pred_np: (512, N_tar)
    y_pred_np = y_pred_batch.squeeze(-1).cpu().numpy()

    # Sort x_tar for clean line plotting
    x_tar_np = x_tar.squeeze().cpu().numpy()
    sort_idx = np.argsort(x_tar_np)
    x_sorted = x_tar_np[sort_idx]

    # Sort predictions according to x
    y_pred_sorted = y_pred_np[:, sort_idx]

    # Mean and Std Dev per point
    mean_pred = np.mean(y_pred_sorted, axis=0)
    std_pred = np.std(y_pred_sorted, axis=0)

    # 4. Visualization
    plt.figure(figsize=(12, 5))

    # --- Plot 1: Functional Output Space ---
    plt.subplot(1, 2, 1)

    # Ground Truth
    y_tar_np = y_tar.squeeze().cpu().numpy()[sort_idx]
    plt.plot(x_sorted, y_tar_np, "k--", linewidth=2, label="Ground Truth", zorder=3)

    # Context Points
    xc_np = x_ctx.squeeze().cpu().numpy()
    yc_np = y_ctx.squeeze().cpu().numpy()
    plt.scatter(xc_np, yc_np, c="black", s=60, marker="x", label="Context", zorder=4)

    # Uncertainty Band (Mean +/- 2 Sigma)
    plt.fill_between(
        x_sorted,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        color="blue",
        alpha=0.2,
        label="Pred $\pm 2\sigma$",
    )
    plt.plot(x_sorted, mean_pred, "b-", linewidth=2, label="Mean Pred", zorder=2)

    # (Optional) Plot a few faint spaghetti lines to show diversity
    for i in range(min(10, num_samples)):
        plt.plot(x_sorted, y_pred_sorted[i], "b-", alpha=0.15, linewidth=0.8)

    plt.title(f"Predictive Posterior ($N={num_samples}$ samples)")
    plt.legend()
    plt.ylim(-5, 5)  # Adjust based on data range
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Latent Space ---
    plt.subplot(1, 2, 2)
    z_np = z_samples.cpu().numpy()

    # Check dimensions
    if z_np.shape[1] == 2:
        plt.scatter(z_np[:, 0], z_np[:, 1], c="blue", alpha=0.5, s=10, edgecolor="none")
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
    else:
        # If High-D, plot first 2 dims or use PCA (simple slicing here)
        plt.scatter(z_np[:, 0], z_np[:, 1], c="blue", alpha=0.5, s=10, edgecolor="none")
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
        plt.title(f"Latent Posterior (Dims 1 & 2 of {z_np.shape[1]})")

    plt.title("Latent Samples $z \sim p(z|D_{ctx})$")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.grid(True, alpha=0.3)

    # Add a unit circle to visualize the Prior N(0,I)
    circle = plt.Circle(
        (0, 0),
        2,
        color="k",
        fill=False,
        linestyle="--",
        alpha=0.3,
        label="Prior $2\sigma$",
    )
    plt.gca().add_patch(circle)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation plot saved to {save_path}")
