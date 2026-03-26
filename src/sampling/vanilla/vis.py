from typing import Dict, List
from matplotlib import pyplot as plt
from torch import Tensor


def visualize_progress(
    epoch: int,
    losses: Dict[str, List[float]],
    real_x: Tensor,
    recon_x: Tensor,
    latents: Tensor,
    save_path: str = "nets_moons.png",
) -> None:
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
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
