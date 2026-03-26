import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        # t: (Batch, 1) or (Batch,)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device)
            / half_dim
        )
        args = t * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, c_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        # MLP to produce scale and shift from conditioning
        self.scale_shift_net = nn.Linear(c_dim, num_features * 2)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # x: (Batch, Features)
        # c: (Batch, c_dim)

        # Normalize
        x_norm: Tensor = self.bn(x)

        # Compute adaptive scale and shift
        scale_shift = self.scale_shift_net(c)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)

        # Apply adaptive transformation
        x_norm = x_norm * (1 + scale) + shift

        return x_norm


class AdaptiveConditionalDriftNetwork(nn.Module):
    def __init__(self, z_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Sinusoidal time embedding
        self.time_embed = SinusoidalEmbedding(r_dim)

        # MLP to compute conditioning vector c from [t_embed, r]
        self.c_net = nn.Sequential(
            nn.Linear(r_dim + r_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network with adaptive batch norm
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.abn1 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
        self.act1 = nn.SiLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.abn2 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
        self.act2 = nn.SiLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.abn3 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
        self.act3 = nn.SiLU()

        self.fc_out = nn.Linear(hidden_dim, z_dim)

    def forward(self, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        # Handle time tensor dimensions
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        if t.dim() == 1:
            t = t.view(-1, 1)

        # Compute sinusoidal time embedding
        t_embed = self.time_embed(t)  # (Batch, t_embed_dim)

        # Compute conditioning vector c from [t_embed, r]
        c = self.c_net(torch.cat([t_embed, r], dim=-1))  # (Batch, hidden_dim)

        # Main network with adaptive batch norm conditioning
        h = self.fc1(z)
        h = self.abn1(h, c)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.abn2(h, c)
        h = self.act2(h)

        h = self.fc3(h)
        h = self.abn3(h, c)
        h = self.act3(h)

        z = self.fc_out(h)

        return z


class AdaptiveConditionalFreeEnergyNetwork(nn.Module):
    def __init__(self, r_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Sinusoidal time embedding
        self.time_embed = SinusoidalEmbedding(r_dim)

        # MLP to compute conditioning vector c from t_embed only
        self.c_net = nn.Sequential(
            nn.Linear(r_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network with adaptive batch norm, r is the input
        self.fc1 = nn.Linear(r_dim, hidden_dim)
        self.abn1 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
        self.act1 = nn.SiLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.abn2 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
        self.act2 = nn.SiLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.abn3 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
        self.act3 = nn.SiLU()

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, r: Tensor, t: Tensor) -> Tensor:
        # Handle time tensor dimensions
        if t.dim() == 0:
            t = t.expand(r.shape[0], 1)
        if t.dim() == 1:
            t = t.view(-1, 1)

        # Compute sinusoidal time embedding
        t_embed = self.time_embed(t)  # (Batch, t_embed_dim)

        # Compute conditioning vector c from t_embed only
        c = self.c_net(t_embed)  # (Batch, hidden_dim)

        # Main network with adaptive batch norm conditioning
        # r is the input
        h = self.fc1(r)
        h = self.abn1(h, c)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.abn2(h, c)
        h = self.act2(h)

        h = self.fc3(h)
        h = self.abn3(h, c)
        h = self.act3(h)

        r = self.fc_out(h)

        return r


class MaskedSetEncoder(nn.Module):
    """
    Encodes a set of (x,y) pairs into a fixed representation r.
    Handles variable set sizes via masking.
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_projector = nn.Linear(hidden_dim, r_dim)

    def forward(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x, y: (Batch, N, Dim)
        # mask: (Batch, N, 1) - 1.0 for valid, 0.0 for padding

        pairs = torch.cat([x, y], dim=2)  # (Batch, N, x_dim + y_dim)
        h = self.feature_extractor(pairs)
        rs = self.out_projector(h)  # (Batch, N, r_dim)

        if mask is not None:
            # Zero out padded items so they don't affect the sum
            rs = rs * mask

            # Sum valid items
            sum_rs = torch.sum(rs, dim=1)

            # Count valid items (avoid div by zero)
            count = torch.sum(mask, dim=1)
            count = torch.clamp(count, min=1.0)

            r = sum_rs / count
        else:
            r = torch.mean(rs, dim=1)

        return r


class ConvSetEncoder(nn.Module):
    """
    Encodes a set of (x,y) pairs by projecting them onto a 1D grid
    and processing the resulting signal with a CNN.
    """

    def __init__(
        self,
        x_dim: int,  # Must be 1 for this 1D grid implementation
        y_dim: int,
        r_dim: int,
        points_per_unit: int = 64,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        assert x_dim == 1, "ConvSetEncoder currently only supports 1D x-inputs."

        self.y_dim = y_dim
        self.grid_range = grid_range
        self.num_grid_points = int((grid_range[1] - grid_range[0]) * points_per_unit)

        # Learnable RBF Kernel width (sigma)
        # We store log_sigma to ensure sigma is always positive
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        # CNN Feature Extractor
        # Input channels = y_dim (signal) + 1 (density)
        in_channels = y_dim + 1

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, r_dim, kernel_size=5, padding=2),
            # Output is (Batch, r_dim, grid_size)
        )

        # Final projector to get a single vector 'r'
        # We simply pool the grid features into one vector
        self.final_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (Batch, N, 1)
        y: (Batch, N, y_dim)
        mask: (Batch, N, 1)
        """
        # 1. CREATE THE GRID (Functional Space)
        # Create a grid of query points [min, max]
        # shape: (1, 1, grid_size)
        grid = torch.linspace(
            self.grid_range[0], self.grid_range[1], self.num_grid_points
        )
        grid = grid.view(1, 1, -1).to(x.device)

        # 2. COMPUTE RBF KERNEL WEIGHTS
        # x shape: (Batch, N, 1) -> (Batch, N, 1)
        # grid shape: (1, 1, G)
        # dists: (Batch, N, G)
        dists = x - grid

        sigma = torch.exp(self.log_sigma)
        # Gaussian Kernel: exp( -0.5 * (dist/sigma)^2 )
        weights = torch.exp(-0.5 * (dists / sigma) ** 2)

        # Apply mask if provided (zero out invalid points)
        if mask is not None:
            weights = (
                weights * mask
            )  # mask is (Batch, N, 1), broadcasts to (Batch, N, G)

        # 3. PROJECT TO GRID (Summation)
        # We need two channels:
        # A. Density Channel: Where is the data? (Sum of weights)
        # B. Signal Channel: What is the value? (Sum of weights * y)

        # Density: Sum over N (dim 1) -> (Batch, 1, G)
        density = weights.sum(dim=1, keepdim=True)

        # Signal: Weighted sum of y
        # weights: (B, N, G), y: (B, N, y_dim)
        # We need to broadcast y to (B, N, 1, y_dim) to mult with weights?
        # Easier approach: einsum or batched matmul.
        # weights: (B, N, G), y: (B, N, Dy) -> result (B, Dy, G)
        signal = torch.einsum("bng,bnd->bdg", weights, y)

        # Normalize signal by density (optional, but standard in Neural Processes)
        # Adding a small epsilon to avoid div by zero in empty regions
        signal = signal / (density + 1e-5)

        # Concatenate Density and Signal -> (Batch, 1+y_dim, G)
        grid_input = torch.cat([density, signal], dim=1)

        # 4. CNN PROCESSING
        # Input: (Batch, Channels, GridSize)
        features = self.cnn(grid_input)

        # 5. POOLING (Get fixed r)
        # Convert (Batch, r_dim, GridSize) -> (Batch, r_dim, 1)
        r: Tensor = self.final_pool(features).squeeze(-1)

        return r


# class AdaptiveSetDecoder(nn.Module):
#     """
#     Decodes target y given latent z and target x.
#     Uses z for conditioning with adaptive batch normalization.
#     """

#     def __init__(self, x_dim: int, y_dim: int, z_dim: int, hidden_dim: int = 128):
#         super().__init__()

#         # MLP to compute conditioning vector c from z
#         self.c_net = nn.Sequential(
#             nn.Linear(z_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         # Main network with adaptive batch norm, x is the input
#         self.fc1 = nn.Linear(x_dim, hidden_dim)
#         self.abn1 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
#         self.act1 = nn.SiLU()

#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.abn2 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
#         self.act2 = nn.SiLU()

#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.abn3 = AdaptiveBatchNorm1d(hidden_dim, hidden_dim)
#         self.act3 = nn.SiLU()

#         self.fc_out = nn.Linear(hidden_dim, y_dim)

#     def forward(self, z: Tensor, x: Tensor) -> Tensor:
#         # z: (Batch, z_dim)
#         # x: (Batch, N, x_dim)

#         batch_size, n_points, x_dim = x.shape

#         # Compute conditioning vector c from z
#         c = self.c_net(z)  # (Batch, hidden_dim)

#         # Expand c to match set size: (Batch, 1, hidden_dim) -> (Batch, N, hidden_dim)
#         c_exp = c.unsqueeze(1).expand(-1, n_points, -1)

#         # Reshape for processing: (Batch, N, dim) -> (Batch * N, dim)
#         x_flat = x.reshape(batch_size * n_points, x_dim)
#         c_flat = c_exp.reshape(batch_size * n_points, -1)

#         # Main network with adaptive batch norm conditioning
#         h = self.fc1(x_flat)
#         h = self.abn1(h, c_flat)
#         h = self.act1(h)

#         h = self.fc2(h)
#         h = self.abn2(h, c_flat)
#         h = self.act2(h)

#         h = self.fc3(h)
#         h = self.abn3(h, c_flat)
#         h = self.act3(h)

#         y_flat = self.fc_out(h)

#         # Reshape back: (Batch * N, y_dim) -> (Batch, N, y_dim)
#         y: Tensor = y_flat.reshape(batch_size, n_points, -1)

#         return y
