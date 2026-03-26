import torch
from torch import nn, Tensor


class ConditionalDriftNetwork(nn.Module):
    def __init__(self, z_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + r_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: Tensor, r: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        if t.dim() == 1:
            t = t.view(-1, 1)
        # Concatenate z, x, and t
        drift: Tensor = self.net(torch.cat([z, r, t], dim=-1))
        return drift


class ConditionalFreeEnergyNetwork(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.expand(r.shape[0], 1)
        if t.dim() == 1:
            t = t.view(-1, 1)
        F: Tensor = self.net(torch.cat([r, t], dim=-1))
        return F


class SetEncoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, r_dim),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # (batch_size, task_size, x_dim)
        # (batch_size, task_size, y_dim)

        # cat
        pairs = torch.cat([x, y], dim=2)
        # (batch_size, task_size, x_dim + y_dim)

        rs = self.encoder(pairs)
        # (batch_size, task_size, r_dim)

        r = torch.mean(rs, dim=1)
        # (batch_size, r_dim)

        return r


class SASetEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        r_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()

        # 1. Point-wise Feature Extractor
        # Maps raw (x,y) pairs to a high-dimensional hidden state
        self.feature_extractor = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 2. Multi-Head Self-Attention (The "Mixing" Layer)
        # We use batch_first=True because our inputs are (Batch, Seq_Len, Dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # LayerNorm is standard to stabilize attention gradients
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Final Projection
        self.out_projector = nn.Linear(hidden_dim, r_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x: (batch_size, n_points, x_dim)
        # y: (batch_size, n_points, y_dim)

        # 1. Concatenate inputs
        pairs = torch.cat([x, y], dim=2)
        # (batch_size, n_points, x_dim + y_dim)

        # 2. Embed points independently
        h = self.feature_extractor(pairs)
        # (batch_size, n_points, hidden_dim)

        # 3. Self-Attention Mixing
        # Q=h, K=h, V=h
        # This allows every point to look at every other point
        attn_out, _ = self.attention(h, h, h)

        # Residual connection + Normalization
        h = self.norm(h + attn_out)
        # (batch_size, n_points, hidden_dim)

        # 4. Project to latent dimension r
        rs = self.out_projector(h)
        # (batch_size, n_points, r_dim)

        # 5. Aggregate
        # Now 'r' is a summary where every point has been contextually weighted
        r = torch.mean(rs, dim=1)
        # (batch_size, r_dim)

        return r


class SetDecoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(r_dim + x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, r: Tensor, x: Tensor) -> Tensor:
        """
        r: (Batch, r_dim) - The global task representation (or z)
        x: (Batch, Num_Points, x_dim) - The query points
        """
        # 1. Unsqueeze r to prepare for broadcasting: (Batch, 1, r_dim)
        r = r.unsqueeze(1)

        # 2. Expand r to match the number of points in x: (Batch, Num_Points, r_dim)
        #    x.size(1) is the number of points (e.g., 10 or 128)
        r_expanded = r.expand(-1, x.size(1), -1)

        # 3. Concatenate along the feature dimension (dim=2)
        #    (Batch, Num_Points, r_dim) cat (Batch, Num_Points, x_dim)
        pairs = torch.cat([r_expanded, x], dim=2)

        # 4. Pass through MLP
        y: Tensor = self.net(pairs)

        return y
