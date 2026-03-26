from typing import Optional

import torch
from torch import Tensor, nn


class ConditionalDriftNetwork(nn.Module):
    def __init__(self, z_dim: int, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Input: z (latent), r (context), t (time)
        self.net = nn.Sequential(
            nn.Linear(z_dim + r_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        return self.net(torch.cat([z, r, t], dim=-1))  # type: ignore


class ConditionalFreeEnergyNetwork(nn.Module):
    def __init__(self, r_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(r_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        return self.net(torch.cat([r, t], dim=-1))  # type: ignore


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


class SAMaskedSetEncoder(nn.Module):
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
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        self.mlp = nn.Sequential(
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


class AttentionSetEncoder(nn.Module):
    """
    State-of-the-Art Set Encoder (Set Transformer style).

    Structure:
    1. Feature Extractor (x,y -> h)
    2. Self-Attention Block (Points contextualize each other)
    3. Pooling Block (Latent Query attends to points to summarize)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        r_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- 1. Feature Extractor ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),  # Norm helps initial stability
        )

        # --- 2. Self-Attention Block (SAB) ---
        # "Points talking to points"
        # self.sa_norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        # self.sa_norm2 = nn.LayerNorm(hidden_dim)
        self.sa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            # nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        # --- 3. Pooling Block (PMA) ---
        # "Latent Query summarizing the set"
        self.latent_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # self.pma_norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.pma_norm2 = nn.LayerNorm(hidden_dim)
        self.pma_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        # --- 4. Final Projection ---
        self.out_projector = nn.Linear(hidden_dim, r_dim)

    def forward(self, x: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (Batch, N, x_dim)
            y: (Batch, N, y_dim)
            mask: (Batch, N, 1) - 1.0 for valid, 0.0 for padding.
        """
        batch_size = x.shape[0]

        # Prepare Mask for MultiheadAttention
        # PyTorch expects: True = Ignore (Padding), False = Keep
        # Input mask is 1.0 (Keep), 0.0 (Ignore).
        key_padding_mask: Optional[Tensor] = None
        if mask is not None:
            # (Batch, N, 1) -> (Batch, N) -> bool
            key_padding_mask = mask.squeeze(-1) == 0

        # ---------------------------------------------------------
        # 1. Feature Extraction
        # ---------------------------------------------------------
        pairs = torch.cat([x, y], dim=2)
        h = self.feature_extractor(pairs)  # (Batch, N, Hidden)

        # ---------------------------------------------------------
        # 2. Self-Attention Block (SAB)
        # ---------------------------------------------------------
        # Residual connection + Pre-Norm architecture
        h_norm = self.sa_norm1(h)

        # Note: We pass key_padding_mask so points don't attend to padding
        attn_out, _ = self.self_attention(
            query=h_norm, key=h_norm, value=h_norm, key_padding_mask=key_padding_mask
        )
        h = h + attn_out

        # FFN Part
        h_norm = self.sa_norm2(h)
        ffn_out = self.sa_ffn(h_norm)
        h = h + ffn_out  # (Batch, N, Hidden)

        # ---------------------------------------------------------
        # 3. Pooling Block (PMA)
        # ---------------------------------------------------------
        # Expand Seed Query
        q = self.latent_query.expand(batch_size, -1, -1)  # (Batch, 1, Hidden)

        # Cross-Attention: Query looks at Context h
        q_norm = self.pma_norm1(q)
        h_norm = h  # Usually we don't re-norm keys/values in cross attn, but keys are already processed.

        pool_out, _ = self.cross_attention(
            query=q_norm, key=h_norm, value=h_norm, key_padding_mask=key_padding_mask
        )
        q = q + pool_out

        # FFN Part
        q_norm = self.pma_norm2(q)
        ffn_out = self.pma_ffn(q_norm)
        q = q + ffn_out  # (Batch, 1, Hidden)

        # ---------------------------------------------------------
        # 4. Output
        # ---------------------------------------------------------
        r: Tensor = self.out_projector(q).squeeze(1)  # (Batch, r_dim)

        return r


class SetDecoder(nn.Module):
    """
    Decodes target y given latent z and target x.
    Broadcasting z to match x's set size.
    """

    def __init__(self, x_dim: int, y_dim: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        # z: (Batch, z_dim)
        # x: (Batch, N, x_dim)

        # 1. Expand z: (Batch, 1, z_dim) -> (Batch, N, z_dim)
        z_exp = z.unsqueeze(1).expand(-1, x.size(1), -1)

        # 2. Cat and Solve
        pairs = torch.cat([z_exp, x], dim=2)
        # (Batch, N, z_dim + x_dim)

        y: Tensor = self.net(pairs)
        return y
