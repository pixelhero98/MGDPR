import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class MultiReDiffusion(nn.Module):
    """
    A multi-relation diffusion module that:
      1. Computes per-relation diffused node features via weighted adjacency.
      2. Applies a relation-specific linear transform.
      3. Mixes information across relations with a 1×1 convolution.

    Args:
        input_dim: Dimensionality of input node features D_in.
        output_dim: Dimensionality of output features D_out.
        num_relations: Number of distinct relations R.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int
    ):
        super().__init__()
        self.num_relations = num_relations
        self.output_dim = output_dim

        # Per-relation linear transformations
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in range(num_relations)
        ])

        # 1×1 conv to mix across relations (acting on channel dimension)
        self.relation_mixer = nn.Conv2d(
            in_channels=num_relations,
            out_channels=num_relations,
            kernel_size=1
        )

        # Activation after mixing
        self.mixer_act = nn.PReLU()

    def forward(
        self,
        theta: Tensor,    # [R, S]
        t:     Tensor,    # [R, S]
        a:     Tensor,    # [R, N, N]
        x:     Tensor     # [R, N, D_in]
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            mixed_features: [R, N, D_out]   # after cross-relation mixing
            diffused_feats: [R, N, D_out]   # per-relation diffused outputs
        """
        R, S = theta.shape
        # Dimension checks
        assert t.shape == (R, S), f"Expected t.shape==(R,S)={(R,S)}, got {tuple(t.shape)}"
        assert a.shape[0] == R, f"Expected a first dim R={R}, got {a.shape[0]}"
        assert x.shape[0] == R, f"Expected x first dim R={R}, got {x.shape[0]}"

        device = theta.device

        # --- 1) Compute weighted adjacency per relation ---
        # weights: [R, S, 1, 1]
        weights = (theta * t).unsqueeze(-1).unsqueeze(-1)
        # diffusion_mats: [R, N, N]
        diffusion_mats = (weights * a.unsqueeze(1)).sum(dim=1)

        # --- 2) Diffuse input features ---
        # diff_feats: [R, N, D_in]
        diff_feats = torch.einsum("rnm,rmd->rnd", diffusion_mats, x)

        # Apply per-relation linear only: [R, N, D_out]
        diffused_feats = torch.stack([
            self.fc_layers[r](diff_feats[r])
            for r in range(R)
        ], dim=0)

        # --- 3) Cross-relation mixing via 1×1 conv ---
        # Expand to [1, R, N, D_out]
        expanded = diffused_feats.unsqueeze(0)
        mixed = self.relation_mixer(expanded)
        mixed = self.mixer_act(mixed)
        # Reshape back to [R, N, D_out]
        mixed_features = mixed.reshape(R, diffused_feats.shape[1], self.output_dim)

        return mixed_features, diffused_feats
