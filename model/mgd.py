"""Multi-relation diffusion module used throughout the project."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MultiReDiffusion(nn.Module):
    """Perform diffusion over multiple relations and aggregate the results.

    The module supports a small pipeline that is repeated for every relation:

    1. Aggregate diffusion matrices across expansion steps using a learned
       weighting ``gamma``.
    2. Apply the resulting diffusion matrix to the node features.
    3. Project the diffused features with a relation specific linear layer.
    4. Mix the relation specific representations with a ``1×1`` convolution and
       sum them to obtain the final features.
    """

    def __init__(self, input_dim: int, output_dim: int, num_relations: int) -> None:
        super().__init__()
        if num_relations <= 0:
            raise ValueError("`num_relations` must be positive.")

        self.num_relations = num_relations
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Per-relation linear transformations.
        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_relations)]
        )

        # 1×1 convolution to mix across relations (acting on channel dimension).
        self.relation_mixer = nn.Conv2d(
            in_channels=num_relations,
            out_channels=num_relations,
            kernel_size=1,
        )

        # Activation after mixing across relations.
        self.mixer_act = nn.PReLU(num_parameters=num_relations)

    def forward(self, gamma: Tensor, T: Tensor, a: Tensor, x: Tensor) -> Tensor:
        """Run the diffusion pipeline."""

        if x.dim() != 2:
            raise ValueError("`x` must be a 2D tensor of shape [N, input_dim].")

        R, S = gamma.shape
        N, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected node features with dimension {self.input_dim}, "
                f"received {feature_dim}."
            )

        expected_T_shape: Tuple[int, int, int, int] = (R, S, N, N)
        expected_a_shape: Tuple[int, int, int] = (R, N, N)
        if T.shape != expected_T_shape:
            raise ValueError(
                "Expected T to have shape (R, S, N, N) but received "
                f"{tuple(T.shape)}."
            )
        if a.shape != expected_a_shape:
            raise ValueError(
                "Expected adjacency tensor to have shape (R, N, N) but "
                f"received {tuple(a.shape)}."
            )

        # Normalize gamma per relation so that for each relation the weights sum to 1.
        gamma_norm = torch.softmax(gamma, dim=1)

        # 1) Combine step matrices into per-relation diffusion matrices.
        weighted_steps = gamma_norm.view(R, S, 1, 1) * T
        combined_diff = weighted_steps.sum(dim=1)

        # 2) Apply adjacency masks to respect graph structure.
        diffusion_mats = combined_diff * a

        # 3) Diffuse features for each relation.
        diff_feats = torch.matmul(diffusion_mats, x)

        # 4) Per-relation transformation.
        diffused = torch.stack(
            [layer(diff_feats[r]) for r, layer in enumerate(self.fc_layers)],
            dim=0,
        )

        # 5) Cross-relation mixing and activation.
        mixed = self.relation_mixer(diffused.unsqueeze(0))
        mixed = self.mixer_act(mixed)
        mixed = mixed.view(R, N, self.output_dim)

        # 6) Aggregate across relations.
        aggregated_features = mixed.sum(dim=0)
        return aggregated_features
