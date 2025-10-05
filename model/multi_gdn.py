"""Implementation of the MGDPR architecture."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from mgd import MultiReDiffusion
from paret import ParallelRetention


class MGDPR(nn.Module):
    """Multi-Graph Diffusion with Parallel Retention (MGDPR).

    Parameters
    ----------
    num_nodes:
        Number of nodes in the graph.
    diffusion_dims:
        Feature dimensions for each diffusion layer (length ``layers + 1``).
    ret_in_dim, ret_inter_dim, ret_hidden_dim, ret_out_dim:
        Per-layer dimensions for the retention blocks.
    post_pro:
        Hidden dimensions for the final post processing MLP.
    num_relation:
        Number of relations in the multi-relational graph.
    expansion_steps:
        Number of diffusion expansion steps per layer.
    zeta:
        Retention decay factor used to build the decay buffer.
    """

    def __init__(
        self,
        num_nodes: int,
        diffusion_dims: Sequence[int],
        ret_in_dim: Sequence[int],
        ret_inter_dim: Sequence[int],
        ret_hidden_dim: Sequence[int],
        ret_out_dim: Sequence[int],
        post_pro: Sequence[int],
        num_relation: int,
        expansion_steps: int,
        zeta: float,
    ) -> None:
        super().__init__()

        if len(diffusion_dims) != len(ret_in_dim) + 1:
            raise ValueError("diffusion_dims must be one element longer than retention dimensions.")
        if not (
            len(ret_in_dim)
            == len(ret_inter_dim)
            == len(ret_hidden_dim)
            == len(ret_out_dim)
        ):
            raise ValueError("Retention dimension lists must all be the same length.")

        self.layers = len(ret_in_dim)
        if self.layers == 0:
            raise ValueError("At least one diffusion/retention layer is required.")

        self.num_nodes = num_nodes

        # Transition tensor T: (layers, R, S, N, N)
        self.T = nn.Parameter(
            torch.empty(self.layers, num_relation, expansion_steps, num_nodes, num_nodes)
        )

        # Diffusion weighting gamma: (layers, R, S)
        self.gamma = nn.Parameter(torch.empty(self.layers, num_relation, expansion_steps))

        # Initialize transition parameters
        self._init_transition_params()

        # Precompute decay buffer D: D[i, j] = zeta^(i-j) for i > j, else 0
        idx = torch.arange(num_nodes)
        i, j = torch.meshgrid(idx, idx, indexing="ij")
        diff = i - j
        decay = torch.where(diff > 0, zeta ** diff, torch.zeros_like(diff, dtype=torch.float32))
        self.register_buffer("D", decay.float())

        # Diffusion and Retention modules
        self.diffusion_layers = nn.ModuleList(
            MultiReDiffusion(in_dim, out_dim, num_relation)
            for in_dim, out_dim in zip(diffusion_dims[:-1], diffusion_dims[1:])
        )
        self.retention_layers = nn.ModuleList(
            ParallelRetention(in_dim, i_dim, h_dim, o_dim)
            for in_dim, i_dim, h_dim, o_dim in zip(
                ret_in_dim, ret_inter_dim, ret_hidden_dim, ret_out_dim
            )
        )

        # Raw feature projection (project x → first retention input)
        self.raw_feat = nn.Linear(diffusion_dims[0], ret_out_dim[0])

        # Post-processing MLP
        self.mlp = nn.ModuleList(
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(post_pro[:-1], post_pro[1:])
        )

    def _init_transition_params(self) -> None:
        """Initialise transition parameters ``T`` and ``gamma``."""

        nn.init.xavier_uniform_(self.T)
        nn.init.constant_(self.gamma, 1.0 / self.gamma.size(-1))

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        """Execute the MGDPR forward pass.

        Parameters
        ----------
        x:
            Input features of shape ``[num_nodes, diffusion_dims[0]]`` or
            ``[batch, num_nodes, diffusion_dims[0]]``.
        a:
            Relation specific adjacency tensor expected by ``MultiReDiffusion`` with
            shape ``[num_relation, num_nodes, num_nodes]``.
        """

        if x.dim() == 3:
            # Collapse batch dimension; downstream operations expect [N, F].
            if x.size(0) != 1:
                raise ValueError("Batch dimension larger than 1 is not supported in this implementation.")
            x = x.squeeze(0)
        elif x.dim() != 2:
            raise ValueError("Input features must be a 2D tensor or a 3D tensor with batch size 1.")

        if x.shape != (self.num_nodes, self.diffusion_layers[0].input_dim):
            raise ValueError(
                "Input features must match the expected shape "
                f"({self.num_nodes}, {self.diffusion_layers[0].input_dim})."
            )

        if a.dim() != 3 or a.shape[1:] != (self.num_nodes, self.num_nodes):
            raise ValueError(
                "Adjacency tensor must have shape (num_relations, num_nodes, num_nodes)."
            )

        if a.shape[0] != self.gamma.size(1):
            raise ValueError(
                "Adjacency tensor must contain the same number of relations as configured "
                f"({self.gamma.size(1)})."
            )

        h: Tensor = x
        residual: Tensor | None = None

        for idx, (diff_module, retention_module) in enumerate(
            zip(self.diffusion_layers, self.retention_layers)
        ):
            h = diff_module(self.gamma[idx], self.T[idx], a, h)
            if idx == 0:
                residual = retention_module(h, self.D, self.raw_feat(x))
            else:
                if residual is None:
                    raise RuntimeError("Residual state was not initialised correctly.")
                residual = residual + retention_module(h, self.D, residual)

        if residual is None:
            raise RuntimeError("Model forward pass did not initialise the residual state.")

        out = residual
        for layer in self.mlp:
            out = layer(out)

        return out

    def reset_parameters(self) -> None:
        """Reinitialise all learnable parameters."""

        self._init_transition_params()
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
