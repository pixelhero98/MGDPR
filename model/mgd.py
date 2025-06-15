import torch
import torch.nn as nn
from torch import Tensor

class MultiReDiffusion(nn.Module):
    """
    A multi-relation diffusion module that:
      1. Computes per-relation diffused node features via weighted diffusion matrices.
      2. Applies a relation-specific linear transform.
      3. Mixes information across relations with a 1×1 convolution and aggregates across relations.

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
        self.input_dim = input_dim
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
        gamma: Tensor,    # [R, S] unnormalized per-relation weights over diffusion steps
        T:     Tensor,    # [R, S, N, N] diffusion matrices per step and relation
        a:     Tensor,    # [R, N, N] adjacency matrices per relation
        x:     Tensor     # [N, D_in] node feature matrix
    ) -> Tensor:
        """
        Args:
            gamma: Unnormalized weights over S diffusion steps for each relation ([R, S]).
                   Will be normalized per relation via softmax over steps (dim=1).
            T:     Diffusion step matrices ([R, S, N, N]).
            a:     Adjacency matrices per relation ([R, N, N]).
            x:     Node feature matrix ([N, D_in]), shared across relations.

        Returns:
            aggregated_features: [N, D_out] matrix after mixing and summing over relations.
        """
        R, S = gamma.shape
        N, _ = x.shape
        assert T.shape == (R, S, N, N), f"Expected T.shape==(R,S,N,N)=({R},{S},{N},{N}), got {tuple(T.shape)}"
        assert a.shape == (R, N, N), f"Expected a.shape==(R,N,N)=({R},{N},{N}), got {tuple(a.shape)}"

        x = x.reshape(N, -1)
        # Normalize gamma per relation so that for each r, sum_s gamma[r,s] == 1
        gamma_norm = torch.softmax(gamma, dim=1)  # [R, S]

        # 1) Combine step matrices into per-relation diffusion: [R, N, N]
        weighted_steps = gamma_norm.view(R, S, 1, 1) * T  # [R, S, N, N]
        combined_diff = weighted_steps.sum(dim=1)         # [R, N, N]

        # 2) Apply adjacency mask: [R, N, N]
        diffusion_mats = combined_diff * a

        # 3) Diffuse features: [R, N, D_in]
        diff_feats = torch.einsum("rnm,md->rnd", diffusion_mats, x)

        # 4) Per-relation transform: [R, N, D_out]
        diffused = torch.stack([
            self.fc_layers[r](diff_feats[r])
            for r in range(R)
        ], dim=0)

        # 5) Cross-relation mixing: [1, R, N, D_out]
        mixed = self.relation_mixer(diffused.unsqueeze(0))
        mixed = self.mixer_act(mixed)
        mixed = mixed.reshape(R, N, self.output_dim)  # [R, N, D_out]

        # 6) Aggregate across relations: [N, D_out]
        aggregated_features = mixed.sum(dim=0)

        return aggregated_features
