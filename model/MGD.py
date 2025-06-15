import torch
import torch.nn as nn
from torch import Tensor

class MultiReDiffusion(nn.Module):
    """
    A multi-relation diffusion module that:
      1. Computes per-relation diffused node features via weighted adjacency.
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
        T:     Tensor,    # [R, S] time-dependent scalars
        a:     Tensor,    # [R, N, N] adjacency matrices
        x:     Tensor     # [N, D_in] node feature matrix
    ) -> Tensor:
        """
        Args:
            gamma: Unnormalized relation weights over S diffusion steps ([R, S]).
                   Will be normalized per relation via softmax over steps (dim=1).
            T:     Time-dependent scalars ([R, S]).
            a:     Adjacency matrices ([R, N, N]).
            x:     Node feature matrix ([N, D_in]), shared across relations.

        Returns:
            aggregated_features: [N, D_out] matrix after mixing and summing over relations.
        """
        R, S = gamma.shape
        N, D_in = x.shape
        assert T.shape == (R, S), f"Expected T.shape==(R,S)={(R,S)}, got {tuple(T.shape)}"
        assert a.shape == (R, N, N), f"Expected a.shape==(R,N,N)=({R},{N},{N}), got {tuple(a.shape)}"

        # Normalize gamma per relation so that for each relation r, sum_s gamma[r,s] == 1
        gamma = torch.softmax(gamma, dim=1)

        # 1) Compute weighted adjacency per relation: [R, N, N]
        weights = (gamma * T).unsqueeze(-1).unsqueeze(-1)
        diffusion_mats = (weights * a.unsqueeze(1)).sum(dim=1)

        # 2) Diffuse features: [R, N, D_in]
        # x is [N, D_in], diffusion_mats is [R, N, N]
        diff_feats = torch.einsum("rnm,md->rnd", diffusion_mats, x)

        # 3) Per-relation transform: [R, N, D_out]
        diffused = torch.stack([
            self.fc_layers[r](diff_feats[r])
            for r in range(R)
        ], dim=0)

        # 4) Cross-relation mixing: [1, R, N, D_out]
        mixed = self.relation_mixer(diffused.unsqueeze(0))
        mixed = self.mixer_act(mixed)
        # Reshape to [R, N, D_out]
        mixed = mixed.reshape(R, N, self.output_dim)

        # 5) Aggregate across relations: [N, D_out]
        aggregated_features = mixed.sum(dim=0)

        return aggregated_features

