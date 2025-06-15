import torch
import torch.nn as nn

class ParallelRetention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        inter_dim: int,
        out_dim: int,
        num_groups: int = 16
    ):
        super(ParallelRetention, self).__init__()
        self.in_dim = in_dim
        self.inter_dim = inter_dim
        self.hidden_dim = out_dim
        self.out_dim = out_dim

        # Activation
        self.activation = nn.GELU()

        # Linear projections
        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.ret_feat = nn.Linear(self.inter_dim, self.hidden_dim)
        self.ret_proj = nn.Linear(self.hidden_dim, self.out_dim)

        # GroupNorm after activation
        # num_groups should divide out_dim
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=self.out_dim)

    def forward(self, x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (N, in_dim)
        D: Tensor of shape (N, N) for scaling feature interactions
        returns: Tensor of shape (N, out_dim)
        """
        # Ensure D is on the same device
        D = D.to(x.device)

        # Projections
        Q = self.Q_layers(x)  # (N, inter_dim)
        K = self.K_layers(x)  # (N, inter_dim)
        V = self.V_layers(x)  # (N, inter_dim)

        # Compute pairwise interactions: (N, N)
        inter_feat = Q @ K.transpose(0, 1)

        # Retention operation: (D * inter_feat) @ V -> (N, inter_dim)
        x_ret = (D * inter_feat) @ V

        # Project to output dim and activate
        x_out = self.ret_feat(x_ret)
        x_out = self.activation(x_out)
        x_out = self.ret_proj(x_out)
        # Apply GroupNorm
        x_out = self.group_norm(x_out)

        return x_out
