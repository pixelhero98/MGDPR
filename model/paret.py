import torch
import torch.nn as nn

class ParallelRetention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        inter_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_groups: int = 16
    ):
        super(ParallelRetention, self).__init__()
        self.in_dim = in_dim
        self.inter_dim = inter_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Activations
        self.ret_activation = nn.GELU()
        self.cat_activation = nn.PReLU(num_parameters=self.out_dim)

        # Linear projections
        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.ret_feat = nn.Linear(self.inter_dim, self.hidden_dim)
        self.cat_feat = nn.Linear(self.out_dim, self.hidden_dim)
        self.ret_proj = nn.Linear(2 * self.hidden_dim, self.out_dim)

        # GroupNorm after activation
        # num_groups should divide out_dim
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=self.out_dim)

    def forward(self, h: torch.Tensor, D: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """
        h: Tensor of shape (N, in_dim)
        D: Tensor of shape (N, N) for scaling feature interactions
        h_prime: Tensor of shape (N, out_dim) to concatenate
        returns: Tensor of shape (N, out_dim)
        """
        # Ensure D is on the same device as h
        D = D.to(h.device)

        # Compute Q, K, V projections
        Q = self.Q_layers(h)  # (N, inter_dim)
        K = self.K_layers(h)  # (N, inter_dim)
        V = self.V_layers(h)  # (N, inter_dim)

        # Compute pairwise interactions: (N, N)
        inter_feat = Q @ K.transpose(0, 1)

        # Retention operation: (D * inter_feat) @ V -> (N, inter_dim)
        h_ret = (D * inter_feat) @ V

        # Project and activate retention features
        h_ret = self.ret_feat(h_ret)
        h_ret = self.ret_activation(h_ret)

        # Transform h_prime
        h_cat = self.cat_feat(h_prime)

        # Concatenate retention and prime, project and activate
        combined = torch.cat((h_ret, h_cat), dim=-1)  # (N, 2*hidden_dim)
        x_out = self.ret_proj(combined)
        x_out = self.cat_activation(x_out)

        # Apply GroupNorm
        x_out = self.group_norm(x_out)

        return x_out
