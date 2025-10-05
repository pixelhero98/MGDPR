"""Parallel retention layer used within the MGDPR model."""

from __future__ import annotations

import math

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
        """Combine diffused features ``h`` with retained state ``h_prime``.

        Parameters
        ----------
        h:
            Diffused node representations of shape ``[N, in_dim]``.
        D:
            Decay matrix used to scale interactions, shape ``[N, N]``.
        h_prime:
            Previously retained features of shape ``[N, out_dim]``.

        Returns
        -------
        torch.Tensor
            Updated retained features with the same shape as ``h_prime``.
        """

        if h.shape[0] != D.shape[0] or D.shape[0] != D.shape[1]:
            raise ValueError("Decay matrix must be square with side length equal to the number of nodes.")

        if h_prime.shape[0] != h.shape[0]:
            raise ValueError("Retention input must share the same node dimension as h.")

        device = h.device
        D = D.to(device)

        # Compute Q, K, V projections
        Q = self.Q_layers(h)
        K = self.K_layers(h)
        V = self.V_layers(h)

        # Compute pairwise interactions: (N, N)
        inter_feat = (Q @ K.transpose(0, 1)) / math.sqrt(self.inter_dim)

        # Retention operation: (D * inter_feat) @ V -> (N, inter_dim)
        attn = torch.softmax(D * inter_feat, dim=-1)
        h_ret = attn @ V

        # Project and activate retention features
        h_ret = self.ret_feat(h_ret)
        h_ret = self.ret_activation(h_ret)

        # Transform h_prime
        h_cat = self.cat_feat(h_prime)

        # Concatenate retention and prime, project and activate
        combined = torch.cat((h_ret, h_cat), dim=-1)
        x_out = self.ret_proj(combined)
        x_out = self.cat_activation(x_out)

        # Apply GroupNorm (expects shape [B, C, *])
        x_out = x_out.transpose(0, 1).unsqueeze(0)
        x_out = self.group_norm(x_out)
        x_out = x_out.squeeze(0).transpose(0, 1)

        return x_out
