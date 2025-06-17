class MGDPR(nn.Module):
    def __init__(self,
                 num_nodes:         int,
                 diffusion_dims:    list[int],
                 ret_in_dim:        list[int],
                 ret_inter_dim:     list[int],
                 ret_hidden_dim:    list[int],
                 ret_out_dim:       list[int],
                 post_pro:          list[int],
                 num_relation:      int,
                 expansion_steps:   int,
                 zeta:              float):
        super().__init__()

        # Derive number of layers
        assert len(diffusion_dims) == len(ret_in_dim) + 1
        assert len(ret_in_dim) == len(ret_inter_dim) == len(ret_hidden_dim) == len(ret_out_dim)
        self.layers = len(ret_in_dim)
        self.num_nodes = num_nodes
        # Transition tensor T: (layers, R, S, N, N)
        self.T = nn.Parameter(torch.empty(self.layers,
                                           num_relation,
                                           expansion_steps,
                                           num_nodes,
                                           num_nodes))
        # Diffusion weighting gamma: (layers, R, S)
        self.gamma = nn.Parameter(torch.empty(self.layers,
                                              num_relation,
                                              expansion_steps))

        # Initialize transition parameters
        self._init_transition_params()

                # Precompute decay buffer D: D[i,j] = zeta^(i-j) for i>j, else 0
        idx = torch.arange(num_nodes)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        diff = i - j
        D = (zeta ** diff) * (diff > 0)
        self.register_buffer('D', D.float())

        # Diffusion and Retention modules
        self.diffusion_layers = nn.ModuleList(
            MultiReDiffusion(in_dim, out_dim, num_relation)
            for in_dim, out_dim in zip(diffusion_dims[:-1],
                                       diffusion_dims[1:])
        )
        self.retention_layers = nn.ModuleList(
            ParallelRetention(in_dim, i_dim, h_dim, o_dim)
            for in_dim, i_dim, h_dim, o_dim
            in zip(ret_in_dim, ret_inter_dim, ret_hidden_dim, ret_out_dim)
        )

        # Raw feature projection (proj x → first retention input)
        self.raw_feat = nn.Linear(diffusion_dims[0], ret_out_dim[0])

        # Post-processing MLP
        self.mlp = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(post_pro[:-1], post_pro[1:])
        )

    def _init_transition_params(self):
        """
        Initialize transition parameters:
        - T with Xavier uniform
        - gamma with constant normalized values
        """
        nn.init.xavier_uniform_(self.T)
        nn.init.constant_(self.gamma, 1.0 / self.gamma.size(-1))

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        """
        x: (batch, N, F_in)
        a: adjacency or relation tensor
        """
        # Initial graph repr
        x = x.reshape(self.num_nodes, -1)
        h = x

        for idx, (diff, ret) in enumerate(zip(self.diffusion_layers,
                                              self.retention_layers)):
            # MultiReDiffusion might return (h_new, u); adjust if so
            h = diff(self.gamma[idx], self.T[idx], a, h)
            if idx == 0:
                # first retention sees raw_feat(x)
                h_prime = ret(h, self.D, self.raw_feat(x))
            else:
                h_prime = h_prime + ret(h, self.D, h_prime)


        # Post-MLP
        out = h_prime
        for layer in self.mlp:
            out = layer(out)

        return out

    def reset_parameters(self):
        """Reinitialize all parameters."""
        self._init_transition_params()
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
