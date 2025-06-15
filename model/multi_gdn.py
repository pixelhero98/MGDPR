import torch
import torch.nn as nn
from paret import ParallelRetention
from mgd import MultiReDiffusion

class MGDPR(nn.Module):
    def __init__(self, diffusion, ret_in_dim, ret_inter_dim, ret_hidden_dim, ret_out_dim, post_pro, 
                 layers, num_nodes, num_relation, expansion_steps, zeta):
        super(MGDPR, self).__init__()
        
        self.layers = layers
        
        # Learnable parameters for multi-relational transitions:
        # T is used as a transition (or weighting) tensor.
        # Initialized using standard Xavier uniform initialization.
        self.T = nn.Parameter(torch.empty(layers, num_relation, expansion_steps, num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.T)
        
        # Theta is used for weighting coefficients in the diffusion process.
        self.gamma = nn.Parameter(torch.empty(layers, num_relation, expansion_steps))
        nn.init.xavier_uniform_(self.theta)
        
        # Create a lower triangular mask. Only positions with lower_tri != 0 (i.e., strictly lower triangular)
        # will be assigned a decay value computed as zeta ** -lower_tri.
        lower_tri = torch.tril(torch.ones(num_nodes, num_nodes), diagonal=-1)
        D = torch.where(lower_tri == 0, torch.tensor(0.0), zeta ** -lower_tri)
        # Register as a buffer so it moves with the model's device and is saved/loaded with state_dict.
        self.register_buffer('D', D)
        
        # Initialize Multi-relational Graph Diffusion layers.
        self.diffusion_layers = nn.ModuleList(
            [MultiReDiffusion(diffusion[i], diffusion[i + 1], num_relation) 
             for i in range(len(diffusion) - 1)]
        )
        
        # Initialize Parallel Retention layers.
        self.retention_layers = nn.ModuleList(
            [ParallelRetention(ret_in_dim[i], ret_inter_dim[i], ret_hidden_dim[i], ret_out_dim[i]) 
             for i in range(len(retention))]
        )
        
        # MLP layers for post-processing.
        self.mlp = nn.ModuleList(
            [nn.Linear(post_pro[i], post_pro[i + 1]) for i in range(len(post_pro) - 1)]
        )

        self.raw_feat = nn.Linear(diffusion[0], ret_out_dim)
    
    def forward(self, x, a):
        """
        x: input tensor (e.g., node features); expected shape should be (batch_size, num_nodes, feature_dim)
        a: adjacency (or relation) information for graph diffusion.
        """

        # Information diffusion and graph representation learning
        for l in range(self.layers):
            # Multi-relational Graph Diffusion layer:
            # The diffusion layer returns updated h and an intermediate representation u.
            h = self.diffusion_layers[l](self.gamma[l], self.T[l], a, h)
            
            # Parallel Retention layer:
            if l == 0:
                h_prime = self.retention_layers[l](h, self.D, self.raw_feat(x))
            else:
                h_prime = self.retention_layers[l](h, self.D, h_prime)
                                                   
        # Post-processing with MLP layers to generate final graph representation.
        for mlp_layer in self.mlp:
            h_prime = mlp_layer(h_prime)
            
        return h_prime
    
    
    def reset_parameters(self):
        """
        Reset learnable model parameters using appropriate initialization methods.
        Note that D_gamma is not learnable and is registered as a buffer.
        """
        # Reinitialize T and theta with Xavier uniform initialization.
        nn.init.xavier_uniform_(self.T)
        nn.init.xavier_uniform_(self.theta)
        
        # Optionally, you could also reset the parameters of the submodules.
        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module not in [self]:
                module.reset_parameters()


