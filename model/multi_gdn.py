import torch
import torch.nn as nn
from paret import ParallelRetention
from mgd import MultiReDiffusion

class MGDPR(nn.Module):
    def __init__(self, diffusion, retention, ret_linear_1, ret_linear_2, post_pro, 
                 layers, num_nodes, time_dim, num_relation, gamma, expansion_steps):
        super(MGDPR, self).__init__()
        
        self.layers = layers
        
        # Learnable parameters for multi-relational transitions:
        # T is used as a transition (or weighting) tensor.
        # Initialized using standard Xavier uniform initialization.
        self.T = nn.Parameter(torch.empty(layers, num_relation, expansion_steps, num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.T)
        
        # Theta is used for weighting coefficients in the diffusion process.
        self.theta = nn.Parameter(torch.empty(layers, num_relation, expansion_steps))
        nn.init.xavier_uniform_(self.theta)
        
        # Create a lower triangular mask. Only positions with lower_tri != 0 (i.e., strictly lower triangular)
        # will be assigned a decay value computed as gamma ** -lower_tri.
        lower_tri = torch.tril(torch.ones(time_dim, time_dim), diagonal=-1)
        D_gamma_tensor = torch.where(lower_tri == 0, torch.tensor(0.0), gamma ** -lower_tri)
        # Register as a buffer so it moves with the model's device and is saved/loaded with state_dict.
        self.register_buffer('D_gamma', D_gamma_tensor)
        
        # Initialize Multi-relational Graph Diffusion layers.
        self.diffusion_layers = nn.ModuleList(
            [MultiReDiffusion(diffusion[i], diffusion[i + 1], num_relation) 
             for i in range(len(diffusion) - 1)]
        )
        
        # Initialize Parallel Retention layers.
        self.retention_layers = nn.ModuleList(
            [ParallelRetention(time_dim, retention[3 * i], retention[3 * i + 1], retention[3 * i + 2]) 
             for i in range(len(retention) // 3)]
        )
        
        # Initialize decoupled transformation layers.
        self.ret_linear_1 = nn.ModuleList(
            [nn.Linear(ret_linear_1[2 * i], ret_linear_1[2 * i + 1]) 
             for i in range(len(ret_linear_1) // 2)]
        )
        self.ret_linear_2 = nn.ModuleList(
            [nn.Linear(ret_linear_2[2 * i], ret_linear_2[2 * i + 1]) 
             for i in range(len(ret_linear_2) // 2)]
        )
        
        # MLP layers for post-processing.
        self.mlp = nn.ModuleList(
            [nn.Linear(post_pro[i], post_pro[i + 1]) for i in range(len(post_pro) - 1)]
        )
    
    
    def forward(self, x, a):
        """
        x: input tensor (e.g., node features); expected shape should be (batch_size, num_nodes, feature_dim)
        a: adjacency (or relation) information for graph diffusion.
        """
        # Use the same device as the input x.
        device = x.device
        
        # Ensure h is on the proper device.
        h = x.to(device)
        
        # Information diffusion and graph representation learning
        for l in range(self.layers):
            # Multi-relational Graph Diffusion layer:
            # The diffusion layer returns updated h and an intermediate representation u.
            h, u = self.diffusion_layers[l](self.theta[l], self.T[l], a, h)
            
            # Ensure u is on the same device as D_gamma.
            u = u.to(device)
            
            # Parallel Retention layer:
            # The retention layer expects u and the decay matrix D_gamma.
            eta = self.retention_layers[l](u, self.D_gamma)
            
            # Decoupled representation transform:
            # For the first layer, combine the eta representation with a transformed version of the
            # original input.
            if l == 0:
                # Reshape x to (num_nodes, -1) so that it aligns with eta.
                x_reshaped = x.view(x.shape[1], -1)
                h_concat = torch.cat((eta, self.ret_linear_1[l](x_reshaped)), dim=1)
                h_prime = self.ret_linear_2[l](h_concat)
            else:
                h_concat = torch.cat((eta, self.ret_linear_1[l](h_prime)), dim=1)
                h_prime = self.ret_linear_2[l](h_concat)
                
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


