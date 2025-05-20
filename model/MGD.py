import torch
import torch.nn as nn

class MultiReDiffusion(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_relation):
        super(MultiReDiffusion, self).__init__()
        self.output = output_dim
        self.fc_layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relation)])
        self.update_layer = torch.nn.Conv2d(num_relation, num_relation, kernel_size=1)
        self.activation1 = torch.nn.PReLU()
        self.activation0 = torch.nn.PReLU()
        self.num_relation = num_relation

    def forward(self, theta, t, a, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusions = torch.zeros(theta.shape[0], a.shape[1], self.output).to(device)

        # Normalize diffusion coefficients so that they sum to one per relation
        theta = torch.softmax(theta, dim=-1)

        for rel in range(theta.shape[0]):
            diffusion_mat = torch.zeros_like(a[rel])
            for step in range(theta.shape[-1]):
                # Normalize transition matrix to be column stochastic
                t_norm = torch.softmax(t[rel][step], dim=0)
                diffusion_mat += theta[rel][step] * t_norm * a[rel]
            
            diffusion_feat = torch.matmul(diffusion_mat, x[rel])
            diffusions[rel] = self.activation0(self.fc_layers[rel](diffusion_feat))

        latent_feat = self.activation1(self.update_layer(diffusions.unsqueeze(0)))
        latent_feat = latent_feat.reshape(self.num_relation, a.shape[1], -1)

        return latent_feat, diffusions
