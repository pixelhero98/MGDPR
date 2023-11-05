import torch
import torch.nn as nn

class MultiReDiffusion(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_relation):
        super(MultiReDiffusion, self).__init()
        self.output = output_dim
        self.fc_layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relation)])
        self.update_layer = torch.nn.Conv2d(num_relation, 1, kernel_size=1)
        self.activation1 = torch.nn.GELU()
        self.activation0 = torch.nn.PReLU()

    def forward(self, theta, t, a, x):

        u = torch.zeros_like(theta.shape[0], a.shape[1], self.output)

        for i in range(theta.shape[0]):
            s = torch.zeros_like(t)
            for j in range(theta.shape[-1]):
                s += (theta[i][j] * t[i][j]) * a[i]
            u[i] = self.activation0(self.fc_layers[i](s @ x))

        h = u.unsqueeze(0)
        h = self.activation1(self.update_layer(h))
        h = h.squeeze(0)

        return h, u
