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

        u = torch.zeros(theta.shape[0], a.shape[1], self.output)

        for i in range(theta.shape[0]):
            s = torch.zeros_like(a[i])

            for j in range(theta.shape[-1]):
                s += (theta[i][j] * t[i][j]) * a[i]

            u[i] = self.activation0(self.fc_layers[i](s @ x[i]))

        h = u.unsqueeze(0).to('cuda')
        h = self.activation1(self.update_layer(h))
        h = h.reshape(self.num_relation, a.shape[1], -1)

        return h, u
