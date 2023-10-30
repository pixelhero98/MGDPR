import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDPR(nn.Module):

  def __init__(self, relation_size, inter_layer_size, node_feature_size, readout_size, retention_size, layers,
                 num_nodes, num_relation, time_steps, expansion_steps):
    super(MGDPR, self).__init__()


    # Initialize diffusion matrices for all layers, and causal masking and exponential decay matrix
    self.Q = nn.Parameter(torch.randn(layers, num_relation, expansion_steps, num_nodes, num_nodes))
    self.theta = nn.Parameter(torch.randn(layers, num_relation, expansion_steps))
    self.D_gamma = nn.Parameter(torch.randn(time_steps, time_steps))
    self.K = expansion_steps

    # Initialize different module layers at all levels
    self.diffusion_layers = nn.ModuleList(
      [nn.Linear(relation_size[i], relation_size[i + 1]) for i in range(len(relation_size) - 1)])
    self.inter_layers = nn.ModuleList(
      [nn.Linear(inter_layer_size[2 * i], inter_layer_size[2 * i + 1]) for i in range(len(inter_layer_size) // 2)])
    self.node_feature_layers = nn.ModuleList(
      [nn.Linear(node_feature_size[2 * i], node_feature_size[2 * i + 1]) for i in range(len(node_feature_size) // 2)])
    self.MLP = nn.ModuleList(
      [nn.Linear(readout_size[i], readout_size[i + 1]) for i in range(len(readout_size) - 1)])
    self.retention_layers = nn.ModuleList([nn.Linear(retention_size[2 * i], retention_size[2 * i + 1]) for i in range(len(retention_size) // 2)])

    # Initialize activations
    self.activation1 = nn.PReLU()
    self.activation2 = nn.ReLU()

  # x[num_relation, num_node, time_steps], a[num_relation, num_node, num_node]
  def forward(self, x, a):

    # Retention produce the learned input tensor x[num_node, hidden_dim]
    time_length = x.shape[2]
    x = x.permute(2, 1, 0).contiguous().view(time_length, -1)
    x0 = self.retention_layers[0](x) @ self.retention_layers[1](x).transpose(0, 1)
    x = (self.D_gamma * x0) @ self.retention_layers[2](x)
    x = self.activation2(x)

    # Initialize latent representation with retention feature matrix
    z = x.view(1026, -1)

    # Information diffusion and graph feature learning
    for q in range(self.Q.shape[0]):

      # Information diffusion
      z_sum = torch.zeros_like(z)
      box = torch.zeros((a.shape[0], z.shape[0], z.shape[1])).to(device)

    # Diffusion with different relations
      for i in range(a.shape[0]):
        temp_box = torch.zeros_like(z)
        for j in range(self.K)
        temp_box += (self.theta[q][i][j] * self.Q[q][i][j] * a[i]) @ z
        # Copy current outputs for graph feature transform
        box[i] =  temp_box
        z_sum += temp_box

      # Information propagation transform
      z = self.activation1(self.diffusion_layers[q](z_sum))

      # Graph feature transform
      if q != 0:
        s = self.node_feature_layers[q](box.view(box.shape[1], -1))
        s = self.activation2(s)
        f = self.activation2(self.inter_layers[q](torch.cat((f, s), dim=1)))
      else:
        s = self.node_feature_layers[q](box.view(box.shape[1], -1))
        s = self.activation2(s)
        f = self.activation2(self.inter_layers[q](s))

    # Readout process to generate final graph representation
    for mlp in self.MLP:
      f = mlp(f)

      if mlp is not self.MLP[-1]:
        f = self.activation2(f)

    return f

  def reset_parameters(self):
    # Initialize the model parameters with corresponding methods

    nn.init.normal_(self.Q)
    nn.init.normal_(self.D_gamma)
    nn.init.normal_(self.theta)

    for layer in self.diffusion_layers:
      nn.init.kaiming_uniform_(layer.weight)
    for layer in self.inter_layers:
      nn.init.kaiming_uniform_(layer.weight)
    for layer in self.node_feature_layers:
      nn.init.kaiming_uniform_(layer.weight)
    for layer in self.retention_layers:
      nn.init.kaiming_uniform_(layer.weight)
    for layer in self.MLP:
      if layer is self.MLP[-1]:
        nn.init.xavier_uniform_(layer.weight)
      else:
        nn.init.kaiming_uniform_(layer.weight)






