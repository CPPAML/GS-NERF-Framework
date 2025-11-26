import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, hidden_layers: int = 8, skip_connect=[4]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.skip_connect = skip_connect
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for i in range(1, hidden_layers):
            if i in skip_connect:
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, 4)
        torch.nn.init.constant_(self.output_layer.bias[3], -0.25)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, views, num_rays, samples_per_ray):
        input_x = x
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connect:
                h = torch.cat([h, input_x], dim=-1)
            h = self.relu(layer(h))
            
        out = self.output_layer(h)
        
        sigma = self.softplus(out[:, -1])
        rgb = self.sigmoid(out[:, :-1])
        
        return sigma.view(views, num_rays, samples_per_ray, 1), rgb.view(views, num_rays, samples_per_ray, 3)
