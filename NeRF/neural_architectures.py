import torch

class NeRF(torch.nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 4)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        sigma = self.relu(x[:, -1])
        rgb = self.sigmoid(x[:, :-1])
        return sigma, rgb