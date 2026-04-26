import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, heads=4):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=0.5
        )

        self.gat_mu = GATConv(
            hidden_channels * heads,
            latent_dim,
            heads=1,
            concat=False
        )

        self.gat_logstd = GATConv(
            hidden_channels * heads,
            latent_dim,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))

        mu = self.gat_mu(x, edge_index)
        logstd = self.gat_logstd(x, edge_index)

        return mu, logstd


class VGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()

        self.encoder = GATEncoder(
            in_channels,
            hidden_channels,
            latent_dim
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 3, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

    def decode(self, z, task_idx):
        num_nodes = z.size(0) - 1
        zt = z[task_idx]

        probs = []

        for i in range(num_nodes):
            row = []
            for j in range(num_nodes):
                edge_input = torch.cat([z[i], z[j], zt], dim=0)
                prob = torch.sigmoid(self.decoder(edge_input))
                row.append(prob)
            probs.append(torch.stack(row))

        return torch.stack(probs).squeeze(-1)

    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(
            torch.sum(
                1 + 2 * logstd - mu ** 2 - torch.exp(2 * logstd),
                dim=1
            )
        )