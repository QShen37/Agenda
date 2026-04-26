# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualHeadThresholdNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, N):
        super().__init__()

        self.shared_fc = nn.Linear(input_dim, hidden_dim)

        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.gamma_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        self.N = N

    def forward(self, e_t):
        h = F.relu(self.shared_fc(e_t))

        beta = self.beta_head(h)                  # [B, 1]
        gamma = self.gamma_head(h) * self.N  # [B, 1]

        return beta, gamma
