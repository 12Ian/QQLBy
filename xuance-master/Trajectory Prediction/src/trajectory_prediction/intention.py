"""Paper-style intention recognition: target BiLSTM plus defender-state branch."""

import torch
from torch import nn


class IntentionBiLSTM(nn.Module):
    def __init__(self, hidden_size: int = 128, projection_size: int = 64,
                 dropout: float = 0.2, classes: int = 9):
        super().__init__()
        self.sequence = nn.LSTM(6, hidden_size, batch_first=True,
                                bidirectional=True)
        self.sequence_norm = nn.LayerNorm(2 * hidden_size)
        self.sequence_projection = nn.Linear(2 * hidden_size, projection_size)
        self.defender_projection = nn.Linear(6, projection_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * projection_size, projection_size),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(projection_size, classes),
        )

    def forward(self, target_history: torch.Tensor,
                defender_state: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.sequence(target_history)
        temporal = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        temporal = torch.relu(self.sequence_projection(
            self.sequence_norm(temporal)))
        static = torch.relu(self.defender_projection(defender_state))
        return self.classifier(torch.cat((self.dropout(temporal), static), dim=-1))
