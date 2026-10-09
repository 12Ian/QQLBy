"""Masked encounter-plane attention representations for CRL-G MADDPG."""

import torch
from torch import nn


class _MaskedAttention(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.width = width

    def forward(self, query, tokens, mask):
        scores = torch.sum(self.query(query).unsqueeze(1) *
                           self.key(tokens), dim=-1) / self.width ** 0.5
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(weights.unsqueeze(-1) * self.value(tokens), dim=1)


class CRLGActorRepresentation(nn.Module):
    """Separate attention over two near and three far projected teammates."""

    def __init__(self, input_shape, hidden_sizes=None, **kwargs):
        super().__init__()
        if input_shape[0] != 19:
            raise ValueError(f"CRLG actor requires 19 observations, got {input_shape}")
        self.output_shapes = {"state": (96,)}
        self.own = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))
        self.peer = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32))
        self.near = _MaskedAttention(32)
        self.far = _MaskedAttention(32)
        self.device = torch.device(kwargs.get("device") or "cpu")
        self.to(self.device)

    def forward(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32,
                                       device=self.device)
        shape = observations.shape[:-1]
        obs = observations.reshape(-1, 19)
        own = self.own(obs[:, :3])
        peers = self.peer(obs[:, 3:13].reshape(-1, 5, 2))
        mask = obs[:, 13:18] > 0.5
        near = self.near(own, peers[:, :2], mask[:, :2])
        far = self.far(own, peers[:, 2:], mask[:, 2:])
        result = torch.cat((own, near, far), dim=-1)
        result = result * obs[:, 18:19]
        return {"state": result.reshape(*shape, 96)}


class CRLGCriticRepresentation(nn.Module):
    """Centralized masked attention over six observation-action tokens."""

    def __init__(self, input_shape, hidden_sizes=None, **kwargs):
        super().__init__()
        if input_shape[0] != 126:
            raise ValueError(f"CRLG critic requires 126 joint features, got {input_shape}")
        self.output_shapes = {"state": (192,)}
        self.embed = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32))
        self.attention = _MaskedAttention(32)
        self.device = torch.device(kwargs.get("device") or "cpu")
        self.to(self.device)

    def forward(self, joint_input):
        joint_input = torch.as_tensor(joint_input, dtype=torch.float32,
                                      device=self.device)
        shape = joint_input.shape[:-1]
        flat = joint_input.reshape(-1, 126)
        obs = flat[:, :114].reshape(-1, 6, 19)
        actions = flat[:, 114:].reshape(-1, 6, 2)
        active = obs[:, :, 18] > 0.5
        token = self.embed(torch.cat((obs[:, :, :3], actions,
                                      obs[:, :, 18:19]), dim=-1))
        attended = []
        for i in range(6):
            features = token[:, i] + self.attention(token[:, i], token, active)
            attended.append(features * active[:, i:i + 1].float())
        return {"state": torch.cat(attended, dim=-1).reshape(*shape, 192)}
