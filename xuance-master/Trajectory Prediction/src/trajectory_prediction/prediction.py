"""Intention-conditioned interaction graph, candidate cropper, and GRU decoder.

The paper leaves candidate construction and several cropper thresholds open. The
fixed-6g primitive library and ground-plane mask below are explicit choices,
not parameters claimed to be published by Yu et al.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cov_map import CovMap


class InteractionBlock(nn.Module):
    """Typed same-time target/defender attention, then per-agent temporal attention."""

    def __init__(self, width: int = 256, spatial_heads: int = 6,
                 head_width: int = 64, temporal_heads: int = 4):
        super().__init__()
        self.heads = spatial_heads
        self.head_width = head_width
        spatial_width = spatial_heads * head_width
        self.query = nn.Linear(width, spatial_width)
        self.key = nn.Linear(width, spatial_width)
        self.value = nn.Linear(width, spatial_width)
        self.edge_bias = nn.Sequential(nn.Linear(6, 64), nn.ReLU(),
                                       nn.Linear(64, spatial_heads))
        self.spatial_output = nn.Linear(spatial_width, width)
        self.spatial_norm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, temporal_heads,
                                               batch_first=True)
        self.temporal_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(),
                                         nn.Linear(width * 2, width))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, features: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch, times, agents, width = features.shape
        shape = (batch, times, agents, self.heads, self.head_width)
        query = self.query(features).reshape(shape)
        key = self.key(features).reshape(shape)
        value = self.value(features).reshape(shape)
        logits = torch.einsum("btihd,btjhd->bthij", query, key) / math.sqrt(self.head_width)
        pair_difference = states.unsqueeze(3) - states.unsqueeze(2)
        logits = logits + self.edge_bias(pair_difference).permute(0, 1, 4, 2, 3)
        weights = logits.softmax(dim=-1)
        spatial = torch.einsum("bthij,btjhd->btihd", weights, value)
        spatial = self.spatial_output(spatial.reshape(batch, times, agents, -1))
        features = self.spatial_norm(features + spatial)
        per_agent = features.transpose(1, 2).reshape(batch * agents, times, width)
        temporal, _ = self.temporal(per_agent, per_agent, per_agent,
                                    need_weights=False)
        per_agent = self.temporal_norm(per_agent + temporal)
        per_agent = self.output_norm(per_agent + self.feedforward(per_agent))
        return per_agent.reshape(batch, agents, times, width).transpose(1, 2)


def primitive_candidates(position: torch.Tensor, velocity: torch.Tensor,
                         steps: int = 60, dt: float = 0.1,
                         acceleration: float = 6.0 * 9.81) -> torch.Tensor:
    """Nine constant-speed, velocity-normal maneuver hypotheses [B,9,T,3]."""
    signs = velocity.new_tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                                 [1, 1], [-1, 1], [1, -1], [-1, -1]])
    batch = len(position)
    pos = position[:, None, :].expand(batch, 9, 3).clone()
    speed = velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = F.normalize(velocity, dim=-1)[:, None, :].expand(batch, 9, 3).clone()
    up = velocity.new_tensor([0.0, 0.0, 1.0]).expand(batch, 9, 3)
    alternative = velocity.new_tensor([0.0, 1.0, 0.0]).expand(batch, 9, 3)
    positions = []
    for _ in range(steps):
        left = torch.linalg.cross(up, direction)
        left = torch.where(left.norm(dim=-1, keepdim=True) < 1e-6,
                           torch.linalg.cross(alternative, direction), left)
        left = F.normalize(left, dim=-1)
        climb = F.normalize(torch.linalg.cross(direction, left), dim=-1)
        command = signs[None, :, 0:1] * left + signs[None, :, 1:2] * climb
        command = torch.where(command.norm(dim=-1, keepdim=True) > 0,
                              F.normalize(command, dim=-1), torch.zeros_like(command))
        next_direction = F.normalize(direction + command * (acceleration * dt) /
                                     speed[:, None, :], dim=-1)
        pos = pos + (direction + next_direction) * speed[:, None, :] * (dt / 2)
        direction = next_direction
        positions.append(pos)
    return torch.stack(positions, dim=2)


class TrajectoryPredictor(nn.Module):
    """Graph context -> soft/Top-K candidate cropper -> autoregressive deltas."""

    def __init__(self, width: int = 256, graph_steps: int = 16,
                 future_steps: int = 60, top_k: int = 8,
                 temperature: float = 0.1, dt: float = 0.1,
                 candidate_acceleration_g: float = 6.0,
                 min_altitude_m: float = 0.0,
                 escape_rate_weight: float = 0.1,
                 escape_rate_scale_mps: float = 200.0,
                 threat_range_m: float = 3000.0,
                 threat_gate_scale_m: float = 500.0):
        super().__init__()
        if (not 1 <= top_k <= 9 or temperature <= 0 or dt <= 0 or
                not 0 <= candidate_acceleration_g <= 9 or
                escape_rate_weight < 0 or escape_rate_scale_mps <= 0 or
                threat_range_m <= 0 or threat_gate_scale_m <= 0):
            raise ValueError("Invalid candidate or time configuration")
        self.graph_steps = graph_steps
        self.future_steps = future_steps
        self.top_k = top_k
        self.temperature = temperature
        self.dt = dt
        self.candidate_acceleration_g = candidate_acceleration_g
        self.min_altitude_m = min_altitude_m
        self.escape_rate_weight = escape_rate_weight
        self.escape_rate_scale_mps = escape_rate_scale_mps
        self.threat_range_m = threat_range_m
        self.threat_gate_scale_m = threat_gate_scale_m
        self.state_embedding = nn.Linear(6, width)
        self.agent_embedding = nn.Embedding(2, width)
        self.blocks = nn.ModuleList([InteractionBlock(width) for _ in range(2)])
        self.context = nn.Sequential(nn.Linear(3 * width + 9 + 8, width),
                                     nn.ReLU(), nn.LayerNorm(width))
        self.candidate_embedding = nn.Sequential(nn.Linear(width + 8, width),
                                                 nn.ReLU())
        # Six candidate heads form three fixed two-head groups: E, R, T.
        self.candidate_heads = nn.Linear(width, 6)
        self.candidate_score = nn.Linear(width, 1)
        self.decoder = nn.GRUCell(4, width)
        self.delta_correction = nn.Linear(width, 3)
        nn.init.zeros_(self.delta_correction.weight)
        nn.init.zeros_(self.delta_correction.bias)

    def forward(self, target_history: torch.Tensor,
                defender_history: torch.Tensor,
                intention_prob: torch.Tensor,
                cov_map: CovMap | None = None) -> dict[str, torch.Tensor]:
        if target_history.shape != defender_history.shape or target_history.shape[-1] != 6:
            raise ValueError("Expected matched target/defender histories [B,T,6]")
        if intention_prob.shape != (len(target_history), 9):
            raise ValueError("Expected intention probabilities [B,9]")
        if cov_map is not None:
            cov_map.assert_starts_inside(target_history[:, -1, :3],
                                         defender_history[:, -1, :3])
        indices = torch.linspace(0, target_history.shape[1] - 1,
                                 self.graph_steps, device=target_history.device).long()
        origin = target_history[:, -1, :3]
        target = target_history[:, indices].clone()
        defender = defender_history[:, indices].clone()
        target[:, :, :3] = (target[:, :, :3] - origin[:, None, :]) / 1000.0
        defender[:, :, :3] = (defender[:, :, :3] - origin[:, None, :]) / 1000.0
        target[:, :, 3:] = target[:, :, 3:] / 300.0
        defender[:, :, 3:] = defender[:, :, 3:] / 300.0
        states = torch.stack((target, defender), dim=2)
        agents = self.agent_embedding(torch.arange(2, device=states.device))
        features = self.state_embedding(states) + agents[None, None, :, :]
        for block in self.blocks:
            features = block(features, states)
        rel_pos = defender[:, -1, :3] - target[:, -1, :3]
        rel_vel = defender[:, -1, 3:] - target[:, -1, 3:]
        range_scaled = rel_pos.norm(dim=-1, keepdim=True)
        closure = (rel_pos * rel_vel).sum(dim=-1, keepdim=True) / range_scaled.clamp_min(1e-6)
        engagement = torch.cat((rel_pos, rel_vel, range_scaled / 8.0, closure), dim=-1)
        context = self.context(torch.cat((features[:, -1, 0], features[:, -1, 1],
                                          features[:, :, 0].mean(dim=1), intention_prob,
                                          engagement), dim=-1))
        with torch.no_grad():
            candidates = primitive_candidates(origin, target_history[:, -1, 3:],
                                               self.future_steps, self.dt,
                                               self.candidate_acceleration_g * 9.81)
        final_displacement = (candidates[:, :, -1] - origin[:, None, :]) / 1000.0
        defender_final = (defender_history[:, -1, :3] +
                          defender_history[:, -1, 3:] * (self.future_steps * self.dt))
        separation = (candidates[:, :, -1] - defender_final[:, None, :]) / 1000.0
        candidate_features = torch.cat((final_displacement, separation,
                                        intention_prob.unsqueeze(-1),
                                        separation.norm(dim=-1, keepdim=True)), dim=-1)
        candidate_hidden = self.candidate_embedding(torch.cat((
            context[:, None, :].expand(-1, 9, -1), candidate_features), dim=-1))
        head_logits = self.candidate_heads(candidate_hidden).transpose(1, 2)
        scores = self.candidate_score(candidate_hidden).squeeze(-1)
        current_relative = defender_history[:, -1, :3] - origin
        current_range = current_relative.norm(dim=-1).clamp_min(1e-6)
        visible_cosine = ((current_relative * target_history[:, -1, 3:]).sum(dim=-1) /
                          (current_range * target_history[:, -1, 3:].norm(dim=-1).clamp_min(1e-6)))
        visibility_gate = torch.sigmoid(10.0 * (visible_cosine - 0.5))
        threat_gate = torch.sigmoid((self.threat_range_m - current_range) /
                                    self.threat_gate_scale_m) * visibility_gate
        escape_rate = ((candidates[:, :, -1] - defender_final[:, None, :]).norm(dim=-1) -
                       current_range[:, None]) / (self.future_steps * self.dt)
        scores = (scores + self.escape_rate_weight * threat_gate[:, None] *
                  torch.tanh(escape_rate / self.escape_rate_scale_mps))
        feasible = (candidates[:, :, :, 2] >= self.min_altitude_m).all(dim=-1)
        if cov_map is not None:
            feasible &= cov_map.candidate_valid(candidates)
        fallback = ~feasible.any(dim=-1)
        selectable = feasible.clone()
        selectable[fallback, 0] = True  # Numerical fallback, reported as infeasible.
        scores = scores.masked_fill(~selectable, -1e4)
        keep = torch.zeros_like(feasible)
        keep[:, 0] = selectable[:, 0]
        if self.top_k > 1:
            keep[:, 1:].scatter_(1, scores[:, 1:].topk(self.top_k - 1, dim=-1).indices,
                                 True)
        keep &= selectable
        weights = (scores / self.temperature).masked_fill(~keep, -1e4).softmax(dim=-1)
        soft_weights = (scores / self.temperature).masked_fill(
            ~selectable, -1e4).softmax(dim=-1)
        head_weights = (head_logits / self.temperature).masked_fill(
            ~keep[:, None, :], -1e4).softmax(dim=-1)
        candidate_deltas = torch.cat((candidates[:, :, :1] - origin[:, None, None, :],
                                      candidates[:, :, 1:] - candidates[:, :, :-1]), dim=2)
        mean_deltas = (candidate_deltas * weights[:, :, None, None]).sum(dim=1)
        hidden = context
        deltas = []
        for step in range(self.future_steps):
            fraction = mean_deltas.new_full((len(mean_deltas), 1),
                                            (step + 1) / self.future_steps)
            hidden = self.decoder(torch.cat((mean_deltas[:, step] / 50.0,
                                             fraction), dim=-1), hidden)
            deltas.append(mean_deltas[:, step] + self.delta_correction(hidden))
        future = origin[:, None, :] + torch.stack(deltas, dim=1).cumsum(dim=1)
        return {"future_xyz": future, "candidate_weights": weights,
                "candidate_soft_weights": soft_weights,
                "candidate_head_weights": head_weights,
                "candidate_valid": feasible, "candidate_kept": keep,
                "candidate_fallback": fallback,
                "candidate_xyz": candidates, "escape_rate_mps": escape_rate,
                "engagement_gate": threat_gate}
