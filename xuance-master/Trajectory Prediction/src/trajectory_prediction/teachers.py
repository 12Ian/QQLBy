"""Training-only energy, threat, and task teacher signals for candidate heads.

The paper gives the score families but omits normalization, a defender forecast,
and a mission corridor dataset. Here the defender follows observed constant
velocity and the mission corridor follows the target's observed straight path.
These are replaceable reproduction assumptions, never future truth inputs.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _smooth_max(values: torch.Tensor, sharpness: float, dim: int) -> torch.Tensor:
    return torch.logsumexp(sharpness * values, dim=dim) / sharpness


@torch.no_grad()
def teacher_scores(candidates: torch.Tensor, target_history: torch.Tensor,
                   defender_history: torch.Tensor, config: dict,
                   dt: float = 0.1) -> torch.Tensor:
    """Return [B,3,9] scores; larger E/T means preferable, larger R means threat."""
    if candidates.ndim != 4 or candidates.shape[1] != 9 or candidates.shape[-1] != 3:
        raise ValueError("Expected candidate trajectories [B,9,T,3]")
    batch, modes, steps, _ = candidates.shape
    origin = target_history[:, -1, :3]
    target_velocity = target_history[:, -1, 3:6]
    defender_position = defender_history[:, -1, :3]
    defender_velocity = defender_history[:, -1, 3:6]
    times = torch.arange(1, steps + 1, device=candidates.device,
                         dtype=candidates.dtype) * dt

    previous_position = torch.cat((origin[:, None, None, :].expand(-1, modes, 1, -1),
                                   candidates[:, :, :-1]), dim=2)
    velocity = (candidates - previous_position) / dt
    previous_velocity = torch.cat((target_velocity[:, None, None, :].expand(-1, modes, 1, -1),
                                   velocity[:, :, :-1]), dim=2)
    acceleration = (velocity - previous_velocity) / dt
    observed_acceleration = (target_history[:, -1, 3:6] -
                             target_history[:, -2, 3:6]) / dt
    previous_acceleration = torch.cat((
        observed_acceleration[:, None, None, :].expand(-1, modes, 1, -1),
        acceleration[:, :, :-1]), dim=2)
    jerk = (acceleration - previous_acceleration) / dt
    speed = target_velocity.norm(dim=-1).clamp_min(1e-6)
    acceleration_cost = acceleration.norm(dim=-1).mean(dim=-1) / (9.0 * 9.81)
    jerk_cost = jerk.norm(dim=-1).mean(dim=-1) / config["jerk_reference_mps3"]
    braking_cost = ((speed[:, None, None] - velocity.norm(dim=-1)).relu() /
                    speed[:, None, None]).mean(dim=-1)
    energy_components = torch.stack((acceleration_cost.square(), jerk_cost.square(),
                                     braking_cost.square()), dim=-1)
    energy = -_smooth_max(energy_components, config["energy_kappa"], dim=-1)

    defender_future = (defender_position[:, None, None, :] +
                       defender_velocity[:, None, None, :] * times[None, None, :, None])
    separation = (candidates - defender_future).norm(dim=-1)
    initial_separation = (origin - defender_position).norm(dim=-1)
    previous_separation = torch.cat((
        initial_separation[:, None, None].expand(-1, modes, 1),
        separation[:, :, :-1]), dim=-1)
    range_rate = (separation - previous_separation) / dt
    reachable_radius = 0.5 * config["defender_acceleration_g"] * 9.81 * times.square()
    risk_components = torch.stack((
        torch.exp(-separation / config["distance_scale_m"]),
        torch.sigmoid(-range_rate / config["closure_scale_mps"]),
        torch.sigmoid((reachable_radius[None, None, :] - separation) /
                      config["reachability_scale_m"])), dim=-1)
    risk_over_time = _smooth_max(risk_components, config["risk_kappa"], dim=-1)
    risk = (_smooth_max(risk_over_time, config["risk_time_lambda"], dim=-1) -
            math.log(steps) / config["risk_time_lambda"])

    nominal = origin[:, None, None, :] + target_velocity[:, None, None, :] * times[None, None, :, None]
    deviation = candidates - nominal
    forward = F.normalize(target_velocity, dim=-1)[:, None, None, :]
    radial = (deviation * forward).sum(dim=-1).abs()
    normal = (deviation - (deviation * forward).sum(dim=-1, keepdim=True) * forward).norm(dim=-1)
    radial_penalty = ((radial - config["corridor_radial_m"]) /
                      config["corridor_radial_m"]).relu()
    normal_penalty = ((normal - config["corridor_normal_m"]) /
                      config["corridor_normal_m"]).relu()
    corridor = (_smooth_max(torch.stack((radial_penalty, normal_penalty), dim=-1),
                            config["task_kappa"], dim=-1) -
                math.log(2) / config["task_kappa"])
    goal = nominal[:, :, -1:, :]
    goal_distance = (candidates - goal).norm(dim=-1) / config["goal_scale_m"]
    remaining_time = ((times[-1] - times) / times[-1])[None, None, :]
    gate = torch.sigmoid(config["task_kappa"] * (times / times[-1] - 1))
    task_penalty = ((1 - gate)[None, None, :] * corridor +
                    gate[None, None, :] * (goal_distance + remaining_time))
    task = -task_penalty.mean(dim=-1)
    return torch.stack((energy, risk, task), dim=1)


def _teacher_distribution(scores: torch.Tensor, mask: torch.Tensor,
                          temperature: float) -> torch.Tensor:
    return (scores / temperature).masked_fill(~mask, -1e4).softmax(dim=-1)


class TeacherObjective(nn.Module):
    """Factor-specific head KL, entropy, separation, cropper KL, and ADE/FDE."""

    def __init__(self, config: dict, ade_weight: float, fde_weight: float):
        super().__init__()
        positive = ("temperature", "energy_kappa", "jerk_reference_mps3",
                    "risk_kappa", "risk_time_lambda", "distance_scale_m",
                    "closure_scale_mps", "reachability_scale_m", "task_kappa",
                    "corridor_radial_m", "corridor_normal_m", "goal_scale_m")
        if any(float(config[name]) <= 0 for name in positive):
            raise ValueError("Teacher temperatures and physical scales must be positive")
        self.config = config
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.log_sigma = nn.Parameter(torch.zeros(4))  # geometry, E, R, T

    def forward(self, prediction: dict[str, torch.Tensor],
                target_history: torch.Tensor, defender_history: torch.Tensor,
                truth: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        distances = (prediction["future_xyz"] - truth).norm(dim=-1)
        geometry = (self.ade_weight * distances.mean() +
                    self.fde_weight * distances[:, -1].mean()) / 1000.0
        if not self.config["enabled"]:
            return geometry, {"geometry": float(geometry.detach()), "teacher": 0.0}
        with torch.no_grad():
            scores = teacher_scores(prediction["candidate_xyz"], target_history,
                                    defender_history, self.config)
        heads = prediction["candidate_head_weights"].clamp_min(1e-8)
        kept = prediction["candidate_kept"]
        temperature = self.config["temperature"]
        group_losses = []
        group_means = []
        for group in range(3):
            teacher = _teacher_distribution(scores[:, group], kept, temperature)
            student = heads[:, group * 2:group * 2 + 2]
            kl = (teacher[:, None, :] * (
                teacher[:, None, :].clamp_min(1e-8).log() - student.log())).sum(-1).mean()
            entropy = -(student * student.log()).sum(-1)
            anticollapse = F.relu(self.config["entropy_min"] - entropy).mean()
            group_losses.append(kl + self.config["entropy_weight"] * anticollapse)
            group_means.append(student.mean(dim=1))
        geometry_weighted = (0.5 * torch.exp(-2 * self.log_sigma[0]) * geometry +
                             self.log_sigma[0])
        weighted = geometry_weighted
        for group, group_loss in enumerate(group_losses):
            index = group + 1
            weighted = (weighted + 0.5 * torch.exp(-2 * self.log_sigma[index]) *
                        group_loss + self.log_sigma[index])
        separation = sum((F.cosine_similarity(group_means[i], group_means[j], dim=-1)
                          .square().mean() for i, j in ((0, 1), (0, 2), (1, 2)))) / 3
        centered = ((scores - scores.mean(dim=-1, keepdim=True)) /
                    scores.std(dim=-1, keepdim=True).clamp_min(0.1))
        desirability = (self.config["energy_desirability_weight"] * centered[:, 0] -
                        self.config["risk_desirability_weight"] * centered[:, 1] +
                        self.config["task_desirability_weight"] * centered[:, 2])
        crop_teacher = _teacher_distribution(
            desirability, prediction["candidate_valid"], temperature)
        crop_student = prediction["candidate_soft_weights"].clamp_min(1e-8)
        cropper_kl = (crop_teacher * (crop_teacher.clamp_min(1e-8).log() -
                                      crop_student.log())).sum(-1).mean()
        loss = (weighted + self.config["separation_weight"] * separation +
                self.config["cropper_weight"] * cropper_kl)
        return loss, {"geometry": float(geometry.detach()),
                      "teacher": float((loss - geometry_weighted).detach()),
                      "energy_kl": float(group_losses[0].detach()),
                      "risk_kl": float(group_losses[1].detach()),
                      "task_kl": float(group_losses[2].detach()),
                      "cropper_kl": float(cropper_kl.detach())}
