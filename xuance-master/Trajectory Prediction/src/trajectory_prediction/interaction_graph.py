"""Temporal and proximity edges for synchronized civilian measured histories."""
import torch


def interaction_edges(features, valid, cfg, progress=1.):
    batch, nodes, _ = features.shape
    length = cfg.observation_steps
    index = torch.arange(nodes, device=features.device)
    time, entity = index % length, index // length
    lag = (time[:, None] - time[None]).abs().to(features.dtype) * cfg.dt
    same = entity[:, None] == entity[None]
    position = features[:, :, None, :3] - features[:, None, :, :3]
    velocity = features[:, :, None, 3:] - features[:, None, :, 3:]
    speed = cfg.max_speed_mps or 1.5
    edges = torch.cat((position / (speed * (length - 1) * cfg.dt), velocity / speed,
                       (lag / ((length - 1) * cfg.dt))[None, :, :, None].expand(batch, -1, -1, -1)), -1)
    temporal = same[None] & (lag <= cfg.temporal_radius * cfg.dt)[None]
    if cfg.historical_reachability:
        temporal = temporal & (position.norm(dim=-1) <= cfg.max_speed_mps * lag + cfg.reachability_position_tolerance_m * (2 - progress))
        temporal = temporal & (velocity.norm(dim=-1) <= cfg.max_acceleration_mps2 * lag + cfg.reachability_velocity_tolerance_mps * (2 - progress))
    spatial = (~same)[None] & (lag <= cfg.interaction_time_radius * cfg.dt)[None] & (position.norm(dim=-1) <= cfg.interaction_radius_m)
    allowed = (temporal | spatial) & valid[:, :, None] & valid[:, None]
    if cfg.max_speed_mps:
        plausible = features[..., 3:].norm(dim=-1) <= cfg.max_speed_mps
        allowed = allowed & plausible[:, None]
    # Isolated unavailable query rows retain only self; excluded from supervision/memory.
    allowed = allowed | torch.eye(nodes, dtype=torch.bool, device=features.device)[None]
    return edges, allowed
