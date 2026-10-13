"""Edges between observed samples of one civilian trajectory, without external entities."""
import torch


def history_edges(features, cfg, progress=1.0):
    length = features.shape[1]
    index = torch.arange(length, device=features.device, dtype=features.dtype)
    lag = (index[:, None] - index[None]).abs() * cfg.dt
    position = features[:, :, None, :3] - features[:, None, :, :3]
    velocity = features[:, :, None, 3:] - features[:, None, :, 3:]
    speed_scale = cfg.max_speed_mps or 6.0
    edge = torch.cat((position / (speed_scale * max(cfg.dt, (length - 1) * cfg.dt)),
                      velocity / speed_scale,
                      (lag / ((length - 1) * cfg.dt))[None, :, :, None].expand(len(features), -1, -1, -1)), -1)
    reachable = None
    if cfg.historical_reachability:
        # Measurement tolerance narrows during warm-up; no future state is used.
        tolerance = 2 - progress
        reachable = ((position.norm(dim=-1) <= cfg.max_speed_mps * lag + cfg.reachability_position_tolerance_m * tolerance)
                     & (velocity.norm(dim=-1) <= cfg.max_acceleration_mps2 * lag + cfg.reachability_velocity_tolerance_mps * tolerance))
        reachable = reachable | torch.eye(length, dtype=torch.bool, device=features.device)[None]
    return edge, reachable
