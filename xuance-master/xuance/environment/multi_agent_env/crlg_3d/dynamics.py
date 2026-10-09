"""Discrete acceleration-command UAV dynamics shared by flight and reachability."""

from dataclasses import dataclass
import numpy as np


@dataclass
class FlightState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float = 0.0

    def copy(self):
        return FlightState(self.position.copy(), self.velocity.copy(),
                           self.acceleration.copy(), float(self.yaw))


def bounded_vector(value, limit):
    value = np.asarray(value, dtype=np.float64)
    length = np.linalg.norm(value)
    return value * min(1.0, float(limit) / max(length, 1e-12))


def advance(state, command, dt=0.1, lag=0.5, accel_limit=1.0,
            min_speed=0.0, max_speed=11.0, jerk_limit=np.inf,
            yaw_rate=0.0, yaw_rate_limit=np.inf):
    """Apply the specified discrete model with bounded command, jerk and yaw rate."""
    if (dt < 0. or lag <= 0. or accel_limit < 0. or jerk_limit < 0. or
            yaw_rate_limit < 0. or min_speed < 0. or max_speed < min_speed):
        raise ValueError("time constant and motion limits must be valid")
    command = bounded_vector(command, accel_limit)
    jerk = bounded_vector((command - state.acceleration) / lag, jerk_limit)
    position = state.position + state.velocity * dt + 0.5 * state.acceleration * dt ** 2
    velocity = state.velocity + state.acceleration * dt
    acceleration = bounded_vector(state.acceleration + jerk * dt, accel_limit)
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity *= max_speed / speed
    elif 0 < speed < min_speed:
        velocity *= min_speed / speed
    omega = float(np.clip(yaw_rate, -yaw_rate_limit, yaw_rate_limit))
    yaw = (float(state.yaw) + omega * dt + np.pi) % (2. * np.pi) - np.pi
    return FlightState(position, velocity, acceleration, yaw)


def rollout(state, command, horizon, **dynamics):
    result = state.copy()
    full_steps = int(max(0, horizon) / dynamics.get("dt", 0.1))
    for _ in range(full_steps):
        result = advance(result, command, **dynamics)
    remainder = max(0.0, horizon - full_steps * dynamics.get("dt", 0.1))
    if remainder > 1e-9:
        result = advance(result, command, dt=remainder,
                         **{k: v for k, v in dynamics.items() if k != "dt"})
    return result
