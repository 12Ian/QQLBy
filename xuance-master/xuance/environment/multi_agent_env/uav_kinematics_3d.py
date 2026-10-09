"""Discrete 3D acceleration-command dynamics for the pursuit environment."""

import numpy as np


def limit_norm(vector, limit):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > limit > 0.0:
        return vector * (limit / norm)
    if limit <= 0.0:
        return np.zeros_like(vector)
    return vector


def advance_translation(position, velocity, acceleration, acceleration_command,
                        dt, acceleration_lag, acceleration_limit, jerk_limit,
                        speed_limit):
    """按加速度滞后和 jerk 限制推进三维平动状态。"""
    if (dt <= 0.0 or acceleration_lag <= 0.0 or acceleration_limit < 0.0 or
            jerk_limit < 0.0 or speed_limit < 0.0):
        raise ValueError("Invalid UAV dynamics parameter")
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    command = limit_norm(acceleration_command, acceleration_limit)
    jerk = limit_norm((command - acceleration) / acceleration_lag, jerk_limit)

    next_position = position + velocity * dt + 0.5 * acceleration * dt ** 2
    next_velocity = limit_norm(velocity + acceleration * dt, speed_limit)
    next_acceleration = limit_norm(acceleration + jerk * dt, acceleration_limit)
    return (next_position.astype(np.float32),
            next_velocity.astype(np.float32),
            next_acceleration.astype(np.float32))


def advance(position, velocity, acceleration, yaw, acceleration_command,
            yaw_rate_command, dt, acceleration_lag, acceleration_limit,
            jerk_limit, speed_limit, yaw_rate_limit):
    """Advance one substep using the specified Euler acceleration-lag model.

    Position and velocity use the acceleration at the start of the substep;
    the jerk-limited acceleration update takes effect on the next substep.
    """
    if (dt <= 0.0 or acceleration_lag <= 0.0 or acceleration_limit < 0.0 or
            jerk_limit < 0.0 or speed_limit < 0.0 or yaw_rate_limit < 0.0):
        raise ValueError("Invalid UAV dynamics parameter")
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    command = limit_norm(acceleration_command, acceleration_limit)
    jerk = limit_norm((command - acceleration) / acceleration_lag, jerk_limit)

    next_position = position + velocity * dt + 0.5 * acceleration * dt ** 2
    next_velocity = velocity + acceleration * dt
    next_acceleration = limit_norm(acceleration + jerk * dt, acceleration_limit)
    next_velocity = limit_norm(next_velocity, speed_limit)
    omega = float(np.clip(yaw_rate_command, -yaw_rate_limit, yaw_rate_limit))
    next_yaw = (float(yaw) + omega * dt + np.pi) % (2.0 * np.pi) - np.pi
    return (next_position.astype(np.float32),
            next_velocity.astype(np.float32),
            next_acceleration.astype(np.float32), float(next_yaw))
