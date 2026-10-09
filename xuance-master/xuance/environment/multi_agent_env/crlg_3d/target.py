"""CV, CA and Singer target commands under the shared UAV motion model."""

import numpy as np
from .dynamics import advance, bounded_vector


def advance_target(state, previous_command, mode, dt, rng, accel_limit,
                   singer_alpha, singer_sigma, *, lag, jerk_limit,
                   yaw_rate=0., yaw_rate_limit=np.inf, max_speed=11.,
                   maneuver_plane=None):
    """Advance the target and return its new acceleration command.

    CV, CA and Singer describe the commanded acceleration. The vehicle's actual
    acceleration follows that command through the same lag and jerk limit as a
    pursuer, so a Singer command is filtered before it changes velocity.
    """
    if mode == "CV":
        command = np.zeros(3)
    elif mode == "CA":
        command = np.asarray(previous_command, dtype=np.float64).copy()
    elif mode == "Singer":
        persistence = np.exp(-singer_alpha * dt)
        command = (persistence * previous_command + singer_sigma *
                   np.sqrt(max(0., 1. - persistence ** 2)) * rng.normal(size=3))
    else:
        raise ValueError(f"Unknown target mode: {mode}")
    if maneuver_plane is not None:
        command = maneuver_plane.T @ (maneuver_plane @ command)
    command = bounded_vector(command, accel_limit)
    return (advance(state, command, dt=dt, lag=lag, accel_limit=accel_limit,
                    jerk_limit=jerk_limit, yaw_rate=yaw_rate,
                    yaw_rate_limit=yaw_rate_limit, max_speed=max_speed), command)
