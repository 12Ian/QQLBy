"""Midcourse action projection and terminal proportional navigation."""

import numpy as np
from .dynamics import bounded_vector


def midcourse_command(action, plane, acceleration_limit):
    action = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
    return bounded_vector(action @ plane, acceleration_limit)


def yaw_rate_toward(current_yaw, desired_yaw, horizon, rate_limit):
    """Point independently of translation, subject to the yaw-rate limit."""
    error = (desired_yaw - current_yaw + np.pi) % (2. * np.pi) - np.pi
    return float(np.clip(error / max(horizon, 1e-9), -rate_limit, rate_limit))


def png_command(pursuer, target, navigation_constant=3., acceleration_limit=2.):
    displacement = target.position - pursuer.position
    distance = max(np.linalg.norm(displacement), 1e-6)
    line = displacement / distance
    relative_velocity = target.velocity - pursuer.velocity
    closing = max(0., -np.dot(relative_velocity, line))
    los_rate = np.cross(displacement, relative_velocity) / distance ** 2
    command = navigation_constant * closing * np.cross(los_rate, line)
    return bounded_vector(command, acceleration_limit)
