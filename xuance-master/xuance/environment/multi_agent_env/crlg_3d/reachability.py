"""Calibrated lateral endpoint radius under the implemented flight dynamics."""

import numpy as np
from .dynamics import FlightState, advance, rollout


class ReachabilityTable:
    def __init__(self, max_horizon=35., step=0.5, **dynamics):
        max_speed = float(dynamics.get("max_speed", 11.))
        self.speeds = np.unique(np.r_[np.arange(0., max_speed, 1.), max_speed])
        self.times = np.arange(0., max_horizon + step / 2, step)
        self.values = np.empty((len(self.speeds), len(self.times)))
        self.dynamics = dynamics
        for i, speed in enumerate(self.speeds):
            state = FlightState(np.zeros(3), np.array([speed, 0., 0.]),
                                np.zeros(3))
            previous = 0.
            for j, horizon in enumerate(self.times):
                state = rollout(state, np.array([0., dynamics.get("accel_limit", 1.), 0.]),
                                float(horizon - previous), **dynamics)
                self.values[i, j] = max(0., state.position[1])
                previous = horizon

    def radius(self, speed, horizon):
        clipped_t = np.clip(horizon, self.times[0], self.times[-1])
        at_speeds = [np.interp(clipped_t, self.times, row) for row in self.values]
        return float(np.interp(np.clip(speed, self.speeds[0], self.speeds[-1]),
                               self.speeds, at_speeds))


def directional_polygon(state, horizon, plane, zero_endpoint, bounds,
                        dt=0.1, lag=0.5, accel_limit=1.0, jerk_limit=np.inf,
                        max_speed=11., directions=12):
    """Project boundary-feasible extreme-control endpoints from the current state.

    Used near the encounter, when actuator state and speed clipping matter most.
    A direction whose path touches a wall collapses to the zero-control center.
    """
    center = plane @ zero_endpoint
    points = []
    steps = max(1, int(np.ceil(horizon / dt)))
    sub_dt = horizon / steps
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    for angle in np.linspace(0., 2. * np.pi, directions, endpoint=False):
        command = accel_limit * (np.cos(angle) * plane[0] +
                                 np.sin(angle) * plane[1])
        current = state.copy()
        valid = True
        for _ in range(steps):
            current = advance(current, command, sub_dt, lag, accel_limit,
                              jerk_limit=jerk_limit, max_speed=max_speed)
            if np.any(current.position <= lower) or np.any(current.position >= upper):
                valid = False
                break
        points.append(plane @ current.position if valid else center)
    return np.asarray(points)
