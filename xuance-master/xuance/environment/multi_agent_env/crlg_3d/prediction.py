"""Gaussian target endpoint prediction without future truth access."""

import numpy as np
from functools import lru_cache


def transition(mode, horizon, alpha):
    t = max(float(horizon), 0.)
    if mode == "CV":
        phi, psi, eta = 0., 0., 0.
    elif mode == "CA":
        phi, psi, eta = 1., 1., 1.
    elif mode == "Singer":
        x = alpha * t
        eta = np.exp(-x)
        phi = 2 * (x - 1 + eta) / max(x * x, 1e-12)
        psi = (1 - eta) / max(x, 1e-12)
    else:
        raise ValueError(mode)
    block = np.array([[1., t, phi * t * t / 2],
                      [0., 1., psi * t], [0., 0., eta]])
    return np.kron(block, np.eye(3))


@lru_cache(maxsize=256)
def process_covariance(mode, dt, alpha, spectral_sigma):
    """Integrate continuous white maneuver noise over one prediction step."""
    points, weights = np.polynomial.legendre.leggauss(8)
    one_axis = np.zeros((3, 3), dtype=np.float64)
    for point, weight in zip(points, weights):
        s = dt * (point + 1.) / 2.
        if mode == "CV":
            response = np.array([s, 1., 0.])
        elif mode == "CA" or alpha < 1e-6:
            response = np.array([s * s / 2., s, 1.])
        else:
            decay = np.exp(-alpha * s)
            response = np.array([s / alpha - (1. - decay) / alpha ** 2,
                                 (1. - decay) / alpha, decay])
        one_axis += weight * np.outer(response, response) * dt / 2.
    return spectral_sigma ** 2 * np.kron(one_axis, np.eye(3))


def predict_endpoint(measured_state, mode, horizon, alpha, position_sigma,
                     velocity_sigma, acceleration_sigma, process_sigma,
                     actuator_lag=None, physics_dt=0.1):
    if mode == "Singer" and actuator_lag is not None:
        if actuator_lag <= 0. or physics_dt <= 0.:
            raise ValueError("actuator_lag and physics_dt must be positive")
        # The Singer process drives acceleration commands. Actual acceleration
        # follows through the target's first-order actuator, as in target.py.
        # The hidden current command is estimated by current acceleration.
        mean = np.stack((measured_state.position, measured_state.velocity,
                         measured_state.acceleration, measured_state.acceleration))
        covariance = np.diag([position_sigma ** 2, velocity_sigma ** 2,
                              acceleration_sigma ** 2, acceleration_sigma ** 2])
        covariance[2, 3] = covariance[3, 2] = acceleration_sigma ** 2
        remaining = max(float(horizon), 0.)
        while remaining > 1e-9:
            dt = min(float(physics_dt), remaining)
            persistence = np.exp(-alpha * dt)
            response = dt / actuator_lag
            matrix = np.array([[1., dt, dt * dt / 2., 0.],
                               [0., 1., dt, 0.],
                               [0., 0., 1. - response, response * persistence],
                               [0., 0., 0., persistence]])
            noise_variance = (process_sigma ** 2 *
                              ((-np.expm1(-2. * alpha * dt)) / (2. * alpha)
                               if alpha > 1e-9 else dt))
            noise_response = np.array([0., 0., response, 1.])
            mean = matrix @ mean
            covariance = (matrix @ covariance @ matrix.T +
                          noise_variance * np.outer(noise_response, noise_response))
            remaining -= dt
        return mean[0], np.eye(3) * max(covariance[0, 0], 0.)

    mean = np.concatenate((measured_state.position, measured_state.velocity,
                           measured_state.acceleration))
    covariance = np.diag([position_sigma ** 2] * 3 +
                         [velocity_sigma ** 2] * 3 +
                         [acceleration_sigma ** 2] * 3)
    remaining = max(float(horizon), 0.)
    while remaining > 1e-9:
        dt = min(0.5, remaining)
        matrix = transition(mode, dt, alpha)
        covariance = (matrix @ covariance @ matrix.T +
                      process_covariance(mode, dt, alpha, process_sigma))
        mean = matrix @ mean
        remaining -= dt
    return mean[:3], (covariance[:3, :3] + covariance[:3, :3].T) / 2.
