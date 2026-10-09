"""Shared encounter plane and geometric time-to-go estimates."""

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(np.linalg.norm(vector), 1e-9)


def encounter_basis(pursuer_centroid, target_position):
    forward = unit(target_position - pursuer_centroid)
    reference = np.array([0., 0., 1.])
    if abs(float(np.dot(forward, reference))) > 0.95:
        reference = np.array([0., 1., 0.])
    first = unit(np.cross(reference, forward))
    second = unit(np.cross(forward, first))
    return np.stack((first, second)), forward


def encounter_time(pursuers, target, minimum=0.0, maximum=35.0):
    values = []
    for state in pursuers:
        line = unit(target.position - state.position)
        closing = np.dot(state.velocity - target.velocity, line)
        values.append(np.linalg.norm(target.position - state.position) /
                      max(closing, 1e-3))
    return float(np.clip(np.median(values), minimum, maximum))


def geometric_tgo(pursuer, target):
    line = unit(target.position - pursuer.position)
    closing = np.dot(pursuer.velocity - target.velocity, line)
    return float(np.linalg.norm(target.position - pursuer.position) /
                 max(closing, 1e-3))
