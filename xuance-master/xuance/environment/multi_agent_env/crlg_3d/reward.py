"""CRL-G coverage, prediction error, energy, and terminal reward terms."""

import numpy as np


def process_reward(coverage, projected_misses, commands, scales, acceleration_limit=1.):
    count = len(projected_misses)
    result = []
    components = []
    for i in range(count):
        prediction_raw = -float(
            np.dot(projected_misses[i], projected_misses[i]))
        energy_raw = -float(np.dot(commands[i], commands[i])) / acceleration_limit ** 2
        cov = scales["coverage"] * coverage
        prediction = scales["prediction"] * prediction_raw
        energy = scales["energy"] * energy_raw
        result.append(cov + prediction + energy)
        components.append({"coverage_raw": coverage,
                           "prediction_raw": prediction_raw,
                           "energy_raw": energy_raw,
                           "coverage": cov, "prediction": prediction,
                           "energy": energy, "terminal_raw": 0.,
                           "terminal": 0., "boundary": 0.})
    return result, components


def terminal_reward(misses, own_weight=0.5, team_weight=0.5, bias=5.):
    misses = np.asarray(misses, dtype=np.float64)
    team = 1. / (float(np.min(misses)) ** 2 + bias ** 2)
    return (own_weight / (misses ** 2 + bias ** 2) +
            team_weight * team).tolist()
