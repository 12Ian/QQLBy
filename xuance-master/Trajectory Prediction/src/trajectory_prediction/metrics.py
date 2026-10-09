"""Displacement metrics and nonlearned trajectory prediction baselines."""

import numpy as np


def displacement_errors(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Expected matching [prediction_steps, 3] arrays")
    distances = np.linalg.norm(predicted - truth, axis=-1)
    return float(distances.mean()), float(distances[-1])


def constant_velocity(history: np.ndarray, dt: float, steps: int) -> np.ndarray:
    future_times = np.arange(1, steps + 1)[:, None] * dt
    return history[-1, :3] + future_times * history[-1, 3:6]


def constant_acceleration(history: np.ndarray, acceleration: np.ndarray,
                          dt: float, steps: int) -> np.ndarray:
    future_times = np.arange(1, steps + 1)[:, None] * dt
    return (history[-1, :3] + future_times * history[-1, 3:6] +
            0.5 * future_times ** 2 * acceleration)
