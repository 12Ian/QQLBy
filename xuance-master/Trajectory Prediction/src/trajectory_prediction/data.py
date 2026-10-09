"""Track-level splits and leakage-free windows for the paper experiments."""

from pathlib import Path

import numpy as np


def track_files(directory: Path) -> list[Path]:
    paths = sorted(Path(directory).glob("track_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No track_*.npz files in {directory}")
    return paths


def split_tracks(paths: list[Path], seed: int = 1,
                 fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
                 ) -> dict[str, list[Path]]:
    """Split complete tracks before generating any overlapping windows."""
    if len(paths) < 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Need at least three tracks and fractions summing to one")
    order = np.random.default_rng(seed).permutation(len(paths))
    n_train = min(max(1, int(len(paths) * fractions[0])), len(paths) - 2)
    n_val = min(max(1, int(len(paths) * fractions[1])), len(paths) - n_train - 1)
    return {
        "train": [paths[i] for i in order[:n_train]],
        "val": [paths[i] for i in order[n_train:n_train + n_val]],
        "test": [paths[i] for i in order[n_train + n_val:]],
    }


def split_prediction_tracks(paths: list[Path], seed: int = 1,
                            train_fraction: float = 0.8
                            ) -> dict[str, list[Path]]:
    """Paper trajectory experiment: 80% complete tracks for train, 20% for eval."""
    if len(paths) < 2 or not 0.0 < train_fraction < 1.0:
        raise ValueError("Need at least two tracks and a fraction inside (0, 1)")
    order = np.random.default_rng(seed).permutation(len(paths))
    n_train = min(max(1, int(len(paths) * train_fraction)), len(paths) - 1)
    return {
        "train": [paths[i] for i in order[:n_train]],
        "eval": [paths[i] for i in order[n_train:]],
    }


def _dominant_intent(values: np.ndarray) -> int:
    counts = np.bincount(values, minlength=9)
    return int(np.argmax(counts))


def intention_windows(paths: list[Path], dt: float = 0.1,
                      observation_seconds: float = 10.0,
                      sample_seconds: float = 0.5,
                      stride_samples: int = 5
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dt <= 0 or sample_seconds <= 0 or stride_samples <= 0:
        raise ValueError("Sampling intervals and stride must be positive")
    factor = int(round(sample_seconds / dt))
    samples = int(round(observation_seconds / sample_seconds))
    if factor < 1 or samples < 2 or not np.isclose(factor * dt, sample_seconds):
        raise ValueError("Observation sampling must be an integer multiple of dt")
    target_rows, defender_rows, labels = [], [], []
    offsets = np.arange(samples - 1, -1, -1) * factor
    for path in paths:
        with np.load(path, allow_pickle=False) as track:
            target = track["target"]
            defender = track["defender"]
            intents = track["intent"]
            ends = range(int(offsets[0]), len(target), stride_samples * factor)
            for end in ends:
                indices = end - offsets
                origin = target[end, :3]
                target_features = target[indices, :6].copy()
                target_features[:, :3] -= origin
                defender_features = defender[end, :6].copy()
                defender_features[:3] -= origin
                target_rows.append(target_features)
                defender_rows.append(defender_features)
                labels.append(_dominant_intent(intents[indices]))
    if not labels:
        raise ValueError("Tracks are too short for the requested intention window")
    return (np.asarray(target_rows, dtype=np.float32),
            np.asarray(defender_rows, dtype=np.float32),
            np.asarray(labels, dtype=np.int64))


def fit_intention_scaler(target: np.ndarray, defender: np.ndarray) -> dict[str, np.ndarray]:
    """Fit on training windows only; preserve statistics for inference."""
    target_mean = target.mean(axis=(0, 1))
    target_std = target.std(axis=(0, 1)).clip(min=1e-5)
    defender_mean = defender.mean(axis=0)
    defender_std = defender.std(axis=0).clip(min=1e-5)
    return {"target_mean": target_mean, "target_std": target_std,
            "defender_mean": defender_mean, "defender_std": defender_std}


def scale_intention(target: np.ndarray, defender: np.ndarray,
                    scaler: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return ((target - scaler["target_mean"]) / scaler["target_std"],
            (defender - scaler["defender_mean"]) / scaler["defender_std"])


def prediction_windows(paths: list[Path], dt: float = 0.1,
                       observation_seconds: float = 8.0,
                       prediction_seconds: float = 6.0,
                       stride_seconds: float = 1.0):
    """Yield only information available at the final observed time and future truth."""
    obs_steps = int(round(observation_seconds / dt))
    pred_steps = int(round(prediction_seconds / dt))
    stride_steps = int(round(stride_seconds / dt))
    if min(obs_steps, pred_steps, stride_steps) < 1:
        raise ValueError("Window and stride lengths must be positive")
    for path in paths:
        with np.load(path, allow_pickle=False) as track:
            target = track["target"]
            defender = track["defender"]
            for end in range(obs_steps - 1, len(target) - pred_steps, stride_steps):
                yield {
                    "target_history": target[end - obs_steps + 1:end + 1, :6].copy(),
                    "defender_history": defender[end - obs_steps + 1:end + 1, :6].copy(),
                    "target_acceleration": target[end, 6:9].copy(),
                    "future_xyz": target[end + 1:end + 1 + pred_steps, :3].copy(),
                    "track_id": path.stem,
                }
