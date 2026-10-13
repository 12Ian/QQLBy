"""Track-level splitting and observed-history-only feature processing."""

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["straight", "left", "right", "up", "down",
               "left_up", "right_up", "left_down", "right_down"]

# Eleven overload-intensity intention levels (paper IV-A, second experiment):
# normal overload quantized to {+-10, +-8, +-6, +-4, +-2, 0} g plus the two
# outermost saturating bins, ordered from the most negative to the most
# positive commanded overload.
OVERLOAD_CLASSES = 11
OVERLOAD_NAMES = ["-10g", "-8g", "-6g", "-4g", "-2g", "0g",
                  "2g", "4g", "6g", "8g", "10g"]


@dataclass
class Track:
    track_id: str
    time: np.ndarray
    states: np.ndarray  # [T, 6]: position (m), velocity (m/s)
    labels: np.ndarray  # [T], -1 means unavailable
    context: np.ndarray | None = None  # [T,N,6], synchronized measured peers
    context_mask: np.ndarray | None = None  # [T,N], measurement availability
    scene_id: str = ""
    scenario: str = "independent"
    overload_labels: np.ndarray | None = None  # [T], 11-level overload intensity
    evasion_mask: np.ndarray | None = None  # [T], evasion trigger active (military)
    engagement: np.ndarray | None = None  # [T,9] engagement descriptors (military)
    # [T] raw signed normal overload (m/s^2); the eleven-level window label is
    # the quantization of the observation-window mean of this series (paper
    # IV-A: a_bar_n = (1/to) * sum a_n(k), label = bin containing a_bar_n).
    overload_series: np.ndarray | None = None
    # Whether the generator widened the two extreme overload bins to
    # (-inf, I_min+1.5] / [I_max-1.5, +inf); the window-label recomputation
    # must reuse the same bin convention or the +/-10 g rows silently vanish.
    overload_censored: bool = False


def intention_label_source(track, task):
    """Resolve the per-step label array for the configured intention task.

    "semantic9" uses the nine-class dominant maneuver primitive; "overload11"
    uses the quantized normal-overload intensity (paper IV-A). Both are
    per-step arrays defined on the observation window only.
    """
    source = track.overload_labels if task == "overload11" else track.labels
    if task == "overload11" and source is None:
        raise ValueError("intention_task 'overload11' requires overload_labels in the dataset"
                         " (simulate with engagement_scenarios.simulate_engagements)")
    return source


def window_intention_label(track, start, stop, classes, task="semantic9"):
    """Window-level intention label on the observation window [start, stop).

    Only observed steps are aggregated; the future window is never consulted.
    Returns -1 when no observed label is available.

    * "semantic9" (paper IV-A first experiment): dominant maneuver semantics,
      i.e. the majority per-step semantic label inside the window.
    * "overload11" (paper IV-A second experiment): the paper defines the label
      by the *mean* overload a_bar_n over the observation window, assigned by
      locating a_bar_n in (I_m - 0.5, I_m + 0.5] -- ONE quantization of the
      window mean, not a vote over per-step quantizations.  When the raw
      overload series is available (engagement datasets store it), the mean is
      quantized directly; legacy datasets without the series fall back to
      majority voting over the stored per-step bins.
    """
    if task == "overload11":
        series = getattr(track, "overload_series", None)
        if series is not None:
            from .engagement_scenarios import overload_label
            window = np.asarray(series[start:stop], dtype=np.float64)
            if window.size == 0:
                return -1
            return int(overload_label(window, censored_ends=track.overload_censored))
    labels = intention_label_source(track, task)[start:stop]
    available = labels[labels >= 0]
    return int(np.bincount(available, minlength=classes).argmax()) if len(available) else -1


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_labels(values, classes, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain finite integers")
    array = array.astype(np.int64)
    if np.any((array < -1) | (array >= classes)):
        raise ValueError(f"{name} must be -1 or in [0,{classes - 1}]")
    return array


def load_tracks(path, dt, minimum_steps, classes=9, overload_classes=OVERLOAD_CLASSES):
    path = Path(path)
    engagement_raw = None          # CSV carries no engagement descriptors
    overload_series_raw = None     # npz-only: raw signed normal overload series
    overload_censored_raw = False  # npz-only: generator widened the extreme bins
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {"track_id", "time", "x", "y", "z", "vx", "vy", "vz"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"CSV needs columns {sorted(required)} and optional label")
        ids = np.asarray([r["track_id"] for r in rows])
        time = np.asarray([float(r["time"]) for r in rows], dtype=np.float64)
        states = np.asarray([[float(r[k]) for k in ("x", "y", "z", "vx", "vy", "vz")]
                             for r in rows], dtype=np.float32)
        labels_raw = np.asarray([float(r.get("label", "") or -1) for r in rows])
        scene_ids = np.asarray([r.get("scene_id") or r["track_id"] for r in rows])
        explicit_scenes = "scene_id" in rows[0] and all((r.get("scene_id") or "").strip() for r in rows)
        scenarios = np.asarray([r.get("scenario") or "independent" for r in rows])
        peer_indices = sorted({int(key.split("_")[1]) for key in rows[0] if key.startswith("context_")})
        if peer_indices != list(range(len(peer_indices))):
            raise ValueError("CSV context indices must be contiguous, starting at 0")
        context_states, context_masks = [], []
        for peer in peer_indices:
            columns = [f"context_{peer}_{key}" for key in ("x", "y", "z", "vx", "vy", "vz")]
            if not set(columns).issubset(rows[0]):
                raise ValueError("Each CSV context entity needs six state columns")
            context_states.append([[float(row[key]) if row[key].strip() else np.nan for key in columns] for row in rows])
            context_masks.append([float(row.get(f"context_{peer}_mask", "1")) for row in rows])
        context_states = np.asarray(context_states).transpose(1, 0, 2) if peer_indices else None
        context_masks = np.asarray(context_masks).T if peer_indices else None
        overload_raw = (np.asarray([float(r.get("overload_label", "") or -1) for r in rows])
                        if "overload_label" in rows[0] else None)
        evasion_raw = (np.asarray([float(r.get("evasion", "0") or 0) for r in rows])
                       if "evasion" in rows[0] else None)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            ids, time, states = (archive[k] for k in ("track_id", "time", "states"))
            labels_raw = archive["labels"] if "labels" in archive else np.full(len(time), -1)
            scene_ids = archive["scene_id"] if "scene_id" in archive else ids
            explicit_scenes = "scene_id" in archive
            scenarios = archive["scenario"] if "scenario" in archive else np.full(len(time), "independent")
            context_states = archive["context_states"] if "context_states" in archive else None
            context_masks = archive["context_mask"] if "context_mask" in archive else None
            overload_raw = archive["overload_labels"] if "overload_labels" in archive else None
            overload_series_raw = archive["overload_series"] if "overload_series" in archive else None
            overload_censored_raw = (bool(np.asarray(archive["overload_censored"]).item())
                                     if "overload_censored" in archive else False)
            evasion_raw = archive["evasion_mask"] if "evasion_mask" in archive else None
            engagement_raw = archive["engagement_states"] if "engagement_states" in archive else None
    else:
        raise ValueError("Data must be a .npz or .csv file")
    if states.ndim != 2 or states.shape[1] != 6 or len(states) != len(time):
        raise ValueError("states must have shape [rows,6] matching time")
    if ids.ndim != 1 or time.ndim != 1 or labels_raw.ndim != 1:
        raise ValueError("track_id, time, labels must be one-dimensional")
    if len(ids) != len(time) or len(labels_raw) != len(time):
        raise ValueError("track_id/time/labels row counts do not match")
    if not np.isfinite(states).all() or not np.isfinite(time).all():
        raise ValueError("Nonfinite states or timestamps")
    labels = _integer_labels(labels_raw, classes, "labels")
    overload = None
    if overload_raw is not None:
        if len(overload_raw) != len(time):
            raise ValueError("overload_labels/time row counts do not match")
        overload = _integer_labels(overload_raw, overload_classes, "overload_labels")
    overload_series = None
    if overload_series_raw is not None:
        series = np.asarray(overload_series_raw, dtype=np.float64)
        if series.shape != time.shape or not np.isfinite(series).all():
            raise ValueError("overload_series must be finite with one value per row")
        overload_series = series
        if overload is None:
            raise ValueError("overload_series requires overload_labels (bin convention reference)")
    evasion = None
    if evasion_raw is not None:
        flags = np.asarray(evasion_raw)
        if flags.ndim == 2 and flags.shape[1] == 1:
            flags = flags[:, 0]
        if flags.shape != time.shape or not np.isin(flags, [0, 1]).all():
            raise ValueError("evasion_mask must be 0/1 with one flag per row")
        evasion = flags.astype(bool)
    engagement = None
    if engagement_raw is not None:
        engagement = np.asarray(engagement_raw, dtype=np.float32)
        if engagement.ndim != 2 or engagement.shape[0] != len(time):
            raise ValueError("engagement_states must be [rows,9] matching time")
        if not np.isfinite(engagement).all():
            raise ValueError("engagement_states must be finite")
    ids = ids.astype(str)
    if np.any(ids == ""):
        raise ValueError("Empty track_id")
    scene_ids, scenarios = np.asarray(scene_ids).astype(str), np.asarray(scenarios).astype(str)
    if scene_ids.shape != ids.shape or scenarios.shape != ids.shape or np.any(np.char.strip(scene_ids) == ""):
        raise ValueError("scene_id/scenario must match rows; scene IDs must be nonempty")
    if context_states is not None:
        if not explicit_scenes:
            raise ValueError("Neighbor context requires explicit nonempty scene_id for independent scene grouping")
        context_states = np.asarray(context_states, dtype=np.float32)
        if context_states.ndim != 3 or context_states.shape[0] != len(time) or context_states.shape[2] != 6 or context_states.shape[1] < 1:
            raise ValueError("context_states must be [rows,N,6] with N>=1")
        context_masks = np.ones(context_states.shape[:2], dtype=bool) if context_masks is None else np.asarray(context_masks)
        if context_masks.shape != context_states.shape[:2] or not np.isin(context_masks, [0, 1]).all():
            raise ValueError("context_mask must contain boolean/0/1 values matching [rows,N]")
        context_masks = context_masks.astype(bool)
        if not np.isfinite(context_states[context_masks]).all():
            raise ValueError("Available context states must be finite")
        context_states = np.where(context_masks[..., None], context_states, 0)
    elif context_masks is not None:
        raise ValueError("context_mask requires context_states")
    tracks = []
    for track_id in sorted(np.unique(ids)):
        selection = np.flatnonzero(ids == track_id)
        selection = selection[np.argsort(time[selection], kind="stable")]
        t = time[selection].astype(np.float64)
        if len(t) < minimum_steps:
            raise ValueError(f"Track {track_id} has {len(t)} rows; needs {minimum_steps}")
        if not np.allclose(np.diff(t), dt, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Track {track_id} has duplicate, irregular, or incorrect dt timestamps")
        if len(np.unique(scene_ids[selection])) != 1 or len(np.unique(scenarios[selection])) != 1:
            raise ValueError("Each track must belong to one scene and scenario")
        tracks.append(Track(str(track_id), t, states[selection].astype(np.float32), labels[selection],
                            context_states[selection] if context_states is not None else None,
                            context_masks[selection] if context_masks is not None else None,
                            str(scene_ids[selection[0]]), str(scenarios[selection[0]]),
                            overload[selection] if overload is not None else None,
                            evasion[selection] if evasion is not None else None,
                            engagement[selection] if engagement is not None else None,
                            overload_series[selection] if overload_series is not None else None,
                            bool(overload_censored_raw)))
    if not tracks:
        raise ValueError("No tracks found")
    scene_scenarios = {}
    for track in tracks:
        scene_scenarios.setdefault(track.scene_id, set()).add(track.scenario)
    if any(len(values) > 1 for values in scene_scenarios.values()):
        raise ValueError("Scene members must share one scenario category")
    return tracks


def split_tracks(tracks, train_fraction, validation_fraction, seed, holdout_test=True):
    """Scene-grouped train/validation/test partition.

    holdout_test=False reproduces the paper IV-B protocol: an 80/20
    train/validation split with no separate test partition (the held-out
    validation set is the reporting set).
    """
    groups = {t.scene_id or t.track_id for t in tracks}
    needed = 3 if holdout_test else 2
    if len(groups) < needed:
        raise ValueError(f"Need at least {needed} independent tracks/scenes for the requested partitions")
    ids = np.asarray(sorted(groups))
    ids = np.random.default_rng(seed).permutation(ids)
    reserved = 2 if holdout_test else 1      # groups kept back for val/test
    ntrain = min(max(1, int(np.floor(len(ids) * train_fraction + 1e-12))), len(ids) - reserved)
    if holdout_test:
        nval = min(max(1, int(np.floor(len(ids) * validation_fraction + 1e-12))), len(ids) - ntrain - 1)
    else:
        # Paper IV-B: no test partition, so validation takes every remaining scene.
        nval = len(ids) - ntrain
    grouped = {"train": set(ids[:ntrain]), "validation": set(ids[ntrain:ntrain + nval]), "test": set(ids[ntrain + nval:])}
    members = {}
    for track in tracks:
        members.setdefault(track.scene_id or track.track_id, []).append(str(track.track_id))
    # Preserve the legacy shuffled track order when every group has one member.
    return {phase: [tid for group in ids if group in selected for tid in members[group]]
            for phase, selected in grouped.items()}


def validate_split(tracks, splits, holdout_test=True):
    phases = ("train", "validation", "test") if holdout_test else ("train", "validation")
    flat = [tid for phase in phases for tid in splits[phase]]
    if any(not splits[k] for k in phases):
        raise ValueError("Empty data partition")
    if len(flat) != len(set(flat)) or set(flat) != {t.track_id for t in tracks}:
        raise ValueError("Partitions overlap or fail to cover the dataset")
    lookup = {t.track_id: t.scene_id or t.track_id for t in tracks}
    groups = [{lookup[tid] for tid in splits[phase]} for phase in phases]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
        raise ValueError("Scene members must stay together; scene leakage across partitions")


class Windows(Dataset):
    def __init__(self, tracks, ids, cfg):
        wanted = set(ids)
        self.tracks = [t for t in tracks if t.track_id in wanted]
        self.cfg = cfg
        if cfg.context_entities and any(t.context is None or t.context.shape[1] != cfg.context_entities for t in self.tracks):
            raise ValueError("Data must contain the configured number of context entities")
        if getattr(cfg, "intention_task", "semantic9") == "overload11" and any(t.overload_labels is None for t in self.tracks):
            raise ValueError("Data must contain overload_labels for the overload11 intention task")
        total = cfg.observation_steps + cfg.prediction_steps
        self.index = [(i, start) for i, t in enumerate(self.tracks)
                      for start in range(0, len(t.time) - total + 1, cfg.stride)]
        if not self.index:
            raise ValueError("No complete windows in partition")
        self.class_counts = self._count_classes()
        if getattr(cfg, "balance_intention_classes", False):
            self.index = self._balance_classes(self.index, cfg)
            self.class_counts = self._count_classes()

    def __len__(self):
        return len(self.index)

    def _window_label(self, item):
        i, start = item
        track = self.tracks[i]
        return window_intention_label(track, start, start + self.cfg.observation_steps,
                                      self.cfg.intention_classes,
                                      getattr(self.cfg, "intention_task", "semantic9"))

    def _count_classes(self):
        counts = [0] * self.cfg.intention_classes
        for item in self.index:
            label = self._window_label(item)
            if label >= 0:
                counts[label] += 1
        return counts

    def _balance_classes(self, index, cfg):
        """Equalise the number of windows per intention class (IV-A).

        The paper samples the maneuver classes with balanced coverage, but a
        raw engagement run is dominated by the post-trigger evasion maneuver,
        which leaves the "no maneuver" class almost empty.  Resampling is done
        purely at the window level (trajectories are never modified), inside
        each partition so no scene crosses the train/validation/test boundary:
        over-represented classes are subsampled and under-represented ones are
        repeated with replacement.
        """
        rng = np.random.default_rng(cfg.split_seed)
        by_class = {}
        unlabeled = []
        for item in index:
            label = self._window_label(item)
            (by_class.setdefault(label, []) if label >= 0 else unlabeled).append(item)
        present = [len(v) for v in by_class.values() if v]
        if not present:
            return index
        target = cfg.balance_samples_per_class or int(np.median(present))
        target = max(1, int(target))
        balanced = list(unlabeled)
        for label, items in sorted(by_class.items()):
            if len(items) >= target:
                pick = rng.choice(len(items), size=target, replace=False)
            else:
                pick = rng.choice(len(items), size=target, replace=True)
            balanced.extend(items[int(k)] for k in pick)
        return balanced

    def __getitem__(self, index):
        i, start = self.index[index]
        track, cfg = self.tracks[i], self.cfg
        stop = start + cfg.observation_steps
        observed = track.states[start:stop].copy()
        future = track.states[stop:stop + cfg.prediction_steps, :3].copy()
        # The label uses only the observation window; never future labels.
        label = window_intention_label(track, start, stop, cfg.intention_classes,
                                       getattr(cfg, "intention_task", "semantic9"))
        result = {"observed": torch.from_numpy(observed), "future": torch.from_numpy(future),
                "label": torch.tensor(label), "track_id": track.track_id,
                "scene_id": track.scene_id or track.track_id, "scenario": track.scenario,
                "last_time": torch.tensor(track.time[stop - 1], dtype=torch.float64)}
        if cfg.context_entities:
            result.update(context=torch.from_numpy(track.context[start:stop].transpose(1, 0, 2).copy()),
                          context_mask=torch.from_numpy(track.context_mask[start:stop].T.copy()))
        if track.evasion_mask is not None:
            # Grouping flag: was the evasion trigger already active at the last
            # observed step? Pre/post-trigger errors are reported separately.
            result["evaded"] = torch.tensor(bool(track.evasion_mask[stop - 1]))
        if track.engagement is not None and cfg.engagement_edges:
            # Engagement descriptors r, rdot, eL, eLdot, visibility: they feed
            # the intention query and the temporal node features (Eq. 15/42-53).
            result["engagement"] = torch.from_numpy(track.engagement[start:stop].copy())
        return result

    def labeled_count(self):
        return sum(int(self[i]["label"]) >= 0 for i in range(len(self)))


def rotate_xy(value, heading, inverse=False):
    """World/local planar rotation; heading shape [B], value [B,T,3]."""
    sign = -1 if inverse else 1
    cosine, sine = heading.cos()[:, None], sign * heading.sin()[:, None]
    x = cosine * value[..., 0] + sine * value[..., 1]
    y = -sine * value[..., 0] + cosine * value[..., 1]
    return torch.stack((x, y, value[..., 2]), dim=-1)


def relative_features(observed, frame="world"):
    """Translate positions to the last observation; retain velocity in m/s."""
    position = observed[..., :3] - observed[:, -1:, :3]
    velocity = observed[..., 3:]
    if frame == "yaw":
        heading = torch.atan2(observed[:, -1, 4], observed[:, -1, 3])
        position, velocity = rotate_xy(position, heading), rotate_xy(velocity, heading)
    return torch.cat((position, velocity), dim=-1)


def relative_context(observed, context, frame="world"):
    """Peer [B,N,O,6] expressed in the SAME focal origin/yaw frame."""
    position = context[..., :3] - observed[:, None, -1:, :3]
    velocity = context[..., 3:]
    if frame == "yaw":
        batch, peers, length = context.shape[:3]
        heading = torch.atan2(observed[:, -1, 4], observed[:, -1, 3]).repeat_interleave(peers)
        position = rotate_xy(position.reshape(batch * peers, length, 3), heading).reshape_as(position)
        velocity = rotate_xy(velocity.reshape(batch * peers, length, 3), heading).reshape_as(velocity)
    return torch.cat((position, velocity), -1)


def fit_normalizer(dataset):
    # Streaming moments, fitted solely on training observations.
    total = torch.zeros(6, dtype=torch.float64)
    square = torch.zeros_like(total)
    count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        x = relative_features(sample["observed"].unsqueeze(0), dataset.cfg.relative_frame)[0].double()
        if dataset.cfg.context_entities:
            peers = relative_context(sample["observed"][None], sample["context"][None], dataset.cfg.relative_frame)[0]
            x = torch.cat((x, peers[sample["context_mask"]].double()))
        total += x.sum(0)
        square += (x * x).sum(0)
        count += len(x)
    mean = total / count
    std = (square / count - mean.square()).clamp_min(1e-6).sqrt()
    return {"mean": mean.float(), "std": std.float()}


def generate_tracks(path, count=180, steps=120, dt=0.5, seed=42):
    """Low-speed, independent moving points with smooth civilian motion.

    No other entity, engagement geometry, or pursuit/evasion decision is present.
    """
    if count < 3 or steps < 4 or dt <= 0:
        raise ValueError("Need count>=3, steps>=4, dt>0")
    rng = np.random.default_rng(seed)
    all_states, all_labels, all_ids, all_time = [], [], [], []
    rates = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for index in range(count):
        position = rng.uniform(-30, 30, 3)
        heading = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-0.15, 0.15)
        speed = rng.uniform(2, 6)
        label = index % len(rates)
        remaining = 0
        states, labels = [], []
        yaw_rate, pitch_rate = 0.0, 0.0
        for step in range(steps):
            if remaining == 0:
                if step > 0:
                    label = int(rng.integers(len(rates)))
                remaining = int(rng.integers(25, 45))
                yaw_rate = rates[label][0] * rng.uniform(0.025, 0.06)
                pitch_rate = rates[label][1] * rng.uniform(0.01, 0.025)
            velocity = speed * np.array([np.cos(pitch) * np.cos(heading),
                                         np.cos(pitch) * np.sin(heading), np.sin(pitch)])
            states.append(np.concatenate((position, velocity)))
            labels.append(label)
            next_heading = heading + dt * yaw_rate
            next_pitch = np.clip(pitch + dt * pitch_rate, -0.7, 0.7)
            next_velocity = speed * np.array([np.cos(next_pitch) * np.cos(next_heading),
                                             np.cos(next_pitch) * np.sin(next_heading), np.sin(next_pitch)])
            position = position + dt * (velocity + next_velocity) / 2
            heading, pitch = next_heading, next_pitch
            remaining -= 1
        all_states.extend(states)
        all_labels.extend(labels)
        all_ids.extend([f"track_{index:06d}"] * steps)
        all_time.extend(np.arange(steps) * dt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace data: {path}")
    with path.open("wb") as stream:
        np.savez_compressed(stream, states=np.asarray(all_states, dtype=np.float32),
                            labels=np.asarray(all_labels, dtype=np.int64),
                            track_id=np.asarray(all_ids), time=np.asarray(all_time, dtype=np.float64))
    return {"path": str(path.resolve()), "tracks": count, "steps": steps, "dt": dt,
            "seed": seed, "sha256": fingerprint(path), "domain": "independent_civilian_points"}
