"""A 3-D point-mass engagement for the paper's one-target/one-defender setting.

The paper specifies the kinematics and terminal anti-LOS evasion, but omits the
defender guidance law and exact maneuver-primitive schedule. Those choices are
explicit here so generated datasets remain reproducible.
"""

from dataclasses import dataclass

import numpy as np


G = 9.81
INTENT_NAMES = (
    "none", "left", "right", "climb", "dive", "left_climb",
    "right_climb", "left_dive", "right_dive",
)
_MODE_SIGNS = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
     [1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=np.float64,
)


@dataclass(frozen=True)
class ScenarioConfig:
    dt: float = 0.1
    duration: float = 24.0
    initial_range_min: float = 4000.0
    initial_range_max: float = 8000.0
    target_speed_min: float = 250.0
    target_speed_max: float = 350.0
    defender_speed_min: float = 380.0
    defender_speed_max: float = 450.0
    target_accel_max: float = 9.0 * G
    defender_accel_max: float = 15.0 * G
    fov_half_angle_deg: float = 60.0
    initial_visible_fraction: float = 1.0
    maneuver_min_seconds: float = 2.0
    maneuver_max_seconds: float = 5.0
    evasion_trigger_fraction: float = 2.0 / 3.0
    evasion_weight: float = 0.7


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def _normal_basis(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = _unit(velocity)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    if np.linalg.norm(left) < 1e-8:
        left = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    left = _unit(left)
    climb = _unit(np.cross(forward, left))
    return left, climb


def advance_point_mass(position: np.ndarray, velocity: np.ndarray,
                       command: np.ndarray, acceleration_limit: float,
                       dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply velocity-normal acceleration while preserving airspeed."""
    if dt <= 0 or acceleration_limit < 0:
        raise ValueError("dt must be positive and acceleration_limit nonnegative")
    speed = float(np.linalg.norm(velocity))
    direction = _unit(velocity)
    acceleration = np.asarray(command, dtype=np.float64)
    acceleration = acceleration - np.dot(acceleration, direction) * direction
    norm = float(np.linalg.norm(acceleration))
    if norm > acceleration_limit:
        acceleration *= acceleration_limit / norm
    next_direction = _unit(direction + acceleration * dt / speed)
    next_velocity = speed * next_direction
    next_position = position + 0.5 * (velocity + next_velocity) * dt
    return next_position, next_velocity, acceleration


def _mode_command(mode: int, velocity: np.ndarray, magnitude: float) -> np.ndarray:
    left, climb = _normal_basis(velocity)
    lateral, vertical = _MODE_SIGNS[mode]
    direction = lateral * left + vertical * climb
    norm = float(np.linalg.norm(direction))
    return magnitude * direction / norm if norm else np.zeros(3)


def _intent_from_acceleration(acceleration: np.ndarray,
                              velocity: np.ndarray) -> int:
    """Give a signed semantic label to the actual applied normal acceleration."""
    left, climb = _normal_basis(velocity)
    lateral = float(np.dot(acceleration, left))
    vertical = float(np.dot(acceleration, climb))
    threshold = 0.75 * G
    signs = (int(lateral > threshold) - int(lateral < -threshold),
             int(vertical > threshold) - int(vertical < -threshold))
    return next(i for i, pair in enumerate(_MODE_SIGNS) if tuple(pair) == signs)


def _sample_initial_state(config: ScenarioConfig, rng: np.random.Generator):
    range0 = rng.uniform(config.initial_range_min, config.initial_range_max)
    yaw = rng.uniform(-np.pi, np.pi)
    pitch = rng.uniform(-np.pi / 6, np.pi / 6)
    los = np.array([np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    target_position = np.array([0.0, 0.0, rng.uniform(5000.0, 7000.0)])
    defender_position = target_position - range0 * los
    desired_visible = rng.random() < config.initial_visible_fraction
    for _ in range(1000):
        target_heading = rng.uniform(-np.pi, np.pi)
        target_elevation = rng.uniform(-np.pi / 12, np.pi / 12)
        target_direction = np.array([
            np.cos(target_elevation) * np.cos(target_heading),
            np.cos(target_elevation) * np.sin(target_heading),
            np.sin(target_elevation),
        ])
        visible = np.dot(-los, target_direction) >= np.cos(
            np.deg2rad(config.fov_half_angle_deg))
        if visible == desired_visible:
            break
    else:
        raise RuntimeError("Could not sample the requested FOV geometry")
    # Reproduction assumption: the defender initially flies toward the target.
    defender_direction = _unit(los + rng.normal(0.0, 0.15, size=3))
    return (target_position, target_direction * rng.uniform(
                config.target_speed_min, config.target_speed_max),
            defender_position, defender_direction * rng.uniform(
                config.defender_speed_min, config.defender_speed_max), range0)


def simulate_track(config: ScenarioConfig, seed: int) -> dict[str, np.ndarray]:
    """Return synchronized state and intention sequences for one engagement."""
    if config.dt <= 0 or config.duration <= 0:
        raise ValueError("dt and duration must be positive")
    if not 0.0 <= config.initial_visible_fraction <= 1.0:
        raise ValueError("initial_visible_fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    tp, tv, dp, dv, range0 = _sample_initial_state(config, rng)
    steps = int(round(config.duration / config.dt))
    target = np.zeros((steps + 1, 9), dtype=np.float32)
    defender = np.zeros_like(target)
    labels = np.zeros(steps + 1, dtype=np.int64)
    fov = np.zeros(steps + 1, dtype=np.bool_)
    evading = np.zeros(steps + 1, dtype=np.bool_)
    mode = int(rng.integers(len(INTENT_NAMES)))
    magnitude = rng.uniform(1.0, 9.0) * G
    segment_end = 0
    triggered = False

    for step in range(steps + 1):
        relative = dp - tp
        fov[step] = np.dot(_unit(relative), _unit(tv)) >= np.cos(
            np.deg2rad(config.fov_half_angle_deg))
        target[step, :6] = np.concatenate((tp, tv))
        defender[step, :6] = np.concatenate((dp, dv))
        if step == steps:
            break
        if step >= segment_end:
            mode = int(rng.integers(len(INTENT_NAMES)))
            magnitude = rng.uniform(1.0, 9.0) * G
            segment_end = step + max(1, int(round(rng.uniform(
                config.maneuver_min_seconds, config.maneuver_max_seconds) / config.dt)))
        command = _mode_command(mode, tv, magnitude)
        distance = float(np.linalg.norm(tp - dp))
        triggered |= distance <= config.evasion_trigger_fraction * range0
        evading[step] = triggered
        if triggered and fov[step]:
            # Paper anti-LOS command, projected to the target's velocity-normal plane.
            anti_los = _unit(tp - dp)
            direction = _unit(tv)
            evasive = anti_los - np.dot(anti_los, direction) * direction
            if np.linalg.norm(evasive) > 1e-8:
                evasive = _unit(evasive) * config.target_accel_max
                command = ((1.0 - config.evasion_weight) * command +
                           config.evasion_weight * evasive)
        next_tp, next_tv, applied_target = advance_point_mass(
            tp, tv, command, config.target_accel_max, config.dt)
        # Acceleration at t+1 describes the completed t -> t+1 interval.
        # Storing it at t would expose a future command to predictors.
        labels[step + 1] = _intent_from_acceleration(applied_target, tv)
        target[step + 1, 6:] = applied_target
        # Reproduction assumption: the defender turns toward current LOS.
        defender_command = _unit(tp - dp) * config.defender_accel_max
        next_dp, next_dv, applied_defender = advance_point_mass(
            dp, dv, defender_command, config.defender_accel_max, config.dt)
        defender[step + 1, 6:] = applied_defender
        tp, tv, dp, dv = next_tp, next_tv, next_dp, next_dv

    evading[-1] = triggered
    return {"target": target, "defender": defender, "intent": labels,
            "fov": fov, "evading": evading,
            "time": np.arange(steps + 1, dtype=np.float32) * config.dt,
            "seed": np.array(seed, dtype=np.int64),
            "initial_range": np.array(range0, dtype=np.float32)}
