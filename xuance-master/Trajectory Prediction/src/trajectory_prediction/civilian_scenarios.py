"""Low-speed pedestrians/carts following fixed routes and right-of-way rules."""
from pathlib import Path

import numpy as np

from .data import fingerprint


SCENARIOS = ("crossing", "shared_aisle", "merge", "ramp_crossing", "priority_change", "separated_control")


def _routes(name):
    if name == "merge":
        return np.array([[[-14, -5, 0], [0, 0, 0], [14, -1, 0]],
                         [[-14, 5, 0], [0, 0, 0], [14, 1, 0]]], dtype=float)
    if name == "shared_aisle":
        return np.array([[[-14, -2, 0], [-2, 0, 0], [2, 0, 0], [14, -2, 0]],
                         [[-14, 2, 0], [-2, 0, 0], [2, 0, 0], [14, 2, 0]]], dtype=float)
    if name == "ramp_crossing":
        return np.array([[[-14, 0, 0], [0, 0, 1.5], [14, 0, 3]],
                         [[0, -14, 1.5], [0, 0, 1.5], [0, 14, 1.5]]], dtype=float)
    if name == "separated_control":
        return np.array([[[-14, 0, 0], [0, 0, 0], [14, 0, 0]],
                         [[-14, 12, 0], [0, 12, 0], [14, 12, 0]]], dtype=float)
    return np.array([[[-14, 0, 0], [0, 0, 0], [14, 0, 0]],
                     [[0, -14, 0], [0, 0, 0], [0, 14, 0]]], dtype=float)


def _motion_labels(states, dt):
    velocity = states[:, 3:]
    yaw = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    yaw_rate = np.r_[0., np.diff(yaw) / dt]
    turning = np.where(yaw_rate > .015, 1, np.where(yaw_rate < -.015, -1, 0))
    vertical = np.where(velocity[:, 2] > .025, 1, np.where(velocity[:, 2] < -.025, -1, 0))
    turning[np.linalg.norm(velocity[:, :2], axis=1) < .15] = 0
    mapping = {(0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, 1): 3, (0, -1): 4,
               (1, 1): 5, (-1, 1): 6, (1, -1): 7, (-1, -1): 8}
    return np.asarray([mapping[a, b] for a, b in zip(turning, vertical)], dtype=np.int64)


def simulate_scenes(path, scenes=60, steps=80, dt=.5, seed=42, measurement_dropout=0.):
    if not isinstance(scenes, int) or scenes < 3 or not isinstance(steps, int) or steps < 4:
        raise ValueError("Need at least three independent scenes and four steps")
    if not np.isfinite(dt) or dt <= 0 or not 0 <= measurement_dropout < 1:
        raise ValueError("Invalid sampling interval or measurement dropout")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    records = {name: [] for name in ("track_id", "scene_id", "scenario", "time", "states", "labels", "context_states", "context_mask")}
    diagnostics = []
    for scene in range(scenes):
        name = SCENARIOS[scene % len(SCENARIOS)]
        routes = _routes(name)
        angle = rng.uniform(-np.pi, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        origin = rng.uniform([-4, -4, 0], [4, 4, .5])
        routes = routes @ rotation.T + origin
        center = origin + np.array([0., 0., 1.5 if name == "ramp_crossing" else 0.])
        positions, velocities = routes[:, 0].copy(), np.zeros((2, 3))
        waypoints = np.ones(2, dtype=int)
        speeds = rng.uniform(.9, 1.5, 2)
        delays = rng.uniform(0, 2.5, 2)
        priority = int(rng.integers(2))
        initial_priority = priority
        switch_time = rng.uniform(6, 8)
        owner, cleared, entered = None, np.zeros(2, dtype=bool), np.zeros(2, dtype=bool)
        states, waits = np.zeros((steps, 2, 6)), np.zeros(2, dtype=int)
        for step in range(steps):
            states[step] = np.concatenate((positions, velocities), axis=1)
            clock = step * dt
            if name == "priority_change" and clock >= switch_time and owner is None and not cleared.any():
                priority = 1 - initial_priority
            distance = np.linalg.norm(positions - center, axis=1)
            if owner is not None:
                entered[owner] |= distance[owner] < 2.5
                if entered[owner] and distance[owner] > 6 and waypoints[owner] >= 2:
                    cleared[owner], owner = True, None
            if name != "separated_control" and owner is None:
                candidates = [i for i in (priority, 1 - priority) if not cleared[i] and distance[i] < 6
                              and (i == priority or cleared[priority])]
                if candidates:
                    owner = candidates[0]
            next_velocity = velocities.copy()
            for agent in range(2):
                goal = routes[agent, waypoints[agent]]
                remaining = np.linalg.norm(goal - positions[agent])
                if remaining < .35 and waypoints[agent] < len(routes[agent]) - 1:
                    waypoints[agent] += 1
                    goal = routes[agent, waypoints[agent]]
                    remaining = np.linalg.norm(goal - positions[agent])
                desired = (goal - positions[agent]) / max(remaining, 1e-8) * min(speeds[agent], np.sqrt(2 * remaining))
                blocked = (name != "separated_control" and not cleared[agent] and distance[agent] < 6
                           and ((owner is not None and owner != agent)
                                or (owner is None and agent != priority and not cleared[priority])))
                if clock < delays[agent] or blocked or (remaining < .1 and waypoints[agent] == len(routes[agent]) - 1):
                    desired = np.zeros(3)
                    waits[agent] += int(blocked)
                change = desired - velocities[agent]
                next_velocity[agent] += change * min(1., dt / max(np.linalg.norm(change), 1e-8))
            # Trapezoidal position integration; acceleration magnitude <= 1 m/s².
            positions += dt * (velocities + next_velocity) / 2
            velocities = next_velocity
        separation = np.linalg.norm(states[:, 0, :3] - states[:, 1, :3], axis=1)
        diagnostics.append(dict(scene_id=f"scene_{scene:06d}", scenario=name,
                                initial_priority=initial_priority, minimum_separation_m=float(separation.min()),
                                clearance_below_075m_fraction=float((separation < .75).mean()), waiting_steps=waits.tolist()))
        for agent in range(2):
            availability = rng.random((steps, 1)) >= measurement_dropout
            records["track_id"].extend([f"scene_{scene:06d}_agent_{agent}"] * steps)
            records["scene_id"].extend([f"scene_{scene:06d}"] * steps)
            records["scenario"].extend([name] * steps)
            records["time"].extend(np.arange(steps) * dt)
            records["states"].extend(states[:, agent])
            records["labels"].extend(_motion_labels(states[:, agent], dt))
            records["context_states"].extend(states[:, 1 - agent, None])
            records["context_mask"].extend(availability)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(file, **{name: np.asarray(values, dtype=np.float32 if name in {"states", "context_states"}
                                    else np.float64 if name == "time" else np.int64 if name == "labels"
                                    else bool if name == "context_mask" else str) for name, values in records.items()})
    return dict(path=str(path.resolve()), seed=seed, scenes=scenes, tracks=2 * scenes, steps=steps, dt=dt,
                sha256=fingerprint(path), domain="civilian_fixed_route_right_of_way", context_entities=1,
                measurement_dropout=measurement_dropout, max_speed_mps=1.5, max_acceleration_mps2=1.,
                scenario_types=list(SCENARIOS), diagnostics=diagnostics,
                prediction_inputs="measured focal/peer histories only; route plans, priority, future states excluded",
                safety_claim="Scripted right-of-way benchmark; clearance diagnostics are recorded, not a guaranteed collision-free controller")
