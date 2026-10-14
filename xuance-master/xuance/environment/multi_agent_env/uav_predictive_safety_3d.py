"""Substep action filter with swept-volume checks and a certified brake tail.

The policy action remains the action of the composite (filtered) environment.
Certification is conditional on this simulator's dynamics and static geometry;
an infeasible state is reported, never hidden by clipping or teleportation.
"""
import numpy as np


def _limit(vectors, maximum):
    return vectors * np.minimum(1.0, maximum / np.maximum(
        np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12))


def swept_pair_distance(first, second):
    """Minimum separation over all linearly interpolated substep segments."""
    relative = first - second
    change = np.diff(relative, axis=-2)
    start = relative[..., :-1, :]
    fraction = np.clip(-np.sum(start * change, axis=-1) / np.maximum(
        np.sum(change * change, axis=-1), 1e-12), 0.0, 1.0)
    closest = start + fraction[..., None] * change
    return np.min(np.linalg.norm(closest, axis=-1), axis=-1)


class PredictiveSafetyFilter3D:
    def __init__(self, env, config):
        self.env = env
        self.margin = float(getattr(config, 'safety_margin', 2.0))
        self.horizon = float(getattr(config, 'safety_horizon_seconds', 8.0))
        self.dt = env.decision_dt / max(1, int(np.ceil(env.decision_dt / env.physics_dt)))
        lag = env.acceleration_lag
        if (not np.isfinite(self.margin) or self.margin <= 0 or
                not np.isfinite(self.horizon) or self.horizon < self.dt or
                not 0 < self.dt < 2 * lag or env.uav_jerk_max <= 0):
            raise ValueError('Safety filter needs positive margin/horizon/jerk and dt < 2*lag')
        self.steps = int(np.ceil(self.horizon / self.dt))
        # Unsaturated backup: u=-v/lag-a. z=(v,a), z_next=A*z.
        matrix = np.array([[1., self.dt],
                           [-self.dt / lag ** 2, 1. - 2. * self.dt / lag]])
        self.lyapunov = np.linalg.solve(
            np.eye(4) - np.kron(matrix.T, matrix.T), np.eye(2).reshape(-1)).reshape(2, 2)
        self.lyapunov = (self.lyapunov + self.lyapunov.T) / 2
        eigenvalues = np.linalg.eigvalsh(self.lyapunov)
        if np.min(eigenvalues) <= 0:
            raise ValueError('Brake feedback has no positive terminal certificate')
        inverse = np.linalg.inv(self.lyapunov)
        contraction = np.sqrt(max(0., 1. - 1. / np.max(eigenvalues)))
        travel_row = np.array([self.dt, .5 * self.dt ** 2])
        self.tail_gain = np.sqrt(travel_row @ inverse @ travel_row) / (1. - contraction)
        rows = np.array([[1., 0.], [0., 1.], [-1. / lag, -1.],
                         [-1. / lag ** 2, -2. / lag]])
        self.tail_bounds = np.sqrt(np.einsum('ij,jk,ik->i', rows, inverse, rows))

    def brake(self, velocity, acceleration):
        return _limit(-velocity / self.env.acceleration_lag - acceleration, self.env.max_accel)

    def _rollout(self, commands):
        """Vectorized copy of advance_translation; no geometry clipping."""
        env, dt = self.env, self.dt
        count = commands.shape[1]
        position = np.repeat(env.uav_positions[:, None, :], count, axis=1)
        velocity = np.repeat(env.uav_velocities[:, None, :], count, axis=1)
        acceleration = np.repeat(env.uav_accelerations[:, None, :], count, axis=1)
        path = [position.copy()]
        for index in range(self.steps):
            p, v, a = (value.astype(np.float64) for value in (position, velocity, acceleration))
            command = commands if index == 0 else self.brake(v, a)
            command = _limit(command, env.max_accel)
            jerk = _limit((command - a) / env.acceleration_lag, env.uav_jerk_max)
            position = (p + v * dt + .5 * a * dt ** 2).astype(np.float32)
            velocity = _limit(v + a * dt, env.uav_max_speed).astype(np.float32)
            acceleration = _limit(a + jerk * dt, env.max_accel).astype(np.float32)
            path.append(position.copy())
        path = np.stack(path, axis=2)
        z = np.stack([velocity, acceleration], axis=-1).astype(np.float64)
        energy = np.einsum('...ki,ij,...kj->...', z, self.lyapunov, z)
        scale = np.sqrt(np.maximum(energy, 0.))
        bounds = scale[..., None] * self.tail_bounds
        linear_tail = np.all(bounds <= np.array([
            env.uav_max_speed, env.max_accel, env.max_accel, env.uav_jerk_max]), axis=-1)
        # The Lyapunov bound encloses every subsequent backup position.
        radius = self.tail_gain * scale + .01  # numerical reserve, separate from flight margin
        radius[~linear_tail] = np.inf
        return path, radius

    def _individual_safe(self, path, tail_radius):
        env = self.env
        padding = env.uav_radius + self.margin
        lower = np.array([padding, padding, env.z_min + padding])
        upper = np.array([env.map_size - padding, env.map_size - padding, env.z_max - padding])
        safe = np.all((path > lower) & (path < upper), axis=(-1, -2))
        endpoint = path[..., -1, :]
        safe &= np.all((endpoint - tail_radius[..., None] > lower) &
                       (endpoint + tail_radius[..., None] < upper), axis=-1)
        if not len(env.buildings):
            return safe
        boxes = env.buildings.astype(np.float64)
        lo = np.stack([boxes[:, 0] - padding, boxes[:, 2] - padding,
                       np.full(len(boxes), -np.inf)], axis=1)
        hi = np.stack([boxes[:, 1] + padding, boxes[:, 3] + padding,
                       boxes[:, 4] + padding], axis=1)
        for agent in range(env.num_agents):
            tracks = path[agent].astype(np.float64)
            # Broad phase retains every box which any candidate segment can cross.
            nearby = np.all((lo <= tracks.max(axis=(0, 1))) &
                            (hi >= tracks.min(axis=(0, 1))), axis=1)
            start, change = tracks[:, :-1, None, :], np.diff(tracks, axis=1)[:, :, None, :]
            if np.any(nearby):
                parallel = np.abs(change) < 1e-12
                divisor = np.where(parallel, 1., change)
                first = (lo[nearby] - start) / divisor
                last = (hi[nearby] - start) / divisor
                entry = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
                leave = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
                within = np.all(~parallel | ((start >= lo[nearby]) & (start <= hi[nearby])), axis=-1)
                hit = within & (np.maximum(entry, 0.) <= np.minimum(leave, 1.))
                safe[agent] &= ~np.any(hit, axis=(-1, -2))
            distance = np.linalg.norm(np.maximum(np.maximum(
                lo[None] - endpoint[agent, :, None], endpoint[agent, :, None] - hi[None]), 0.), axis=-1)
            safe[agent] &= np.all(distance > tail_radius[agent, :, None], axis=-1)
        return safe

    def _pair_safe(self, paths, radii, first, second, a, b):
        clearance = self.env.uav_collision_distance + 2 * self.margin
        return (swept_pair_distance(paths[first, a], paths[second, b]) > clearance and
                np.linalg.norm(paths[first, a, -1] - paths[second, b, -1]) >
                clearance + radii[first, a] + radii[second, b])

    def _joint_safe(self, paths, radii, safe, selected):
        if not all(safe[i, selected[i]] for i in range(self.env.num_agents)):
            return False
        return all(self._pair_safe(paths, radii, i, j, selected[i], selected[j])
                   for i in range(self.env.num_agents) for j in range(i))

    def filter(self, proposed):
        env = self.env
        proposed = np.asarray(proposed, dtype=np.float64)
        if proposed.shape != (env.num_agents, 3) or not np.all(np.isfinite(proposed)):
            raise ValueError('Safety filter requires finite three-axis commands')
        proposed = _limit(proposed, env.max_accel)
        brake = self.brake(env.uav_velocities.astype(np.float64),
                           env.uav_accelerations.astype(np.float64))
        commands = np.stack([proposed, brake], axis=1)
        paths, radii = self._rollout(commands)
        safe = self._individual_safe(paths, radii)
        nominal = np.zeros(env.num_agents, dtype=int)
        if self._joint_safe(paths, radii, safe, nominal):
            return proposed.astype(np.float32), dict(certified=True, intervened=[False] * env.num_agents)
        selected = np.ones(env.num_agents, dtype=int)
        if not self._joint_safe(paths, radii, safe, selected):
            # No certified backup: disclose this and execute best-effort braking.
            # Never turn an unavoidable collision into a fake successful certificate.
            return brake.astype(np.float32), dict(certified=False, intervened=(
                np.linalg.norm(brake - proposed, axis=-1) > 1e-6).tolist())
        axes = np.concatenate([np.eye(3), -np.eye(3)]) * env.max_accel
        commands = np.concatenate([commands, np.zeros_like(proposed[:, None]),
                                   proposed[:, None] * .5,
                                   np.broadcast_to(axes, (env.num_agents, 6, 3))], axis=1)
        paths, radii = self._rollout(commands)
        safe = self._individual_safe(paths, radii)
        # Preserve a certified joint solution while changing one agent at a time.
        # This is the closest feasible candidate in this set, not a global MPC optimum.
        for agent in range(env.num_agents):
            cost = np.linalg.norm(commands[agent] - proposed[agent], axis=-1)
            for candidate in np.argsort(cost, kind='stable'):
                if safe[agent, candidate] and all(self._pair_safe(
                        paths, radii, agent, other, candidate, selected[other])
                        for other in range(env.num_agents) if other != agent):
                    selected[agent] = candidate
                    break
        result = commands[np.arange(env.num_agents), selected].astype(np.float32)
        return result, dict(certified=True, intervened=(
            np.linalg.norm(result - proposed, axis=-1) > 1e-6).tolist())
