"""Three dimensional CRL-G method transfer environment for MADDPG.

Midcourse updates the estimated encounter time from current geometry. The
deadline is locked for the final 5 s; evaluation then switches to PNG.
"""

import gymnasium as gym
import numpy as np

from xuance.environment import RawMultiAgentEnv
from .coverage import coverage_probability
from .dynamics import FlightState, advance, bounded_vector, rollout
from .geometry import encounter_basis, encounter_time, geometric_tgo
from .guidance import midcourse_command, png_command, yaw_rate_toward
from .prediction import predict_endpoint
from .reachability import ReachabilityTable, directional_polygon
from .reward import process_reward, terminal_reward
from .target import advance_target


class CRLG3DEnv(RawMultiAgentEnv):
    """Fixed six agent API with a sampled active team of three to six."""

    def __init__(self, config):
        super().__init__()
        self.agents = [f"agent_{i}" for i in range(6)]
        self.num_agents = len(self.agents)
        self.obs_dim = 19  # tgo, own miss(2), peers(10), peer mask(5), own mask
        self.action_space = {a: gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
                             for a in self.agents}
        self.observation_space = {
            a: gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for a in self.agents}
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(6 * 11 + 11,),
                                          dtype=np.float32)
        self.decision_dt = float(getattr(config, "decision_dt", 0.5))
        self.physics_dt = float(getattr(config, "physics_dt", 0.1))
        self.lag = float(getattr(config, "actuator_lag", 0.5))
        self.max_speed = float(getattr(config, "max_speed", 11.))
        self.mid_accel = float(getattr(config, "mid_accel", 0.11))
        self.terminal_accel = float(getattr(config, "terminal_accel", 0.22))
        self.pursuer_jerk_max = float(getattr(config, "pursuer_jerk_max", 0.22))
        self.terminal_jerk_max = float(getattr(config, "terminal_jerk_max", 0.44))
        self.target_jerk_max = float(getattr(config, "target_jerk_max", 0.14))
        self.yaw_rate_max = float(getattr(config, "yaw_rate_max", np.pi / 4.))
        self.z_min = float(getattr(config, "z_min", 50.))
        self.z_max = float(getattr(config, "z_max", 350.))
        self.map_size = float(getattr(config, "map_size", 1000.))
        self.max_episode_steps = int(getattr(config, "episode_length", 70))
        self.terminal_seconds = float(getattr(config, "terminal_seconds", 5.))
        self.eval_png = bool(getattr(config, "evaluation_png", False))
        self.nav_constant = float(getattr(config, "navigation_constant", 3.))
        self.alpha = float(getattr(config, "singer_alpha", 0.12))
        self.target_accel = float(getattr(config, "target_accel", 0.07))
        self.target_sigma = float(getattr(config, "singer_sigma", 0.025))
        self.pursuer_initial_speed_min = float(getattr(
            config, "pursuer_initial_speed_min", 7.6))
        self.pursuer_initial_speed_max = float(getattr(
            config, "pursuer_initial_speed_max", 8.4))
        self.target_initial_speed_ratio = float(getattr(
            config, "target_initial_speed_ratio", 1.25))
        self.cloud_size = int(getattr(config, "coverage_samples", 256))
        self.exact_reachability_horizon = float(getattr(
            config, "exact_reachability_horizon_s", 5.))
        self.reachability_directions = int(getattr(config, "reachability_directions", 12))
        self.fixed_team_size = getattr(config, "team_size", None)
        self.fixed_target_mode = getattr(config, "target_mode", None)
        self.meas_position_sigma = float(getattr(config, "position_sigma", 0.75))
        self.meas_velocity_sigma = float(getattr(config, "velocity_sigma", 0.07))
        self.prediction_accel_sigma = float(getattr(config, "prediction_accel_sigma", 0.006))
        self.process_sigma = float(getattr(config, "process_sigma", 0.002))
        self.singer_process_sigma = float(getattr(
            config, "singer_process_sigma",
            self.target_sigma * np.sqrt(2. * self.alpha)))
        self.eval_noise_scale = float(getattr(config, "evaluation_noise_scale", 1.))
        self.reward_scales = {
            "coverage": float(getattr(config, "reward_coverage", 0.6)),
            "prediction": float(getattr(config, "reward_prediction", 0.3)),
            "energy": float(getattr(config, "reward_energy", 0.1)),
        }
        self.miss_normalization = float(getattr(config, "miss_normalization_m", 30.))
        self.terminal_scale = float(getattr(config, "reward_terminal", 1000.))
        self.terminal_bias = float(getattr(config, "terminal_bias_m", 5.))
        self.terminal_own_weight = float(getattr(config, "terminal_own_weight", 0.5))
        self.boundary_penalty = float(getattr(config, "boundary_penalty", 5.))
        self.rng = np.random.default_rng(getattr(config, "env_seed",
                                                  getattr(config, "seed", None)))
        self.reachability = ReachabilityTable(
            max_horizon=self.max_episode_steps * self.decision_dt,
            dt=self.physics_dt, lag=self.lag, accel_limit=self.mid_accel,
            jerk_limit=self.pursuer_jerk_max, max_speed=self.max_speed)
        self._episode_step = 0
        self.reset()

    def _in_bounds(self, position):
        return (0. < position[0] < self.map_size and
                0. < position[1] < self.map_size and
                self.z_min < position[2] < self.z_max)

    def _estimate_encounter_at(self):
        remaining_limit = max(0., self.max_episode_steps * self.decision_dt - self.elapsed)
        tgo = encounter_time(self.pursuers[:self.active_count], self.target,
                             maximum=remaining_limit)
        return self.elapsed + tgo

    def reset(self):
        self._episode_step = 0
        self.elapsed = 0.
        self.active_count = (int(self.fixed_team_size) if self.fixed_team_size is not None
                             else int(self.rng.integers(3, 7)))
        if self.active_count not in (3, 4, 5, 6):
            raise ValueError("team_size must be one of 3, 4, 5, 6")
        self.active = np.zeros(6, dtype=bool)
        self.active[:self.active_count] = True
        self.mode = (str(self.fixed_target_mode) if self.fixed_target_mode is not None
                     else str(self.rng.choice(["CV", "CA", "Singer"])))
        if self.mode not in ("CV", "CA", "Singer"):
            raise ValueError("target_mode must be CV, CA, or Singer")
        target_position = np.array([650., 500., 200.]) + self.rng.normal(
            0., [20., 20., 12.])
        self.initial_pursuer_speed = float(self.rng.uniform(
            self.pursuer_initial_speed_min, self.pursuer_initial_speed_max))
        self.initial_target_speed = (self.target_initial_speed_ratio *
                                     self.initial_pursuer_speed)
        target_heading = np.array([-1., 0., 0.]) + self.rng.normal(
            0., [0., 0.05, 0.03])
        target_velocity = (self.initial_target_speed * target_heading /
                           np.linalg.norm(target_heading))
        target_acceleration = (np.zeros(3) if self.mode == "CV" else
                               bounded_vector(self.rng.normal(
                                   0., self.target_accel / 3., 3), self.target_accel))
        target_yaw = float(np.arctan2(target_velocity[1], target_velocity[0]))
        self.target = FlightState(target_position, target_velocity,
                                  target_acceleration, target_yaw)
        self.pursuers = []
        for i in range(6):
            # Concentrated same-side formation; inactive slots retain zero state.
            if not self.active[i]:
                self.pursuers.append(FlightState(np.zeros(3), np.zeros(3),
                                                 np.zeros(3), 0.))
                continue
            position = np.array([250., 500. + (i - (self.active_count - 1) / 2) * 12.,
                                 200. + ((i % 3) - 1) * 8.])
            position += self.rng.normal(0., [8., 3., 3.])
            velocity = np.array([self.initial_pursuer_speed, 0., 0.])
            self.pursuers.append(FlightState(position, velocity, np.zeros(3), 0.))
        centroid = np.mean([s.position for s in self.pursuers[:self.active_count]], axis=0)
        self.plane, self.forward = encounter_basis(centroid, self.target.position)
        self.target.acceleration = self.plane.T @ (
            self.plane @ self.target.acceleration)
        self.target_command = self.target.acceleration.copy()
        self.encounter_at = self._estimate_encounter_at()
        self.initial_encounter_at = self.encounter_at
        self.terminal_phase = False
        self.termination_reason = None
        self.normal_samples = self.rng.standard_normal((self.cloud_size, 2))
        self._last_measured_velocity = None
        self._last_measurement_time = None
        self._geometry_cache = None
        self.last_coverage = self._geometry()[2]
        self.individual_episode_reward = {a: 0. for a in self.agents}
        self.episode_sub_rewards = {a: {
            name: 0. for name in ("coverage_raw", "prediction_raw", "energy_raw",
                                 "coverage", "prediction", "energy",
                                 "terminal_raw", "terminal", "boundary")}
            for a in self.agents}
        self.episode_command_energy = {a: 0. for a in self.agents}
        self.min_actual_miss = np.full(6, np.inf)
        self.invalid_reason = None
        return self._get_obs(), self._info()

    def _geometry(self):
        if self._geometry_cache is not None:
            return self._geometry_cache
        tgo = max(0., self.encounter_at - self.elapsed)
        measured = self.target.copy()
        if self.eval_png and self.eval_noise_scale > 0:
            measured.position += self.rng.normal(0., self.meas_position_sigma *
                                                 self.eval_noise_scale, 3)
            measured.velocity += self.rng.normal(0., self.meas_velocity_sigma *
                                                 self.eval_noise_scale, 3)
        if self._last_measured_velocity is None:
            measured.acceleration = np.zeros(3)
        else:
            delta = max(self.elapsed - self._last_measurement_time, 1e-6)
            measured.acceleration = bounded_vector(
                (measured.velocity - self._last_measured_velocity) / delta,
                self.target_accel)
        measured.acceleration = self.plane.T @ (
            self.plane @ measured.acceleration)
        self._last_measured_velocity = measured.velocity.copy()
        self._last_measurement_time = self.elapsed
        self.measured_target = measured
        future, covariance = predict_endpoint(
            measured, self.mode, tgo, self.alpha, self.meas_position_sigma,
            self.meas_velocity_sigma, self.prediction_accel_sigma,
            self.singer_process_sigma if self.mode == "Singer" else self.process_sigma,
            actuator_lag=self.lag, physics_dt=self.physics_dt)
        projected_target = self.plane @ future
        projected_covariance = self.plane @ covariance @ self.plane.T
        centers, radii, misses, polygons, zero_endpoints = [], [], [], [], []
        bounds = (np.array([0., 0., self.z_min]),
                  np.array([self.map_size, self.map_size, self.z_max]))
        for i in range(self.active_count):
            state = self.pursuers[i]
            zero_endpoint = rollout(state, np.zeros(3), tgo, dt=self.physics_dt,
                                    lag=self.lag, accel_limit=self.mid_accel,
                                    jerk_limit=self.pursuer_jerk_max,
                                    max_speed=self.max_speed)
            center = self.plane @ zero_endpoint.position
            centers.append(center)
            zero_endpoints.append(zero_endpoint.position)
            radii.append(self.reachability.radius(np.linalg.norm(state.velocity), tgo))
            misses.append((center - projected_target) / self.miss_normalization)
            polygons.append(directional_polygon(
                state, tgo, self.plane, zero_endpoint.position, bounds,
                dt=self.physics_dt, lag=self.lag, accel_limit=self.mid_accel,
                jerk_limit=self.pursuer_jerk_max, max_speed=self.max_speed,
                directions=self.reachability_directions)
                if 0. < tgo <= self.exact_reachability_horizon else None)
        coverage = coverage_probability(projected_target, projected_covariance,
                                        centers, radii, self.normal_samples,
                                        polygons=polygons, plane=self.plane,
                                        zero_endpoints=zero_endpoints,
                                        bounds=bounds)
        self._projection_geometry = {
            "plane_axes_world": self.plane.tolist(),
            "plane_origin_world_m": future.tolist(),
            "target_mean_2d_m": projected_target.tolist(),
            "target_covariance_2d_m2": projected_covariance.tolist(),
            "reach_centers_2d_m": np.asarray(centers).tolist(),
            "reach_radii_m": radii,
            "reach_polygons_2d_m": [vertices.tolist() if vertices is not None else None
                                     for vertices in polygons],
        }
        self._geometry_cache = np.asarray(misses), np.asarray(radii), coverage
        return self._geometry_cache

    def projection_geometry(self):
        """Return the exact 2D geometry used by the current coverage calculation."""
        self._geometry()
        return self._projection_geometry

    def _get_obs(self):
        misses, _, _ = self._geometry()
        obs = {}
        for i, agent in enumerate(self.agents):
            vector = np.zeros(self.obs_dim, dtype=np.float32)
            if self.active[i]:
                vector[0] = (self.encounter_at - self.elapsed) / 35.
                vector[1:3] = misses[i]
                peers = sorted((j for j in range(self.active_count) if j != i),
                               key=lambda j: np.linalg.norm(misses[j] - misses[i]))
                for slot, peer in enumerate(peers):
                    vector[3 + 2 * slot:5 + 2 * slot] = misses[peer] - misses[i]
                    vector[13 + slot] = 1.
                vector[18] = 1.
            obs[agent] = vector
        return obs

    def state(self):
        values = []
        for i, state in enumerate(self.pursuers):
            values.extend(state.position)
            values.extend(state.velocity)
            values.extend(state.acceleration)
            values.append(state.yaw)
            values.append(float(self.active[i]))
        values.extend(self.target.position)
        values.extend(self.target.velocity)
        values.extend(self.target.acceleration)
        values.append(self.target.yaw)
        values.append(self.encounter_at - self.elapsed)
        return np.asarray(values, dtype=np.float32)

    def agent_mask(self):
        return {a: bool(self.active[i]) for i, a in enumerate(self.agents)}

    def avail_actions(self):
        return None

    def _info(self):
        active_misses = [float(x) for x in self.min_actual_miss[:self.active_count]]
        return {
            "individual_episode_rewards": self.individual_episode_reward.copy(),
            "episode_sub_rewards": {a: values.copy()
                                    for a, values in self.episode_sub_rewards.items()},
            "episode_command_energy": self.episode_command_energy.copy(),
            "active_agents": self.active_count,
            "target_mode": self.mode,
            "pursuer_yaw_rad": [float(state.yaw)
                                for state in self.pursuers[:self.active_count]],
            "target_yaw_rad": float(self.target.yaw),
            "initial_pursuer_speed_mps": self.initial_pursuer_speed,
            "initial_target_speed_mps": self.initial_target_speed,
            "encounter_time": float(self.encounter_at),
            "initial_encounter_time": float(self.initial_encounter_at),
            "terminal_phase_locked": self.terminal_phase,
            "termination_reason": self.termination_reason,
            "episode_step": self._episode_step,
            "time_to_go": float(max(0., self.encounter_at - self.elapsed)),
            "geometric_tgo": [geometric_tgo(self.pursuers[i], self.target)
                              for i in range(self.active_count)],
            "closest_approach_m": active_misses,
            "terminal_miss_m": [float(np.linalg.norm(self.pursuers[i].position -
                                                     self.target.position))
                                for i in range(self.active_count)],
            "predicted_projected_miss_m": [float(np.linalg.norm(miss) *
                                                self.miss_normalization)
                                           for miss in self._geometry()[0]],
            "invalid_episode": self.invalid_reason is not None,
            "invalid_reason": self.invalid_reason,
            "calibrated_radius_m": [self.reachability.radius(
                np.linalg.norm(self.pursuers[i].velocity),
                max(0., self.encounter_at - self.elapsed))
                                    for i in range(self.active_count)],
            "ideal_radius_m": 0.5 * self.mid_accel *
                              max(0., self.encounter_at - self.elapsed) ** 2,
        }

    def step(self, actions_dict):
        remaining = self.encounter_at - self.elapsed
        interval = min(self.decision_dt, max(remaining, 0.))
        if interval <= 1e-9:
            raise RuntimeError("step called after encounter time")
        if remaining <= self.terminal_seconds:
            self.terminal_phase = True
        use_png = self.eval_png and self.terminal_phase
        commands = []
        yaw_rates = []
        for i, agent in enumerate(self.agents[:self.active_count]):
            if use_png:
                command = png_command(self.pursuers[i], self.measured_target,
                                      self.nav_constant, self.terminal_accel)
            else:
                command = midcourse_command(actions_dict[agent], self.plane,
                                             self.mid_accel)
            commands.append(command)
            relative = self.target.position - self.pursuers[i].position
            desired_yaw = np.arctan2(relative[1], relative[0])
            yaw_rates.append(yaw_rate_toward(self.pursuers[i].yaw, desired_yaw,
                                              interval, self.yaw_rate_max))
        steps = int(np.ceil(interval / self.physics_dt))
        sub_dt = interval / steps
        advanced = 0.
        for _ in range(steps):
            target_heading = np.arctan2(self.target.velocity[1],
                                        self.target.velocity[0])
            target_yaw_rate = yaw_rate_toward(self.target.yaw, target_heading,
                                               sub_dt, self.yaw_rate_max)
            self.target, self.target_command = advance_target(
                self.target, self.target_command, self.mode, sub_dt, self.rng,
                self.target_accel, self.alpha, self.target_sigma, lag=self.lag,
                jerk_limit=self.target_jerk_max, yaw_rate=target_yaw_rate,
                yaw_rate_limit=self.yaw_rate_max, max_speed=self.max_speed,
                maneuver_plane=self.plane)
            for i, command in enumerate(commands):
                self.pursuers[i] = advance(self.pursuers[i], command, sub_dt,
                                           self.lag,
                                           self.terminal_accel if use_png else self.mid_accel,
                                           jerk_limit=(self.terminal_jerk_max if use_png
                                                       else self.pursuer_jerk_max),
                                           yaw_rate=yaw_rates[i],
                                           yaw_rate_limit=self.yaw_rate_max,
                                           max_speed=self.max_speed)
                miss = np.linalg.norm(self.pursuers[i].position - self.target.position)
                self.min_actual_miss[i] = min(self.min_actual_miss[i], miss)
            if not self._in_bounds(self.target.position):
                self.invalid_reason = "target_boundary"
            elif any(not self._in_bounds(self.pursuers[i].position)
                     for i in range(self.active_count)):
                self.invalid_reason = "pursuer_boundary"
            advanced += sub_dt
            if self.invalid_reason:
                break
        self.elapsed += advanced
        self._episode_step += 1
        reached_encounter = self.elapsed >= self.encounter_at - 1e-8
        reached_step_limit = self._episode_step >= self.max_episode_steps
        if not (self.invalid_reason or reached_encounter or reached_step_limit or
                self.terminal_phase):
            self.encounter_at = self._estimate_encounter_at()
            reached_encounter = self.encounter_at - self.elapsed <= 1e-8
        self.termination_reason = (self.invalid_reason or
                                   ("max_steps" if reached_step_limit else None) or
                                   ("encounter" if reached_encounter else None))
        self._geometry_cache = None
        misses, _, coverage = self._geometry()
        coverage_delta = coverage - self.last_coverage
        process, parts = process_reward(
            coverage, misses, commands, self.reward_scales,
            self.terminal_accel if use_png else self.mid_accel)
        self.last_coverage = coverage
        finished = self.termination_reason is not None
        if finished and self.invalid_reason is None:
            actual = [np.linalg.norm(self.pursuers[i].position - self.target.position)
                      for i in range(self.active_count)]
            terminal = terminal_reward(actual, self.terminal_own_weight,
                                       1. - self.terminal_own_weight,
                                       self.terminal_bias)
            for i in range(self.active_count):
                weighted_terminal = self.terminal_scale * terminal[i]
                process[i] += weighted_terminal
                parts[i]["terminal_raw"] = terminal[i]
                parts[i]["terminal"] = weighted_terminal
        elif self.invalid_reason:
            process = [value - self.boundary_penalty for value in process]
            for part in parts:
                part["boundary"] = -self.boundary_penalty
        rewards = {a: float(process[i]) if self.active[i] else 0.
                   for i, a in enumerate(self.agents)}
        for agent, value in rewards.items():
            self.individual_episode_reward[agent] += value
        for i, agent in enumerate(self.agents[:self.active_count]):
            for name, value in parts[i].items():
                self.episode_sub_rewards[agent][name] += float(value)
            self.episode_command_energy[agent] += float(np.dot(commands[i], commands[i]) *
                                                         advanced)
        info = self._info()
        info.update({"coverage_probability": coverage,
                     "coverage_delta": coverage_delta,
                     "reward_components": parts,
                     "guidance_phase": "PNG" if use_png else "CRLG"})
        terminated = {a: bool(finished) for a in self.agents}
        return self._get_obs(), rewards, terminated, False, info

    def close(self):
        pass

    def render(self, *args, **kwargs):
        """The training environment is headless; evaluation writes metrics."""
        return None
