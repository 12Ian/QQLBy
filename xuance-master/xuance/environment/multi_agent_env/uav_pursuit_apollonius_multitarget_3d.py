"""Multi-target 3D cooperative pursuit (Chapter 4). Extends the single-target env to M
evaders via dynamic balanced pursuer->target allocation + per-sub-team Apollonius
solid-angle encirclement and closure reward. Reuses the single-target dynamics,
geometry, evader policy, and reward helpers; overrides reset/step/obs/state."""
from collections import deque
import numpy as np
import gymnasium as gym
from . import geometry3d as g3
from . import apollonius3d as ap3
from .task_allocation import assign_pursuers_to_targets, apollonius_coverage_cost
from .uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv

FULL_SOLID = 4.0 * np.pi


class UAVPursuitApolloniusMultiTarget3DEnv(UAVPursuitApollonius3DEnv):
    def __init__(self, config):
        self.num_targets = int(getattr(config, "num_targets", 2))
        # dynamic (default) re-optimises the pursuer->target assignment every step;
        # fixed keeps the reset assignment and only reassigns pursuers whose target died.
        self.dynamic_alloc = bool(getattr(config, "dynamic_alloc", True))
        self.strict_capture = bool(getattr(config, "strict_capture", False))
        self.last_guide_dist = {}
        self.last_sub_dists = {}
        super().__init__(config)
        M, N = self.num_targets, self.num_agents
        # obs adds the M-1 other targets' relative positions; state carries all M targets
        base_obs = 17 + 3 * (N - 1) + self.num_radar_rays + 3 * (M - 1)
        self.obs_dim = base_obs + (self.obstacle_gat_k * self.obstacle_gat_feat_dim
                                   if self.use_obstacle_gat else 0)
        self.observation_space = {a: gym.spaces.Box(-1.0, 1.0, (self.obs_dim,), np.float32)
                                  for a in self.agents}
        self.state_dim = 9 * N + 6 * M
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)

        self.target_positions = np.zeros((M, 3), np.float32)
        self.target_yaws = np.zeros(M, np.float32)
        self.target_pitches = np.zeros(M, np.float32)
        self.target_speeds = np.full(M, self.target_min_speed, np.float32)
        self.target_alive = np.ones(M, dtype=bool)
        self.assign = np.full(N, -1, dtype=int)
        self.target_trails = [deque(maxlen=100) for _ in range(M)]
        # Apollonius-based allocation. Default False so every published multi-target run
        # reproduces byte-for-byte; the distance criterion stays the baseline arm.
        self.apollonius_alloc = bool(getattr(config, "apollonius_alloc", False))
        self.sub_fields = [None] * M
        self.last_min_dist = np.full(M, 1e9, np.float32)


    def _alloc_cost(self):
        """Cost matrix for allocation. Returns None to fall back to plain distance.

        With apollonius_alloc on, pursuers are matched to the target whose escape directions
        they can actually dominate, which is the quantity the capture criterion measures --
        distance is only a proxy for it and a poor one when a pursuer sits on the wrong side.
        """
        if not getattr(self, "apollonius_alloc", False):
            return None
        alive_idx = np.where(self.target_alive)[0]
        if len(alive_idx) == 0:
            return None
        dirs = self.escape_dirs
        free_all, r_f_all = {}, {}
        for m in alive_idx:
            T = self.target_positions[m]
            free_all[m] = g3.ray_free_distance(T, dirs, self.map_size, self.z_min,
                                               self.z_max, self.buildings)
            self.target_speed = float(self.target_speeds[m])
            r_f_all[m] = self._compute_r_f()
        return apollonius_coverage_cost(
            self.uav_positions, self.target_positions, self.target_speeds,
            self.uav_max_speed, dirs, free_all, self.catch_radius, r_f_all,
            self.target_alive)

    # ---- per-target Apollonius field + guide points for a sub-team ----
    def _field_and_guides(self, m):
        sub = np.where(self.assign == m)[0]
        T = self.target_positions[m]
        guides = {}
        if len(sub) == 0:
            return None, sub, guides
        self.target_speed = float(self.target_speeds[m])   # _compute_r_f reads this
        dirs = self.escape_dirs
        free = g3.ray_free_distance(T, dirs, self.map_size, self.z_min, self.z_max, self.buildings)
        r_f = self._compute_r_f(target_pos=T, target_speed=self.target_speeds[m])
        field = ap3.compute_escape_field_3d(
            T, self.target_speeds[m], self.uav_positions[sub], self.uav_max_speed,
            dirs, free, self.catch_radius, r_f,
            criterion_mode=self.criterion_mode, buildings=self.buildings)
        esa = float(field.get("escape_solid_angle", 0.6 * FULL_SOLID))
        cand = np.where(field["dangerous"])[0]
        if len(cand) < len(sub):
            cand = np.argsort(-free)[:max(len(sub), len(cand))]
        chosen = [int(cand[c]) for c in self._farthest_point_indices(dirs[cand], len(sub))]
        while len(chosen) < len(sub):
            chosen.append(chosen[-1])
        for k, i in enumerate(sub):
            d = dirs[chosen[k]]
            if self.guide_collapse:
                rad = self._encircle_guide_radius(esa, r_f, float(free[chosen[k]]))
            else:
                rad = min(max(r_f, self.catch_radius * 1.5),
                          max(self.catch_radius * 1.2, float(free[chosen[k]]) * 0.85))
            p = (T + rad * d).astype(np.float32)
            p[0] = np.clip(p[0], 50.0, self.map_size - 50.0)
            p[1] = np.clip(p[1], 50.0, self.map_size - 50.0)
            p[2] = np.clip(p[2], self.z_min, self.z_max)
            guides[int(i)] = p
        return field, sub, guides

    def _evader_step_multi(self):
        for m in range(self.num_targets):
            if not self.target_alive[m]:
                continue
            self.target_position = self.target_positions[m].copy()
            self.target_yaw = float(self.target_yaws[m])
            self.target_pitch = float(self.target_pitches[m])
            self.target_speed = float(self.target_speeds[m])
            self.target_trail = self.target_trails[m]
            self._evader_step()      # flees all pursuers; updates target_* in place
            self.target_positions[m] = self.target_position
            self.target_yaws[m] = self.target_yaw
            self.target_pitches[m] = self.target_pitch
            self.target_speeds[m] = self.target_speed

    def reset(self):
        super().reset()   # spawns pursuers + one target; sets self.target_position etc.
        M = self.num_targets
        rng = np.random
        for m in range(M):
            self.target_positions[m] = np.array([
                rng.uniform(150, self.map_size - 150),
                rng.uniform(150, self.map_size - 150),
                rng.uniform(self.z_min + 40, self.z_max - 40)], np.float32)
            self.target_yaws[m] = float(rng.uniform(0, 2 * np.pi))
            self.target_pitches[m] = 0.0
            self.target_speeds[m] = self.target_min_speed
            self.target_alive[m] = True
            self.target_trails[m] = deque([self.target_positions[m].copy()], maxlen=100)
        self.last_guide_dist = {}      # per-episode; stale values would fake progress on step 1
        self.last_sub_dists = {}
        if self.surround_spawn:
            self._place_pursuers_around_targets()
        self.assign = assign_pursuers_to_targets(self.uav_positions, self.target_positions,
                                                 np.full(self.num_agents, -1), self.target_alive,
                                                 base_cost=self._alloc_cost())
        for m in range(M):
            f, sub, g = self._field_and_guides(m)
            self.sub_fields[m] = f
            for i, p in g.items():
                self.current_guide_points[self.agents[i]] = p
            if len(sub):
                self.last_min_dist[m] = float(np.min(
                    np.linalg.norm(self.uav_positions[sub] - self.target_positions[m], axis=1)))
        self._episode_step = 0
        return self._get_obs(), {"state": self.state()}


    def _place_pursuers_around_targets(self):
        """Split the team between targets and ring each sub-team around its own target.

        reset() lays the pursuers out relative to the single inherited target and then moves
        every target to a fresh uniform position, which discards that geometry: the team ends
        up roughly 220 m from the nearest target, on one side of it. Pursuers cap at 11 m/s and
        the evader at 10, so the net closing rate is 1 m/s and 200 m cannot be closed inside a
        400-step episode once the evader manoeuvres -- the physical-reach criterion is out of
        reach before the policy makes its first decision. The Apollonius trap, by contrast, is
        satisfiable at that range given the speed edge, which is why the learned behaviour holds
        station far out: it optimises the only criterion it can actually reach.

        Ringing each sub-team around its target is what made the same criterion attainable in
        the single-target chapter.
        """
        M, N = self.num_targets, self.num_agents
        groups = [[] for _ in range(M)]
        for i in range(N):                       # round-robin keeps sub-teams balanced
            groups[i % M].append(i)
        for m, members in enumerate(groups):
            if not members:
                continue
            dirs = g3.fibonacci_sphere(max(len(members), 3))
            for k, i in enumerate(members):
                p = (self.target_positions[m] + self.spawn_radius * dirs[k]).astype(np.float32)
                p[0] = np.clip(p[0], self.uav_radius, self.map_size - self.uav_radius)
                p[1] = np.clip(p[1], self.uav_radius, self.map_size - self.uav_radius)
                p[2] = np.clip(p[2], self.z_min, self.z_max)
                tries = 0
                while self._is_in_building(p, self.uav_radius) and tries < 12:
                    p = p + np.random.uniform(-40, 40, 3).astype(np.float32)
                    p[2] = np.clip(p[2], self.z_min, self.z_max)
                    tries += 1
                self.uav_positions[i] = p
                d = self.target_positions[m] - p
                self.uav_yaws[i] = float(np.arctan2(d[1], d[0]))
                self.uav_pitches[i] = float(np.clip(
                    np.arctan2(d[2], float(np.linalg.norm(d[:2])) + 1e-6),
                    -self.pitch_max, self.pitch_max))
                self.uav_speeds[i] = self.uav_min_speed

    def step(self, actions_dict):
        self._episode_step += 1
        old_positions = self.uav_positions.copy()
        hit_flags = []
        for i, a in enumerate(self.agents):
            act = np.clip(actions_dict[a], -1.0, 1.0)
            self.uav_speeds[i] = np.clip(self.uav_speeds[i] + act[0] * self.max_accel,
                                         self.uav_min_speed, self.uav_max_speed)
            self.uav_yaws[i] = (self.uav_yaws[i] + act[1] * self.max_yaw_rate) % (2 * np.pi)
            self.uav_pitches[i] = np.clip(self.uav_pitches[i] + act[2] * self.max_pitch_rate,
                                          -self.pitch_max, self.pitch_max)
            new_pos, hit = self._move_with_clip_3d(self.uav_positions[i], self.uav_yaws[i],
                                                   self.uav_pitches[i], self.uav_speeds[i], self.uav_radius)
            self.uav_positions[i] = new_pos
            hit_flags.append(hit)
            self.uav_trails[a].append(new_pos.copy())

        self._update_radar_cache()
        radar_min = [float(self._radar_cache[i].min()) * self.radar_range for i in range(self.num_agents)]
        self._evader_step_multi()

        # re-allocation over still-alive targets
        if np.any(self.target_alive):
            if self.dynamic_alloc:
                self.assign = assign_pursuers_to_targets(self.uav_positions, self.target_positions,
                                                         self.assign, self.target_alive,
                                                         base_cost=self._alloc_cost())
            else:  # fixed: only reassign pursuers whose assigned target has died
                alive_idx = np.where(self.target_alive)[0]
                for i in range(self.num_agents):
                    if self.assign[i] < 0 or not self.target_alive[self.assign[i]]:
                        d = np.linalg.norm(self.target_positions[alive_idx] - self.uav_positions[i], axis=1)
                        self.assign[i] = int(alive_idx[int(np.argmin(d))])
        max_rel = self.uav_max_speed + self.target_max_speed
        rewards = {a: 0.0 for a in self.agents}
        newly_caught = 0
        for m in range(self.num_targets):
            if not self.target_alive[m]:
                continue
            field, sub, guides = self._field_and_guides(m)
            self.sub_fields[m] = field
            for i, p in guides.items():
                self.current_guide_points[self.agents[i]] = p
            if len(sub) == 0:
                continue
            dists = np.linalg.norm(self.uav_positions[sub] - self.target_positions[m], axis=1)
            min_new = float(np.min(dists))
            # Distance to each pursuer's own guide point, before and after this step. The guide
            # radius collapses toward 1.3 * catch_radius as the escape angle closes, but until
            # now nothing rewarded flying to it, so the collapse changed the guide and not the
            # behaviour -- which is why the team held station ~155 m out and the strict
            # criterion scored zero. Chapter 3 rewards this progress via w_pos/r_guide.
            g_new = np.array([
                float(np.linalg.norm(self.uav_positions[i] - self.current_guide_points[self.agents[i]]))
                if self.agents[i] in self.current_guide_points else 0.0 for i in sub])
            g_old = np.array([float(self.last_guide_dist.get(self.agents[i], gn))
                              for i, gn in zip(sub, g_new)])
            esa = float(field.get("escape_solid_angle", 0.6 * FULL_SOLID)) if field else 0.6 * FULL_SOLID
            # Chapter 3 scores a target as captured only when a pursuer actually reaches it.
            # This environment additionally accepted a collapsed escape solid angle, so the two
            # chapters were not reporting the same quantity; strict_capture aligns them.
            reached = bool(np.any(dists <= self.catch_radius))
            caught = reached if self.strict_capture else (reached or esa <= 0.1)
            closure = self.closure_weight * self._closure_bonus(
                esa, self.last_min_dist[m], min_new, max_rel)
            pincer = 0.0
            if self.pincer_weight > 0.0 and len(sub) > 1:
                pincer = self.pincer_weight * self._pincer_bonus(
                    esa, np.sort(self.last_sub_dists.get(m, dists)), np.sort(dists), max_rel)
            self.last_sub_dists[m] = dists.copy()
            r_gap = float(np.exp(-esa))
            for k, i in enumerate(sub):
                ag = self.agents[i]
                prog = float(np.clip((self.last_min_dist[m] - dists[k]) / max_rel, -1.0, 1.0))
                r_guide = float(np.clip((g_old[k] - g_new[k]) / max_rel, -1.0, 1.0))
                # Chapter 3 mixes guide-point progress with team progress rather than using
                # team progress alone.
                prog = (self.reward_mix["guide_progress"] * r_guide
                        + self.reward_mix["team_progress"] * prog)
                if caught:
                    r_finish = 80.0
                elif dists[k] < self.catch_radius * 3:
                    # Continuous ramp over the last three catch radii. Without it the terminal
                    # reward is 0 until it is 80, so the final approach carries no gradient at
                    # all and the closure bonus alone has to discover it.
                    r_finish = (self.catch_radius * 3 - dists[k]) / (self.catch_radius * 2)
                else:
                    r_finish = 0.0
                r_safe = -15.0 if hit_flags[i] else -0.5 * ((radar_min[i] - self.radar_range) / self.radar_range) ** 2
                rewards[ag] = (self.reward_weights["w_near"] * prog
                               + self.reward_weights["w_gap"] * r_gap
                               + self.reward_weights["w_safe"] * r_safe
                               + self.reward_weights["w_finish"] * (r_finish + closure + pincer))
            self.last_min_dist[m] = min_new
            for k, i in enumerate(sub):
                self.last_guide_dist[self.agents[i]] = float(g_new[k])
            if caught:
                self.target_alive[m] = False
                newly_caught += 1

        all_caught = not np.any(self.target_alive)
        timeout = self._episode_step >= self.max_episode_steps
        done = all_caught or timeout
        terminated = {a: all_caught for a in self.agents}
        truncated = {a: (timeout and not all_caught) for a in self.agents}
        info = {"is_success": all_caught, "num_caught": int(np.sum(~self.target_alive)),
                "num_targets": self.num_targets, "state": self.state()}
        return self._get_obs(), rewards, terminated, truncated, info

    def _get_obs(self):
        obs = {}
        ms = self.map_size
        zr = max(self.z_max - self.z_min, 1e-6)
        for i, ag in enumerate(self.agents):
            m = int(self.assign[i]) if self.assign[i] >= 0 else 0
            pos, yaw, pit = self.uav_positions[i], self.uav_yaws[i], self.uav_pitches[i]
            norm_pos = np.array([pos[0] / ms * 2 - 1, pos[1] / ms * 2 - 1,
                                 (pos[2] - self.z_min) / zr * 2 - 1], np.float32)
            heading = np.array([np.cos(yaw), np.sin(yaw), np.sin(pit)], np.float32)
            nspd = np.array([(self.uav_speeds[i] - self.uav_min_speed) /
                             (self.uav_max_speed - self.uav_min_speed) * 2 - 1], np.float32)
            gp = self.current_guide_points.get(ag, self.target_positions[m])
            rel_guide = ((gp - pos) / ms).astype(np.float32)
            rel_target = ((self.target_positions[m] - pos) / ms).astype(np.float32)
            mates = [((self.uav_positions[j] - pos) / ms) for j in range(self.num_agents) if j != i]
            rel_mates = np.concatenate(mates).astype(np.float32) if mates else np.array([], np.float32)
            others = [((self.target_positions[mm] - pos) / ms) for mm in range(self.num_targets) if mm != m]
            rel_others = np.concatenate(others).astype(np.float32) if others else np.array([], np.float32)
            t_head = np.array([np.cos(self.target_yaws[m]), np.sin(self.target_yaws[m]),
                               np.sin(self.target_pitches[m])], np.float32)
            ntspd = np.array([self.target_speeds[m] / self.target_max_speed * 2 - 1], np.float32)
            radar = (self._radar_cache[i] * 2 - 1).astype(np.float32)
            extra = [self._nearest_obstacle_features(pos, self.obstacle_gat_k).flatten()] if self.use_obstacle_gat else []
            obs[ag] = np.concatenate([norm_pos, heading, nspd, rel_guide, rel_target,
                                      rel_mates, rel_others, t_head, ntspd, radar] + extra).astype(np.float32)
        return obs

    def state(self):
        guides = np.concatenate([self.current_guide_points.get(a, self.uav_positions[i])
                                 for i, a in enumerate(self.agents)])
        tstate = np.concatenate([
            self.target_positions.flatten(), self.target_yaws, self.target_pitches, self.target_speeds])
        return np.concatenate([
            self.uav_positions.flatten(), self.uav_yaws, self.uav_pitches, self.uav_speeds,
            guides, tstate]).astype(np.float32)
