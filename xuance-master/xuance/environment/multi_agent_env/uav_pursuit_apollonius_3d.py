"""3D multi-UAV cooperative pursuit with 2.5D prism obstacles (P1 increment 1).

Ports the proven 2D environment (``uav_pursuit_apollonius_obs_5.py``) to 3D:
position (x,y,z), heading yaw + flight-path pitch, action [accel, yaw_rate,
pitch_rate]. Obstacles are full-height vertical prisms (xy AABB footprints), so
collision and line-of-sight reuse the 2D footprint logic. The Apollonius escape
field is generalized to a solid angle on the sphere (Euclidean criterion this
increment). Visibility-constraint and obstacle-target GAT are later increments.
"""
import os
import json
import copy
import numpy as np
from .cbf_filter import filter_action as cbf_filter_action
import gymnasium as gym
from collections import deque
from scipy.optimize import linear_sum_assignment

from xuance.environment import RawMultiAgentEnv
from . import geometry3d as g3
from . import apollonius3d as ap3


class UAVPursuitApollonius3DEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(UAVPursuitApollonius3DEnv, self).__init__()

        # ---------------- 1. physical parameters ----------------
        self.map_size = 1000.0
        self.z_min = float(getattr(config, "z_min", 50.0))
        self.z_max = float(getattr(config, "z_max", 350.0))

        self.uav_min_speed = 9.0
        self.uav_max_speed = 11.0
        self.cbf_enabled = bool(getattr(config, "cbf_enabled", False))
        self.cbf_eta = float(getattr(config, "cbf_eta", 0.25))
        self.cbf_margin = float(getattr(config, "cbf_margin", 6.0))
        self.max_accel = 1.0
        self.max_yaw_rate = np.pi / 4
        self.max_pitch_rate = np.pi / 12
        self.pitch_max = np.pi / 6           # +/- 30 deg flight-path angle
        self.gravity = float(getattr(config, "gravity", 9.81))
        self.max_load_factor = float(getattr(config, "max_load_factor", 4.0))
        self.uav_radius = 0.5
        self.spawn_safety_enabled = bool(getattr(config, "spawn_safety_enabled", True))
        self.spawn_safety_margin = float(getattr(config, "spawn_safety_margin", 1.5))
        self.spawn_search_resolution = float(getattr(config, "spawn_search_resolution", 1.0))
        if not np.isfinite(self.spawn_safety_margin) or self.spawn_safety_margin < 0.0:
            raise ValueError("spawn_safety_margin must be >= 0")
        if not np.isfinite(self.spawn_search_resolution) or self.spawn_search_resolution <= 0.0:
            raise ValueError("spawn_search_resolution must be > 0")
        self.initial_step_reach = min(self.uav_max_speed, self.uav_min_speed + self.max_accel)
        self.spawn_clearance = self.uav_radius + self.initial_step_reach + self.spawn_safety_margin
        self.spawn_min_pair_distance = 2.0 * self.spawn_clearance

        # configurable (used directly when curriculum disabled; curriculum levels
        # override per-level when enabled). Lets us set a learnable lambda.
        self.target_min_speed = float(getattr(config, "target_min_speed", 4.0))
        self.target_max_speed = float(getattr(config, "target_max_speed", 11.0))
        # Keeps the evader off the walls so captures have to be earned in open air; see
        # _evader_step. 0 reproduces the previous behaviour exactly.
        self.evader_center_pull = float(getattr(config, "evader_center_pull", 0.0))
        self.target_accel = 0.5
        self.target_speed = self.target_min_speed
        self.target_radius = 0.5
        self.catch_radius = 15.0
        self.cartesian_lambda_cap = getattr(config, "cartesian_lambda_cap", 0.8)
        # euclidean | visibility (visibility voids pursuers occluded from probe points)
        self.criterion_mode = getattr(config, "criterion_mode", "euclidean")
        # GAT-O: conditionally expose nearest-k obstacle node features in obs
        self.use_obstacle_gat = bool(getattr(config, "use_obstacle_gat", False))
        self.obstacle_gat_k = int(getattr(config, "obstacle_gat_k", 4))
        self.obstacle_gat_feat_dim = 4
        self.obstacle_sense = 200.0
        # reward-term ablation (3.2): names in this set get zero weight
        self.reward_disable = set(getattr(config, "reward_disable", []) or [])
        # closure: drive capture once encircled (anti "encircle-but-never-close" optimum)
        self.closure_weight = float(getattr(config, "closure_weight", 4.0))
        # pincer: a FASTER reactive evader flees the single nearest pursuer, so lone closing
        # fails (it outruns the one closer). This rewards ALSO reducing the 2nd-nearest
        # distance once encircled -> two pursuers close together and the evader flees one
        # INTO the other. Opt-in (0.0 -> flagship reward byte-identical). See _pincer_bonus.
        self.pincer_weight = float(getattr(config, "pincer_weight", 0.0))
        # close_k: LAYERED encirclement. Once encircled, guide_collapse pulls EVERY pursuer onto
        # a catch_radius*1.3 (~19.5 m) sphere. Since pursuers cannot fly slower than
        # uav_min_speed (9 m/s) they cannot hold station there, so N bodies orbit one small
        # volume and collide -- which is why the collision rate grows with N (30% at N=3 to 77%
        # at N=6). With close_k>0 only the close_k most dangerous escape directions collapse
        # inward (sealing the capture sphere needs 3-4 agents); the remaining pursuers hold the
        # outer containment ring. 0 = legacy "everyone collapses".
        self.close_k = int(getattr(config, "close_k", 0))
        # obs_avoid_weight: scales the radar-based obstacle penalty so avoidance engages EARLIER
        # (client-approved simplification). 1.0 = legacy.
        self.obs_avoid_weight = float(getattr(config, "obs_avoid_weight", 1.0))
        # Inter-UAV collision. The crash test (_move_with_clip_3d / _prospective_collision_sources)
        # only ever sees ONE uav's own state, so it detects buildings and boundaries and nothing
        # else: pursuer-pursuer contact has never been a terminal event, and never entered
        # collision_rate. Teammate proximity was only ever shaped in reward (r_mate under 10 m,
        # plus the optional separation barrier). The pairwise gap is now measured on every step
        # and reported unconditionally, so the size of that omission is a number rather than an
        # assumption; making it terminal stays opt-in so published runs remain reproducible.
        self.uav_collision_enabled = bool(getattr(config, "uav_collision_enabled", False))
        self.uav_collision_distance = float(getattr(config, "uav_collision_distance",
                                                    2.0 * self.uav_radius))
        self.episode_min_uav_gap = float("inf")
        self.episode_uav_conflicts = 0
        self.episode_cbf_steps = 0
        self.episode_cbf_effort = 0.0
        # guide_collapse: shrink guide points inward when encircled (set False for
        # the closure-design ablation -> reverts to fixed far-ring guides).
        self.guide_collapse = bool(getattr(config, "guide_collapse", True))
        # surround_spawn: pursuers start distributed on a sphere AROUND the evader (radius
        # spawn_radius) instead of one-sided at x=100. Enables Apollonius encirclement of a
        # FASTER evader (a one-sided chase can never catch/surround a faster target).
        self.surround_spawn = bool(getattr(config, "surround_spawn", False))
        self.spawn_radius = float(getattr(config, "spawn_radius", 200.0))
        # Training-only curriculum switch.  With soft collisions the UAV still
        # receives the full crash penalty and slides off the obstacle, but the
        # episode continues so the replay buffer contains recovery trajectories.
        # Evaluation and all legacy runs keep hard termination by default.
        self.terminate_on_collision = bool(getattr(config, "terminate_on_collision", True))

        self._setup_curriculum(config)       # sets reward_weights/mix, building_mode, spawn_offset, target_pitch_max

        self.radar_range = 100.0
        self.num_radar_rays = 16

        self.num_agents = int(getattr(config, "num_agents", 6))
        self.agents = [f"uav_{i}" for i in range(self.num_agents)]
        self.spawn_offsets = np.zeros(self.num_agents, dtype=np.float32)
        self._safe_spawn_cache = {}
        # Optional early separation shaping.  The legacy reward only penalized
        # teammates once they were already within 10 m, which is too late for
        # N=4 during the final close-in maneuver.  Keep the feature opt-in so
        # published legacy runs remain bit-compatible.
        self.separation_weight = float(getattr(config, "separation_weight", 0.0))
        self.separation_distance = float(getattr(config, "separation_distance", 20.0))

        self.escape_num_dirs = int(getattr(config, "escape_num_dirs", 200))
        self.escape_dirs = g3.fibonacci_sphere(self.escape_num_dirs)

        self.buildings_dir = getattr(
            config, "buildings_dir",
            os.path.join(os.path.dirname(__file__), "buildings"))
        self.buildings = np.array(self._generate_city_blocks(), dtype=np.float32)

        # domain randomization: if set, each reset draws a random density from this
        # pool -> one model robust across obstacle densities (no XuanCe curriculum surgery).
        self.randomize_density = getattr(config, "randomize_density", None)
        if self.randomize_density:
            self.randomize_density = list(self.randomize_density)
            self._density_pool = {m: np.array(self._load_blocks(m), dtype=np.float32)
                                  for m in self.randomize_density}

        z_mid = 0.5 * (self.z_min + self.z_max)
        ys = np.linspace(150.0, 850.0, self.num_agents)
        self.init_uav_positions = np.array(
            [[100.0, float(y), z_mid] for y in ys], dtype=np.float32)

        plot_config = getattr(config, "plot_config", {})
        self.uav_plot_radius = plot_config.get("uav_radius", 4)
        self.target_plot_radius = plot_config.get("target_radius", 4)

        # ---------------- 2. spaces ----------------
        self.action_space = {a: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                             for a in self.agents}
        # obs: norm_pos(3)+heading(3)+nspd(1)+rel_guide(3)+rel_target(3)
        #      +rel_mates(3*(N-1))+t_head(3)+ntspd(1)+radar(16) [+ obstacle nodes k*4]
        base_obs = 17 + 3 * (self.num_agents - 1) + self.num_radar_rays
        self.obs_dim = base_obs + (self.obstacle_gat_k * self.obstacle_gat_feat_dim
                                   if self.use_obstacle_gat else 0)
        self.observation_space = {a: gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                                  for a in self.agents}
        # state: pursuer pos(3N)+vel(3N)+yaw(N)+target(6)+guides(3N)
        self.state_dim = 10 * self.num_agents + 6
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # ---------------- 3. runtime ----------------
        self.max_episode_steps = getattr(config, "episode_length", 400)
        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

        self.uav_positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.uav_velocities = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.uav_yaws = np.zeros(self.num_agents, dtype=np.float32)
        # Compatibility fields derived from uav_velocities for radar/reward code.
        self.uav_pitches = np.zeros(self.num_agents, dtype=np.float32)
        self.uav_speeds = np.zeros(self.num_agents, dtype=np.float32)

        self.target_position = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.last_distances = np.zeros(self.num_agents, dtype=np.float32)
        self.episode_target_speed_max = self.target_speed

        self._radar_cache = np.ones((self.num_agents, self.num_radar_rays), dtype=np.float32)
        self.uav_trails = {a: deque(maxlen=100) for a in self.agents}
        self.target_trail = deque(maxlen=100)

        self.episode_sub_rewards = {
            a: {"r_near": 0.0, "r_safe": 0.0, "r_pos": 0.0, "r_gap": 0.0, "r_finish": 0.0}
            for a in self.agents
        }
        self.debug_rf = 0.0
        self.apollonius_escape = {}
        self.apollonius_margin_ema = None
        self.apollonius_dangerous_prev = None
        self.guide_polar_state = {}
        self.current_guide_points = {}

    # ============================================================
    # Curriculum (ported from 2D, + vertical-escape ramp)
    # ============================================================
    def _default_curriculum_schedule(self):
        base = [
            # building_mode names map to JSON maps: empty=[] , open(4), medium(16), complex(26)
            (0, 3.0, 5.0, 0.2, 20.0, "empty", 2.5, 0.2, 0.2, 3.0, 0.3, 0.0),
            (1, 4.0, 7.0, 0.3, 40.0, "empty", 2.2, 0.4, 0.4, 3.0, 0.35, 0.25),
            (2, 4.0, 9.0, 0.4, 60.0, "open", 2.0, 0.6, 0.8, 3.0, 0.4, 0.5),
            (3, 4.0, 10.0, 0.5, 80.0, "medium", 2.0, 0.8, 1.0, 3.0, 0.45, 0.75),
            (4, 4.0, 12.0, 0.5, 100.0, "medium", 2.0, 0.8, 1.0, 3.5, 0.5, 1.0),
        ]
        sched = []
        for (lvl, tmin, tmax, tacc, off, bmode, wn, wp, wg, wf, mix, zsc) in base:
            sched.append({
                "level": lvl, "target_min_speed": tmin, "target_max_speed": tmax,
                "target_accel": tacc, "spawn_offset": off, "building_mode": bmode,
                "reward_weights": {"w_near": wn, "w_safe": 1.0, "w_pos": wp, "w_gap": wg, "w_finish": wf},
                "reward_mix": {"guide_progress": mix, "team_progress": 1.0 - mix},
                "z_escape_scale": zsc,
            })
        return sched

    def _setup_curriculum(self, config):
        self.curriculum_enabled = bool(getattr(config, "curriculum_enabled", True))
        self.curriculum_schedule = getattr(config, "curriculum_schedule", None)
        if self.curriculum_schedule is None:
            self.curriculum_schedule = self._default_curriculum_schedule()
        self.curriculum_min_steps_per_level = int(getattr(config, "curriculum_min_steps_per_level", 300000))
        self.curriculum_success_threshold = float(getattr(config, "curriculum_success_threshold", 0.75))
        self.curriculum_drop_threshold = float(getattr(config, "curriculum_drop_threshold", 0.45))
        self.curriculum_level = int(getattr(config, "curriculum_start_level", 0))
        self.curriculum_last_update_step = 0

        self.spawn_offset = 50.0
        self.building_mode = getattr(config, "building_mode", "medium")
        # base weights; config-overridable (used directly when curriculum disabled;
        # curriculum levels override per-level when enabled)
        self.reward_weights = {
            "w_near": float(getattr(config, "w_near", 2.0)),
            "w_safe": float(getattr(config, "w_safe", 1.0)),
            "w_pos": float(getattr(config, "w_pos", 1.2)),
            "w_gap": float(getattr(config, "w_gap", 1.5)),
            "w_finish": float(getattr(config, "w_finish", 2.0)),
        }
        self.reward_mix = {"guide_progress": 0.6, "team_progress": 0.4}
        self.curriculum_config = {}
        self.target_pitch_max = self.pitch_max

        if self.curriculum_enabled:
            self.set_curriculum_level(self.curriculum_level, reload_buildings=False)

    def set_curriculum_level(self, level, reload_buildings=True):
        if not self.curriculum_enabled:
            return self.curriculum_level
        max_level = len(self.curriculum_schedule) - 1
        self.curriculum_level = int(np.clip(level, 0, max_level))
        self.curriculum_config = copy.deepcopy(self.curriculum_schedule[self.curriculum_level])
        self.target_min_speed = float(self.curriculum_config["target_min_speed"])
        self.target_max_speed = float(self.curriculum_config["target_max_speed"])
        self.target_accel = float(self.curriculum_config["target_accel"])
        self.target_speed = min(max(float(getattr(self, "target_speed", self.target_min_speed)),
                                    self.target_min_speed), self.target_max_speed)
        self.spawn_offset = float(self.curriculum_config["spawn_offset"])
        self.building_mode = self.curriculum_config["building_mode"]
        self.reward_weights = copy.deepcopy(self.curriculum_config["reward_weights"])
        self.reward_mix = copy.deepcopy(self.curriculum_config["reward_mix"])
        self.target_pitch_max = self.pitch_max * float(self.curriculum_config.get("z_escape_scale", 1.0))
        if reload_buildings and hasattr(self, "buildings"):
            self.buildings = np.array(self._generate_city_blocks(), dtype=np.float32)
        return self.curriculum_level

    def update_curriculum(self, success_rate, global_step):
        if not self.curriculum_enabled:
            return self.curriculum_level
        if int(global_step) - int(self.curriculum_last_update_step) < self.curriculum_min_steps_per_level:
            return self.curriculum_level
        old = self.curriculum_level
        if success_rate >= self.curriculum_success_threshold:
            self.set_curriculum_level(self.curriculum_level + 1)
        elif success_rate <= self.curriculum_drop_threshold:
            self.set_curriculum_level(self.curriculum_level - 1)
        if self.curriculum_level != old:
            self.curriculum_last_update_step = int(global_step)
        return self.curriculum_level

    def get_curriculum_info(self):
        return {
            "curriculum_enabled": self.curriculum_enabled,
            "curriculum_level": self.curriculum_level,
            "building_mode": self.building_mode,
            "spawn_offset": self.spawn_offset,
            "target_min_speed": self.target_min_speed,
            "target_max_speed": self.target_max_speed,
            "target_accel": self.target_accel,
            "target_speed_current": float(getattr(self, "target_speed", self.target_min_speed)),
            "target_speed_max_episode": float(getattr(self, "episode_target_speed_max",
                                                       getattr(self, "target_speed", self.target_min_speed))),
            "building_count": int(len(getattr(self, "buildings", []))),
            "target_pitch_max": float(getattr(self, "target_pitch_max", self.pitch_max)),
            "separation_weight": float(self.separation_weight),
            "separation_distance": float(self.separation_distance),
            "terminate_on_collision": bool(self.terminate_on_collision),
            "spawn_safety_enabled": bool(self.spawn_safety_enabled),
            "spawn_clearance": float(self.spawn_clearance),
            "spawn_adjusted_count": int(np.count_nonzero(self.spawn_offsets > 1e-6)),
            "spawn_max_offset": float(np.max(self.spawn_offsets)) if len(self.spawn_offsets) else 0.0,
            "spawn_offsets": [float(value) for value in self.spawn_offsets],
            "reward_weights": copy.deepcopy(self.reward_weights),
            "reward_mix": copy.deepcopy(self.reward_mix),
        }

    def _load_blocks(self, mode):
        """Load and validate one obstacle map; only 'empty' may omit an asset."""
        if mode == "empty":
            return []
        config_file = os.path.join(self.buildings_dir, f"buildings_{mode}.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"Missing obstacle map for mode={mode!r}: {config_file}")
        try:
            with open(config_file, "r", encoding="utf-8") as stream:
                blocks = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid obstacle JSON: {config_file}: {exc}") from exc
        try:
            array = np.asarray(blocks, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid obstacle footprints in {config_file}: {exc}") from exc
        if array.size == 0:
            raise ValueError(f"Empty obstacle map is invalid for mode={mode!r}: {config_file}")
        valid_shape = array.ndim == 2 and array.shape[1] == 4
        valid_values = np.all(np.isfinite(array))
        valid_extents = valid_shape and np.all(array[:, 0] < array[:, 1]) and np.all(array[:, 2] < array[:, 3])
        valid_bounds = valid_shape and np.all(array >= 0.0) and np.all(array <= self.map_size)
        if not (valid_shape and valid_values and valid_extents and valid_bounds):
            raise ValueError(f"Invalid obstacle footprints in {config_file}: shape={array.shape}")
        return array.tolist()

    def _generate_city_blocks(self):
        return self._load_blocks(getattr(self, "building_mode", "medium"))

    # ============================================================
    # Geometry / motion helpers
    # ============================================================
    def _is_in_building(self, pos, radius):
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < pos[0] < xmax + radius) and (ymin - radius < pos[1] < ymax + radius):
                return True
        return False

    def _spawn_clearance_xy(self, pos):
        x, y = float(pos[0]), float(pos[1])
        best = min(x, self.map_size - x, y, self.map_size - y)
        for xmin, xmax, ymin, ymax in self.buildings:
            dx = max(float(xmin) - x, 0.0, x - float(xmax))
            dy = max(float(ymin) - y, 0.0, y - float(ymax))
            if float(xmin) <= x <= float(xmax) and float(ymin) <= y <= float(ymax):
                obstacle_clearance = -min(x - xmin, xmax - x, y - ymin, ymax - y)
            else:
                obstacle_clearance = float(np.hypot(dx, dy))
            best = min(best, obstacle_clearance)
        return float(best)

    def _is_valid_pursuer_spawn(self, pos, assigned=()):
        pos = np.asarray(pos, dtype=np.float32)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            return False
        if not (self.z_min <= float(pos[2]) <= self.z_max):
            return False
        if self._spawn_clearance_xy(pos) < self.spawn_clearance - 1e-6:
            return False
        return all(np.linalg.norm(pos - np.asarray(other))
                   >= self.spawn_min_pair_distance - 1e-6 for other in assigned)

    def _allocate_safe_pursuer_spawns(self, desired_positions):
        import heapq
        desired = np.asarray(desired_positions, dtype=np.float32)
        layout_key = (desired.tobytes(), np.asarray(self.buildings, np.float32).tobytes(),
                      self.spawn_clearance, self.spawn_search_resolution)
        use_cache = not bool(getattr(self, "surround_spawn", False))
        cached = self._safe_spawn_cache.get(layout_key) if use_cache else None
        if cached is not None:
            return cached[0].copy(), cached[1].copy()
        assigned, offsets = [], []
        limit = int(np.ceil(self.map_size / self.spawn_search_resolution))
        for agent_index, origin in enumerate(desired):
            heap = [(0, 0, 0)]
            visited = {(0, 0)}
            chosen = None
            while heap:
                _, ix, iy = heapq.heappop(heap)
                candidate = origin.copy()
                candidate[:2] += self.spawn_search_resolution * np.array([ix, iy], np.float32)
                if self._is_valid_pursuer_spawn(candidate, assigned):
                    chosen = candidate
                    break
                for nx, ny in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                    if (nx, ny) in visited or abs(nx) > limit or abs(ny) > limit:
                        continue
                    visited.add((nx, ny))
                    heapq.heappush(heap, (nx * nx + ny * ny, nx, ny))
            if chosen is None:
                raise RuntimeError(
                    f"No safe spawn: mode={self.building_mode}, agent={agent_index}, "
                    f"desired={origin.tolist()}, clearance={self.spawn_clearance}, limit={limit}")
            assigned.append(chosen)
            offsets.append(float(np.linalg.norm(chosen - origin)))
        result = np.asarray(assigned, np.float32), np.asarray(offsets, np.float32)
        if use_cache:
            self._safe_spawn_cache[layout_key] = (result[0].copy(), result[1].copy())
        return result

    def _validate_pursuer_spawns(self, positions):
        assigned = []
        for index, pos in enumerate(np.asarray(positions)):
            if not self._is_valid_pursuer_spawn(pos, assigned):
                raise RuntimeError(
                    f"Spawn invariant failed: mode={self.building_mode}, agent={index}, "
                    f"position={pos.tolist()}, clearance={self.spawn_clearance}")
            assigned.append(pos)


    def _sync_uav_kinematics_from_velocity(self, indices=None):
        if indices is None:
            indices = range(self.num_agents)
        for i in indices:
            v = np.asarray(self.uav_velocities[i], dtype=np.float32)
            speed = float(np.linalg.norm(v))
            self.uav_speeds[i] = speed
            if speed > 1e-6:
                self.uav_pitches[i] = float(np.arcsin(np.clip(v[2] / speed, -1.0, 1.0)))

    def _velocity_from_yaw_pitch_speed(self, yaw, pitch, speed):
        return speed * np.array([np.cos(pitch) * np.cos(yaw),
                                 np.cos(pitch) * np.sin(yaw),
                                 np.sin(pitch)], dtype=np.float32)

    def _clip_speed_vector(self, velocity):
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-6:
            return np.array([self.uav_min_speed, 0.0, 0.0], dtype=np.float32)
        if speed < self.uav_min_speed:
            return (velocity / speed * self.uav_min_speed).astype(np.float32)
        if speed > self.uav_max_speed:
            return (velocity / speed * self.uav_max_speed).astype(np.float32)
        return velocity.astype(np.float32)

    def _limit_acceleration_by_load(self, accel, velocity):
        accel = np.asarray(accel, dtype=np.float32)
        an = float(np.linalg.norm(accel))
        if an > self.max_accel > 0.0:
            accel = accel / an * self.max_accel
        speed = float(np.linalg.norm(velocity))
        if speed <= 1e-6 or self.max_load_factor <= 1.0:
            return accel.astype(np.float32)
        vhat = velocity / speed
        a_parallel = float(np.dot(accel, vhat)) * vhat
        a_normal = accel - a_parallel
        normal_limit = self.gravity * np.sqrt(max(self.max_load_factor ** 2 - 1.0, 0.0))
        nn = float(np.linalg.norm(a_normal))
        if nn > normal_limit > 0.0:
            a_normal = a_normal / nn * normal_limit
        return (a_parallel + a_normal).astype(np.float32)

    def _limit_yaw_rate_by_load(self, yaw_rate, velocity):
        speed = float(np.linalg.norm(velocity))
        if speed <= 1e-6 or self.max_load_factor <= 1.0:
            return 0.0
        load_yaw = self.gravity * np.sqrt(max(self.max_load_factor ** 2 - 1.0, 0.0)) / speed
        limit = min(abs(self.max_yaw_rate), load_yaw)
        return float(np.clip(yaw_rate, -limit, limit))

    def _move_velocity_with_clip_3d(self, pos, velocity, radius):
        next_pos = (pos + velocity).astype(np.float32)
        hit = False
        if next_pos[0] < radius:
            next_pos[0] = radius; hit = True
        elif next_pos[0] > self.map_size - radius:
            next_pos[0] = self.map_size - radius; hit = True
        if next_pos[1] < radius:
            next_pos[1] = radius; hit = True
        elif next_pos[1] > self.map_size - radius:
            next_pos[1] = self.map_size - radius; hit = True
        next_pos[2] = np.clip(next_pos[2], self.z_min, self.z_max)
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < next_pos[0] < xmax + radius) and (ymin - radius < next_pos[1] < ymax + radius):
                hit = True
                px = np.array([next_pos[0], pos[1], next_pos[2]], dtype=np.float32)
                if not ((xmin - radius < px[0] < xmax + radius) and (ymin - radius < px[1] < ymax + radius)):
                    next_pos = px; break
                py = np.array([pos[0], next_pos[1], next_pos[2]], dtype=np.float32)
                if not ((xmin - radius < py[0] < xmax + radius) and (ymin - radius < py[1] < ymax + radius)):
                    next_pos = py; break
                next_pos = np.array([pos[0], pos[1], next_pos[2]], dtype=np.float32); break
        return next_pos, hit

    def _move_with_clip_3d(self, pos, yaw, pitch, speed, radius):
        vel = speed * np.array([np.cos(pitch) * np.cos(yaw),
                                np.cos(pitch) * np.sin(yaw),
                                np.sin(pitch)], dtype=np.float32)
        next_pos = (pos + vel).astype(np.float32)
        hit = False
        # xy borders count as a crash (like 2D)
        if next_pos[0] < radius:
            next_pos[0] = radius; hit = True
        elif next_pos[0] > self.map_size - radius:
            next_pos[0] = self.map_size - radius; hit = True
        if next_pos[1] < radius:
            next_pos[1] = radius; hit = True
        elif next_pos[1] > self.map_size - radius:
            next_pos[1] = self.map_size - radius; hit = True
        # z floor/ceiling: clip only (altitude limit, not a crash)
        next_pos[2] = np.clip(next_pos[2], self.z_min, self.z_max)
        # building footprints (xy), full height -> slide like 2D
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < next_pos[0] < xmax + radius) and (ymin - radius < next_pos[1] < ymax + radius):
                hit = True
                px = np.array([next_pos[0], pos[1], next_pos[2]], dtype=np.float32)
                if not ((xmin - radius < px[0] < xmax + radius) and (ymin - radius < px[1] < ymax + radius)):
                    next_pos = px; break
                py = np.array([pos[0], next_pos[1], next_pos[2]], dtype=np.float32)
                if not ((xmin - radius < py[0] < xmax + radius) and (ymin - radius < py[1] < ymax + radius)):
                    next_pos = py; break
                next_pos = np.array([pos[0], pos[1], next_pos[2]], dtype=np.float32); break
        return next_pos, hit

    def _prospective_collision_sources(self, pos, yaw, pitch, speed, radius):
        velocity = speed * np.array([
            np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)
        ], dtype=np.float32)
        candidate = np.asarray(pos, np.float32) + velocity
        sources = []
        if candidate[0] < radius:
            sources.append("boundary:x_min")
        if candidate[0] > self.map_size - radius:
            sources.append("boundary:x_max")
        if candidate[1] < radius:
            sources.append("boundary:y_min")
        if candidate[1] > self.map_size - radius:
            sources.append("boundary:y_max")
        for index, (xmin, xmax, ymin, ymax) in enumerate(self.buildings):
            if xmin - radius < candidate[0] < xmax + radius and ymin - radius < candidate[1] < ymax + radius:
                sources.append(f"building:{index}")
        return sources

    def _update_radar_cache(self):
        for i in range(self.num_agents):
            self._radar_cache[i] = g3.radar_distances(
                self.uav_positions[i], float(self.uav_yaws[i]), float(self.uav_pitches[i]),
                self.radar_range, self.map_size, self.z_min, self.z_max, self.buildings)

    def _nearest_obstacle_features(self, pos, k):
        """(k, 4) features of the k nearest building footprints from `pos`:
        [rel_dx/map, rel_dy/map, clip(dist/sense,0,1), valid_flag]. Zero-padded
        when fewer than k buildings (valid_flag stays 0)."""
        feats = np.zeros((k, 4), dtype=np.float32)
        if len(self.buildings) == 0:
            return feats
        bx = np.clip(pos[0], self.buildings[:, 0], self.buildings[:, 1])
        by = np.clip(pos[1], self.buildings[:, 2], self.buildings[:, 3])
        dvx = bx - pos[0]
        dvy = by - pos[1]
        dists = np.hypot(dvx, dvy)
        order = np.argsort(dists)[:k]
        for i, idx in enumerate(order):
            feats[i] = [dvx[idx] / self.map_size, dvy[idx] / self.map_size,
                        np.clip(dists[idx] / self.obstacle_sense, 0.0, 1.0), 1.0]
        return feats

    def _compute_r_f(self, target_pos=None, target_speed=None):
        # The multi-target subclass evaluates one target at a time and used to pass its speed by
        # assigning self.target_speed, leaving self.target_position pointing at whatever the
        # inherited single-target field last held -- in practice a phantom 442-965 m away from
        # the target actually being processed. Taking both explicitly removes that coupling;
        # omitting them keeps the single-target behaviour unchanged.
        tp = self.target_position if target_pos is None else np.asarray(target_pos, np.float32)
        ts = self.target_speed if target_speed is None else float(target_speed)
        cur = np.linalg.norm(self.uav_positions - tp[None, :], axis=1)
        avg_dist = float(np.mean(cur))
        speed_ratio = ts / max(self.uav_max_speed, 1e-6)
        r_by_dist = avg_dist * (0.25 + 0.35 * speed_ratio)
        r_by_speed = ts * 5.0
        return float(np.clip(max(r_by_dist, r_by_speed, self.catch_radius * 1.5),
                             self.catch_radius * 1.2, self.catch_radius * 8.0))

    def _encircle_guide_radius(self, escape_solid_angle, r_f, free_k):
        """Guide radius blends from r_f (target not yet encircled -> establish the
        ring) toward catch_radius*1.3 (encircled -> close in for the capture)."""
        near = self.catch_radius * 1.3
        far = min(max(r_f, self.catch_radius * 1.5),
                  max(self.catch_radius * 1.2, float(free_k) * 0.85))
        frac = float(np.clip(escape_solid_angle / (0.6 * 4 * np.pi), 0.0, 1.0))
        return near + frac * (far - near)

    def _closure_bonus(self, escape_solid_angle, min_dist_old, min_dist_new, max_rel_speed):
        """Progress-based incentive to close the last gap once the target is
        encircled. Rewards REDUCING the team's min distance (not being close), so
        it cannot be farmed by hovering just outside the capture radius."""
        frac = float(np.clip(escape_solid_angle / (0.6 * 4 * np.pi), 0.0, 1.0))
        encircle = 1.0 - frac
        progress = float(np.clip((min_dist_old - min_dist_new) / max(max_rel_speed, 1e-6),
                                 -1.0, 1.0))
        return encircle * progress

    def _pincer_bonus(self, escape_solid_angle, dists_old_sorted, dists_new_sorted, max_rel_speed):
        """Second-nearest closure incentive (the 'pincer'). _closure_bonus rewards the single
        nearest pursuer closing, but a strictly-faster reactive evader flees that nearest
        threat and outruns it, so lone closure stalls near the capture radius. Rewarding a
        REDUCTION of the 2nd-nearest distance (only when encircled) drives a second pursuer in
        simultaneously; combined with the Apollonius boundary-control reward (which keeps the
        team spread over escape directions) the two closers sit on opposite sides, so the
        evader flees one into the other. Progress-based -> cannot be farmed by hovering."""
        if self.num_agents < 2:
            return 0.0
        frac = float(np.clip(escape_solid_angle / (0.6 * 4 * np.pi), 0.0, 1.0))
        encircle = 1.0 - frac
        progress = float(np.clip((dists_old_sorted[1] - dists_new_sorted[1]) / max(max_rel_speed, 1e-6),
                                 -1.0, 1.0))
        return encircle * progress

    def _teammate_separation_penalty(self, agent_index):
        """Smooth pre-collision barrier for one pursuer.

        Returns an unweighted, non-positive penalty.  A 20 m threshold leaves
        the nominal N=4 tetrahedral capture spacing untouched (~24.5 m at a
        15 m radius), while providing roughly two control steps of warning.
        """
        if self.separation_weight <= 0.0 or self.num_agents <= 1:
            return 0.0
        delta = self.uav_positions - self.uav_positions[agent_index]
        dists = np.linalg.norm(delta, axis=1)
        mask = (np.arange(self.num_agents) != agent_index) & (dists < self.separation_distance)
        if not np.any(mask):
            return 0.0
        x = (self.separation_distance - dists[mask]) / max(self.separation_distance, 1e-6)
        return -float(np.sum(np.clip(x, 0.0, 1.0) ** 2))

    @staticmethod
    def _farthest_point_indices(vectors, k):
        """Greedy farthest-point sampling: indices of k spread-out unit vectors."""
        n = len(vectors)
        if n == 0:
            return []
        chosen = [0]
        if k <= 1:
            return chosen[:k]
        min_d = 1.0 - vectors @ vectors[0]
        while len(chosen) < k:
            nxt = int(np.argmax(min_d))
            chosen.append(nxt)
            min_d = np.minimum(min_d, 1.0 - vectors @ vectors[nxt])
        return chosen[:k]

    @staticmethod
    def _ideal_spherical_nn_angle(n):
        """Target nearest-neighbour angle for the distribution reward.

        Four points in 3D form a regular tetrahedron, whose pairwise angle is
        acos(-1/3), not the 60 degree equal-area heuristic used previously.
        Keep the legacy approximation for other team sizes so existing curves
        remain comparable while fixing the N=4 pathological incentive.
        """
        if n == 4:
            return float(np.arccos(-1.0 / 3.0))
        return float(np.arccos(np.clip(1.0 - 2.0 / max(n, 1), -1.0, 1.0)))

    @staticmethod
    def _guide_candidate_indices(dangerous, free, min_count):
        """Preserve all dangerous directions, then supplement if necessary."""
        chosen = [int(i) for i in np.flatnonzero(dangerous)]
        if len(chosen) >= min_count:
            return np.asarray(chosen, dtype=np.int64)
        seen = set(chosen)
        for idx in np.argsort(-free):
            j = int(idx)
            if j not in seen:
                chosen.append(j)
                seen.add(j)
            if len(chosen) >= min_count:
                break
        return np.asarray(chosen, dtype=np.int64)

    # ============================================================
    # Guide-point assignment (3D)
    # ============================================================
    def _assign_target_points(self):
        target = self.target_position
        dirs = self.escape_dirs
        free = g3.ray_free_distance(target, dirs, self.map_size, self.z_min, self.z_max, self.buildings)
        r_f = self._compute_r_f()
        self.debug_rf = r_f

        field = ap3.compute_escape_field_3d(
            target, self.target_speed, self.uav_positions, self.uav_max_speed,
            dirs, free, self.catch_radius, r_f,
            margin_ema=self.apollonius_margin_ema,
            dangerous_prev=self.apollonius_dangerous_prev,
            criterion_mode=self.criterion_mode, buildings=self.buildings)
        self.apollonius_escape = field
        self.apollonius_margin_ema = field["margin_ema"]
        self.apollonius_dangerous_prev = field["dangerous_prev"]

        cand_idx = self._guide_candidate_indices(field["dangerous"], free, self.num_agents)
        chosen = self._farthest_point_indices(dirs[cand_idx], self.num_agents)
        chosen_idx = [int(cand_idx[c]) for c in chosen]
        while len(chosen_idx) < self.num_agents:           # safety pad
            chosen_idx.append(chosen_idx[-1])

        safe_margin = 50.0
        target_points = []
        escape_angle = float(field.get("escape_solid_angle", 0.6 * 4 * np.pi))
        for rank, idx in enumerate(chosen_idx):
            d = dirs[idx]
            # chosen_idx comes from greedy farthest-point sampling over the DANGEROUS escape
            # directions, and any prefix of that traversal is itself a well-spread cover. So the
            # first close_k are inner closers spread maximally around the target -- the right
            # shape for sealing the capture sphere -- while the rest hold the outer ring.
            closes_in = self.close_k <= 0 or rank < self.close_k
            if self.guide_collapse and closes_in:
                rad = self._encircle_guide_radius(escape_angle, r_f, float(free[idx]))
            else:   # outer containment ring (also the no-collapse ablation)
                rad = min(max(r_f, self.catch_radius * 1.5),
                          max(self.catch_radius * 1.2, float(free[idx]) * 0.85))
            p = (target + rad * d).astype(np.float32)
            p[0] = np.clip(p[0], safe_margin, self.map_size - safe_margin)
            p[1] = np.clip(p[1], safe_margin, self.map_size - safe_margin)
            p[2] = np.clip(p[2], self.z_min, self.z_max)
            target_points.append(p)

        # Hungarian assignment
        n = self.num_agents
        cost = np.zeros((n, len(target_points)), dtype=np.float32)
        for i in range(n):
            upos = self.uav_positions[i]
            yaw = self.uav_yaws[i]
            vel = self.uav_velocities[i]
            speed = max(float(np.linalg.norm(vel)), 1e-6)
            vdir = (vel / speed).astype(np.float32)
            prev = self.current_guide_points.get(self.agents[i], None)
            for j, tp in enumerate(target_points):
                vec = tp - upos
                dist = float(np.linalg.norm(vec))
                ideal = vec / max(dist, 1e-6)
                ang = float(np.arccos(np.clip(np.dot(vdir, ideal), -1.0, 1.0)))
                los = 0.0 if g3.footprint_los(upos, tp, self.buildings) else 1500.0
                cons = 0.8 * float(np.linalg.norm(tp - prev)) if prev is not None else 0.0
                tt = float(np.linalg.norm(tp - target)) / max(self.target_speed, 1e-6)
                ut = dist / max(self.uav_max_speed, 1e-6)
                late = max(0.0, ut - tt)
                cost[i, j] = dist + 10.0 * ang + los + cons + 25.0 * late
        row, col = linear_sum_assignment(cost)
        return {self.agents[int(r)]: target_points[int(c)].astype(np.float32)
                for r, c in zip(row, col)}

    # ============================================================
    # Observation / state
    # ============================================================
    def _get_obs(self):
        obs = {}
        ms = self.map_size
        zr = max(self.z_max - self.z_min, 1e-6)
        for i, ag in enumerate(self.agents):
            pos, yaw = self.uav_positions[i], self.uav_yaws[i]
            vel = self.uav_velocities[i]
            spd = float(np.linalg.norm(vel))
            pit = float(np.arcsin(np.clip(vel[2] / max(spd, 1e-6), -1.0, 1.0)))
            norm_pos = np.array([pos[0] / ms * 2 - 1, pos[1] / ms * 2 - 1,
                                 (pos[2] - self.z_min) / zr * 2 - 1], np.float32)
            heading = np.array([np.cos(yaw), np.sin(yaw), np.sin(pit)], np.float32)
            nspd = np.array([(spd - self.uav_min_speed) /
                             (self.uav_max_speed - self.uav_min_speed) * 2 - 1], np.float32)
            rel_guide = ((self.current_guide_points[ag] - pos) / ms).astype(np.float32)
            rel_target = ((self.target_position - pos) / ms).astype(np.float32)
            mates = [((self.uav_positions[j] - pos) / ms) for j in range(self.num_agents) if j != i]
            rel_mates = np.concatenate(mates).astype(np.float32) if mates else np.array([], np.float32)
            t_head = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw),
                               np.sin(self.target_pitch)], np.float32)
            ntspd = np.array([self.target_speed / self.target_max_speed * 2 - 1], np.float32)
            radar = (self._radar_cache[i] * 2 - 1).astype(np.float32)
            extra = []
            if self.use_obstacle_gat:
                extra = [self._nearest_obstacle_features(pos, self.obstacle_gat_k).flatten()]
            obs[ag] = np.concatenate([norm_pos, heading, nspd, rel_guide, rel_target,
                                      rel_mates, t_head, ntspd, radar] + extra).astype(np.float32)
        return obs

    def state(self):
        guides = np.concatenate([self.current_guide_points[a] for a in self.agents])
        return np.concatenate([
            self.uav_positions.flatten(), self.uav_velocities.flatten(), self.uav_yaws,
            self.target_position, [self.target_yaw], [self.target_pitch], [self.target_speed],
            guides]).astype(np.float32)

    def agent_mask(self):
        return {a: True for a in self.agents}

    def avail_actions(self):
        return None

    # ============================================================
    # reset / step
    # ============================================================
    def reset(self):
        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        for a in self.agents:
            for k in self.episode_sub_rewards[a]:
                self.episode_sub_rewards[a][k] = 0.0
            self.uav_trails[a].clear()
        self.target_trail.clear()
        self.apollonius_escape = {}
        self.apollonius_margin_ema = None
        self.apollonius_dangerous_prev = None
        self.guide_polar_state = {}
        self.episode_min_uav_gap = float("inf")   # per-episode; otherwise the min would carry over
        self.episode_uav_conflicts = 0
        self.episode_cbf_steps = 0                # how often the barrier filter had to correct
        self.episode_cbf_effort = 0.0

        if self.randomize_density:
            mode = str(np.random.choice(self.randomize_density))
            self.building_mode = mode
            self.buildings = self._density_pool[mode]

        self.uav_positions = self.init_uav_positions.copy()
        center = self.map_size / 2.0
        so = self.spawn_offset
        z_mid = 0.5 * (self.z_min + self.z_max)
        while True:
            xy = np.random.uniform(center - so, center + so, size=2).astype(np.float32)
            z = np.random.uniform(max(self.z_min, z_mid - so), min(self.z_max, z_mid + so))
            t = np.array([xy[0], xy[1], z], np.float32)
            if not self._is_in_building(t, self.target_radius):
                self.target_position = t
                break

        self.uav_speeds = np.ones(self.num_agents, np.float32) * self.uav_min_speed
        self.uav_yaws = np.random.uniform(0, 2 * np.pi, size=self.num_agents).astype(np.float32)
        self.uav_pitches = np.zeros(self.num_agents, np.float32)
        self.uav_velocities = np.array([
            self._velocity_from_yaw_pitch_speed(self.uav_yaws[i], self.uav_pitches[i], self.uav_speeds[i])
            for i in range(self.num_agents)
        ], dtype=np.float32)
        if self.surround_spawn:
            self._place_pursuers_around_target()   # encircling spawn (override one-sided)
        desired_positions = self.uav_positions.copy()
        if self.spawn_safety_enabled:
            self.uav_positions, self.spawn_offsets = self._allocate_safe_pursuer_spawns(desired_positions)
            self._validate_pursuer_spawns(self.uav_positions)
        else:
            self.spawn_offsets = np.zeros(self.num_agents, dtype=np.float32)
        if getattr(self, "surround_spawn", False):
            for i in range(self.num_agents):
                vector = self.target_position - self.uav_positions[i]
                self.uav_yaws[i] = float(np.arctan2(vector[1], vector[0]))
                self.uav_pitches[i] = float(np.clip(
                    np.arctan2(vector[2], np.hypot(vector[0], vector[1]) + 1e-6),
                    -self.pitch_max, self.pitch_max))
        self.target_yaw = float(np.random.uniform(0, 2 * np.pi))
        self.target_pitch = 0.0
        self.target_speed = self.target_min_speed
        self.episode_target_speed_max = self.target_speed

        for i in range(self.num_agents):
            self.last_distances[i] = np.linalg.norm(self.uav_positions[i] - self.target_position)
        for i, a in enumerate(self.agents):
            self.uav_trails[a].append(self.uav_positions[i].copy())
        self.target_trail.append(self.target_position.copy())

        self._update_radar_cache()
        self.current_guide_points = self._assign_target_points()
        return self._get_obs(), {
            "infos": self.get_curriculum_info(),
            "individual_episode_rewards": self.individual_episode_reward,
        }

    def _place_pursuers_around_target(self):
        """Distribute pursuers on a Fibonacci sphere of radius spawn_radius around the
        evader, clipped to the arena and nudged out of buildings; face each toward the
        target. Enables cooperative encirclement of a faster evader from step 0."""
        dirs = g3.fibonacci_sphere(self.num_agents)
        for i in range(self.num_agents):
            p = (self.target_position + self.spawn_radius * dirs[i]).astype(np.float32)
            p[0] = np.clip(p[0], self.uav_radius, self.map_size - self.uav_radius)
            p[1] = np.clip(p[1], self.uav_radius, self.map_size - self.uav_radius)
            p[2] = np.clip(p[2], self.z_min, self.z_max)
            tries = 0
            while self._is_in_building(p, self.uav_radius) and tries < 12:
                p = p + np.random.uniform(-40, 40, 3).astype(np.float32)
                p[2] = np.clip(p[2], self.z_min, self.z_max)
                tries += 1
            self.uav_positions[i] = p
            v = self.target_position - self.uav_positions[i]
            self.uav_yaws[i] = float(np.arctan2(v[1], v[0]))
            self.uav_pitches[i] = float(np.clip(np.arctan2(v[2], np.hypot(v[0], v[1]) + 1e-6),
                                                -self.pitch_max, self.pitch_max))

    def _evader_step(self):
        rep = np.zeros(3, np.float32)
        tx, ty, tz = self.target_position
        # strong evader: wide sensing + firm pursuer-repulsion so it actively/early evades
        # (the old 60m + weak 6.0 coeff made it look passive; per client, that under-models
        # the adversary). Coefficient is now comparable to the wall/obstacle repulsion so
        # fleeing pursuers genuinely drives motion.
        sense = getattr(self, "evader_sense", 250.0)
        rep_coef = getattr(self, "evader_rep_coef", 40.0)
        is_sensed = False
        for pos in self.uav_positions:
            vec = self.target_position - pos
            dist = float(np.linalg.norm(vec))
            if dist < sense:
                is_sensed = True
                if dist > 0.1:
                    rep += (vec / dist) * (rep_coef / (dist + 5.0))
        if is_sensed:
            self.target_speed = min(self.target_max_speed, self.target_speed + self.target_accel)
        else:
            self.target_speed = max(self.target_min_speed, self.target_speed - self.target_accel)
        self.episode_target_speed_max = max(self.episode_target_speed_max, self.target_speed)

        wall = 100.0
        if tx < wall:
            rep[0] += 50.0 / (max(tx, 0.1) ** 1.5)
        if self.map_size - tx < wall:
            rep[0] -= 50.0 / (max(self.map_size - tx, 0.1) ** 1.5)
        if ty < wall:
            rep[1] += 50.0 / (max(ty, 0.1) ** 1.5)
        if self.map_size - ty < wall:
            rep[1] -= 50.0 / (max(self.map_size - ty, 0.1) ** 1.5)
        if tz - self.z_min < wall:
            rep[2] += 50.0 / (max(tz - self.z_min, 0.1) ** 1.5)
        if self.z_max - tz < wall:
            rep[2] -= 50.0 / (max(self.z_max - tz, 0.1) ** 1.5)

        if len(self.buildings) > 0:
            xmins, xmaxs = self.buildings[:, 0], self.buildings[:, 1]
            ymins, ymaxs = self.buildings[:, 2], self.buildings[:, 3]
            cx = np.clip(tx, xmins, xmaxs)
            cy = np.clip(ty, ymins, ymaxs)
            vx, vy = tx - cx, ty - cy
            dists = np.hypot(vx, vy)
            m = (dists < 20.0) & (dists > 0.1)
            if np.any(m):
                s = 40.0 / (dists[m] ** 1.5)
                rep[0] += float(np.sum((vx[m] / dists[m]) * s))
                rep[1] += float(np.sum((vy[m] / dists[m]) * s))
            if np.any(dists <= 0.1):
                rep[:2] += np.random.randn(2).astype(np.float32) * 100.0

        # Restoring pull toward the centre of the airspace.
        #
        # This term was removed earlier as a demo crutch. That was wrong for the question the
        # experiments are meant to answer. Without it the evader is pushed into the boundary and
        # pinned there: every capture measured, at both speed ratios, landed within 50 m of a
        # side wall (median 14-16 m), while the nearest building sat ~100 m away and played no
        # part. Three pursuers cannot bound an escape set in three dimensions on their own -- the
        # intersection of three half-spaces is unbounded -- so the walls were supplying the
        # missing faces. Real airspace has no such walls, which makes those captures artefacts of
        # the arena rather than results.
        #
        # Scaled so the pull at the arena edge is comparable to the pursuer and wall repulsions
        # (both O(0.1-1.5)); it is near zero in the middle, so the evader is free there and only
        # gets turned back as it runs out of room.
        if self.evader_center_pull > 0.0:
            centre = np.array([self.map_size * 0.5, self.map_size * 0.5,
                               0.5 * (self.z_min + self.z_max)], np.float32)
            off = centre - self.target_position
            # Each axis normalised by its own half-extent. The airspace is 1000 m across but
            # only 300 m tall, so scaling all three by the horizontal half-span made the
            # vertical pull ~3.3x too weak and simply moved the problem: captures left the side
            # walls (16 m -> 177 m) and reappeared against the floor (median altitude 80 m,
            # 30 m above the 50 m floor).
            span = np.array([self.map_size * 0.5, self.map_size * 0.5,
                             0.5 * max(self.z_max - self.z_min, 1e-6)], np.float32)
            u = off / span                       # componentwise fraction of the way out
            m = float(np.linalg.norm(u))
            if m > 1e-6:
                rep += self.evader_center_pull * u * min(1.0, 1.5 / m)

        horiz = rep[:2]
        hmag = float(np.linalg.norm(horiz))
        if hmag > 1e-3:
            desired_yaw = np.arctan2(horiz[1], horiz[0])
            self.target_yaw += np.clip((desired_yaw - self.target_yaw + np.pi) % (2 * np.pi) - np.pi,
                                       -np.pi / 4, np.pi / 4) + np.random.normal(0, 0.05)
        else:
            self.target_yaw += np.random.normal(0, 0.05)

        zpm = float(getattr(self, "target_pitch_max", self.pitch_max))
        desired_pitch = np.clip(np.arctan2(rep[2], max(hmag, 1e-3)), -zpm, zpm)
        self.target_pitch += float(np.clip(desired_pitch - self.target_pitch,
                                           -self.max_pitch_rate, self.max_pitch_rate))
        self.target_pitch = float(np.clip(self.target_pitch, -zpm, zpm))

        new_t, t_hit = self._move_with_clip_3d(self.target_position, self.target_yaw,
                                               self.target_pitch, self.target_speed, self.target_radius)
        self.target_position = new_t
        if t_hit:
            self.target_yaw += np.random.uniform(np.pi / 2, np.pi) * np.random.choice([-1, 1])
        self.target_trail.append(new_t.copy())

    def step(self, actions_dict):
        self._episode_step += 1
        rewards_dict = {a: 0.0 for a in self.agents}
        hit_flags = []
        collision_sources = {}
        old_positions = self.uav_positions.copy()

        for i, a in enumerate(self.agents):
            act = np.clip(actions_dict[a], -1.0, 1.0)
            if self.cbf_enabled:
                # Safety layer of Cheng et al. (AAAI-19): the policy proposes, a model-based
                # barrier filter applies the smallest correction that keeps the vehicle in the
                # safe set. Unlike relaxing the crash rule, this removes collisions rather than
                # rescoring them.
                act, _did, _eff = cbf_filter_action(
                    act, self.uav_positions[i], float(self.uav_yaws[i]),
                    float(self.uav_pitches[i]), float(self.uav_speeds[i]),
                    self.buildings, self.map_size, self.uav_radius,
                    self.uav_min_speed, self.uav_max_speed, self.max_accel,
                    self.max_yaw_rate, self.max_pitch_rate, self.pitch_max,
                    eta=self.cbf_eta, margin=self.cbf_margin)
                self.episode_cbf_steps += int(_did)
                self.episode_cbf_effort += float(_eff)
            self.uav_speeds[i] = np.clip(self.uav_speeds[i] + act[0] * self.max_accel,
                                         self.uav_min_speed, self.uav_max_speed)
            self.uav_yaws[i] = (self.uav_yaws[i] + act[1] * self.max_yaw_rate) % (2 * np.pi)
            self.uav_pitches[i] = np.clip(self.uav_pitches[i] + act[2] * self.max_pitch_rate,
                                          -self.pitch_max, self.pitch_max)
            sources = self._prospective_collision_sources(
                self.uav_positions[i], self.uav_yaws[i], self.uav_pitches[i],
                self.uav_speeds[i], self.uav_radius)
            new_pos, hit = self._move_with_clip_3d(self.uav_positions[i], self.uav_yaws[i],
                                                   self.uav_pitches[i], self.uav_speeds[i], self.uav_radius)
            self.uav_positions[i] = new_pos
            hit_flags.append(hit)
            if hit:
                collision_sources[a] = sources or ["unclassified"]
            self.uav_trails[a].append(new_pos.copy())

        # Smallest pursuer-pursuer gap this step. Always measured, so the cost of the historical
        # omission is quantifiable; only terminal when uav_collision_enabled is set.
        if self.num_agents > 1:
            _d = np.linalg.norm(self.uav_positions[:, None, :] - self.uav_positions[None, :, :], axis=-1)
            np.fill_diagonal(_d, np.inf)
            min_uav_gap = float(_d.min())
        else:
            min_uav_gap = float("inf")
        self.episode_min_uav_gap = min(self.episode_min_uav_gap, min_uav_gap)
        uav_conflict = bool(min_uav_gap <= self.uav_collision_distance)
        if uav_conflict:
            self.episode_uav_conflicts += 1
        if self.uav_collision_enabled and uav_conflict:
            for _i in range(self.num_agents):
                if float(np.min(np.delete(_d[_i], _i))) <= self.uav_collision_distance:
                    hit_flags[_i] = True
                    collision_sources.setdefault(self.agents[_i], []).append("uav:teammate")

        self._update_radar_cache()
        radar_min = [float(self._radar_cache[i].min()) * self.radar_range for i in range(self.num_agents)]

        self._evader_step()
        self.current_guide_points = self._assign_target_points()

        cur_dists = np.linalg.norm(self.uav_positions - self.target_position[None, :], axis=1)
        is_caught = bool(np.any(cur_dists <= self.catch_radius))

        w = self.reward_weights
        max_rel_speed = self.uav_max_speed + self.target_max_speed
        min_old = float(np.min(self.last_distances))
        min_new = float(np.min(cur_dists))
        r_team = float(np.clip((min_old - min_new) / max_rel_speed, -1.0, 1.0))

        # sphere distribution uniformity
        vecs = self.uav_positions - self.target_position[None, :]
        nv = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-6)
        uvecs = vecs / nv
        if self.num_agents > 1:
            cosM = np.clip(uvecs @ uvecs.T, -1.0, 1.0)
            np.fill_diagonal(cosM, 1.0)
            angle_pair = np.arccos(cosM) + np.eye(self.num_agents) * 10.0
            nn_angle = np.min(angle_pair, axis=1)
            mean_nn = float(np.mean(nn_angle))
            ideal = self._ideal_spherical_nn_angle(self.num_agents)
            r_sphere_uniform = float(np.exp(-abs(mean_nn - ideal)))
        else:
            r_sphere_uniform = 1.0

        field = self.apollonius_escape
        if field:
            geo = field["geometry_open"]
            controlled = field["controlled"]
            cs = field["control_strength"]
            owner = field["best_owner"]
            r_gap_global = float(np.exp(-field["escape_solid_angle"]))
            open_idx = np.where(geo)[0]
            if len(open_idx) == 0:
                r_boundary, r_owner = 1.0, 1.0
            else:
                r_boundary = float(np.mean(cs[open_idx]))
                ctrl_idx = np.where(geo & controlled & (owner >= 0))[0]
                if len(ctrl_idx) == 0:
                    r_owner = 0.0
                else:
                    counts = np.bincount(owner[ctrl_idx], minlength=self.num_agents).astype(np.float32)
                    probs = counts / max(float(np.sum(counts)), 1e-6)
                    ap = probs[probs > 0]
                    entropy = -float(np.sum(ap * np.log(ap + 1e-8)))
                    r_owner = entropy / np.log(self.num_agents)
            r_pos_global = 0.7 * (r_boundary * r_owner) + 0.3 * r_sphere_uniform
        else:
            r_pos_global = r_sphere_uniform
            r_gap_global = 0.0

        esa = float(field["escape_solid_angle"]) if field else 0.6 * 4 * np.pi
        closure_term = self.closure_weight * self._closure_bonus(
            esa, min_old, min_new, max_rel_speed)
        pincer_term = 0.0
        if self.pincer_weight > 0.0:
            pincer_term = self.pincer_weight * self._pincer_bonus(
                esa, np.sort(self.last_distances), np.sort(cur_dists), max_rel_speed)

        for i, a in enumerate(self.agents):
            dist = float(cur_dists[i])
            g_new = float(np.linalg.norm(self.uav_positions[i] - self.current_guide_points[a]))
            g_old = float(np.linalg.norm(old_positions[i] - self.current_guide_points[a]))
            r_guide = float(np.clip((g_old - g_new) / max_rel_speed, -1.0, 1.0))

            if hit_flags[i]:
                r_near = 0.0
                r_safe = -15.0
            else:
                r_near = (self.reward_mix["guide_progress"] * r_guide
                          + self.reward_mix["team_progress"] * r_team)
                radar_pen = ((radar_min[i] - self.radar_range) / self.radar_range) ** 2
                r_obs = -0.5 * self.obs_avoid_weight * radar_pen
                r_turn = -float(np.clip(actions_dict[a][1], -1.0, 1.0) ** 2)
                r_pitch = -float(np.clip(actions_dict[a][2], -1.0, 1.0) ** 2)
                r_mate = 0.0
                for j in range(self.num_agents):
                    if i == j:
                        continue
                    dmate = float(np.linalg.norm(self.uav_positions[i] - self.uav_positions[j]))
                    if dmate < 10.0:
                        r_mate -= (10.0 - dmate) / 10.0
                r_separation = self._teammate_separation_penalty(i)
                z = float(self.uav_positions[i, 2])
                thr = 30.0
                r_zlimit = 0.0
                if z - self.z_min < thr:
                    r_zlimit -= (thr - (z - self.z_min)) / thr
                if self.z_max - z < thr:
                    r_zlimit -= (thr - (self.z_max - z)) / thr
                r_safe = (r_obs + 0.5 * r_mate + 0.2 * r_turn + 0.2 * r_pitch
                          + 0.5 * r_zlimit + self.separation_weight * r_separation)

            r_finish = 0.0
            if is_caught:
                r_finish = 80.0
            elif dist < self.catch_radius * 3:
                r_finish = (self.catch_radius * 3 - dist) / (self.catch_radius * 2)
            # closure is carried on the finish channel at the same w_finish weighting as
            # the validated method. Split disable gates keep the full reward byte-identical
            # (neither flag set) while allowing a CLEAN independent ablation of the terminal
            # bonus ("r_finish") and the closure bonus ("r_closure").
            fin_part = 0.0 if "r_finish" in self.reward_disable else r_finish
            clo_part = 0.0 if "r_closure" in self.reward_disable else closure_term
            pin_part = 0.0 if "r_pincer" in self.reward_disable else pincer_term

            wr_near = 0.0 if "r_near" in self.reward_disable else w["w_near"] * r_near
            wr_safe = 0.0 if "r_safe" in self.reward_disable else w["w_safe"] * r_safe
            wr_pos = 0.0 if "r_pos" in self.reward_disable else w["w_pos"] * r_pos_global
            wr_gap = 0.0 if "r_gap" in self.reward_disable else w["w_gap"] * r_gap_global
            wr_finish = w["w_finish"] * (fin_part + clo_part + pin_part)
            rewards_dict[a] = wr_near + wr_safe + wr_pos + wr_gap + wr_finish
            sr = self.episode_sub_rewards[a]
            sr["r_near"] += wr_near
            sr["r_safe"] += wr_safe
            sr["r_pos"] += wr_pos
            sr["r_gap"] += wr_gap
            sr["r_finish"] += wr_finish

        self.last_distances = cur_dists.copy()
        for k, v in rewards_dict.items():
            self.individual_episode_reward[k] += v

        is_crashed = any(hit_flags)
        terminate_for_crash = bool(is_crashed and self.terminate_on_collision)
        terminated = {a: bool(is_caught or terminate_for_crash) for a in self.agents}
        truncated = (self._episode_step >= self.max_episode_steps) if not terminated[self.agents[0]] else False
        info = {
            "infos": self.get_curriculum_info(),
            "individual_episode_rewards": self.individual_episode_reward,
            "episode_sub_rewards": copy.deepcopy(self.episode_sub_rewards),
            "is_success": bool(is_caught),
            "is_collision": bool(is_crashed),
        }
        info.update({
            "collision_agents": [agent for agent in self.agents if agent in collision_sources],
            "collision_sources": collision_sources,
            "min_uav_gap": min_uav_gap,
            "cbf_steps": int(self.episode_cbf_steps),
            "cbf_effort": float(self.episode_cbf_effort),
            "episode_min_uav_gap": self.episode_min_uav_gap,
            "uav_conflict": uav_conflict,
            "episode_uav_conflicts": self.episode_uav_conflicts,
        })
        return self._get_obs(), rewards_dict, terminated, truncated, info

    # ============================================================
    # 3D render
    # ============================================================
    def render(self, mode="rgb_array", save_path=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(11, 8), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Multi-UAV Cooperative Pursuit")

        # buildings as full-height prisms
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            dx, dy, dz = xmax - xmin, ymax - ymin, self.z_max - self.z_min
            ax.bar3d(xmin, ymin, self.z_min, dx, dy, dz,
                     color="gray", alpha=0.25, shade=True)

        colors = ["#FF8C00", "#32CD32", "#00BFFF", "#9400D3", "#FF1493", "#1f77b4",
                  "#8c564b", "#e377c2"]
        # target
        tp = self.target_position
        if len(self.target_trail) > 1:
            tt = np.array(self.target_trail)
            ax.plot(tt[:, 0], tt[:, 1], tt[:, 2], color="red", alpha=0.6, linewidth=1.2)
        ax.scatter(tp[0], tp[1], tp[2], color="red", s=40, marker="o", label="Target")

        for i, a in enumerate(self.agents):
            c = colors[i % len(colors)]
            up = self.uav_positions[i]
            trail = list(self.uav_trails[a])
            if len(trail) > 1:
                tr = np.array(trail)
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, alpha=0.7, linewidth=1.5)
            ax.scatter(up[0], up[1], up[2], color=c, s=30, marker="o", edgecolors="black")
            if a in self.current_guide_points:
                gp = self.current_guide_points[a]
                ax.scatter(gp[0], gp[1], gp[2], color=c, s=40, marker="*")
                ax.plot([up[0], gp[0]], [up[1], gp[1]], [up[2], gp[2]],
                        linestyle="--", color=c, alpha=0.5, linewidth=1.0)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if mode == "rgb_array":
            fig.canvas.draw()
            img = np.array(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)
            return img
        plt.close(fig)
        return None

    def close(self):
        pass
