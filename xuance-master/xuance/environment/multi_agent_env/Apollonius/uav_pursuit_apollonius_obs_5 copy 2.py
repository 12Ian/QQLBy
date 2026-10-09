import numpy as np
import gymnasium as gym
import cv2
from collections import deque
from xuance.environment import RawMultiAgentEnv
import copy
from scipy.optimize import linear_sum_assignment

class UAVPursuitApolloniusObs5Env(RawMultiAgentEnv):
    def __init__(self, config):
        super(UAVPursuitApolloniusObs5Env, self).__init__()

        # ---------------- 1. 真实物理环境参数 ----------------
        self.map_size = 1000.0

        self.uav_min_speed = 9.0
        self.uav_max_speed = 11.0
        self.max_accel = 1
        self.max_yaw_rate = np.pi / 4  
        self.uav_radius = 0.5
 
        # [新增]: 逃逸机动态速度参数
        self.target_min_speed = 4.0
        self.target_max_speed = 11.0
        self.target_accel = 0.5     # 每次 step 提速/减速的步长
        self.target_speed = self.target_min_speed
        
        self.target_radius = 0.5
        self.catch_radius = 15.0
        self.cartesian_lambda_cap = getattr(config, "cartesian_lambda_cap", 0.8)
        self.cartesian_strength_weight = getattr(config, "cartesian_strength_weight", 0.3)
        self._setup_curriculum(config)
        
        self.radar_range = 100.0
        self.num_radar_rays = 16

        self.num_agents = 4
        self.agents = [f"uav_{i}" for i in range(self.num_agents)]

        self.buildings = np.array(self._generate_city_blocks(), dtype=np.float32)

        self.init_uav_positions = np.array([
            [100.0, 200.0],
            [100.0, 400.0],
            [100.0, 600.0],
            [100.0, 800.0]
        ], dtype=np.float32)

        # ---------------- 2. 绘图样式参数 ----------------
        plot_config = getattr(config, "plot_config", {})
        self.uav_plot_radius = plot_config.get("uav_radius", 4)
        self.target_plot_radius = plot_config.get("target_radius", 4)
        self.font_size_base = plot_config.get("font_size_base", 20)

        # ---------------- 3. 定义状态与动作空间 ----------------
        self.action_space = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent in self.agents}

        # 动态计算 obs_dim，防止后续加减特征时维度对不上
        # 组成: norm_pos(2) + yaw_vec(2) + norm_speed(1) + rel_guide(2) + rel_mates(2*3=6) + radar(16) + target_yaw(2) + norm_target_speed(1) = 32
        self.obs_dim = 2 + 2 + 1 + 2 + (self.num_agents - 1) * 2 + self.num_radar_rays + 2 + 1
        self.observation_space = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32) for agent in self.agents}
        
        # 动态计算 state_dim，与 state() 函数的返回值严格对齐
        # 组成: uav_pos(4*2=8) + uav_yaws(4) + uav_speeds(4) + target_pos(2) + target_yaw(1) + target_speed(1) + guides(4*2=8) = 28
        self.state_dim = self.num_agents * 2 + self.num_agents + self.num_agents + 2 + 1 + 1 + self.num_agents * 2
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # ---------------- 4. 运行时变量与轨迹缓存初始化 ----------------
        self.max_episode_steps = getattr(config, "episode_length", 400)
        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

        self.uav_positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.uav_yaws = np.zeros(self.num_agents, dtype=np.float32)
        self.uav_speeds = np.zeros(self.num_agents, dtype=np.float32)

        self.target_position = np.zeros(2, dtype=np.float32)
        self.target_yaw = 0.0
        self.last_distances = np.zeros(self.num_agents, dtype=np.float32)

        self.uav_trails = {agent: deque(maxlen=100) for agent in self.agents}
        self.target_trail = deque(maxlen=100)

        self.episode_sub_rewards = {
            agent: {
                "r_near": 0.0,
                "r_safe": 0.0,
                "r_pos": 0.0,
                "r_gap": 0.0,
                "r_finish": 0.0,
            }
            for agent in self.agents
        }
        # [新增]: 用于存储渲染所需的射线与扇区可视化数据
        self.debug_rays = []
        self.debug_rf = 0.0
        self.apollonius_escape = {}
        self.apollonius_margin_ema = None
        self.apollonius_dangerous_prev = None
        self.guide_polar_state = {}
        self.current_guide_points = {}


    def _default_curriculum_schedule(self):
        return [
            {
                "level": 0,
                "target_min_speed": 3.0,
                "target_max_speed": 5.0,
                "target_accel": 0.2,
                "spawn_offset": 20.0,
                "building_mode": "empty",
                "reward_weights": {
                    "w_near": 2.5,
                    "w_safe": 1.0,
                    "w_pos": 0.2,
                    "w_gap": 0.2,
                    "w_finish": 3.0,
                },
                "reward_mix": {"guide_progress": 0.3, "team_progress": 0.7},
            },
            {
                "level": 1,
                "target_min_speed": 4.0,
                "target_max_speed": 7.0,
                "target_accel": 0.3,
                "spawn_offset": 40.0,
                "building_mode": "empty",
                "reward_weights": {
                    "w_near": 2.2,
                    "w_safe": 1.0,
                    "w_pos": 0.4,
                    "w_gap": 0.4,
                    "w_finish": 3.0,
                },
                "reward_mix": {"guide_progress": 0.35, "team_progress": 0.65},
            },
            {
                "level": 2,
                "target_min_speed": 4.0,
                "target_max_speed": 9.0,
                "target_accel": 0.4,
                "spawn_offset": 60.0,
                "building_mode": "easy",
                "reward_weights": {
                    "w_near": 2.0,
                    "w_safe": 1.0,
                    "w_pos": 0.6,
                    "w_gap": 0.8,
                    "w_finish": 3.0,
                },
                "reward_mix": {"guide_progress": 0.4, "team_progress": 0.6},
            },
            {
                "level": 3,
                "target_min_speed": 4.0,
                "target_max_speed": 10.0,
                "target_accel": 0.5,
                "spawn_offset": 80.0,
                "building_mode": "medium",
                "reward_weights": {
                    "w_near": 2.0,
                    "w_safe": 1.0,
                    "w_pos": 0.8,
                    "w_gap": 1.0,
                    "w_finish": 3.0,
                },
                "reward_mix": {"guide_progress": 0.45, "team_progress": 0.55},
            },
            {
                "level": 4,
                "target_min_speed": 4.0,
                "target_max_speed": 11.0,
                "target_accel": 0.5,
                "spawn_offset": 100.0,
                "building_mode": "medium",
                "reward_weights": {
                    "w_near": 2.0,
                    "w_safe": 1.0,
                    "w_pos": 0.8,
                    "w_gap": 1.0,
                    "w_finish": 3.5,
                },
                "reward_mix": {"guide_progress": 0.5, "team_progress": 0.5},
            },
        ]

    def _setup_curriculum(self, config):
        self.curriculum_enabled = bool(getattr(config, "curriculum_enabled", True))
        self.curriculum_schedule = getattr(config, "curriculum_schedule", None)
        if self.curriculum_schedule is None:
            self.curriculum_schedule = self._default_curriculum_schedule()

        self.curriculum_min_steps_per_level = int(
            getattr(config, "curriculum_min_steps_per_level", 300000)
        )
        self.curriculum_success_threshold = float(
            getattr(config, "curriculum_success_threshold", 0.75)
        )
        self.curriculum_drop_threshold = float(
            getattr(config, "curriculum_drop_threshold", 0.45)
        )
        self.curriculum_level = int(getattr(config, "curriculum_start_level", 0))
        self.curriculum_last_update_step = 0

        self.spawn_offset = 50.0
        self.building_mode = getattr(config, "building_mode", "medium")
        self.reward_weights = {
            "w_near": 2.0,
            "w_safe": 1.0,
            "w_pos": 1.2,
            "w_gap": 1.5,
            "w_finish": 2.0,
        }
        self.reward_mix = {"guide_progress": 0.6, "team_progress": 0.4}
        self.curriculum_config = {}

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
        self.target_speed = min(
            max(float(getattr(self, "target_speed", self.target_min_speed)), self.target_min_speed),
            self.target_max_speed,
        )
        self.spawn_offset = float(self.curriculum_config["spawn_offset"])
        self.building_mode = self.curriculum_config["building_mode"]
        self.reward_weights = copy.deepcopy(self.curriculum_config["reward_weights"])
        self.reward_mix = copy.deepcopy(self.curriculum_config["reward_mix"])

        if reload_buildings and hasattr(self, "buildings"):
            self.buildings = np.array(self._generate_city_blocks(), dtype=np.float32)

        return self.curriculum_level

    def update_curriculum(self, success_rate, global_step):
        if not self.curriculum_enabled:
            return self.curriculum_level

        steps_since_update = int(global_step) - int(self.curriculum_last_update_step)
        if steps_since_update < self.curriculum_min_steps_per_level:
            return self.curriculum_level

        old_level = self.curriculum_level
        if success_rate >= self.curriculum_success_threshold:
            self.set_curriculum_level(self.curriculum_level + 1)
        elif success_rate <= self.curriculum_drop_threshold:
            self.set_curriculum_level(self.curriculum_level - 1)

        if self.curriculum_level != old_level:
            self.curriculum_last_update_step = int(global_step)

        return self.curriculum_level


    def _generate_city_blocks(self):
        import json
        import os
        difficulty = getattr(self, "building_mode", "medium")
        if difficulty == "empty":
            return []
        # 请根据你的实际路径修改
        config_file = os.path.join(r"/home/ryy/UAV_Project/xuance-master/xuance/environment/buildings",
                                   f"buildings_{difficulty}.json")

        if os.path.exists(config_file):
            print(f"🌍 [Env Info] Loading {difficulty} map from: {config_file}")
            with open(config_file, 'r', encoding='utf-8') as f:
                buildings = json.load(f)
            return buildings
        else:
            print(f"⚠️ [Warning] Config file {config_file} not found! Loading empty map.")
            return []

    def _check_los(self, p1, p2):
        """
        快速视线检测 (Line-of-Sight Check)
        判断两点 p1 和 p2 之间的连线是否被建筑物阻挡。
        返回: True (无遮挡), False (被建筑物阻挡)
        """
        # 提前计算 AABB 快速排除（Bounding Box Check）
        min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
        min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])

        D = p2 - p1
        
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            
            # 1. 快速排除：如果线段的包围盒与建筑物不相交，直接跳过
            if max_x < xmin or min_x > xmax or max_y < ymin or min_y > ymax:
                continue

            # 2. 精确参数化线段求交
            t_enter = 0.0
            t_exit = 1.0

            # X 轴方向碰撞测试
            if D[0] == 0:
                if p1[0] < xmin or p1[0] > xmax:
                    continue
            else:
                tx1 = (xmin - p1[0]) / D[0]
                tx2 = (xmax - p1[0]) / D[0]
                t_enter = max(t_enter, min(tx1, tx2))
                t_exit = min(t_exit, max(tx1, tx2))

            # Y 轴方向碰撞测试
            if D[1] == 0:
                if p1[1] < ymin or p1[1] > ymax:
                    continue
            else:
                ty1 = (ymin - p1[1]) / D[1]
                ty2 = (ymax - p1[1]) / D[1]
                t_enter = max(t_enter, min(ty1, ty2))
                t_exit = min(t_exit, max(ty1, ty2))

            # 命中判断：如果进入时间小于退出时间，且交点在线段内部 [0, 1]
            if t_enter <= t_exit and t_exit >= 0 and t_enter <= 1.0:
                return False  # 视线被阻挡
                
        return True  # 视线畅通

    def _raycast(self, pos, angles, max_range):
        dx, dy = np.cos(angles), np.sin(angles)
        tx = np.where(dx > 0, (self.map_size - pos[0]) / (dx + 1e-8), (0 - pos[0]) / (dx - 1e-8))
        ty = np.where(dy > 0, (self.map_size - pos[1]) / (dy + 1e-8), (0 - pos[1]) / (dy - 1e-8))
        d_border = np.maximum(0, np.minimum(tx, ty))
        if len(self.buildings) > 0:
            t1 = (self.buildings[:, 0:1] - pos[0]) / (dx + 1e-8)
            t2 = (self.buildings[:, 1:2] - pos[0]) / (dx + 1e-8)
            t3 = (self.buildings[:, 2:3] - pos[1]) / (dy + 1e-8)
            t4 = (self.buildings[:, 3:4] - pos[1]) / (dy + 1e-8)
            t_enter = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
            t_exit = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            valid_hit = (t_exit >= 0) & (t_enter <= t_exit) & (t_enter > 0)
            d_buildings = np.min(np.where(valid_hit, t_enter, np.inf), axis=0)
        else:
            d_buildings = np.full_like(angles, np.inf)
        return np.minimum(np.minimum(d_border, d_buildings), max_range) / max_range

    def _angle_diff(self, a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def _split_open_sectors(self, open_indices, num_rays):
        if len(open_indices) == 0:
            return []

        sectors = []
        current = [int(open_indices[0])]

        for k in range(1, len(open_indices)):
            idx = int(open_indices[k])
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                sectors.append(current)
                current = [idx]
        sectors.append(current)

        if len(sectors) > 1 and sectors[0][0] == 0 and sectors[-1][-1] == num_rays - 1:
            sectors[0] = sectors[-1] + sectors[0]
            sectors.pop()

        return sectors

    def _sector_mid_angle(self, sector, num_rays):
        angles = np.array(sector, dtype=np.float32) * (2 * np.pi / num_rays)
        if sector[0] > sector[-1]:
            angles = np.where(angles < np.pi, angles + 2 * np.pi, angles)
        return float(np.mean(angles) % (2 * np.pi))

    def _sector_width(self, sector, num_rays):
        return max(1, len(sector)) * (2 * np.pi / num_rays)

    def _make_point(self, angle, radius, safe_margin):
        p = self.target_position + radius * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        p[0] = np.clip(p[0], safe_margin, self.map_size - safe_margin)
        p[1] = np.clip(p[1], safe_margin, self.map_size - safe_margin)
        return p.astype(np.float32)

    def _cartesian_half_occupied_angles(self):
        dists = np.linalg.norm(
            self.uav_positions - self.target_position.reshape(1, 2),
            axis=1,
        )
        dists = np.maximum(dists, 1e-6)

        capture_ratio = np.clip(self.catch_radius / dists, 0.0, 1.0)
        if hasattr(self, "uav_speeds") and len(self.uav_speeds) == len(self.uav_positions):
            pursuer_speeds = np.asarray(self.uav_speeds, dtype=np.float32)
        else:
            pursuer_speeds = np.full(len(self.uav_positions), self.uav_max_speed, dtype=np.float32)
        lambda_cap = float(getattr(self, "cartesian_lambda_cap", 0.8))
        speed_ratio = np.clip(
            pursuer_speeds / max(float(self.target_speed), 1e-6),
            0.0,
            lambda_cap,
        )

        half_angles = np.arcsin(capture_ratio) + np.arcsin(speed_ratio)
        return np.clip(half_angles, 0.0, np.pi).astype(np.float32)

    def _compute_cartesian_escape_coverage(self, angles):
        bearings = np.arctan2(
            self.uav_positions[:, 1] - self.target_position[1],
            self.uav_positions[:, 0] - self.target_position[0],
        )
        half_angles = self._cartesian_half_occupied_angles()

        angle_diffs = np.abs(
            self._angle_diff(angles.reshape(1, -1), bearings.reshape(-1, 1))
        )
        angular_margins = half_angles.reshape(-1, 1) - angle_diffs
        owners = np.argmax(angular_margins, axis=0).astype(np.int32)
        best_margin = angular_margins[owners, np.arange(len(angles))]
        controlled = best_margin >= 0.0

        owner_half_angles = np.maximum(half_angles[owners], 1e-6)
        strength = np.clip(best_margin / owner_half_angles, 0.0, 1.0).astype(np.float32)

        return {
            "controlled": controlled,
            "best_owner": owners,
            "best_margin": best_margin.astype(np.float32),
            "strength": strength,
            "half_angles": half_angles,
            "bearings": bearings.astype(np.float32),
        }

    def _compute_apollonius_escape_field(self, angles, min_dists, r_f):
        """Classify each escape ray by time reachability with Cartesian support.

        Cartesian oval coverage is used as a soft confidence signal. It should
        not remove a ray from the dangerous set by itself because turn limits,
        acceleration limits, obstacles, and decentralized control make the
        theoretical occupied angle optimistic in this environment.
        """
        vt = max(float(self.target_speed), 1e-6)
        vu = max(float(self.uav_max_speed), 1e-6)
        n_rays = len(angles)

        # Geometric openness is only the first filter. Apollonius reachability
        # decides whether the open ray is actually dangerous.
        geometry_open = min_dists > max(self.catch_radius * 1.2, r_f * 0.8)
        cartesian_coverage = self._compute_cartesian_escape_coverage(angles)
        cartesian_controlled = geometry_open & cartesian_coverage["controlled"]

        controlled = np.zeros(n_rays, dtype=bool)
        dangerous = np.zeros(n_rays, dtype=bool)
        best_owner = np.full(n_rays, -1, dtype=np.int32)
        best_radius = np.full(n_rays, r_f, dtype=np.float32)
        best_margin = np.full(n_rays, np.inf, dtype=np.float32)
        control_strength = np.zeros(n_rays, dtype=np.float32)

        min_probe = max(self.catch_radius * 1.2, 1.0)
        max_probe_base = max(r_f * 1.8, self.catch_radius * 4.0)
        probe_count = 12

        for k, angle in enumerate(angles):
            if not geometry_open[k]:
                continue

            ray_limit = min(float(min_dists[k]), max_probe_base)
            if ray_limit <= min_probe:
                continue

            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            radii = np.linspace(min_probe, ray_limit, probe_count, dtype=np.float32)
            ray_points = self.target_position.reshape(1, 2) + radii.reshape(-1, 1) * direction.reshape(1, 2)
            target_times = radii / vt

            for i, uav_pos in enumerate(self.uav_positions):
                uav_times = np.linalg.norm(ray_points - uav_pos.reshape(1, 2), axis=1) / vu
                margins = uav_times - target_times
                local_idx = int(np.argmin(margins))
                local_margin = float(margins[local_idx])

                if local_margin < best_margin[k]:
                    best_margin[k] = local_margin
                    best_owner[k] = i
                    best_radius[k] = radii[local_idx]

        finite_margin = np.where(np.isfinite(best_margin), best_margin, 10.0)
        if self.apollonius_margin_ema is None or len(self.apollonius_margin_ema) != n_rays:
            margin_ema = finite_margin.astype(np.float32)
        else:
            margin_ema = (0.65 * self.apollonius_margin_ema + 0.35 * finite_margin).astype(np.float32)

        prev_dangerous = self.apollonius_dangerous_prev
        if prev_dangerous is None or len(prev_dangerous) != n_rays:
            prev_dangerous = np.zeros(n_rays, dtype=bool)

        # Hysteresis: once a ray is dangerous, require a stronger safety margin
        # before removing it; once safe, require a stronger late margin to add it.
        time_dangerous = np.where(prev_dangerous, margin_ema > -0.10, margin_ema > 0.20)
        dangerous = geometry_open & time_dangerous
        controlled = geometry_open & ~dangerous
        time_strength = 1.0 / (1.0 + np.exp(np.clip(1.25 * margin_ema, -60.0, 60.0)))
        cartesian_weight = np.clip(float(getattr(self, "cartesian_strength_weight", 0.3)), 0.0, 1.0)
        control_strength = (
            (1.0 - cartesian_weight) * time_strength
            + cartesian_weight * cartesian_coverage["strength"]
        ).astype(np.float32)

        replace_owner = cartesian_controlled & (
            (best_owner < 0) | (cartesian_coverage["strength"] >= time_strength)
        )
        best_owner = np.where(replace_owner, cartesian_coverage["best_owner"], best_owner)

        self.apollonius_margin_ema = margin_ema
        self.apollonius_dangerous_prev = dangerous.copy()

        self.apollonius_escape = {
            "angles": angles,
            "geometry_open": geometry_open,
            "cartesian_controlled": cartesian_controlled,
            "cartesian_margin": cartesian_coverage["best_margin"],
            "cartesian_strength": cartesian_coverage["strength"],
            "cartesian_half_angles": cartesian_coverage["half_angles"],
            "cartesian_bearings": cartesian_coverage["bearings"],
            "controlled": controlled,
            "dangerous": dangerous,
            "best_owner": best_owner,
            "best_radius": best_radius,
            "best_margin": margin_ema,
            "raw_margin": best_margin,
            "control_strength": control_strength,
        }
        return self.apollonius_escape

    def _assign_target_points(self):
        cur_dists_to_target = np.array(
            [np.linalg.norm(pos - self.target_position) for pos in self.uav_positions],
            dtype=np.float32,
        )
        avg_dist = float(np.mean(cur_dists_to_target))

        safe_margin = 50.0
        num_rays = 72
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        dx = np.cos(angles)
        dy = np.sin(angles)

        speed_ratio = self.target_speed / max(self.uav_max_speed, 1e-6)

        # Apollonius-style interception radius:
        # faster evaders and larger initial gaps need earlier interception points.
        r_by_dist = avg_dist * (0.25 + 0.35 * speed_ratio)
        r_by_speed = self.target_speed * 5.0
        r_f = np.clip(
            max(r_by_dist, r_by_speed, self.catch_radius * 1.5),
            self.catch_radius * 1.2,
            self.catch_radius * 8.0,
        )
        self.debug_rf = r_f

        tx_bound = np.where(
            dx > 0,
            (self.map_size - safe_margin - self.target_position[0]) / (dx + 1e-8),
            (safe_margin - self.target_position[0]) / (dx - 1e-8),
        )
        ty_bound = np.where(
            dy > 0,
            (self.map_size - safe_margin - self.target_position[1]) / (dy + 1e-8),
            (safe_margin - self.target_position[1]) / (dy - 1e-8),
        )

        tx_bound = np.maximum(0.0, tx_bound)
        ty_bound = np.maximum(0.0, ty_bound)
        min_dists = np.minimum(tx_bound, ty_bound)

        if len(self.buildings) > 0:
            xmins = self.buildings[:, 0:1]
            xmaxs = self.buildings[:, 1:2]
            ymins = self.buildings[:, 2:3]
            ymaxs = self.buildings[:, 3:4]

            dx_b = dx.reshape(1, -1)
            dy_b = dy.reshape(1, -1)

            tx1 = (xmins - self.target_position[0]) / (dx_b + 1e-8)
            tx2 = (xmaxs - self.target_position[0]) / (dx_b + 1e-8)
            ty1 = (ymins - self.target_position[1]) / (dy_b + 1e-8)
            ty2 = (ymaxs - self.target_position[1]) / (dy_b + 1e-8)

            tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))

            valid_hit = (tmax >= 0) & (tmin <= tmax) & (tmin > 0)
            building_dists = np.min(np.where(valid_hit, tmin, np.inf), axis=0)
            min_dists = np.minimum(min_dists, building_dists)

        escape_field = self._compute_apollonius_escape_field(angles, min_dists, r_f)
        is_open = escape_field["geometry_open"]
        is_dangerous = escape_field["dangerous"]
        best_radius = escape_field["best_radius"]

        plot_dists = np.minimum(min_dists, r_f * 1.3)
        self.debug_rays = list(zip(angles, plot_dists, is_open))

        open_indices = np.where(is_dangerous)[0]
        target_points = []

        if len(open_indices) == 0:
            # All geometric escapes are already contestable. Tighten the net
            # around the target while still staying outside the capture radius.
            close_r = self.catch_radius * 1.4
            base_angles = [
                self.target_yaw,
                self.target_yaw + np.pi / 2,
                self.target_yaw - np.pi / 2,
                self.target_yaw + np.pi,
            ]
            target_points = [self._make_point(a, close_r, safe_margin) for a in base_angles]
        elif len(open_indices) == num_rays:
            # Open field: generate interception points in the target-heading frame.
            base_angles = [
                self.target_yaw,
                self.target_yaw + np.pi / 2,
                self.target_yaw - np.pi / 2,
                self.target_yaw + np.pi,
            ]
            radii = [r_f * 1.2, r_f, r_f, r_f * 0.85]
            target_points = [
                self._make_point(angle, radius, safe_margin)
                for angle, radius in zip(base_angles, radii)
            ]
        else:
            sectors = self._split_open_sectors(open_indices, num_rays)
            sector_infos = []

            for sector in sectors:
                mid_angle = self._sector_mid_angle(sector, num_rays)
                width = self._sector_width(sector, num_rays)

                heading_align = np.cos(self._angle_diff(mid_angle, self.target_yaw))
                heading_score = 0.25 + 0.75 * ((heading_align + 1.0) / 2.0)

                sector_margins = escape_field["best_margin"][sector]
                sector_control = escape_field["control_strength"][sector]
                mean_lateness = float(np.mean(np.maximum(sector_margins, 0.0)))
                mean_uncontrolled = float(np.mean(1.0 - sector_control))

                # Wider, forward-facing, less controllable sectors get more UAVs.
                score = width * heading_score * (1.0 + mean_uncontrolled + 0.3 * mean_lateness)
                sector_infos.append(
                    {
                        "sector": sector,
                        "mid_angle": mid_angle,
                        "width": width,
                        "score": float(score),
                    }
                )

            sector_infos = sorted(sector_infos, key=lambda x: x["score"], reverse=True)
            if len(sector_infos) > self.num_agents:
                sector_infos = sector_infos[: self.num_agents]

            total_score = sum(s["score"] for s in sector_infos) + 1e-8
            agents_left = self.num_agents
            sectors_left = len(sector_infos)

            for idx, info in enumerate(sector_infos):
                if idx == len(sector_infos) - 1:
                    num_for_sector = agents_left
                else:
                    raw = info["score"] / total_score * self.num_agents
                    num_for_sector = int(np.round(raw))
                    num_for_sector = max(1, num_for_sector)
                    num_for_sector = min(
                        num_for_sector, agents_left - (sectors_left - 1)
                    )

                sector = info["sector"]
                start = sector[0] * (2 * np.pi / num_rays)
                end = sector[-1] * (2 * np.pi / num_rays)
                if sector[0] > sector[-1]:
                    end += 2 * np.pi

                width = max(end - start, 2 * np.pi / num_rays)
                step = width / num_for_sector

                for k in range(num_for_sector):
                    raw_angle = start + step * (k + 0.5)
                    ray_idx = int(np.round((raw_angle % (2 * np.pi)) / (2 * np.pi) * num_rays)) % num_rays
                    angle = angles[ray_idx]
                    frontness = (np.cos(self._angle_diff(angle, self.target_yaw)) + 1.0) / 2.0
                    # Place the guide near the Apollonius contest boundary for
                    # that ray, not on a fixed enclosing circle.
                    boundary_radius = float(best_radius[ray_idx])
                    radius = np.clip(
                        0.75 * boundary_radius + 0.25 * r_f * (0.85 + 0.35 * frontness),
                        self.catch_radius * 1.2,
                        min(float(min_dists[ray_idx]) * 0.9, r_f * 1.8),
                    )
                    target_points.append(self._make_point(angle, radius, safe_margin))

                agents_left -= num_for_sector
                sectors_left -= 1

        while len(target_points) < self.num_agents:
            angle = self.target_yaw + 2 * np.pi * len(target_points) / self.num_agents
            target_points.append(self._make_point(angle, r_f, safe_margin))

        target_points = target_points[: self.num_agents]
        cost_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)

        for i, uav_pos in enumerate(self.uav_positions):
            agent = self.agents[i]
            uav_yaw = self.uav_yaws[i]
            prev_guide = self.current_guide_points.get(agent, None)

            for j, tp in enumerate(target_points):
                vec_to_tp = tp - uav_pos
                dist = np.linalg.norm(vec_to_tp)
                ideal_yaw = np.arctan2(vec_to_tp[1], vec_to_tp[0])
                angle_cost = abs(self._angle_diff(uav_yaw, ideal_yaw))

                los_cost = 0.0 if self._check_los(uav_pos, tp) else 1500.0

                consistency_cost = 0.0
                if prev_guide is not None:
                    consistency_cost = 0.8 * np.linalg.norm(tp - prev_guide)

                target_to_tp = np.linalg.norm(tp - self.target_position)
                target_time = target_to_tp / max(self.target_speed, 1e-6)
                uav_time = dist / max(self.uav_max_speed, 1e-6)
                late_time = max(0.0, uav_time - target_time)
                apollonius_cost = 25.0 * late_time

                cost_matrix[i, j] = (
                    dist
                    + 10.0 * angle_cost
                    + los_cost
                    + consistency_cost
                    + apollonius_cost
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment = {}
        target_speed_ratio = self.target_speed / max(self.target_max_speed, 1e-6)
        old_w = 0.65 - 0.25 * target_speed_ratio
        new_w = 1.0 - old_w

        for idx in range(self.num_agents):
            agent = self.agents[row_ind[idx]]
            new_target = target_points[col_ind[idx]]

            new_vec = new_target - self.target_position
            new_radius = float(np.linalg.norm(new_vec))
            new_angle = float(np.arctan2(new_vec[1], new_vec[0]))

            if agent in self.guide_polar_state:
                old_angle, old_radius = self.guide_polar_state[agent]
                angle_delta = self._angle_diff(new_angle, old_angle)
                radius_delta = abs(new_radius - old_radius)

                # Large sector changes should be allowed to respond quickly,
                # while ordinary jitter is smoothed in target-relative polar space.
                if abs(angle_delta) < np.pi * 0.75 and radius_delta < 70.0:
                    smooth_angle = old_angle + new_w * angle_delta
                    smooth_radius = old_w * old_radius + new_w * new_radius
                else:
                    smooth_angle = new_angle
                    smooth_radius = new_radius

                smooth_target = self._make_point(smooth_angle, smooth_radius, 50.0)
                self.guide_polar_state[agent] = (smooth_angle % (2 * np.pi), smooth_radius)
                assignment[agent] = smooth_target.astype(np.float32)
            else:
                self.guide_polar_state[agent] = (new_angle % (2 * np.pi), new_radius)
                assignment[agent] = new_target.astype(np.float32)

        return assignment

    def _get_obs(self):
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            pos, yaw = self.uav_positions[i], self.uav_yaws[i]
            
            norm_pos = (pos / self.map_size) * 2.0 - 1.0
            yaw_vec = np.array([np.cos(yaw), np.sin(yaw)])
            
            norm_speed = np.array([(self.uav_speeds[i] - self.uav_min_speed) / (self.uav_max_speed - self.uav_min_speed)])
            norm_speed_scaled = norm_speed * 2.0 - 1.0
            
            # 指向各自专属引导点的位置向量
            guide_pos = self.current_guide_points[agent]
            rel_guide = (guide_pos - pos) / self.map_size
            
            rel_mates = []
            for j in range(self.num_agents):
                if i != j: 
                    rel_mates.append((self.uav_positions[j] - pos) / self.map_size)
            rel_mates = np.concatenate(rel_mates) if rel_mates else np.array([])

            radar_data = self._raycast(pos, yaw + np.linspace(0, 2 * np.pi, self.num_radar_rays, endpoint=False), self.radar_range)
            radar_data_scaled = radar_data * 2.0 - 1.0
            
            target_yaw_vec = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw)])
            
            # 动态反馈逃逸机的实时速度映射
            norm_target_speed = np.array([(self.target_speed / self.target_max_speed) * 2.0 - 1.0])

            obs_dict[agent] = np.concatenate(
                [norm_pos, yaw_vec, norm_speed_scaled, rel_guide, rel_mates, radar_data_scaled, target_yaw_vec, norm_target_speed]
            ).astype(np.float32)
            
        return obs_dict

    def _move_with_clip(self, pos, yaw, speed, radius):
        dx, dy = speed * np.cos(yaw), speed * np.sin(yaw)
        next_pos = pos + np.array([dx, dy])
        hit_obstacle = False
        if next_pos[0] < radius:
            next_pos[0] = radius; hit_obstacle = True
        elif next_pos[0] > self.map_size - radius:
            next_pos[0] = self.map_size - radius; hit_obstacle = True
        if next_pos[1] < radius:
            next_pos[1] = radius; hit_obstacle = True
        elif next_pos[1] > self.map_size - radius:
            next_pos[1] = self.map_size - radius; hit_obstacle = True
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < next_pos[0] < xmax + radius) and (ymin - radius < next_pos[1] < ymax + radius):
                hit_obstacle = True
                pos_x = np.array([next_pos[0], pos[1]])
                if not ((xmin - radius < pos_x[0] < xmax + radius) and (ymin - radius < pos_x[1] < ymax + radius)): 
                    next_pos = pos_x; break
                pos_y = np.array([pos[0], next_pos[1]])
                if not ((xmin - radius < pos_y[0] < xmax + radius) and (ymin - radius < pos_y[1] < ymax + radius)): 
                    next_pos = pos_y; break
                next_pos = pos; break
        return next_pos, hit_obstacle

    def _is_in_building(self, pos, radius):
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < pos[0] < xmax + radius) and (ymin - radius < pos[1] < ymax + radius):
                return True
        return False

    def reset(self):
        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

        for agent in self.agents:
            for k in self.episode_sub_rewards[agent].keys():
                self.episode_sub_rewards[agent][k] = 0.0

        for agent in self.agents: self.uav_trails[agent].clear()
        self.target_trail.clear()
        self.apollonius_escape = {}
        self.apollonius_margin_ema = None
        self.apollonius_dangerous_prev = None
        self.guide_polar_state = {}
        
        self.uav_positions = self.init_uav_positions.copy()

        center_pos = self.map_size / 2.0
        spawn_offset = self.spawn_offset

        while True:
            t_pos = np.random.uniform(low=center_pos - spawn_offset, high=center_pos + spawn_offset, size=2).astype(np.float32)
            if not self._is_in_building(t_pos, self.target_radius):
                self.target_position = t_pos
                break

        self.uav_speeds = np.ones(self.num_agents, dtype=np.float32) * self.uav_min_speed
        self.uav_yaws = np.random.uniform(0, 2 * np.pi, size=self.num_agents).astype(np.float32)
        
        self.target_yaw = np.random.uniform(0, 2 * np.pi)
        self.target_speed = self.target_min_speed # 初始未被发现，速度为4

        for i in range(self.num_agents):
            self.last_distances[i] = np.linalg.norm(self.uav_positions[i] - self.target_position)

        for i, agent in enumerate(self.agents): self.uav_trails[agent].append(self.uav_positions[i].copy())
        self.target_trail.append(self.target_position.copy())

        self.current_guide_points = self._assign_target_points()

        return self._get_obs(), {
            "infos": {"curriculum_level": self.curriculum_level},
            "individual_episode_rewards": self.individual_episode_reward,
        }

    def step(self, actions_dict):
        self._episode_step += 1
        rewards_dict = {agent: 0.0 for agent in self.agents}
        hit_flags, radar_distances = [], []

        old_uav_positions = self.uav_positions.copy()

        # ==================== 1. 物理运动与观测更新 ====================
        for i, agent in enumerate(self.agents):
            action = np.clip(actions_dict[agent], -1.0, 1.0)
            self.uav_speeds[i] = np.clip(self.uav_speeds[i] + action[0] * self.max_accel, self.uav_min_speed, self.uav_max_speed)
            self.uav_yaws[i] = (self.uav_yaws[i] + action[1] * self.max_yaw_rate) % (2 * np.pi)

            new_pos, hit_obs = self._move_with_clip(self.uav_positions[i], self.uav_yaws[i], self.uav_speeds[i], self.uav_radius)
            self.uav_positions[i] = new_pos
            hit_flags.append(hit_obs)
            self.uav_trails[agent].append(new_pos.copy())

            radar_dists = self._raycast(new_pos, self.uav_yaws[i] + np.linspace(0, 2 * np.pi, self.num_radar_rays, endpoint=False), self.radar_range) * self.radar_range
            radar_distances.append(np.min(radar_dists))

# 逃逸机智能人工势场法 & 动态速度感知逻辑
        repulsive_force = np.zeros(2, dtype=np.float32)
        tx, ty = self.target_position  # 提前解包坐标供后续使用
        
        # ---------------- 1. 躲避追捕者 (感知范围扩大，非线性斥力) ----------------
        sense_range = 60.0 
        is_sensed = False  

        for pos in self.uav_positions:
            vec = self.target_position - pos
            dist = np.linalg.norm(vec)
            if dist < sense_range:
                is_sensed = True
                if dist > 0.1:
                    # 距离越近，斥力爆炸性增长
                    strength = 6.0 / (dist + 5.0) 
                    repulsive_force += (vec / dist) * strength

        # [核心动态速度控制]
        if is_sensed:
            self.target_speed = min(self.target_max_speed, self.target_speed + self.target_accel)
        else:
            self.target_speed = max(self.target_min_speed, self.target_speed - self.target_accel)

        # ---------------- 2. 躲避地图边界 (软气垫墙效果) ----------------
        wall_sense = 100.0  
        if tx < wall_sense: 
            repulsive_force[0] += 50.0 / (max(tx, 0.1) ** 1.5)
        if self.map_size - tx < wall_sense: 
            repulsive_force[0] -= 50.0 / (max(self.map_size - tx, 0.1) ** 1.5)
        if ty < wall_sense: 
            repulsive_force[1] += 50.0 / (max(ty, 0.1) ** 1.5)
        if self.map_size - ty < wall_sense: 
            repulsive_force[1] -= 50.0 / (max(self.map_size - ty, 0.1) ** 1.5)

# ---------------- 3. [优化后] 躲避建筑物 (AABB 斥力向量化) ----------------
        building_sense = 20.0  # 建筑物警戒距离
        
        if len(self.buildings) > 0:
            # 批量提取边界
            xmins = self.buildings[:, 0]
            xmaxs = self.buildings[:, 1]
            ymins = self.buildings[:, 2]
            ymaxs = self.buildings[:, 3]
            
            # 批量计算目标到所有建筑物的最近点 (Clamping)
            closest_x = np.clip(tx, xmins, xmaxs)
            closest_y = np.clip(ty, ymins, ymaxs)
            
            # 批量计算向量和距离
            vecs_x = tx - closest_x
            vecs_y = ty - closest_y
            
            # 使用 np.hypot 计算欧氏距离，速度快且稳健
            dists = np.hypot(vecs_x, vecs_y)
            
            # 找出距离小于警戒线 且 大于0.1 的有效建筑物掩码
            valid_mask = (dists < building_sense) & (dists > 0.1)
            
            if np.any(valid_mask):
                # 批量计算受力强度
                strengths = 40.0 / (dists[valid_mask] ** 1.5)
                # 批量累加斥力
                repulsive_force[0] += np.sum((vecs_x[valid_mask] / dists[valid_mask]) * strengths)
                repulsive_force[1] += np.sum((vecs_y[valid_mask] / dists[valid_mask]) * strengths)
            
            # 异常保护：批量处理贴脸或穿模的情况
            if np.any(dists <= 0.1):
                repulsive_force += np.random.randn(2) * 100.0

        # ---------------- 4. 开阔地带向心引力 (打破边缘滑行) ----------------
        map_center = np.array([self.map_size / 2.0, self.map_size / 2.0])
        vec_to_center = map_center - self.target_position
        dist_to_center = np.linalg.norm(vec_to_center)
        if dist_to_center > 200.0:
            attractive_force = (vec_to_center / dist_to_center) * 0.3
            repulsive_force += attractive_force

        # ---------------- 5. 最终航向角平滑更新 ----------------
        if np.linalg.norm(repulsive_force) > 1e-3:
            desired_yaw = np.arctan2(repulsive_force[1], repulsive_force[0])
            # 限制最大转弯速率，引入微小随机扰动避免陷入局部死锁
            self.target_yaw += np.clip((desired_yaw - self.target_yaw + np.pi) % (2 * np.pi) - np.pi, -np.pi / 4, np.pi / 4) + np.random.normal(0, 0.05)
        else:
            self.target_yaw += np.random.normal(0, 0.05)

        new_tpos, t_hit = self._move_with_clip(self.target_position, self.target_yaw, self.target_speed, self.target_radius)
        self.target_position = new_tpos
        if t_hit:
            self.target_yaw += np.random.uniform(np.pi / 2, np.pi) * np.random.choice([-1, 1])
        self.target_trail.append(new_tpos.copy())

        # 逃逸机跑完本步后，更新本帧的 4 个引导点
        self.current_guide_points = self._assign_target_points()


        # ==================== 2. 状态与奖励分配 ====================
        cur_dists = [np.linalg.norm(self.uav_positions[i] - self.target_position) for i in range(self.num_agents)]
        d_capture = self.catch_radius
        is_caught = any(d <= d_capture for d in cur_dists)

        # 五项奖励：接近/引导、安全、位置分布、逃逸缺口、终局捕获。
        w_near = self.reward_weights["w_near"]
        w_safe = self.reward_weights["w_safe"]
        w_pos = self.reward_weights["w_pos"]
        w_gap = self.reward_weights["w_gap"]
        w_finish = self.reward_weights["w_finish"]

        max_relative_speed = self.uav_max_speed + self.target_max_speed

        min_dist_old = float(np.min(self.last_distances))
        min_dist_new = float(np.min(cur_dists))
        r_team_progress = np.clip((min_dist_old - min_dist_new) / max_relative_speed, -1.0, 1.0)

        # 基础角度分布：保留论文中的位置分布思想，但不再作为唯一标准。
        rel_angles = []
        for pos in self.uav_positions:
            v = pos - self.target_position
            rel_angles.append(np.arctan2(v[1], v[0]))
        rel_angles = np.sort(np.array(rel_angles))
        angle_gaps = np.diff(np.concatenate([rel_angles, [rel_angles[0] + 2 * np.pi]]))
        theta_min = float(np.min(angle_gaps))
        ideal_gap = 2 * np.pi / self.num_agents
        r_angle_uniform = np.exp(-abs(theta_min - ideal_gap))

        # Apollonius 奖励核心：只惩罚“几何开放且无人能先到”的危险逃逸角。
        escape_field = getattr(self, "apollonius_escape", {})
        if escape_field:
            geometry_open = escape_field["geometry_open"]
            controlled = escape_field["controlled"]
            dangerous = escape_field["dangerous"]
            control_strength = escape_field["control_strength"]
            best_owner = escape_field["best_owner"]
            n_rays = len(geometry_open)

            dangerous_indices = np.where(dangerous)[0]
            if len(dangerous_indices) == 0:
                max_danger_angle = 0.0
            elif len(dangerous_indices) == n_rays:
                max_danger_angle = 2 * np.pi
            else:
                sectors = self._split_open_sectors(dangerous_indices, n_rays)
                max_danger_angle = max(len(s) for s in sectors) * (2 * np.pi / n_rays)
            r_gap_global = np.exp(-max_danger_angle)

            open_indices = np.where(geometry_open)[0]
            if len(open_indices) == 0:
                r_boundary_coverage = 1.0
                r_owner_balance = 1.0
            else:
                r_boundary_coverage = float(np.mean(control_strength[open_indices]))

                controlled_indices = np.where(geometry_open & controlled & (best_owner >= 0))[0]
                if len(controlled_indices) == 0:
                    r_owner_balance = 0.0
                else:
                    owner_counts = np.bincount(
                        best_owner[controlled_indices],
                        minlength=self.num_agents,
                    ).astype(np.float32)
                    owner_probs = owner_counts / max(float(np.sum(owner_counts)), 1e-6)
                    active_probs = owner_probs[owner_probs > 0]
                    entropy = -float(np.sum(active_probs * np.log(active_probs + 1e-8)))
                    r_owner_balance = entropy / np.log(self.num_agents)

            # 70% 边界控制覆盖 + 30% 传统角度均匀，兼顾可捕获性和队形稳定。
            r_pos_global = 0.7 * (r_boundary_coverage * r_owner_balance) + 0.3 * r_angle_uniform
        else:
            r_pos_global = r_angle_uniform
            r_gap_global = 0.0

        for i, agent in enumerate(self.agents):
            dist = cur_dists[i]

            dist_to_guide_new = np.linalg.norm(self.uav_positions[i] - self.current_guide_points[agent])
            dist_to_guide_old = np.linalg.norm(old_uav_positions[i] - self.current_guide_points[agent])
            r_guide_progress = np.clip(
                (dist_to_guide_old - dist_to_guide_new) / max_relative_speed,
                -1.0,
                1.0,
            )

            if hit_flags[i]:
                r_near = 0.0
                r_safe = -15.0
            else:
                # 保留引导点进度，同时加入真实最近距离进步，避免只学会追引导点。
                r_near = (
                    self.reward_mix["guide_progress"] * r_guide_progress
                    + self.reward_mix["team_progress"] * r_team_progress
                )

                radar_penalty = ((radar_distances[i] - self.radar_range) / self.radar_range) ** 2
                r_obstacle_safe = -0.5 * radar_penalty

                r_turn_smooth = -float(np.clip(actions_dict[agent][1], -1.0, 1.0) ** 2)

                r_mate_safe = 0.0
                mate_safe_dist = 10.0
                for j in range(self.num_agents):
                    if i == j:
                        continue
                    dist_to_mate = np.linalg.norm(self.uav_positions[i] - self.uav_positions[j])
                    if dist_to_mate < mate_safe_dist:
                        r_mate_safe -= (mate_safe_dist - dist_to_mate) / mate_safe_dist

                # 把避障、队友安全、转向平滑合并为一个 r_safe，不增加奖励项数量。
                r_safe = r_obstacle_safe + 0.5 * r_mate_safe + 0.2 * r_turn_smooth

            r_finish = 0.0
            if is_caught:
                r_finish = 80.0
            elif dist < self.catch_radius * 3:
                r_finish = (self.catch_radius * 3 - dist) / (self.catch_radius * 2)

            w_r_near = w_near * r_near
            w_r_safe = w_safe * r_safe
            w_r_pos = w_pos * r_pos_global
            w_r_gap = w_gap * r_gap_global
            w_r_finish = w_finish * r_finish

            rewards_dict[agent] = w_r_near + w_r_safe + w_r_pos + w_r_gap + w_r_finish

            self.episode_sub_rewards[agent]["r_near"] += w_r_near
            self.episode_sub_rewards[agent]["r_safe"] += w_r_safe
            self.episode_sub_rewards[agent]["r_pos"] += w_r_pos
            self.episode_sub_rewards[agent]["r_gap"] += w_r_gap
            self.episode_sub_rewards[agent]["r_finish"] += w_r_finish


        self.last_distances = cur_dists.copy()

        for k, v in rewards_dict.items(): 
            self.individual_episode_reward[k] += v
            
        is_crashed = any(hit_flags)
        terminated = {agent: (is_caught or is_crashed) for agent in self.agents}
        truncated = (self._episode_step >= self.max_episode_steps) if not terminated[self.agents[0]] else False

        info = {
            "infos": {
                "curriculum_level": self.curriculum_level,
                "curriculum_config": copy.deepcopy(self.curriculum_config),
            },
            "individual_episode_rewards": self.individual_episode_reward,
            "episode_sub_rewards": copy.deepcopy(self.episode_sub_rewards),
            "is_success": is_caught 
        }
        return self._get_obs(), rewards_dict, terminated, truncated, info
    
    def state(self):

        guide_points_flat = np.concatenate([self.current_guide_points[agent] for agent in self.agents])
        
        return np.concatenate([
            self.uav_positions.flatten(), 
            self.uav_yaws, 
            self.uav_speeds, 
            self.target_position,
            [self.target_yaw], 
            [self.target_speed],
            guide_points_flat
        ]).astype(np.float32)

    def agent_mask(self):
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        return None

    def render(self, mode="rgb_array", save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap, to_rgba
        import numpy as np

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'

        fig = plt.figure(figsize=(11, 8), dpi=150)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

        ax = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])

        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_aspect('equal')

        ax.set_xticks(np.arange(0, self.map_size + 1, 500))
        ax.set_yticks(np.arange(0, self.map_size + 1, 500))
        ax.grid(True, linestyle='-', color='#EBEBEB', linewidth=1, zorder=0)

        ax.set_xlabel("X Position (m)", fontsize=14)
        ax.set_ylabel("Y Position (m)", fontsize=14)
        ax.set_title("Simulation: Multi-UAV Cooperative Pursuit", fontsize=16, pad=15)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 1. 画建筑物
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=1, edgecolor='black', facecolor='black', zorder=2)
            ax.add_patch(rect)

        # 2. 画逃逸机及其雷达
        tx, ty = self.target_position
        target_radar = patches.Circle((tx, ty), self.radar_range,
                                      color='#FF1493', alpha=0.15, zorder=1, clip_on=True)
        ax.add_patch(target_radar)

        # =========== [新增]: 绘制目标周围的射线与开阔扇区探测情况 ===========
        if hasattr(self, 'debug_rays') and self.debug_rays:
            tx, ty = self.target_position

            # 画出合围半径 r_f 的基准圆 (灰色虚线)
            rf_circle = patches.Circle((tx, ty), self.debug_rf, fill=False,
                                       linestyle='--', color='gray', alpha=0.6, zorder=2)
            ax.add_patch(rf_circle)

            # 画出 r_f * 1.2 的阈值判定圆 (橙色点线)
            threshold_circle = patches.Circle((tx, ty), self.debug_rf * 1.5, fill=False,
                                              linestyle=':', color='orange', alpha=0.8, zorder=2)
            ax.add_patch(threshold_circle)

            dangerous_flags = None
            if hasattr(self, "apollonius_escape") and self.apollonius_escape:
                dangerous_flags = self.apollonius_escape.get("dangerous", None)

            # 绘制 72 根射线
            for ray_idx, (angle, dist, is_open_ray) in enumerate(self.debug_rays):
                dx, dy = np.cos(angle), np.sin(angle)
                end_x = tx + dist * dx
                end_y = ty + dist * dy

                if is_open_ray:
                    is_dangerous_ray = dangerous_flags is not None and bool(dangerous_flags[ray_idx])
                    if is_dangerous_ray:
                        # 几何开放且无人机无法先到的危险逃逸方向
                        ax.plot([tx, end_x], [ty, end_y], color='orangered', alpha=0.65, linewidth=1.8, zorder=2)
                    else:
                        # 几何开放但已被 Apollonius 可达边界控制
                        ax.plot([tx, end_x], [ty, end_y], color='limegreen', alpha=0.5, linewidth=1.5, zorder=2)
                else:
                    # 被建筑物或边界阻挡的路线，用浅红色细线，并在末端画一个碰撞红点
                    ax.plot([tx, end_x], [ty, end_y], color='tomato', alpha=0.3, linewidth=1.0, zorder=2)
                    ax.plot(end_x, end_y, marker='x', markersize=3, color='red', alpha=0.5, zorder=2)
        # =================================================================
        # 逃逸机轨迹 (红色渐变)
        if len(self.target_trail) > 1:
            tx_trail, ty_trail = zip(*self.target_trail)
            pts = np.array(self.target_trail)
            segments = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            norm = plt.Normalize(0, len(self.target_trail))
            c_rgba = to_rgba('red')
            c_trans = (c_rgba[0], c_rgba[1], c_rgba[2], 0.1)
            cmap_target = LinearSegmentedColormap.from_list('target_trail', [c_trans, c_rgba])
            lc = LineCollection(segments, cmap=cmap_target, norm=norm, linewidth=1.5, zorder=3)
            lc.set_array(np.arange(len(self.target_trail)))
            ax.add_collection(lc)

        ax.plot(tx, ty, marker='o', markersize=self.target_plot_radius, color='red', zorder=4)
        # ax.text(tx + 25, ty - 25, "Target", fontsize=11, zorder=5, color='red')
        # ax.arrow(tx, ty, 40 * np.cos(self.target_yaw), 40 * np.sin(self.target_yaw),
        #          head_width=15, head_length=15, fc='red', ec='red', zorder=5)

        # 3. 定义 4 种醒目的颜色来区分不同的 UAV 和它们的引导点
        agent_colors = ['#FF8C00', '#32CD32', '#00BFFF', '#9400D3'] # 橙色, 莱姆绿, 深天蓝, 深紫罗兰色

        # 4. 画追捕者 (UAV), 雷达, 轨迹 和 【专属引导点】
        for i, agent in enumerate(self.agents):
            ux, uy = self.uav_positions[i]
            color = agent_colors[i % len(agent_colors)] # 获取专属颜色

            # UAV 雷达 (颜色统一调淡一点以免遮挡)
            radar = patches.Circle((ux, uy), self.radar_range,
                                   color='#00FFFF', alpha=0.08, zorder=1, clip_on=True)
            ax.add_patch(radar)

            # UAV 轨迹 (使用专属颜色生成透明到实色的渐变)
            trail = list(self.uav_trails[agent])
            if len(trail) > 1:
                pts = np.array(trail)
                segments = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
                norm = plt.Normalize(0, len(trail))

                # --- [核心修改] 动态生成专属颜色的渐变 Colormap ---
                color_rgba = to_rgba(color)
                color_transparent = (color_rgba[0], color_rgba[1], color_rgba[2], 0.05) # 尾部 5% 透明度
                cmap = LinearSegmentedColormap.from_list(f'uav_trail_{i}', [color_transparent, color_rgba])

                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0, zorder=3)
                lc.set_array(np.arange(len(trail)))
                ax.add_collection(lc)

            # 画出当前分配给该 UAV 的引导点及连线
            if agent in self.current_guide_points:
                gx, gy = self.current_guide_points[agent]
                ax.plot(gx, gy, marker='*', markersize=6, color=color, zorder=4)
                ax.plot([ux, gx], [uy, gy], linestyle='--', color=color, alpha=0.6, linewidth=1.5, zorder=3)
                # ax.text(gx + 15, gy + 15, f"G{i}", fontsize=10, color=color, fontweight='bold', zorder=5)

            # 画 UAV 本体 (边缘用黑色加深对比，内部填充专属颜色)
            ax.plot(ux, uy, marker='o', markersize=self.uav_plot_radius, color=color, markeredgecolor='black', zorder=4)
            ax.text(ux + 25, uy - 25, f"UAV {i}", fontsize=11, color=color, fontweight='bold', zorder=5)
            # ax.arrow(ux, uy, 40 * np.cos(self.uav_yaws[i]), 40 * np.sin(self.uav_yaws[i]),
            #          head_width=15, head_length=15, fc=color, ec='black', zorder=5)

        # 5. 右侧 Dashboard 统计信息
        ax_info.axis('off')

        cur_dists = [np.linalg.norm(self.uav_positions[i] - self.target_position) for i in range(self.num_agents)]
        min_dist = min(cur_dists)

        info_text = f"== Dashboard ==\n"
        info_text += f"Step: {self._episode_step} / {self.max_episode_steps}\n"
        info_text += f"Min Dist:   {min_dist:.1f} m\n"
        info_text += f"Target Spd: {self.target_speed:.1f} m/s\n"
        info_text += "-" * 25 + "\n"

        for i, agent in enumerate(self.agents):
            sr = self.episode_sub_rewards[agent]
            total_r = sum(sr.values())

            info_text += f"[{agent.upper()}] Spd: {self.uav_speeds[i]:.1f} m/s\n"
            info_text += f"Total R: {total_r:.2f}\n"

            keys = list(sr.keys())
            if keys:
                max_key_len = max(len(k) for k in keys)

                for j, key in enumerate(keys):
                    prefix = " └ " if j == len(keys) - 1 else " ├ "
                    padded_key = f"{key}:".ljust(max_key_len + 1)
                    info_text += f"{prefix}{padded_key} {sr[key]:.2f}\n"

            if i < self.num_agents - 1:
                info_text += "\n"

        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=11, family='monospace', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CCCCCC', alpha=1.0))

        fig.subplots_adjust(wspace=0.1)

        if save_path is not None:
            plt.savefig(save_path, format=save_path.split('.')[-1], transparent=True)

        if mode == "rgb_array":
            fig.canvas.draw()
            img_rgb = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
            fig.clf()
            plt.close(fig)
            return img_rgb
        else:
            fig.clf()
            plt.close(fig)
            return None

    def close(self):
        pass
