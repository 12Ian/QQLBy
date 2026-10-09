"""采用加速度指令四旋翼动力学的三维多无人机追捕环境。"""
import os
import json
import copy
import numpy as np
import gymnasium as gym
from collections import deque

from xuance.environment import RawMultiAgentEnv
from .Apollonius import geometry3d as g3
from .Apollonius import apollonius3d as ap3
from .uav_kinematics_3d import advance_translation, limit_norm
from .uav_coverage_forecast_3d import forecast_target, reach_hold_coverage, coverage_preparation


class UAVPursuitCoverage3DEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(UAVPursuitCoverage3DEnv, self).__init__()

        # ---------------- 1. 物理参数 ----------------
        self.map_size = 1000.0
        self.z_min = float(getattr(config, "z_min", 10.0))
        self.z_max = float(getattr(config, "z_max", 350.0))

        self.uav_min_speed = 0.0
        self.uav_initial_speed = float(getattr(config, "uav_initial_speed", 1.0))
        self.uav_max_speed = float(getattr(config, "uav_max_speed", 11.0))
        self.max_accel = float(getattr(config, "max_accel", 3.0))
        self.acceleration_lag = float(getattr(config, "acceleration_lag", 0.5))
        self.uav_jerk_max = float(getattr(config, "uav_jerk_max", 6.0))
        self.target_jerk_max = float(getattr(config, "target_jerk_max", 8.0))
        self.target_velocity_response = float(getattr(config, "target_velocity_response", 1.0))
        self.decision_dt = float(getattr(config, "decision_dt", 1.0))
        self.physics_dt = float(getattr(config, "physics_dt", 0.1))
        if (self.uav_initial_speed < 0.0 or self.uav_initial_speed > self.uav_max_speed or
                self.decision_dt <= 0.0 or self.physics_dt <= 0.0 or
                self.acceleration_lag <= 0.0 or self.uav_jerk_max < 0.0 or
                self.target_jerk_max < 0.0 or self.target_velocity_response <= 0.0):
            raise ValueError("Invalid UAV speed or acceleration response configuration")
        self.uav_radius = 0.5
        self.spawn_safety_enabled = bool(getattr(config, "spawn_safety_enabled", True))
        self.spawn_safety_margin = float(getattr(config, "spawn_safety_margin", 1.5))
        self.spawn_search_resolution = float(getattr(config, "spawn_search_resolution", 1.0))
        if not np.isfinite(self.spawn_safety_margin) or self.spawn_safety_margin < 0.0:
            raise ValueError("spawn_safety_margin must be >= 0")
        if not np.isfinite(self.spawn_search_resolution) or self.spawn_search_resolution <= 0.0:
            raise ValueError("spawn_search_resolution must be > 0")
        self.initial_step_reach = min(
            self.uav_max_speed, self.uav_initial_speed + self.max_accel * self.decision_dt
        ) * self.decision_dt
        self.spawn_clearance = self.uav_radius + self.initial_step_reach + self.spawn_safety_margin
        self.spawn_min_pair_distance = 2.0 * self.spawn_clearance

        # 关闭课程时直接使用配置值；开启课程时由各级课程覆盖，
        # 从而逐步设置可学习的目标速度比。
        self.target_min_speed = float(getattr(config, "target_min_speed", 1.0))
        self.target_max_speed = float(getattr(config, "target_max_speed", 12.0))
        # 防止逃逸机利用地图边缘；具体逻辑见 _evader_step。
        self.evader_center_pull = float(getattr(config, "evader_center_pull", 1.5))
        self.evader_sense = float(getattr(config, "evader_sense", 60.0))
        self.target_accel = float(getattr(config, "target_accel", 4.0))
        self.target_speed = self.target_min_speed
        self.target_radius = 0.5
        self.catch_radius = float(getattr(config, "capture_radius", 15.0))
        self.direct_intercept_radius = float(getattr(config, "direct_intercept_radius", 5.0))
        self.capture_hold_required = float(getattr(config, "capture_hold_required", 2.0))
        if not (0.0 < self.direct_intercept_radius < self.catch_radius and self.capture_hold_required > 0.0):
            raise ValueError("Capture thresholds must satisfy 0 < direct < capture and hold > 0")
        self.cartesian_lambda_cap = getattr(config, "cartesian_lambda_cap", 0.8)
        # 可选欧氏距离或可见性判据；可见性判据不计入被建筑遮挡的追捕机。
        self.criterion_mode = getattr(config, "criterion_mode", "euclidean")
        # 最近建筑观测：每架追捕机获得最近 k 个建筑棱柱的相对几何信息。
        self.use_nearest_building_obs = bool(getattr(config, "use_nearest_building_obs", True))
        self.nearest_building_k = int(getattr(
            config, "nearest_building_k",
            getattr(config, "obstacle_gat_k", 5)))
        self.nearest_building_feat_dim = 6
        self.use_obstacle_gat = bool(getattr(config, "use_obstacle_gat", False))
        self.obstacle_gat_k = int(getattr(config, "obstacle_gat_k", self.nearest_building_k))
        self.obstacle_gat_feat_dim = self.nearest_building_feat_dim
        if self.use_obstacle_gat and (not self.use_nearest_building_obs or
                                      self.obstacle_gat_k != self.nearest_building_k or
                                      int(getattr(config, "obstacle_gat_feat_dim", 6)) != 6):
            raise ValueError("Obstacle attention requires matching K and six building features")
        self.obstacle_avoid_range = 100.0
        # 环境内部进行短时目标轨迹预测；此处为启发式外推，
        # 不依赖学习网络或目标轨迹数据集。
        self.use_target_prediction_obs = bool(getattr(config, "use_target_prediction_obs", True))
        self.future_prediction_steps = int(getattr(config, "future_prediction_steps", 5))
        self.preparation_time_scale = float(getattr(config, "preparation_time_scale", 12.0))
        self.closure_time_scale = float(getattr(config, "closure_time_scale", 30.0))
        self.match_distance = float(getattr(config, "match_distance", 40.0))
        self.match_distance_width = float(getattr(config, "match_distance_width", 10.0))
        self.match_speed_scale = float(getattr(config, "match_speed_scale", 3.0))
        self.discount_gamma = float(getattr(config, "gamma", 0.99))
        if (self.preparation_time_scale <= 0.0 or self.closure_time_scale <= 0.0 or
                self.match_distance <= 0.0 or self.match_distance_width <= 0.0 or
                self.match_speed_scale <= 0.0 or not 0.0 <= self.discount_gamma <= 1.0):
            raise ValueError("Invalid reward scale or discount")
        self.building_collision_penalty = float(getattr(config, "building_collision_penalty", 80.0))
        self.boundary_collision_penalty = float(getattr(config, "boundary_collision_penalty", 100.0))
        self.boundary_warning_distance = float(getattr(config, "boundary_warning_distance", 100.0))
        self.uav_collision_penalty = float(getattr(config, "uav_collision_penalty", 80.0))
        self.collision_contact_penalty_per_second = float(getattr(
            config, "collision_contact_penalty_per_second", 20.0))
        if self.collision_contact_penalty_per_second < 0.0:
            raise ValueError("collision_contact_penalty_per_second must be >= 0")
        # 奖励消融：集合中的奖励项权重置零。
        self.reward_disable = set(getattr(config, "reward_disable", []) or [])
        # obs_avoid_weight 缩放建筑距离避障惩罚；数值增大时更早开始避障。
        # 取 1.0 时沿用原奖励强度。
        self.obs_avoid_weight = float(getattr(config, "obs_avoid_weight", 1.0))
        # 追捕机间碰撞与建筑、边界碰撞分别处理。旧碰撞检测只检查单机位置，
        # 因而没有把机间接触计入终止条件或碰撞率；旧奖励只在队友间距小于
        # 10 米时惩罚，另可启用提前分散项。现在每步记录最小机间距，
        # 是否把机间碰撞作为终止事件仍由开关决定。
        self.uav_collision_enabled = bool(getattr(config, "uav_collision_enabled", True))
        self.uav_collision_distance = float(getattr(config, "uav_collision_distance",
                                                    2.0 * self.uav_radius))
        self.episode_min_uav_gap = float("inf")
        self.episode_uav_conflicts = 0
        # surround_spawn 开启时追捕机在逃逸机周围保持约 spawn_radius 初距，
        # 垂直展开单独限制；关闭时沿 x=100 一侧排布。
        # 环绕出生使初始封锁较快目标成为可能。
        self.surround_spawn = bool(getattr(config, "surround_spawn", False))
        self.spawn_radius = float(getattr(config, "spawn_radius", 200.0))
        self.spawn_vertical_radius = float(getattr(config, "spawn_vertical_radius", 70.0))
        self.spawn_boundary_margin = float(getattr(config, "spawn_boundary_margin", 30.0))
        if (self.spawn_vertical_radius < 0.0 or self.spawn_vertical_radius > self.spawn_radius or
                self.spawn_boundary_margin < self.uav_radius or
                2.0 * (self.spawn_vertical_radius + self.spawn_boundary_margin) >=
                self.z_max - self.z_min):
            raise ValueError("Invalid surround-spawn vertical radius or boundary margin")
        # 建筑碰撞可以独立于边界和机间碰撞设置为软约束：
        # 无人机受罚并沿障碍滑动，但回合继续。
        self.terminate_on_building_collision = bool(
            getattr(config, "terminate_on_building_collision", False))
        # 边界碰撞默认可恢复；此开关控制边界碰撞是否终止回合。
        self.terminate_on_boundary_collision = bool(
            getattr(config, "terminate_on_boundary_collision", False))
        self.terminate_on_collision = bool(getattr(config, "terminate_on_collision", False))

        self._setup_curriculum(config)       # 设置奖励权重、建筑模式和出生偏移
        self.pending_curriculum_level = None

        self.num_agents = int(getattr(config, "num_agents", 6))
        self.agents = [f"uav_{i}" for i in range(self.num_agents)]
        self.spawn_offsets = np.zeros(self.num_agents, dtype=np.float32)
        self._safe_spawn_cache = {}
        # 可选的提前分散奖励。旧奖励在队友间距小于 10 米后才惩罚，
        # 对四机末段闭合可能过晚；默认关闭以保持旧实验数值一致。
        self.separation_weight = float(getattr(config, "separation_weight", 0.5))
        self.separation_distance = float(getattr(config, "separation_distance", 20.0))

        self.escape_num_dirs = int(getattr(config, "escape_num_dirs", 200))
        self.escape_dirs = g3.fibonacci_sphere(self.escape_num_dirs)

        self.buildings_dir = getattr(
            config, "buildings_dir",
            os.path.join(os.path.dirname(__file__), "buildings"))
        self.randomize_building_layout = bool(getattr(config, "randomize_building_layout", True))
        self.buildings = self._as_building_array(self._generate_city_blocks())

        # 域随机化：每次重置从建筑密度池中采样一种场景，
        # 使同一模型适应不同障碍密度。
        self.randomize_density = getattr(config, "randomize_density", None)
        if self.randomize_density:
            self.randomize_density = list(self.randomize_density)
            if not self.randomize_building_layout:
                self._density_pool = {m: self._as_building_array(self._load_blocks(m))
                                      for m in self.randomize_density}

        z_mid = 0.5 * (self.z_min + self.z_max)
        ys = np.linspace(150.0, 850.0, self.num_agents)
        self.init_uav_positions = np.array(
            [[100.0, float(y), z_mid] for y in ys], dtype=np.float32)

        plot_config = getattr(config, "plot_config", {})
        self.uav_plot_radius = plot_config.get("uav_radius", 4)
        self.target_plot_radius = plot_config.get("target_radius", 4)

        # ---------------- 2. 动作与观测空间 ----------------
        self.action_space = {a: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
                             for a in self.agents}
        # 自身(10)与保持进度(1)、队友相对位置与速度(6×(N-1))、边界(6)、目标当前状态(10)。
        self.obs_dim = 27 + 6 * (self.num_agents - 1)
        if self.use_nearest_building_obs:
            self.obs_dim += self.nearest_building_k * self.nearest_building_feat_dim
        if self.use_target_prediction_obs:
            self.obs_dim += self.future_prediction_steps * 3
        self.observation_space = {a: gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                                  for a in self.agents}
        # 全局状态含运动状态和每架追捕机的捕获保持进度。
        self.state_dim = 9 * (self.num_agents + 1) + self.num_agents
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # ---------------- 3. 运行时状态 ----------------
        self.max_episode_steps = getattr(config, "episode_length", 400)
        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

        self.uav_positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.uav_velocities = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.uav_accelerations = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.uav_speeds = np.zeros(self.num_agents, dtype=np.float32)

        self.target_position = np.zeros(3, dtype=np.float32)
        self.target_velocity = np.zeros(3, dtype=np.float32)
        self.target_acceleration = np.zeros(3, dtype=np.float32)
        self.capture_hold_times = np.zeros(self.num_agents, dtype=np.float64)
        self.episode_target_speed_max = self.target_speed

        self.uav_trails = {a: deque(maxlen=100) for a in self.agents}
        self.target_trail = deque(maxlen=100)

        self.episode_sub_rewards = {
            a: {"r_cov": 0.0, "r_prep": 0.0, "r_close": 0.0,
                "r_match": 0.0, "r_hold": 0.0, "r_step": 0.0,
                "r_safe": 0.0, "r_terminal": 0.0}
            for a in self.agents
        }
        self.apollonius_escape = {}
        self.apollonius_margin_ema = None
        self.apollonius_dangerous_prev = None
        self.target_forecast = None
        self.last_coverage = 0.0
        self.last_preparation = 0.0
        self.last_closure = 0.0
        self.last_match_scores = np.zeros(self.num_agents, dtype=np.float32)
        self.last_hold_progress = 0.0
        self.episode_target_wall_contacts = 0
        self.episode_coverage_sum = 0.0
        self.episode_coverage_max = 0.0
        self.episode_preparation_sum = 0.0
        self.episode_safe_route_time_sum = 0.0
        self.episode_safe_route_time_count = 0
        self.episode_min_target_distance = float("inf")
        self.active_contact_kinds = [set() for _ in self.agents]
        self.episode_contact_seconds = {kind: 0.0 for kind in ("building", "boundary", "uav")}
        self.episode_contact_events = {kind: 0 for kind in ("building", "boundary", "uav")}

    # ============================================================
    # 课程设置：从二维版本迁移，并逐步增加垂直逃逸能力
    # ============================================================
    def _default_curriculum_schedule(self, preparation_weight, new_weights):
        base = [
            # 建筑模式对应各自的地图文件：空场景、稀疏、中等和复杂场景。
            (0, 3.0, 5.0, 1.0, 20.0, "empty", 3.0, 0.0),
            (1, 4.0, 7.0, 2.0, 40.0, "empty", 3.5, 0.25),
            (2, 4.0, 9.0, 3.0, 60.0, "open", 4.0, 0.5),
            (3, 4.0, 10.0, 3.5, 80.0, "medium", 4.5, 0.75),
            (4, 4.0, 12.0, 4.0, 100.0, "medium", 5.0, 1.0),
        ]
        sched = []
        for (lvl, tmin, tmax, tacc, off, bmode, wc, zsc) in base:
            sched.append({
                "level": lvl, "target_min_speed": tmin, "target_max_speed": tmax,
                "target_accel": tacc, "spawn_offset": off, "building_mode": bmode,
                "reward_weights": {"w_prep": preparation_weight, "w_cov": wc, "w_hold": 1.0,
                                   "w_safe": 1.0, "w_terminal": 240.0, **new_weights},
                "z_escape_scale": zsc,
            })
        return sched

    def _setup_curriculum(self, config):
        self.curriculum_enabled = bool(getattr(config, "curriculum_enabled", True))
        new_weights = {
            "w_close": float(getattr(config, "w_close", 10.0)),
            "w_match": float(getattr(config, "w_match", 8.0)),
            "w_step": float(getattr(config, "w_step", 0.1)),
            "w_direct_terminal": float(getattr(config, "w_direct_terminal", 300.0)),
        }
        self.curriculum_schedule = getattr(config, "curriculum_schedule", None)
        if self.curriculum_schedule is None:
            self.curriculum_schedule = self._default_curriculum_schedule(
                float(getattr(config, "w_prep", 15.0)), new_weights)
        self.curriculum_min_steps_per_level = int(getattr(config, "curriculum_min_steps_per_level", 300000))
        self.curriculum_success_threshold = float(getattr(config, "curriculum_success_threshold", 0.75))
        self.curriculum_drop_threshold = float(getattr(config, "curriculum_drop_threshold", 0.45))
        self.curriculum_level = int(getattr(config, "curriculum_start_level", 0))
        self.curriculum_last_update_step = 0

        self.spawn_offset = float(getattr(config, "spawn_offset", 50.0))
        self.building_mode = getattr(config, "building_mode", "medium")
        # 基础奖励权重可由配置覆盖；开启课程后由课程级别重新设置。
        self.reward_weights = {
            "w_prep": float(getattr(config, "w_prep", 15.0)),
            "w_cov": float(getattr(config, "w_cov", 4.0)),
            "w_hold": float(getattr(config, "w_hold", 1.0)),
            "w_safe": float(getattr(config, "w_safe", 1.0)),
            "w_terminal": float(getattr(config, "w_terminal", 240.0)),
            **new_weights,
        }
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
        self.target_speed = min(max(float(getattr(self, "target_speed", self.target_min_speed)),
                                    self.target_min_speed), self.target_max_speed)
        if hasattr(self, "target_velocity"):
            self.target_velocity = limit_norm(
                self.target_velocity, self.target_max_speed).astype(np.float32)
            self.target_speed = float(np.linalg.norm(self.target_velocity))
        self.spawn_offset = float(self.curriculum_config["spawn_offset"])
        self.building_mode = self.curriculum_config["building_mode"]
        self.reward_weights = copy.deepcopy(self.curriculum_config["reward_weights"])
        if reload_buildings and hasattr(self, "buildings"):
            self.buildings = self._as_building_array(self._generate_city_blocks())
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

    def schedule_curriculum_level(self, level):
        """让下一次 reset 切换课程，避免在回合中途更换目标动力学和建筑。"""
        if not self.curriculum_enabled:
            return self.curriculum_level
        self.pending_curriculum_level = int(np.clip(level, 0,
                                                    len(self.curriculum_schedule) - 1))
        return self.pending_curriculum_level

    def get_curriculum_info(self):
        return {
            "curriculum_enabled": self.curriculum_enabled,
            "curriculum_level": self.curriculum_level,
            "building_mode": self.building_mode,
            "spawn_offset": self.spawn_offset,
            "uav_initial_speed": float(self.uav_initial_speed),
            "uav_min_speed": float(self.uav_min_speed),
            "uav_max_speed": float(self.uav_max_speed),
            "uav_accel_max": float(self.max_accel),
            "acceleration_lag": float(self.acceleration_lag),
            "uav_jerk_max": float(self.uav_jerk_max),
            "target_jerk_max": float(self.target_jerk_max),
            "decision_dt": float(self.decision_dt),
            "physics_dt": float(self.physics_dt),
            "target_min_speed": self.target_min_speed,
            "target_max_speed": self.target_max_speed,
            "target_accel": self.target_accel,
            "capture_radius": self.catch_radius,
            "direct_intercept_radius": self.direct_intercept_radius,
            "capture_hold_required": self.capture_hold_required,
            "target_speed_current": float(getattr(self, "target_speed", self.target_min_speed)),
            "target_speed_max_episode": float(getattr(self, "episode_target_speed_max",
                                                       getattr(self, "target_speed", self.target_min_speed))),
            "coverage_mean_episode": float(self.episode_coverage_sum / max(self._episode_step, 1)),
            "coverage_max_episode": float(self.episode_coverage_max),
            "preparation_mean_episode": float(self.episode_preparation_sum / max(self._episode_step, 1)),
            "safe_route_time_mean_episode": (
                float(self.episode_safe_route_time_sum / self.episode_safe_route_time_count)
                if self.episode_safe_route_time_count else float("nan")),
            "safe_route_time_valid_steps_episode": int(self.episode_safe_route_time_count),
            "min_target_distance_episode": float(self.episode_min_target_distance),
            "target_wall_contacts_episode": int(self.episode_target_wall_contacts),
            "building_contact_seconds_episode": float(self.episode_contact_seconds["building"]),
            "boundary_contact_seconds_episode": float(self.episode_contact_seconds["boundary"]),
            "uav_contact_seconds_episode": float(self.episode_contact_seconds["uav"]),
            "building_contact_events_episode": int(self.episode_contact_events["building"]),
            "boundary_contact_events_episode": int(self.episode_contact_events["boundary"]),
            "uav_contact_events_episode": int(self.episode_contact_events["uav"]),
            "building_count": int(len(getattr(self, "buildings", []))),
            "randomize_building_layout": bool(self.randomize_building_layout),
            "separation_weight": float(self.separation_weight),
            "separation_distance": float(self.separation_distance),
            "evader_center_pull": float(self.evader_center_pull),
            "evader_sense": float(self.evader_sense),
            "terminate_on_collision": bool(self.terminate_on_collision),
            "terminate_on_boundary_collision": bool(self.terminate_on_boundary_collision),
            "terminate_on_building_collision": bool(self.terminate_on_building_collision),
            "use_nearest_building_obs": bool(self.use_nearest_building_obs),
            "nearest_building_k": int(self.nearest_building_k),
            "use_target_prediction_obs": bool(self.use_target_prediction_obs),
            "future_prediction_steps": int(self.future_prediction_steps),
            "preparation_time_scale": float(self.preparation_time_scale),
            "building_collision_penalty": float(self.building_collision_penalty),
            "boundary_collision_penalty": float(self.boundary_collision_penalty),
            "boundary_warning_distance": float(self.boundary_warning_distance),
            "collision_contact_penalty_per_second": float(self.collision_contact_penalty_per_second),
            "spawn_safety_enabled": bool(self.spawn_safety_enabled),
            "spawn_clearance": float(self.spawn_clearance),
            "spawn_vertical_radius": float(self.spawn_vertical_radius),
            "spawn_boundary_margin": float(self.spawn_boundary_margin),
            "spawn_adjusted_count": int(np.count_nonzero(self.spawn_offsets > 1e-6)),
            "spawn_max_offset": float(np.max(self.spawn_offsets)) if len(self.spawn_offsets) else 0.0,
            "spawn_offsets": [float(value) for value in self.spawn_offsets],
            "reward_weights": copy.deepcopy(self.reward_weights),
        }

    def _load_blocks(self, mode):
        """读取并验证建筑地图；只有空场景允许缺少地图文件。"""
        if mode == "empty":
            return np.zeros((0, 5), dtype=np.float32).tolist()
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
            return np.zeros((0, 5), dtype=np.float32).tolist()
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim == 2 and array.shape[1] == 4:
            heights = np.full((array.shape[0], 1), self.z_max, dtype=np.float32)
            array = np.concatenate([array, heights], axis=1)
        valid_shape = array.ndim == 2 and array.shape[1] == 5
        valid_values = np.all(np.isfinite(array))
        valid_extents = valid_shape and np.all(array[:, 0] < array[:, 1]) and np.all(array[:, 2] < array[:, 3])
        valid_xy_bounds = valid_shape and np.all(array[:, :4] >= 0.0) and np.all(array[:, :4] <= self.map_size)
        valid_heights = valid_shape and np.all(array[:, 4] >= self.z_min) and np.all(array[:, 4] <= self.z_max)
        valid_bounds = valid_xy_bounds and valid_heights
        if not (valid_shape and valid_values and valid_extents and valid_bounds):
            raise ValueError(f"Invalid obstacle prisms in {config_file}: shape={array.shape}")
        return array.tolist()

    def _generate_city_blocks(self):
        templates = np.asarray(self._load_blocks(getattr(self, "building_mode", "medium")),
                               dtype=np.float32)
        if not self.randomize_building_layout or templates.size == 0:
            return templates.tolist() if templates.size else []

        # 保留场景的建筑密度级别，同时在每回合改变数量、位置、长宽和高度。
        base_count = len(templates)
        count = int(np.random.randint(max(0, int(np.floor(base_count * 0.75))),
                                      int(np.ceil(base_count * 1.25)) + 1))
        blocks = []
        for _ in range(count):
            template = templates[np.random.randint(base_count)]
            width = float(np.clip((template[1] - template[0]) * np.random.uniform(0.8, 1.2),
                                  20.0, self.map_size * 0.25))
            depth = float(np.clip((template[3] - template[2]) * np.random.uniform(0.8, 1.2),
                                  20.0, self.map_size * 0.25))
            top = float(np.clip(self.z_min + (template[4] - self.z_min) *
                                np.random.uniform(0.8, 1.2), self.z_min + 10.0, self.z_max))
            for _attempt in range(100):
                xmin = float(np.random.uniform(0.0, self.map_size - width))
                ymin = float(np.random.uniform(0.0, self.map_size - depth))
                xmax, ymax = xmin + width, ymin + depth
                if all(xmax + 5.0 <= b[0] or xmin >= b[1] + 5.0 or
                       ymax + 5.0 <= b[2] or ymin >= b[3] + 5.0 for b in blocks):
                    blocks.append([xmin, xmax, ymin, ymax, top])
                    break
        return blocks

    def _as_building_array(self, blocks):
        array = np.asarray(blocks, dtype=np.float32)
        if array.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] == 4:
            heights = np.full((array.shape[0], 1), self.z_max, dtype=np.float32)
            array = np.concatenate([array, heights], axis=1)
        return array.astype(np.float32)

    def _building_top(self, building):
        return float(building[4]) if len(building) >= 5 else self.z_max

    def _point_hits_building(self, pos, radius, building):
        xmin, xmax, ymin, ymax = building[:4]
        ztop = self._building_top(building)
        return ((xmin - radius < pos[0] < xmax + radius) and
                (ymin - radius < pos[1] < ymax + radius) and
                (pos[2] <= ztop + radius))

    # ============================================================
    # 几何与运动辅助函数
    # ============================================================
    def _is_in_building(self, pos, radius):
        for b in self.buildings:
            if self._point_hits_building(pos, radius, b):
                return True
        return False

    def _spawn_clearance_xy(self, pos):
        x, y = float(pos[0]), float(pos[1])
        best = min(x, self.map_size - x, y, self.map_size - y)
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b[:4]
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
        if not (self.z_min + self.spawn_boundary_margin <= float(pos[2]) <=
                self.z_max - self.spawn_boundary_margin):
            return False
        if min(float(pos[0]), self.map_size - float(pos[0]),
               float(pos[1]), self.map_size - float(pos[1])) < self.spawn_boundary_margin:
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
            self.uav_speeds[i] = float(np.linalg.norm(self.uav_velocities[i]))

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
        if next_pos[2] < self.z_min or next_pos[2] > self.z_max:
            hit = True
            next_pos[2] = np.clip(next_pos[2], self.z_min, self.z_max)
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b[:4]
            if self._point_hits_building(next_pos, radius, b):
                hit = True
                px = np.array([next_pos[0], pos[1], next_pos[2]], dtype=np.float32)
                if not self._point_hits_building(px, radius, b):
                    next_pos = px; break
                py = np.array([pos[0], next_pos[1], next_pos[2]], dtype=np.float32)
                if not self._point_hits_building(py, radius, b):
                    next_pos = py; break
                next_pos = np.array([pos[0], pos[1], next_pos[2]], dtype=np.float32); break
        return next_pos, hit

    def _prospective_collision_sources_velocity(self, pos, velocity, radius):
        candidate = np.asarray(pos, np.float32) + np.asarray(velocity, np.float32)
        sources = []
        if candidate[0] < radius:
            sources.append("boundary:x_min")
        if candidate[0] > self.map_size - radius:
            sources.append("boundary:x_max")
        if candidate[1] < radius:
            sources.append("boundary:y_min")
        if candidate[1] > self.map_size - radius:
            sources.append("boundary:y_max")
        if candidate[2] < self.z_min:
            sources.append("boundary:z_min")
        if candidate[2] > self.z_max:
            sources.append("boundary:z_max")
        for index, b in enumerate(self.buildings):
            if self._point_hits_building(candidate, radius, b):
                sources.append(f"building:{index}")
        return sources

    def _building_distances_3d(self, pos):
        """计算位置到所有建筑棱柱的最短三维距离。"""
        if len(self.buildings) == 0:
            return np.empty(0, dtype=np.float32)
        b = self.buildings
        dx = np.maximum(np.maximum(b[:, 0] - pos[0], pos[0] - b[:, 1]), 0.0)
        dy = np.maximum(np.maximum(b[:, 2] - pos[1], pos[1] - b[:, 3]), 0.0)
        dz = np.maximum(np.maximum(self.z_min - pos[2], pos[2] - b[:, 4]), 0.0)
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def _nearest_obstacle_features(self, pos, k):
        """从全图选最近 k 栋建筑，返回完整边界和有效标志。"""
        feats = np.zeros((k, self.nearest_building_feat_dim), dtype=np.float32)
        if len(self.buildings) == 0:
            return feats
        order = np.argsort(self._building_distances_3d(pos))[:k]
        for i, idx in enumerate(order):
            b = self.buildings[idx]
            feats[i] = [(b[0] - pos[0]) / self.map_size,
                        (b[1] - pos[0]) / self.map_size,
                        (b[2] - pos[1]) / self.map_size,
                        (b[3] - pos[1]) / self.map_size,
                        (b[4] - pos[2]) / max(self.z_max - self.z_min, 1e-6),
                        1.0]
        return feats

    def _predict_target_future(self, origin):
        feats = np.zeros((self.future_prediction_steps, 3), dtype=np.float32)
        if self.future_prediction_steps <= 0:
            return feats
        if self.target_forecast is None:
            self.target_forecast = forecast_target(self)
        path = self.target_forecast["paths"][0]
        scale = np.array([self.map_size, self.map_size,
                          max(self.z_max - self.z_min, 1e-6)], dtype=np.float32)
        for step in range(self.future_prediction_steps):
            index = min(len(path) - 1, int(round((step + 1) * self.decision_dt /
                                                 self.target_forecast["dt"])))
            pred = path[index]
            feats[step] = (pred - origin) / scale
        return feats

    def _compute_r_f(self, target_pos=None, target_speed=None):
        """按追捕距离和目标速度估计逃逸方向覆盖分析半径。"""
        tp = self.target_position if target_pos is None else np.asarray(target_pos, np.float32)
        ts = self.target_speed if target_speed is None else float(target_speed)
        cur = np.linalg.norm(self.uav_positions - tp[None, :], axis=1)
        avg_dist = float(np.mean(cur))
        speed_ratio = ts / max(self.uav_max_speed, 1e-6)
        r_by_dist = avg_dist * (0.25 + 0.35 * speed_ratio)
        r_by_speed = ts * 5.0
        return float(np.clip(max(r_by_dist, r_by_speed, self.catch_radius * 1.5),
                             self.catch_radius * 1.2, self.catch_radius * 8.0))

    def _teammate_separation_penalty(self, agent_index):
        """计算单架追捕机的平滑机间避碰惩罚。

        返回未加权的非正值。20 米阈值不会影响四机在 15 米捕获圈附近
        约 24.5 米的理想四面体间距，并提供约两步提前预警。
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

    # ============================================================
    # 逃逸方向覆盖评估
    # ============================================================
    def _update_escape_field(self):
        """更新逃逸方向覆盖数据，供覆盖奖励和诊断使用。"""
        target = self.target_position
        dirs = self.escape_dirs
        free = g3.ray_free_distance(target, dirs, self.map_size, self.z_min, self.z_max, self.buildings)
        r_f = self._compute_r_f()

        field = ap3.compute_escape_field_3d(
            target, self.target_speed, self.uav_positions, self.uav_max_speed,
            dirs, free, self.catch_radius, r_f,
            margin_ema=self.apollonius_margin_ema,
            dangerous_prev=self.apollonius_dangerous_prev,
            criterion_mode=self.criterion_mode, buildings=self.buildings)
        self.apollonius_escape = field
        self.apollonius_margin_ema = field["margin_ema"]
        self.apollonius_dangerous_prev = field["dangerous_prev"]

    # ============================================================
    # 观测与全局状态
    # ============================================================
    def _get_obs(self):
        obs = {}
        ms = self.map_size
        zr = max(self.z_max - self.z_min, 1e-6)
        scale = np.array([ms, ms, zr], dtype=np.float32)
        for i, ag in enumerate(self.agents):
            pos = self.uav_positions[i]
            vel = self.uav_velocities[i]
            spd = float(np.linalg.norm(vel))
            norm_pos = np.array([pos[0] / ms * 2 - 1, pos[1] / ms * 2 - 1,
                                 (pos[2] - self.z_min) / zr * 2 - 1], np.float32)
            hold_progress = min(self.capture_hold_times[i] / self.capture_hold_required, 1.0)
            own = [norm_pos, np.array([spd / self.uav_max_speed], np.float32),
                   vel / self.uav_max_speed, self.uav_accelerations[i] / self.max_accel,
                   np.array([hold_progress], np.float32)]
            mates = [np.concatenate(((self.uav_positions[j] - pos) / scale,
                                     (self.uav_velocities[j] - vel) /
                                     (2.0 * self.uav_max_speed)))
                     for j in range(self.num_agents) if j != i]
            rel_mates = np.concatenate(mates).astype(np.float32) if mates else np.array([], np.float32)
            boundary = np.array([pos[0] / ms, (ms - pos[0]) / ms,
                                 pos[1] / ms, (ms - pos[1]) / ms,
                                 (pos[2] - self.z_min) / zr,
                                 (self.z_max - pos[2]) / zr], np.float32)
            target = [(self.target_position - pos) / scale,
                      np.array([self.target_speed / self.target_max_speed], np.float32),
                      self.target_velocity / self.target_max_speed,
                      self.target_acceleration / self.target_accel]
            parts = own + [rel_mates, boundary] + target
            if self.use_target_prediction_obs:
                parts.append(self._predict_target_future(pos).flatten())
            # 建筑节点固定放在末尾，供 Actor 的障碍注意力按 K×6 读取。
            if self.use_nearest_building_obs:
                parts.append(self._nearest_obstacle_features(pos, self.nearest_building_k).flatten())
            obs[ag] = np.concatenate(parts).astype(np.float32)
        return obs

    def state(self):
        return np.concatenate([
            self.uav_positions.flatten(), self.uav_velocities.flatten(),
            self.uav_accelerations.flatten(), self.target_position,
            self.target_velocity, self.target_acceleration,
            np.clip(self.capture_hold_times / self.capture_hold_required, 0.0, 1.0)]).astype(np.float32)

    def _closure_score(self, route_time):
        if not np.isfinite(route_time):
            return 0.0
        return float(np.exp(-max(route_time, 0.0) / self.closure_time_scale))

    def _terminal_match_scores(self):
        distances = np.linalg.norm(self.uav_positions - self.target_position[None, :], axis=1)
        relative_speeds = np.linalg.norm(self.uav_velocities - self.target_velocity[None, :], axis=1)
        gate = 1.0 / (1.0 + np.exp(np.clip(
            (distances - self.match_distance) / self.match_distance_width, -40.0, 40.0)))
        return (gate / (1.0 + relative_speeds / self.match_speed_scale)).astype(np.float32)

    def agent_mask(self):
        return {a: True for a in self.agents}

    def avail_actions(self):
        return None

    # ============================================================
    # 回合重置与状态推进
    # ============================================================
    def reset(self):
        if self.pending_curriculum_level is not None:
            self.set_curriculum_level(self.pending_curriculum_level,
                                      reload_buildings=False)
            self.pending_curriculum_level = None
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
        self.episode_min_uav_gap = float("inf")   # 每回合重置，避免沿用上回合最小间距
        self.episode_uav_conflicts = 0
        self.capture_hold_times.fill(0.0)

        if self.randomize_density:
            mode = str(np.random.choice(self.randomize_density))
            self.building_mode = mode
        if self.randomize_building_layout:
            self.buildings = self._as_building_array(self._generate_city_blocks())
            self._safe_spawn_cache.clear()
        elif self.randomize_density:
            self.buildings = self._density_pool[mode]

        self.uav_positions = self.init_uav_positions.copy()
        center = self.map_size / 2.0
        so = self.spawn_offset
        z_mid = 0.5 * (self.z_min + self.z_max)
        z_low = self.z_min + self.spawn_boundary_margin + self.spawn_vertical_radius
        z_high = self.z_max - self.spawn_boundary_margin - self.spawn_vertical_radius
        for attempt in range(200):
            if attempt < 100:
                xy = np.random.uniform(center - so, center + so, size=2).astype(np.float32)
                z = np.random.uniform(z_low, z_high) if self.surround_spawn else np.random.uniform(
                    max(self.z_min, z_mid - so), min(self.z_max, z_mid + so))
            else:
                xy = np.random.uniform(0.0, self.map_size, size=2).astype(np.float32)
                z = np.random.uniform(z_low, z_high) if self.surround_spawn else np.random.uniform(
                    self.z_min, self.z_max)
            t = np.array([xy[0], xy[1], z], np.float32)
            if not self._is_in_building(t, self.target_radius):
                self.target_position = t
                break
        else:
            raise RuntimeError("No free target spawn in generated building layout")

        self.uav_accelerations.fill(0.0)
        if self.surround_spawn:
            self._place_pursuers_around_target()   # 环绕出生，替代同侧出生队形
        desired_positions = self.uav_positions.copy()
        if self.spawn_safety_enabled:
            self.uav_positions, self.spawn_offsets = self._allocate_safe_pursuer_spawns(desired_positions)
            self._validate_pursuer_spawns(self.uav_positions)
        else:
            self.spawn_offsets = np.zeros(self.num_agents, dtype=np.float32)
        if self.surround_spawn:
            directions = self.target_position[None, :] - self.uav_positions
            norms = np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
            directions = directions / norms
        else:
            angles = np.random.uniform(0.0, 2.0 * np.pi, size=self.num_agents)
            directions = np.column_stack([np.cos(angles), np.sin(angles),
                                          np.zeros(self.num_agents)])
        self.uav_velocities = (directions * self.uav_initial_speed).astype(np.float32)
        self._sync_uav_kinematics_from_velocity()
        self.target_speed = self.target_min_speed
        target_angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.target_velocity = (self.target_speed * np.array(
            [np.cos(target_angle), np.sin(target_angle), 0.0], np.float32))
        self.target_acceleration.fill(0.0)
        self.episode_target_speed_max = self.target_speed

        initial_distances = np.linalg.norm(
            self.uav_positions - self.target_position[None, :], axis=1)
        for i, a in enumerate(self.agents):
            self.uav_trails[a].append(self.uav_positions[i].copy())
        self.target_trail.append(self.target_position.copy())

        self.target_forecast = forecast_target(self)
        self.last_coverage = reach_hold_coverage(self, self.target_forecast)
        self.last_preparation, route_time = coverage_preparation(self, self.target_forecast)
        self.last_closure = self._closure_score(route_time)
        self.last_match_scores = self._terminal_match_scores()
        self.last_hold_progress = 0.0
        self.episode_target_wall_contacts = 0
        self.episode_coverage_sum = 0.0
        self.episode_coverage_max = self.last_coverage
        self.episode_preparation_sum = 0.0
        self.episode_safe_route_time_sum = 0.0
        self.episode_safe_route_time_count = 0
        self.episode_min_target_distance = float(np.min(initial_distances))
        self.active_contact_kinds = [set() for _ in self.agents]
        self.episode_contact_seconds = {kind: 0.0 for kind in ("building", "boundary", "uav")}
        self.episode_contact_events = {kind: 0 for kind in ("building", "boundary", "uav")}
        return self._get_obs(), {
            "infos": self.get_curriculum_info(),
            "individual_episode_rewards": self.individual_episode_reward,
        }

    def _place_pursuers_around_target(self):
        """随机旋转水平队形，在保持初距的同时压缩垂直展开并留出上下机动余量。"""
        dirs = g3.fibonacci_sphere(self.num_agents)
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        ca, sa = np.cos(angle), np.sin(angle)
        for i in range(self.num_agents):
            z_offset = self.spawn_vertical_radius * float(dirs[i, 2])
            horizontal_radius = np.sqrt(self.spawn_radius ** 2 - z_offset ** 2)
            horizontal_direction = dirs[i, :2] / max(float(np.linalg.norm(dirs[i, :2])), 1e-6)
            rotated = np.array([ca * horizontal_direction[0] - sa * horizontal_direction[1],
                                sa * horizontal_direction[0] + ca * horizontal_direction[1]])
            p = self.target_position.copy()
            p[:2] += horizontal_radius * rotated
            p[2] += z_offset
            p[0] = np.clip(p[0], self.spawn_boundary_margin,
                           self.map_size - self.spawn_boundary_margin)
            p[1] = np.clip(p[1], self.spawn_boundary_margin,
                           self.map_size - self.spawn_boundary_margin)
            p[2] = np.clip(p[2], self.z_min + self.spawn_boundary_margin,
                           self.z_max - self.spawn_boundary_margin)
            tries = 0
            while self._is_in_building(p, self.uav_radius) and tries < 12:
                p = p + np.random.uniform(-40, 40, 3).astype(np.float32)
                p[0] = np.clip(p[0], self.spawn_boundary_margin,
                               self.map_size - self.spawn_boundary_margin)
                p[1] = np.clip(p[1], self.spawn_boundary_margin,
                               self.map_size - self.spawn_boundary_margin)
                p[2] = np.clip(p[2], self.z_min + self.spawn_boundary_margin,
                               self.z_max - self.spawn_boundary_margin)
                tries += 1
            self.uav_positions[i] = p

    def _evader_control(self, position=None, velocity=None, pursuer_positions=None,
                        deterministic=False):
        position = self.target_position if position is None else np.asarray(position, np.float32)
        velocity = self.target_velocity if velocity is None else np.asarray(velocity, np.float32)
        pursuer_positions = self.uav_positions if pursuer_positions is None else pursuer_positions
        rep = np.zeros(3, np.float32)
        tx, ty, tz = position
        # 目标仅在感知范围内对追捕机产生排斥，并切换至最大期望速度。
        sense = self.evader_sense
        rep_coef = getattr(self, "evader_rep_coef", 40.0)
        is_sensed = False
        for pos in pursuer_positions:
            vec = position - pos
            dist = float(np.linalg.norm(vec))
            if dist < sense:
                is_sensed = True
                if dist > 0.1:
                    rep += (vec / dist) * (rep_coef / (dist + 5.0))
        desired_speed = self.target_max_speed if is_sensed else self.target_min_speed

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
            tops = self.buildings[:, 4]
            cx = np.clip(tx, xmins, xmaxs)
            cy = np.clip(ty, ymins, ymaxs)
            vx, vy = tx - cx, ty - cy
            dists = np.hypot(vx, vy)
            active = tz <= tops + self.target_radius + 20.0
            m = active & (dists < 20.0) & (dists > 0.1)
            if np.any(m):
                s = 40.0 / (dists[m] ** 1.5)
                rep[0] += float(np.sum((vx[m] / dists[m]) * s))
                rep[1] += float(np.sum((vy[m] / dists[m]) * s))
            if np.any(active & (dists <= 0.1)):
                if deterministic:
                    rep[:2] += np.array([1.0, 0.0], np.float32) * 100.0
                else:
                    rep[:2] += np.random.randn(2).astype(np.float32) * 100.0

        # 中心偏置仅在接近空域边缘时提前转向，不能在中心附近覆盖原有航向。
        centre_bias = np.zeros(3, np.float32)
        if self.evader_center_pull > 0.0:
            centre = np.array([self.map_size * 0.5, self.map_size * 0.5,
                               0.5 * (self.z_min + self.z_max)], np.float32)
            off = centre - position
            # 各轴按自身半宽归一化。空域横向约 1000 米、高度仅约 300 米；
            # 若统一按横向半宽缩放，垂直引力会偏弱，使捕获集中到下边界。
            span = np.array([self.map_size * 0.5, self.map_size * 0.5,
                             0.5 * max(self.z_max - self.z_min, 1e-6)], np.float32)
            u = off / span                       # 各轴相对边界的归一化偏移
            m = float(np.linalg.norm(u))
            xy_clearance = min(tx, self.map_size - tx, ty, self.map_size - ty)
            z_clearance = min(tz - self.z_min, self.z_max - tz)
            edge_fraction = max(0.0, 1.0 - xy_clearance / 150.0,
                                1.0 - z_clearance / 75.0)
            if m > 1e-6 and edge_fraction > 0.0:
                centre_bias = self.evader_center_pull * edge_fraction * u * min(1.0, 1.5 / m)

        # 有明确威胁时按排斥势场避让；平时保持惯性航向并缓慢修正回中心。
        rep_norm = float(np.linalg.norm(rep))
        velocity_norm = float(np.linalg.norm(velocity))
        if rep_norm > 1e-3:
            direction = rep / rep_norm
        else:
            cruise = (velocity / velocity_norm if velocity_norm > 1e-6
                      else np.array([1.0, 0.0, 0.0], np.float32))
            heading = cruise * desired_speed + centre_bias
            direction = heading / max(float(np.linalg.norm(heading)), 1e-6)
        desired_velocity = direction * desired_speed
        acceleration_command = limit_norm(
            (desired_velocity - velocity) / self.target_velocity_response,
            self.target_accel)
        return acceleration_command

    def _advance_evader_substep(self, acceleration_command, dt):
        next_pos, next_vel, next_acc = advance_translation(
            self.target_position, self.target_velocity, self.target_acceleration,
            acceleration_command, dt,
            self.acceleration_lag, self.target_accel, self.target_jerk_max,
            self.target_max_speed)
        clipped_pos, hit = self._move_velocity_with_clip_3d(
            self.target_position, next_pos - self.target_position, self.target_radius)
        if hit and any(src.startswith("boundary:") for src in
                       self._prospective_collision_sources_velocity(
                           self.target_position, next_pos - self.target_position,
                           self.target_radius)):
            self.episode_target_wall_contacts += 1
        if np.any(np.abs(clipped_pos - next_pos) > 1e-6):
            next_vel = (clipped_pos - self.target_position) / dt
        self.target_position = clipped_pos
        self.target_velocity = next_vel.astype(np.float32)
        self.target_acceleration = next_acc
        self.target_speed = float(np.linalg.norm(self.target_velocity))
        self.episode_target_speed_max = max(self.episode_target_speed_max, self.target_speed)

    def _evader_step(self):
        """独立推进一个决策步，供环境外部调用。"""
        substeps = max(1, int(np.ceil(self.decision_dt / self.physics_dt)))
        dt = self.decision_dt / substeps
        for _ in range(substeps):
            acceleration_command = self._evader_control()
            self._advance_evader_substep(acceleration_command, dt)
        self.target_trail.append(self.target_position.copy())

    @staticmethod
    def _segment_inside_interval(start, end, radius):
        """返回线性子步轨迹位于捕获球内的时间比例区间。"""
        delta = end - start
        a = float(np.dot(delta, delta))
        c = float(np.dot(start, start) - radius * radius)
        if a < 1e-12:
            return (0.0, 1.0) if c <= 0.0 else None
        b = 2.0 * float(np.dot(start, delta))
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = np.sqrt(discriminant)
        enter = max(0.0, (-b - root) / (2.0 * a))
        leave = min(1.0, (-b + root) / (2.0 * a))
        return (enter, leave) if enter <= leave else None

    def _capture_event(self, old_relative, new_relative, dt):
        """用同步子步轨迹检查直接拦截和单机连续保持。"""
        candidates = []
        hold_intervals = []
        for i in range(self.num_agents):
            start, end = old_relative[i], new_relative[i]
            direct = self._segment_inside_interval(start, end, self.direct_intercept_radius)
            if direct is not None:
                candidates.append((direct[0], 0, i, "direct"))
            hold = self._segment_inside_interval(start, end, self.catch_radius)
            hold_intervals.append(hold)
            if hold is not None:
                enter, leave = hold
                previous = float(self.capture_hold_times[i]) if enter == 0.0 else 0.0
                if previous + (leave - enter) * dt + 1e-9 >= self.capture_hold_required:
                    fraction = np.clip(enter + max(0.0, self.capture_hold_required - previous) / dt,
                                       enter, leave)
                    candidates.append((fraction, 1, i, "sustained"))
        event = min(candidates) if candidates else None
        cutoff = float(event[0]) if event is not None else 1.0
        for i, hold in enumerate(hold_intervals):
            if hold is None or not (hold[0] <= cutoff <= hold[1]):
                self.capture_hold_times[i] = 0.0
                continue
            previous = float(self.capture_hold_times[i]) if hold[0] == 0.0 else 0.0
            self.capture_hold_times[i] = previous + (cutoff - hold[0]) * dt
        return (event[3], int(event[2]), cutoff) if event is not None else None

    def step(self, actions_dict):
        self._episode_step += 1
        rewards_dict = {a: 0.0 for a in self.agents}
        hit_flags = [False] * self.num_agents
        collision_sources = {}
        acceleration_commands = []
        for i, a in enumerate(self.agents):
            raw_act = np.asarray(actions_dict[a], dtype=np.float32).reshape(-1)
            if raw_act.size != 3:
                raise ValueError(f"Action for {a} must have three acceleration components")
            act = np.clip(raw_act, -1.0, 1.0)
            acceleration_commands.append(limit_norm(act * self.max_accel, self.max_accel))
        substeps = max(1, int(np.ceil(self.decision_dt / self.physics_dt)))
        dt = self.decision_dt / substeps
        collision_penalties = np.zeros(self.num_agents, dtype=np.float32)
        penalty_by_kind = {"building": self.building_collision_penalty,
                           "boundary": self.boundary_collision_penalty,
                           "uav": self.uav_collision_penalty}
        capture_event = None
        capture_time = None
        step_min_uav_gap = float("inf")
        for substep_index in range(substeps):
            substep_sources = [[] for _ in self.agents]
            old_positions = self.uav_positions.copy()
            old_velocities = self.uav_velocities.copy()
            old_accelerations = self.uav_accelerations.copy()
            old_target = self.target_position.copy()
            old_target_velocity = self.target_velocity.copy()
            old_target_acceleration = self.target_acceleration.copy()
            target_command = self._evader_control()
            for i, a in enumerate(self.agents):
                old_pos = self.uav_positions[i].copy()
                next_pos, next_vel, next_acc = advance_translation(
                    old_pos, self.uav_velocities[i], self.uav_accelerations[i],
                    acceleration_commands[i], dt,
                    self.acceleration_lag, self.max_accel, self.uav_jerk_max,
                    self.uav_max_speed)
                displacement = next_pos - old_pos
                sources = self._prospective_collision_sources_velocity(
                    old_pos, displacement, self.uav_radius)
                clipped_pos, hit = self._move_velocity_with_clip_3d(
                    old_pos, displacement, self.uav_radius)
                if np.any(np.abs(clipped_pos - next_pos) > 1e-6):
                    next_vel = (clipped_pos - old_pos) / dt
                self.uav_positions[i] = clipped_pos
                self.uav_velocities[i] = next_vel.astype(np.float32)
                self.uav_accelerations[i] = next_acc
                if hit:
                    substep_sources[i].extend(sources or ["unclassified"])
                hit_flags[i] |= hit
            if self.num_agents > 1:
                pair_gaps = np.linalg.norm(self.uav_positions[:, None, :] -
                                           self.uav_positions[None, :, :], axis=-1)
                np.fill_diagonal(pair_gaps, np.inf)
                step_min_uav_gap = min(step_min_uav_gap, float(np.min(pair_gaps)))
                if self.uav_collision_enabled:
                    for i in range(self.num_agents):
                        if float(np.min(pair_gaps[i])) <= self.uav_collision_distance:
                            hit_flags[i] = True
                            substep_sources[i].append("uav:teammate")
            for i, a in enumerate(self.agents):
                if substep_sources[i]:
                    sources = collision_sources.setdefault(a, [])
                    for source in substep_sources[i]:
                        if source not in sources:
                            sources.append(source)
                kinds = {source.split(":", 1)[0] for source in substep_sources[i]}
                kinds.intersection_update(penalty_by_kind)
                for kind in kinds - self.active_contact_kinds[i]:
                    collision_penalties[i] -= penalty_by_kind[kind]
                    self.episode_contact_events[kind] += 1
                for kind in kinds:
                    collision_penalties[i] -= self.collision_contact_penalty_per_second * dt
                    self.episode_contact_seconds[kind] += dt
                self.active_contact_kinds[i] = kinds
            self._advance_evader_substep(target_command, dt)
            old_relative = old_positions - old_target[None, :]
            new_relative = self.uav_positions - self.target_position[None, :]
            capture_event = self._capture_event(old_relative, new_relative, dt)
            if capture_event is not None:
                fraction = capture_event[2]
                self.uav_positions = (old_positions + (self.uav_positions - old_positions) * fraction).astype(np.float32)
                self.uav_velocities = (old_velocities + (self.uav_velocities - old_velocities) * fraction).astype(np.float32)
                self.uav_accelerations = (old_accelerations + (self.uav_accelerations - old_accelerations) * fraction).astype(np.float32)
                self.target_position = (old_target + (self.target_position - old_target) * fraction).astype(np.float32)
                self.target_velocity = (old_target_velocity + (self.target_velocity - old_target_velocity) * fraction).astype(np.float32)
                self.target_acceleration = (old_target_acceleration + (self.target_acceleration - old_target_acceleration) * fraction).astype(np.float32)
                self.target_speed = float(np.linalg.norm(self.target_velocity))
                capture_time = ((self._episode_step - 1) * self.decision_dt
                                + (substep_index + fraction) * dt)
                break
        self._sync_uav_kinematics_from_velocity()
        for i, a in enumerate(self.agents):
            self.uav_trails[a].append(self.uav_positions[i].copy())
        self.target_trail.append(self.target_position.copy())

        # 记录所有物理子步的最小机间距，穿越接触不能被决策步采样漏掉。
        if self.num_agents > 1:
            _d = np.linalg.norm(self.uav_positions[:, None, :] - self.uav_positions[None, :, :], axis=-1)
            np.fill_diagonal(_d, np.inf)
            min_uav_gap = min(step_min_uav_gap, float(_d.min()))
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
                    sources = collision_sources.setdefault(self.agents[_i], [])
                    if "uav:teammate" not in sources:
                        sources.append("uav:teammate")

        building_min = [float(np.min(d)) if len(d) else self.obstacle_avoid_range
                        for d in (self._building_distances_3d(pos) for pos in self.uav_positions)]

        cur_dists = np.linalg.norm(self.uav_positions - self.target_position[None, :], axis=1)
        is_caught = capture_event is not None
        building_crash = any(any(str(src).startswith("building:")
                                 for src in collision_sources.get(a, []))
                             for a in self.agents)
        boundary_crash = any(any(str(src).startswith("boundary:")
                                 for src in collision_sources.get(a, []))
                             for a in self.agents)
        uav_crash = any(any(str(src).startswith("uav:")
                            for src in collision_sources.get(a, []))
                        for a in self.agents)
        terminate_for_crash = bool(
            (boundary_crash and self.terminate_on_boundary_collision) or
            (uav_crash and self.terminate_on_collision) or
            (building_crash and self.terminate_on_building_collision))

        w = self.reward_weights
        min_new = float(np.min(cur_dists))
        previous_coverage = self.last_coverage
        self.target_forecast = forecast_target(self)
        current_coverage = reach_hold_coverage(self, self.target_forecast)
        current_preparation, min_route_time = coverage_preparation(self, self.target_forecast)
        current_closure = self._closure_score(min_route_time)
        current_match_scores = self._terminal_match_scores()
        terminal_state = (is_caught or terminate_for_crash or
                          self._episode_step >= self.max_episode_steps)
        # 接近与匹配采用真实进展差值；折扣势函数差在终止势值归零时
        # 折扣累计只取决于初态，无法奖励最终更接近目标。
        r_preparation = current_preparation - self.last_preparation
        self.last_preparation = current_preparation
        r_closure = current_closure - self.last_closure
        self.last_closure = current_closure
        r_match = current_match_scores - self.last_match_scores
        self.last_match_scores = current_match_scores
        self.episode_preparation_sum += current_preparation
        if np.isfinite(min_route_time):
            self.episode_safe_route_time_sum += min_route_time
            self.episode_safe_route_time_count += 1
        # 折扣势函数差分给出进展，小的水平项鼓励维持已有覆盖。
        r_coverage = (self.discount_gamma * (0.0 if terminal_state else current_coverage)
                      - previous_coverage
                      + 0.02 * current_coverage)
        self.last_coverage = current_coverage
        self.episode_coverage_sum += current_coverage
        self.episode_coverage_max = max(self.episode_coverage_max, current_coverage)
        self.episode_min_target_distance = min(self.episode_min_target_distance, min_new)
        hold_progress = float(np.max(self.capture_hold_times) / self.capture_hold_required)
        r_hold = hold_progress - self.last_hold_progress
        self.last_hold_progress = hold_progress
        elapsed_step = (capture_time - (self._episode_step - 1) * self.decision_dt
                        if capture_time is not None else self.decision_dt)

        for i, a in enumerate(self.agents):
            dist = float(cur_dists[i])
            proximity = max(0.0, 1.0 - building_min[i] / self.obstacle_avoid_range)
            r_obs = -0.5 * self.obs_avoid_weight * proximity ** 2
            action = np.asarray(actions_dict[a], dtype=np.float32).reshape(-1)
            accel_effort = float(np.mean(np.clip(action[:3], -1.0, 1.0) ** 2))
            r_accel = -accel_effort
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
            xy_clearance = min(
                float(self.uav_positions[i, 0]) - self.uav_radius,
                self.map_size - self.uav_radius - float(self.uav_positions[i, 0]),
                float(self.uav_positions[i, 1]) - self.uav_radius,
                self.map_size - self.uav_radius - float(self.uav_positions[i, 1]),
            )
            boundary_fraction = np.clip(
                (self.boundary_warning_distance - xy_clearance) /
                max(self.boundary_warning_distance, 1e-6), 0.0, 1.0)
            r_xy_boundary = -float(boundary_fraction ** 2)
            # 动作为三轴加速度指令，各轴采用相同惩罚。
            r_safe = (r_obs + 0.5 * r_mate + 0.2 * r_accel
                      + 0.5 * r_zlimit + 0.5 * r_xy_boundary
                      + self.separation_weight * r_separation
                      + collision_penalties[i])

            wr_cov = 0.0 if "r_cov" in self.reward_disable else w["w_cov"] * r_coverage
            wr_prep = (0.0 if "r_prep" in self.reward_disable
                       else w["w_prep"] * r_preparation)
            wr_close = (0.0 if "r_close" in self.reward_disable
                        else w.get("w_close", 0.0) * r_closure)
            wr_match = (0.0 if "r_match" in self.reward_disable
                        else w.get("w_match", 0.0) * float(r_match[i]))
            wr_hold = 0.0 if "r_hold" in self.reward_disable else w["w_hold"] * r_hold
            wr_step = (0.0 if "r_step" in self.reward_disable
                       else -w.get("w_step", 0.0) * elapsed_step)
            wr_safe = 0.0 if "r_safe" in self.reward_disable else w["w_safe"] * r_safe
            wr_terminal = (0.0 if "r_terminal" in self.reward_disable or not is_caught
                           else (w.get("w_direct_terminal", w["w_terminal"])
                                 if capture_event[0] == "direct" else w["w_terminal"]))
            rewards_dict[a] = (wr_cov + wr_prep + wr_close + wr_match + wr_hold +
                               wr_step + wr_safe + wr_terminal)
            sr = self.episode_sub_rewards[a]
            sr["r_cov"] += wr_cov
            sr["r_prep"] += wr_prep
            sr["r_close"] += wr_close
            sr["r_match"] += wr_match
            sr["r_hold"] += wr_hold
            sr["r_step"] += wr_step
            sr["r_safe"] += wr_safe
            sr["r_terminal"] += wr_terminal

        for k, v in rewards_dict.items():
            self.individual_episode_reward[k] += v

        hard_crash = boundary_crash or uav_crash
        is_crashed = any(hit_flags)
        terminated = {a: bool(is_caught or terminate_for_crash) for a in self.agents}
        truncated = (self._episode_step >= self.max_episode_steps) if not terminated[self.agents[0]] else False
        info = {
            "infos": self.get_curriculum_info(),
            "individual_episode_rewards": self.individual_episode_reward,
            "episode_sub_rewards": copy.deepcopy(self.episode_sub_rewards),
            "is_success": bool(is_caught),
            "capture_type": capture_event[0] if capture_event is not None else None,
            "capture_agent": self.agents[capture_event[1]] if capture_event is not None else None,
            "capture_time": capture_time,
            "capture_hold_times": self.capture_hold_times.tolist(),
            "min_target_distance": float(np.min(cur_dists)),
            "coverage_score": current_coverage,
            "coverage_change": current_coverage - previous_coverage,
            "preparation_score": current_preparation,
            "closure_score": current_closure,
            "terminal_match_score_max": float(np.max(current_match_scores)),
            "safe_route_time_min": min_route_time,
            "target_wall_contacts": self.episode_target_wall_contacts,
            "is_collision": bool(is_crashed),
            "is_boundary_collision": bool(boundary_crash),
            "is_uav_collision": bool(uav_crash),
            "is_building_collision": bool(building_crash),
            "is_hard_collision": bool(hard_crash),
        }
        info.update({
            "collision_agents": [agent for agent in self.agents if agent in collision_sources],
            "collision_sources": collision_sources,
            "min_uav_gap": min_uav_gap,
            "episode_min_uav_gap": self.episode_min_uav_gap,
            "uav_conflict": uav_conflict,
            "episode_uav_conflicts": self.episode_uav_conflicts,
        })
        return self._get_obs(), rewards_dict, terminated, truncated, info

    # ============================================================
    # 三维渲染
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

        # 以不同高度的棱柱绘制建筑
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b[:4]
            top = self._building_top(b)
            dx, dy, dz = xmax - xmin, ymax - ymin, max(top - self.z_min, 1e-6)
            ax.bar3d(xmin, ymin, self.z_min, dx, dy, dz,
                     color="gray", alpha=0.25, shade=True)

        colors = ["#FF8C00", "#32CD32", "#00BFFF", "#9400D3", "#FF1493", "#1f77b4",
                  "#8c564b", "#e377c2"]
        # 绘制目标
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
