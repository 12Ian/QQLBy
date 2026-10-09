import numpy as np
import gymnasium as gym
import cv2
from collections import deque
from xuance.environment import RawMultiAgentEnv
import copy
from scipy.optimize import linear_sum_assignment

class UAVPursuitApolloniusObs3Env(RawMultiAgentEnv):
    def __init__(self, config):
        super(UAVPursuitApolloniusObs3Env, self).__init__()

        # ---------------- 1. 真实物理环境参数 ----------------
        self.map_size = 1000.0

        self.uav_min_speed = 9.0
        self.uav_max_speed = 11.0
        self.max_accel = 1
        self.max_yaw_rate = np.pi / 3
        self.uav_radius = 0.5

        # [新增]: 逃逸机动态速度参数
        self.target_min_speed = 4.0
        self.target_max_speed = 12.0
        self.target_accel = 0.5     # 每次 step 提速/减速的步长
        self.target_speed = self.target_min_speed
        
        self.target_radius = 0.5
        self.catch_radius = 15.0
        
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

        # obs_dim = 32
        obs_dim = 32
        self.observation_space = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32) for agent in self.agents}
        
        # [修改点]: state_space 变为 28，加入了 4个引导点的坐标(8维)，加速 Critic 网络收敛
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

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

        self.uav_trails = {agent: deque(maxlen=200) for agent in self.agents}
        self.target_trail = deque(maxlen=200)

        self.episode_sub_rewards = {
            agent: {"r_near": 0.0, "r_safe": 0.0, "r_turn": 0.0, "r_finish": 0.0}
            for agent in self.agents
        }

        self.current_guide_points = {}
        self.last_guide_distances = {agent: 0.0 for agent in self.agents}

    def _generate_city_blocks(self):
        import json
        import os
        difficulty = "zero"
        # 请根据你的实际路径修改
        config_file = os.path.join(r"D:\Pragram\UAV\xuance-master\xuance\environment\buildings",
                                   f"buildings_{difficulty}.json")

        if os.path.exists(config_file):
            print(f"🌍 [Env Info] Loading {difficulty} map from: {config_file}")
            with open(config_file, 'r', encoding='utf-8') as f:
                buildings = json.load(f)
            return buildings
        else:
            print(f"⚠️ [Warning] Config file {config_file} not found! Loading empty map.")
            return []

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

    def _assign_target_points(self):
        """
        生成 4 个基于绝对方向的动态收缩引导点，并用匈牙利算法最优分配
        """
        cur_dists_to_target = [np.linalg.norm(pos - self.target_position) for pos in self.uav_positions]
        avg_dist = np.mean(cur_dists_to_target)
        
        # 动态合围半径 (牵引网)：时刻保持在无人机群体前方，随着无人机逼近而收紧
        r_f = np.clip(avg_dist * 0.8, self.catch_radius * 0.5, self.catch_radius * 4.0)
        
        # 绝对坐标系：东、北、西、南
        angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]
        
        target_points = []
        for angle in angles:
            gx = self.target_position[0] + r_f * np.cos(angle)
            gy = self.target_position[1] + r_f * np.sin(angle)
            gx = np.clip(gx, self.target_radius, self.map_size - self.target_radius)
            gy = np.clip(gy, self.target_radius, self.map_size - self.target_radius)
            target_points.append(np.array([gx, gy], dtype=np.float32))
            
        cost_matrix = np.zeros((self.num_agents, 4), dtype=np.float32)
        
        for i, uav_pos in enumerate(self.uav_positions):
            uav_yaw = self.uav_yaws[i]
            for j, tp_pos in enumerate(target_points):
                dist = np.linalg.norm(uav_pos - tp_pos)
                vec_to_tp = tp_pos - uav_pos
                ideal_yaw = np.arctan2(vec_to_tp[1], vec_to_tp[0])
                angle_diff = abs((uav_yaw - ideal_yaw + np.pi) % (2 * np.pi) - np.pi)
                cost_matrix[i, j] = dist + 20.0 * angle_diff

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignment_dict = {}
        for idx in range(self.num_agents):
            agent_name = self.agents[row_ind[idx]]
            assignment_dict[agent_name] = target_points[col_ind[idx]]
            
        return assignment_dict

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
        
        self.uav_positions = self.init_uav_positions.copy()

        center_pos = self.map_size / 2.0
        spawn_offset = 50.0 

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
        for i, agent in enumerate(self.agents):
            self.last_guide_distances[agent] = np.linalg.norm(self.uav_positions[i] - self.current_guide_points[agent])

        return self._get_obs(), {"infos": {}, "individual_episode_rewards": self.individual_episode_reward}

    def step(self, actions_dict):
        self._episode_step += 1
        rewards_dict = {agent: 0.0 for agent in self.agents}
        hit_flags, radar_distances = [], []

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
        sense_range = 100.0
        is_sensed = False  # 警戒标志位

        for pos in self.uav_positions:
            vec = self.target_position - pos
            dist = np.linalg.norm(vec)
            if dist < sense_range:
                is_sensed = True  # 发现追捕者！
                if dist > 0.1:
                    strength = 5 / dist
                    repulsive_force += (vec / dist) * strength

        # [核心动态速度控制]: 发现敌人则加速，安全则缓慢减速
        if is_sensed:
            self.target_speed = min(self.target_max_speed, self.target_speed + self.target_accel)
        else:
            self.target_speed = max(self.target_min_speed, self.target_speed - self.target_accel)

        wall_sense = 50.0 
        tx, ty = self.target_position
        if tx < wall_sense: repulsive_force[0] += 5.0 / max(tx, 0.1)
        if self.map_size - tx < wall_sense: repulsive_force[0] -= 5.0 / max(self.map_size - tx, 0.1)
        if ty < wall_sense: repulsive_force[1] += 5.0 / max(ty, 0.1)
        if self.map_size - ty < wall_sense: repulsive_force[1] -= 5.0 / max(self.map_size - ty, 0.1)

        if np.linalg.norm(repulsive_force) > 1e-3:
            desired_yaw = np.arctan2(repulsive_force[1], repulsive_force[0])
            self.target_yaw += np.clip((desired_yaw - self.target_yaw + np.pi) % (2 * np.pi) - np.pi, -np.pi / 4, np.pi / 4) + np.random.normal(0, 0.05)
        else:
            self.target_yaw += np.random.normal(0, 0.1)

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

        w_near = 2.0  
        w_safe = 1.0
        w_turn = 0.2 
        w_finish = 2.0

        # 分母统一采用双方的最大可能相对速度，保证奖励比例稳定
        max_relative_speed = self.uav_max_speed + self.target_max_speed

        for i, agent in enumerate(self.agents):
            dist = cur_dists[i]
            r_turn = - (np.clip(actions_dict[agent][1], -1.0, 1.0) ** 2)

            # 获取当前无人机距离【专属引导点】的距离
            dist_to_guide = np.linalg.norm(self.uav_positions[i] - self.current_guide_points[agent])

            if hit_flags[i]:
                r_safe, r_near = -15.0, 0.0
            else:
                r_near = (self.last_guide_distances[agent] - dist_to_guide) / max_relative_speed

                radar_penalty = ((radar_distances[i] - self.radar_range) / self.radar_range)**2
                r_safe = - 0.5 * radar_penalty
            
            # 抓捕奖励依然看真实目标的距离
            r_finish = 0.0
            if is_caught:
                r_finish += 50.0 
            elif dist < self.catch_radius * 3:
                soft_r = (self.catch_radius * 3 - dist) / (self.catch_radius * 2)
                r_finish += soft_r * 0.2 

            w_r_near = w_near * r_near
            w_r_safe = w_safe * r_safe
            w_r_turn = w_turn * r_turn
            w_r_finish = w_finish * r_finish

            rewards_dict[agent] = w_r_near + w_r_safe + w_r_turn + w_r_finish 
            self.episode_sub_rewards[agent]["r_near"] += w_r_near
            self.episode_sub_rewards[agent]["r_safe"] += w_r_safe
            self.episode_sub_rewards[agent]["r_turn"] += w_r_turn
            self.episode_sub_rewards[agent]["r_finish"] += w_r_finish

            self.last_guide_distances[agent] = dist_to_guide

        self.last_distances = cur_dists.copy()

        for k, v in rewards_dict.items(): 
            self.individual_episode_reward[k] += v
            
        is_crashed = any(hit_flags)
        terminated = {agent: (is_caught or is_crashed) for agent in self.agents}
        truncated = (self._episode_step >= self.max_episode_steps) if not terminated[self.agents[0]] else False

        info = {
            "infos": {},
            "individual_episode_rewards": self.individual_episode_reward,
            "episode_sub_rewards": copy.deepcopy(self.episode_sub_rewards),
            "is_success": is_caught 
        }
        return self._get_obs(), rewards_dict, terminated, truncated, info
    
    def state(self):
        # 展平所有的分配引导点坐标 (8维)
        guide_points_flat = np.concatenate([self.current_guide_points[agent] for agent in self.agents])
        
        # 返回 28 维的全局上帝视角 State，让 MADDPG 的 Critic 能够明察秋毫
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
        from matplotlib.colors import LinearSegmentedColormap
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

        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=1, edgecolor='black', facecolor='black', zorder=2)
            ax.add_patch(rect)

        tx, ty = self.target_position
        target_radar = patches.Circle((tx, ty), self.radar_range,
                                      color='#FF1493', alpha=0.15, zorder=1, clip_on=True)
        ax.add_patch(target_radar)

        for i in range(self.num_agents):
            ux, uy = self.uav_positions[i]
            radar = patches.Circle((ux, uy), self.radar_range,
                                   color='#00FFFF', alpha=0.15, zorder=1, clip_on=True)
            ax.add_patch(radar)

        if len(self.target_trail) > 1:
            tx_trail, ty_trail = zip(*self.target_trail)
            ax.plot(tx_trail, ty_trail, color='red', linewidth=1.5, zorder=3)

        for agent in self.agents:
            trail = list(self.uav_trails[agent])
            if len(trail) > 1:
                pts = np.array(trail)
                segments = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
                norm = plt.Normalize(0, len(trail))
                cmap = LinearSegmentedColormap.from_list('uav_trail', ['#D0E0FF', '#00008B'])
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, zorder=3)
                lc.set_array(np.arange(len(trail)))
                ax.add_collection(lc)

        ax.plot(tx, ty, marker='o', markersize=self.target_plot_radius, color='red', zorder=4)
        ax.text(tx + 25, ty - 25, "Target", fontsize=11, zorder=5)
        ax.arrow(tx, ty, 40 * np.cos(self.target_yaw), 40 * np.sin(self.target_yaw),
                 head_width=15, head_length=15, fc='black', ec='black', zorder=5)

        for i in range(self.num_agents):
            ux, uy = self.uav_positions[i]
            ax.plot(ux, uy, marker='o', markersize=self.uav_plot_radius, color='#00008B', zorder=4)
            ax.text(ux + 25, uy - 25, f"UAV {i}", fontsize=11, zorder=5)
            ax.arrow(ux, uy, 40 * np.cos(self.uav_yaws[i]), 40 * np.sin(self.uav_yaws[i]),
                     head_width=15, head_length=15, fc='black', ec='black', zorder=5)

        ax_info.axis('off') 

        cur_dists = [np.linalg.norm(self.uav_positions[i] - self.target_position) for i in range(self.num_agents)]
        min_dist = min(cur_dists)

        info_text = f"== Dashboard ==\n"
        info_text += f"Step: {self._episode_step} / {self.max_episode_steps}\n"
        info_text += f"Min Dist: {min_dist:.1f} m\n"
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