import numpy as np
import gymnasium as gym
import cv2
from collections import deque
from xuance.environment import RawMultiAgentEnv


class UAVPursuitEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(UAVPursuitEnv, self).__init__()

        # ---------------- 1. 真实物理环境参数 ----------------
        self.map_size = 1000.0

        self.uav_min_speed = 9.0
        self.uav_max_speed = 11.0

        self.max_accel = 1
        self.max_yaw_rate = np.pi / 3
        self.uav_radius = 0.5

        self.target_speed = 12.0
        self.target_radius = 0.5

        self.catch_radius = 15.0
        self.radar_range = 100.0
        self.num_radar_rays = 16



        self.num_agents = 3
        self.agents = [f"uav_{i}" for i in range(self.num_agents)]

        self.buildings = np.array(self._generate_city_blocks(), dtype=np.float32)

        self.init_uav_positions = np.array([
            [100.0, 200.0],
            [100.0, 500.0],
            [100.0, 800.0]
        ], dtype=np.float32)

        # ---------------- 2. 绘图样式参数 ----------------
        plot_config = getattr(config, "plot_config", {})
        self.uav_plot_radius = plot_config.get("uav_radius", 4)
        self.target_plot_radius = plot_config.get("target_radius", 4)
        self.font_size_base = plot_config.get("font_size_base", 20)

        # ---------------- 3. 定义状态与动作空间 ----------------
        self.action_space = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent in
                             self.agents}

        # [修改点 1]: obs_dim 从 34 降到 27 (减去 7 根视觉射线)
        # 构成: pos(2) + yaw_vec(2) + norm_speed(1) + rel_target(2) + rel_mates(4) + radar_data(16) = 27
        obs_dim = 27
        self.observation_space = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32) for
                                  agent in self.agents}
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

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

    def _generate_city_blocks(self):
        import json
        import os
        difficulty = "open"
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

    def _get_obs(self):
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            pos, yaw = self.uav_positions[i], self.uav_yaws[i]
            yaw_vec = np.array([np.sin(yaw), np.cos(yaw)])
            norm_speed = np.array(
                [(self.uav_speeds[i] - self.uav_min_speed) / (self.uav_max_speed - self.uav_min_speed)])
            rel_target = (self.target_position - pos) / self.map_size
            rel_mates = []
            for j in range(self.num_agents):
                if i != j: rel_mates.append((self.uav_positions[j] - pos) / self.map_size)
            rel_mates = np.concatenate(rel_mates) if rel_mates else np.array([])

            # [修改点 1]: 仅保留雷达射线，剔除视觉射线
            radar_data = self._raycast(pos, yaw + np.linspace(0, 2 * np.pi, self.num_radar_rays, endpoint=False),
                                       self.radar_range)
            obs_dict[agent] = np.concatenate(
                [pos / self.map_size, yaw_vec, norm_speed, rel_target, rel_mates, radar_data]).astype(
                np.float32)
        return obs_dict

    def _move_with_clip(self, pos, yaw, speed, radius):
        dx, dy = speed * np.cos(yaw), speed * np.sin(yaw)
        next_pos = pos + np.array([dx, dy])
        hit_obstacle = False
        if next_pos[0] < radius:
            next_pos[0] = radius;
            hit_obstacle = True
        elif next_pos[0] > self.map_size - radius:
            next_pos[0] = self.map_size - radius;
            hit_obstacle = True
        if next_pos[1] < radius:
            next_pos[1] = radius;
            hit_obstacle = True
        elif next_pos[1] > self.map_size - radius:
            next_pos[1] = self.map_size - radius;
            hit_obstacle = True
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < next_pos[0] < xmax + radius) and (ymin - radius < next_pos[1] < ymax + radius):
                hit_obstacle = True
                pos_x = np.array([next_pos[0], pos[1]])
                if not ((xmin - radius < pos_x[0] < xmax + radius) and (
                        ymin - radius < pos_x[1] < ymax + radius)): next_pos = pos_x; break
                pos_y = np.array([pos[0], next_pos[1]])
                if not ((xmin - radius < pos_y[0] < xmax + radius) and (
                        ymin - radius < pos_y[1] < ymax + radius)): next_pos = pos_y; break
                next_pos = pos;
                break
        return next_pos, hit_obstacle

    def _cal_triangle_area(self, p1, p2, p3):
        return abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))

    # [新增辅助方法 2]: 检查某坐标是否在建筑物内部
    def _is_in_building(self, pos, radius):
        for b in self.buildings:
            xmin, xmax, ymin, ymax = b
            if (xmin - radius < pos[0] < xmax + radius) and (ymin - radius < pos[1] < ymax + radius):
                return True
        return False

    def reset(self):
        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        for agent in self.agents: self.uav_trails[agent].clear()
        self.target_trail.clear()

        # [修改点 2]: 三架无人机固定在初始位置
        self.uav_positions = self.init_uav_positions.copy()

        # [修改点 2]: 逃逸者随机生成，且确保不落在建筑物内
        while True:
            t_pos = np.random.uniform(self.target_radius, self.map_size - self.target_radius, size=2).astype(np.float32)
            if not self._is_in_building(t_pos, self.target_radius):
                self.target_position = t_pos
                break

        self.uav_speeds = np.ones(self.num_agents, dtype=np.float32) * self.uav_min_speed
        self.uav_yaws = np.random.uniform(0, 2 * np.pi, size=self.num_agents).astype(np.float32)
        self.target_yaw = np.random.uniform(0, 2 * np.pi)

        for i in range(self.num_agents):
            self.last_distances[i] = np.linalg.norm(self.uav_positions[i] - self.target_position)

        for i, agent in enumerate(self.agents): self.uav_trails[agent].append(self.uav_positions[i].copy())
        self.target_trail.append(self.target_position.copy())

        return self._get_obs(), {"infos": {}, "individual_episode_rewards": self.individual_episode_reward}

    def step(self, actions_dict):
        self._episode_step += 1
        rewards_dict = {agent: 0.0 for agent in self.agents}
        hit_flags, radar_distances = [], []

        # 1. 物理运动与观测更新 (保持原有逻辑)
        for i, agent in enumerate(self.agents):
            action = np.clip(actions_dict[agent], -1.0, 1.0)
            self.uav_speeds[i] = np.clip(self.uav_speeds[i] + action[0] * self.max_accel, self.uav_min_speed,
                                         self.uav_max_speed)
            self.uav_yaws[i] = (self.uav_yaws[i] + action[1] * self.max_yaw_rate) % (2 * np.pi)

            new_pos, hit_obs = self._move_with_clip(self.uav_positions[i], self.uav_yaws[i], self.uav_speeds[i],
                                                    self.uav_radius)
            self.uav_positions[i] = new_pos
            hit_flags.append(hit_obs)
            self.uav_trails[agent].append(new_pos.copy())

            radar_dists = self._raycast(new_pos, self.uav_yaws[i] + np.linspace(0, 2 * np.pi, self.num_radar_rays,
                                                                                endpoint=False),
                                        self.radar_range) * self.radar_range
            radar_distances.append(np.min(radar_dists))

            # ==================== 逃逸机智能人工势场法 ====================
            repulsive_force = np.zeros(2, dtype=np.float32)

            # 1. 无人机带来的斥力 (感知范围 300 米)
            sense_range = 100.0
            for pos in self.uav_positions:
                vec = self.target_position - pos
                dist = np.linalg.norm(vec)
                if 0.1 < dist < sense_range:
                    # 使用 1/dist 并乘以放大系数，确保斥力在远距离不被忽略
                    strength = 1 / dist
                    repulsive_force += (vec / dist) * strength

            # 2. 墙壁与障碍物带来的斥力 (防止逃逸机撞墙后盲目乱转)
            wall_sense = 50.0  # 离墙 50 米内开始感受到排斥
            tx, ty = self.target_position

            # 边界斥力
            if tx < wall_sense: repulsive_force[0] += 5.0 / max(tx, 0.1)
            if self.map_size - tx < wall_sense: repulsive_force[0] -= 5.0 / max(self.map_size - tx, 0.1)
            if ty < wall_sense: repulsive_force[1] += 5.0 / max(ty, 0.1)
            if self.map_size - ty < wall_sense: repulsive_force[1] -= 5.0 / max(self.map_size - ty, 0.1)

            # (可选) 如果你希望它避开建筑物，也可以在这里加上建筑物的斥力

            # 3. 计算期望方向与移动
            # 只要受力超过一个很小的阈值，就开始逃跑
            if np.linalg.norm(repulsive_force) > 1e-3:
                current_target_speed = self.target_speed
                desired_yaw = np.arctan2(repulsive_force[1], repulsive_force[0])
                # 平滑转向，将噪声加在角度上，而不是向量上 (模拟逃避时的轻微慌乱)
                self.target_yaw += np.clip((desired_yaw - self.target_yaw + np.pi) % (2 * np.pi) - np.pi,
                                           -np.pi / 4, np.pi / 4) + np.random.normal(0, 0.05)
            else:
                # 没有任何危险时，平稳地漫游
                current_target_speed = 4
                self.target_yaw += np.random.normal(0, 0.1)

            new_tpos, t_hit = self._move_with_clip(self.target_position, self.target_yaw, current_target_speed,
                                                   self.target_radius)
            self.target_position = new_tpos

            # 如果因为速度太快还是撞墙了，给予一个强烈的反弹修正
            if t_hit:
                self.target_yaw += np.random.uniform(np.pi / 2, np.pi) * np.random.choice([-1, 1])

            self.target_trail.append(new_tpos.copy())
            # ===============================================================

        # 状态统计
        cur_dists = [np.linalg.norm(self.uav_positions[i] - self.target_position) for i in range(self.num_agents)]
        Sum_d, Sum_last_d = sum(cur_dists), sum(self.last_distances)
        S4 = self._cal_triangle_area(self.uav_positions[0], self.uav_positions[1], self.uav_positions[2])
        Sum_S = self._cal_triangle_area(self.uav_positions[0], self.uav_positions[1], self.target_position) + \
                self._cal_triangle_area(self.uav_positions[1], self.uav_positions[2], self.target_position) + \
                self._cal_triangle_area(self.uav_positions[2], self.uav_positions[0], self.target_position)

        d_capture = self.catch_radius
        is_encircled = abs(Sum_S - S4) <= (1e-3 * S4 + 1e-5)
        is_caught = any(d <= d_capture for d in cur_dists)

        # ==================== 2. 全新归一化奖励计算 ====================
        # 定义权重与极值 (可根据需要提取到 __init__ 中)
        w_near = 1.0
        w_safe = 1.0
        w_course = 3
        w_turn = 0.5  # 降低转向惩罚权重，作为轻微正则化即可
        w_finish = 1.0

        # 理论最大相对速度差 = 无人机最大速度 + 目标最大速度 (13.0 + 10.0 = 23.0)
        max_relative_speed = self.uav_max_speed + self.target_speed

        for i, agent in enumerate(self.agents):
            dist = cur_dists[i]

            # [A. 动作平滑惩罚]: 范围 [-1.0, 0.0]
            r_turn = - (np.clip(actions_dict[agent][1], -1.0, 1.0) ** 2)

            # [B. 安全与逼近奖励]
            if hit_flags[i]:
                # 撞墙重罚缩放至 -10.0，避免梯度爆炸
                r_safe, r_near = -10.0, 0.0
            else:
                # 归一化逼近奖励: 距离差除以理论最大速度差，将其严格限制在约 [-1.0, 1.0] 的范围内
                r_near = (self.last_distances[i] - dist) / max_relative_speed

                # 归一化安全奖励: 存活步进惩罚(-0.02) + 雷达避障引导(离墙越近越惩罚, 最大至 -0.2)
                radar_penalty = (radar_distances[i] - self.radar_range) / self.radar_range
                r_safe = -0.02 + 0.2 * radar_penalty

            # [C. 团队协同合围奖励]
            r_course = 0.0
            if not is_encircled:
                # [原代码] 归一化至 [-1.0, 1.0] 左右的追踪收益
                r_track = np.clip((Sum_last_d - Sum_d) / (self.num_agents * self.uav_max_speed), -1.0, 1.0)

                # [修改点] 将原本的对数惩罚改为指数奖励
                # 1. 计算归一化的面积差 (0 ~ 1 之间的小数)
                area_diff_norm = (Sum_S - S4) / (self.map_size ** 2)

                # 2. 使用指数函数将面积差转化为 [0.0, 1.0] 的正向质量得分
                # 当面积差为 0 时 (完美包围)，r_encircle = 1.0
                # 当面积差很大时，r_encircle 趋近于 0.0
                # 这里的 50.0 是一个平滑系数，可以根据实际收敛速度微调
                r_encircle = np.exp(-50.0 * area_diff_norm)

                # [原代码] 动态权重分配保持不变
                alpha = 1.0 / (1.0 + np.exp(-(Sum_d - 10 * d_capture) / 15.0))
                r_course = alpha * r_track + (1.0 - alpha) * r_encircle

            elif is_encircled and any(d > d_capture for d in cur_dists):
                # [原代码] 包围后收缩保持不变
                r_course = np.clip(np.exp((Sum_last_d - Sum_d) / (3.0 * self.uav_max_speed)) / 3.0, 0.0, 1.0)

            r_finish = 40.0 if is_caught else 0.0

            # [E. 汇总]
            rewards_dict[agent] = (w_near * r_near +
                                   w_safe * r_safe +
                                   w_course * r_course +
                                   w_turn * r_turn +
                                   w_finish * r_finish)

        # ===============================================================

        self.last_distances = cur_dists.copy()

        for k, v in rewards_dict.items(): self.individual_episode_reward[k] += v
        is_crashed = any(hit_flags)
        terminated = {agent: (is_caught or is_crashed) for agent in self.agents}
        truncated = (self._episode_step >= self.max_episode_steps) if not terminated[self.agents[0]] else False

        return self._get_obs(), rewards_dict, terminated, truncated, {"infos": {},
                                                                      "individual_episode_rewards": self.individual_episode_reward}
    def state(self):
        return np.concatenate([self.uav_positions.flatten(), self.uav_yaws, self.uav_speeds, self.target_position,
                               [self.target_yaw]]).astype(np.float32)

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

        # [修改点 1]: 将画布变宽 (11, 8)，并使用 gridspec 分割为左右两块 (比例 3:1)
        fig = plt.figure(figsize=(11, 8), dpi=150)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

        # 左侧：地图区域
        ax = fig.add_subplot(gs[0, 0])
        # 右侧：信息面板区域
        ax_info = fig.add_subplot(gs[0, 1])

        # ==================== 左侧：地图渲染 ====================
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

        # ==================== 右侧：信息面板渲染 ====================
        ax_info.axis('off')  # 隐藏坐标轴

        # 计算当前最短距离
        cur_dists = [np.linalg.norm(self.uav_positions[i] - self.target_position) for i in range(self.num_agents)]
        min_dist = min(cur_dists)

        # 构建信息文本（增加间距，排版更美观）
        info_text = "== Dashboard ==\n\n"
        info_text += f"Step: {self._episode_step}\n\n"
        info_text += f"Min Dist: {min_dist:.1f} m\n\n"
        for i in range(self.num_agents):
            info_text += f"UAV {i} Speed:\n  {self.uav_speeds[i]:.1f} m/s\n\n"

        # 在右侧面板中心偏上位置绘制文字
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                     fontsize=14, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8F9FA', edgecolor='#CCCCCC', alpha=1.0))

        # ========================================================

        # 调整子图之间的间距，避免重叠
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