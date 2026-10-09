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
            agent: {"r_near": 0.0, "r_safe": 0.0, "r_turn": 0.0, "r_finish": 0.0, "r_mate": 0.0} for agent in self.agents
        }
        # [新增]: 用于存储渲染所需的射线与扇区可视化数据
        self.debug_rays = []
        self.debug_rf = 0.0
        self.current_guide_points = {}


    def _generate_city_blocks(self):
        import json
        import os
        difficulty = "medium"  # 你可以根据需要修改为 "easy", "medium", "hard"
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

    def _assign_target_points(self):
        cur_dists_to_target = [np.linalg.norm(pos - self.target_position) for pos in self.uav_positions]
        avg_dist = np.mean(cur_dists_to_target)

        # 1. 动态合围半径
        r_f = np.clip(avg_dist * 0.8, self.catch_radius * 0.5, self.catch_radius * 4.0)
        self.debug_rf = r_f  # [新增]: 记录当前合围半径供渲染使用
        safe_margin = 50.0

        # 2. [极致优化] 射线探测逃生路线 (全向量化)
        num_rays = 72
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        dx = np.cos(angles)  # shape: (72,)
        dy = np.sin(angles)  # shape: (72,)
        
        # (A) 批量检测边界
        tx_bound = np.where(dx > 0, 
                            (self.map_size - safe_margin - self.target_position[0]) / (dx + 1e-8), 
                            (safe_margin - self.target_position[0]) / (dx - 1e-8))
        ty_bound = np.where(dy > 0, 
                            (self.map_size - safe_margin - self.target_position[1]) / (dy + 1e-8), 
                            (safe_margin - self.target_position[1]) / (dy - 1e-8))
        
        tx_bound = np.maximum(0.0, tx_bound)
        ty_bound = np.maximum(0.0, ty_bound)
        min_dists = np.minimum(tx_bound, ty_bound)  # 初始最短距离为主地图边界距离

        # (B) 批量检测建筑物 AABB
        if len(self.buildings) > 0:
            # 将建筑物坐标变为列向量 shape: (N, 1)
            xmins = self.buildings[:, 0:1]
            xmaxs = self.buildings[:, 1:2]
            ymins = self.buildings[:, 2:3]
            ymaxs = self.buildings[:, 3:4]
            
            # 将射线方向变为行向量 shape: (1, 72)
            dx_b = dx.reshape(1, -1)
            dy_b = dy.reshape(1, -1)
            
            # 利用广播机制，一次性计算 N个建筑 x 72根射线 的交点参数
            tx1 = (xmins - self.target_position[0]) / (dx_b + 1e-8)
            tx2 = (xmaxs - self.target_position[0]) / (dx_b + 1e-8)
            ty1 = (ymins - self.target_position[1]) / (dy_b + 1e-8)
            ty2 = (ymaxs - self.target_position[1]) / (dy_b + 1e-8)
            
            tmin_x = np.minimum(tx1, tx2)
            tmax_x = np.maximum(tx1, tx2)
            tmin_y = np.minimum(ty1, ty2)
            tmax_y = np.maximum(ty1, ty2)
            
            tmin = np.maximum(tmin_x, tmin_y)  # 进入 AABB 的时间
            tmax = np.minimum(tmax_x, tmax_y)  # 离开 AABB 的时间
            
            # 命中条件
            valid_hit = (tmax >= 0) & (tmin <= tmax) & (tmin > 0)
            
            # 沿着建筑物维度(axis=0)取最小的有效击中距离
            building_dists = np.min(np.where(valid_hit, tmin, np.inf), axis=0)
            
            # 更新最终的最短距离
            min_dists = np.minimum(min_dists, building_dists)

        # (C) 批量评判缺口与记录渲染数据
        is_open = min_dists > (r_f * 1.5)
        plot_dists = np.minimum(min_dists, r_f * 1.5)
        # 直接使用 zip 快速打包供 render 使用
        self.debug_rays = list(zip(angles, plot_dists, is_open))

        # ---------------- 提取独立扇区与分兵策略 (保持原样) ----------------
        open_indices = np.where(is_open)[0]
        target_points = []
        
        if len(open_indices) == 0:
            # 【阶段二：无路可逃】全部被封死，强制缩小半径捕杀
            r_f_shrink = self.catch_radius * 0.8
            angles_to_use = [0.0, np.pi / 2, np.pi, -np.pi / 2]
            for angle in angles_to_use:
                gx = self.target_position[0] + r_f_shrink * np.cos(angle)
                gy = self.target_position[1] + r_f_shrink * np.sin(angle)
                gx = np.clip(gx, safe_margin, self.map_size - safe_margin)
                gy = np.clip(gy, safe_margin, self.map_size - safe_margin)
                target_points.append(np.array([gx, gy], dtype=np.float32))

        elif len(open_indices) == num_rays:
            # 【阶段一 (A)：完美空旷】东南西北均匀包围
            angles_to_use = [0.0, np.pi / 2, np.pi, -np.pi / 2]
            for angle in angles_to_use:
                gx = self.target_position[0] + r_f * np.cos(angle)
                gy = self.target_position[1] + r_f * np.sin(angle)
                gx = np.clip(gx, safe_margin, self.map_size - safe_margin)
                gy = np.clip(gy, safe_margin, self.map_size - safe_margin)
                target_points.append(np.array([gx, gy], dtype=np.float32))

        else:
            # 【阶段一 (B)：复杂地形，提取多个独立扇区并按比例分兵】
            sectors = []
            current_sector = [open_indices[0]]
            for i in range(1, len(open_indices)):
                if open_indices[i] == open_indices[i-1] + 1:
                    current_sector.append(open_indices[i])
                else:
                    sectors.append(current_sector)
                    current_sector = [open_indices[i]]
            sectors.append(current_sector)

            # 处理 360 度首尾相连的扇区
            if len(sectors) > 1 and sectors[0][0] == 0 and sectors[-1][-1] == num_rays - 1:
                sectors[0] = sectors[-1] + sectors[0]
                sectors.pop()

            # 【新增防御】：如果环境过于破碎，缺口数量超过了无人机数量
            # 战略性放弃最小的缺口，只防守最大的前 num_agents 个缺口
            if len(sectors) > self.num_agents:
                sectors = sorted(sectors, key=len, reverse=True)[:self.num_agents]

            total_open_rays = sum(len(s) for s in sectors)
            agents_left = self.num_agents
            sectors_left = len(sectors)

            for i, sector in enumerate(sectors):
                if i == len(sectors) - 1:
                    num_for_sector = agents_left 
                else:
                    proportion = len(sector) / total_open_rays
                    num_for_sector = int(np.round(proportion * self.num_agents))
                    
                    # 【保底逻辑】：确保每个缺口至少派 1 架，且给后面留出名额
                    if num_for_sector < 1:
                        num_for_sector = 1
                    max_allowed = agents_left - (sectors_left - 1)
                    if num_for_sector > max_allowed:
                        num_for_sector = max_allowed
                
                if num_for_sector > 0:
                    start_angle = sector[0] * (2 * np.pi / num_rays)
                    end_angle = sector[-1] * (2 * np.pi / num_rays)
                    
                    if sector[0] > sector[-1]: 
                        end_angle += 2 * np.pi
                        
                    # 1. 计算扣除边界余量后的实际可用扇区起点、终点和总宽度
                    eff_start = start_angle
                    eff_end = end_angle
                    eff_width = eff_end - eff_start
                    
                    # 2. 将可用扇区等分为 num_for_sector 个子扇区 (计算步长)
                    step = eff_width / num_for_sector
                    
                    # 3. 引导点放在每个子扇区的正中间 (首个点偏移半个步长)
                    angles_to_use = [eff_start + step / 2.0 + k * step for k in range(num_for_sector)]
                        
                    for angle in angles_to_use:
                        gx = self.target_position[0] + r_f * np.cos(angle)
                        gy = self.target_position[1] + r_f * np.sin(angle)
                        gx = np.clip(gx, safe_margin, self.map_size - safe_margin)
                        gy = np.clip(gy, safe_margin, self.map_size - safe_margin)
                        target_points.append(np.array([gx, gy], dtype=np.float32))
                
                agents_left -= num_for_sector
                sectors_left -= 1

        # ---------------- 匈牙利算法最优分配 (引入惯性与防穿墙检测) ----------------
        cost_matrix = np.zeros((self.num_agents, 4), dtype=np.float32)
        
        for i, uav_pos in enumerate(self.uav_positions):
            agent_name = self.agents[i]
            uav_yaw = self.uav_yaws[i]
            prev_guide_pos = self.current_guide_points.get(agent_name, None)

            for j, tp_pos in enumerate(target_points):
                # 基础物理成本：距离 + 航向角变化
                dist = np.linalg.norm(uav_pos - tp_pos)
                vec_to_tp = tp_pos - uav_pos
                ideal_yaw = np.arctan2(vec_to_tp[1], vec_to_tp[0])
                angle_diff = abs((uav_yaw - ideal_yaw + np.pi) % (2 * np.pi) - np.pi)
                
                base_cost = dist + 10.0 * angle_diff

                # 【视线防穿墙惩罚】
                if not self._check_los(uav_pos, tp_pos):
                    base_cost += 1500.0  

                # 分配惯性惩罚
                consistency_penalty = 0.0
                if prev_guide_pos is not None:
                    shift_dist = np.linalg.norm(tp_pos - prev_guide_pos)
                    consistency_penalty = 1.5 * shift_dist
                
                cost_matrix[i, j] = base_cost + consistency_penalty

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignment_dict = {}
        for idx in range(self.num_agents):
            agent_name = self.agents[row_ind[idx]]
            new_target = target_points[col_ind[idx]]
            # [新增]: 引导点平滑逻辑 (EMA)
            if hasattr(self, 'current_guide_points') and agent_name in self.current_guide_points:
                old_target = self.current_guide_points[agent_name]
                
                # 检查是否是极端的瞬移 (比如扇区彻底重组，距离超过50米)
                # 如果是合理的变动，进行 70%历史 + 30%新目标的平滑过滤
                if np.linalg.norm(new_target - old_target) < 50.0:
                    smooth_target = 0.7 * old_target + 0.3 * new_target
                else:
                    smooth_target = new_target # 极端情况允许跳变，靠 修改2 的奖励逻辑兜底
                    
                assignment_dict[agent_name] = smooth_target
            else:
                assignment_dict[agent_name] = new_target
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

        return self._get_obs(), {"infos": {}, "individual_episode_rewards": self.individual_episode_reward}

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


        # 保持你原始的权重定义
        w_near = 2.5  
        w_safe = 1.0
        w_turn = 0.2 
        w_finish = 2.0
        w_mate = 0.5

        max_relative_speed = self.uav_max_speed + self.target_max_speed

        for i, agent in enumerate(self.agents):
            dist = cur_dists[i]
            r_turn = - (np.clip(actions_dict[agent][1], -1.0, 1.0) ** 2)   #利用差值来平滑转向惩罚，鼓励更合理的转向行为

            dist_to_guide_new = np.linalg.norm(self.uav_positions[i] - self.current_guide_points[agent])
            dist_to_guide_old = np.linalg.norm(old_uav_positions[i] - self.current_guide_points[agent])
            if hit_flags[i]:
                r_safe, r_near = -15.0, 0.0
            else:
                # 保持原始的 r_near 计算逻辑
                r_near = (dist_to_guide_old - dist_to_guide_new) / max_relative_speed

                radar_penalty = ((radar_distances[i] - self.radar_range) / self.radar_range)**2
                r_safe = - 0.5 * radar_penalty
            
            # 保持原始的抓捕与软奖励逻辑
            r_finish = 0.0
            if is_caught:
                r_finish += 80.0 
            elif dist < self.catch_radius * 3:
                soft_r = (self.catch_radius * 3 - dist) / (self.catch_radius * 2)
                r_finish += soft_r * 0.5 

            r_mate_collision = 0.0
            mate_safe_dist = 10.0  # 设定无人机互相之间的安全距离阈值（比如 10 米）
            
            for j in range(self.num_agents):
                if i != j:
                    dist_to_mate = np.linalg.norm(self.uav_positions[i] - self.uav_positions[j])
                    if dist_to_mate < mate_safe_dist:
                        # 越靠近，惩罚越大。距离为 0 时惩罚为 -1.0
                        r_mate_collision -= (mate_safe_dist - dist_to_mate) / mate_safe_dist

            w_r_near = w_near * r_near
            w_r_safe = w_safe * r_safe
            w_r_turn = w_turn * r_turn
            w_r_finish = w_finish * r_finish
            w_r_mate = w_mate * r_mate_collision

            
            rewards_dict[agent] = w_r_near + w_r_safe +  w_r_finish 
            self.episode_sub_rewards[agent]["r_near"] += w_r_near
            self.episode_sub_rewards[agent]["r_safe"] += w_r_safe
            self.episode_sub_rewards[agent]["r_turn"] += w_r_turn
            self.episode_sub_rewards[agent]["r_finish"] += w_r_finish
            self.episode_sub_rewards[agent]["r_mate"] += w_r_mate


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

            # 绘制 72 根射线
            for angle, dist, is_open_ray in self.debug_rays:
                dx, dy = np.cos(angle), np.sin(angle)
                end_x = tx + dist * dx
                end_y = ty + dist * dy

                if is_open_ray:
                    # 开阔的路线（视为安全逃逸角/分配引导扇区），用浅绿色实线
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