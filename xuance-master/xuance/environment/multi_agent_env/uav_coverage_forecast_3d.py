"""用现有势场和受限动力学构造五秒目标预测与可达—保持覆盖。"""

import numpy as np

from .Apollonius import geometry3d as g3
from .uav_kinematics_3d import advance_translation, limit_norm
from .uav_coverage_route_3d import _inflated_buildings, estimate_safe_route_time


def _plane_basis(normal):
    """根据预测速度构造连续可用的截面基。"""
    normal = np.asarray(normal, np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    reference = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(float(np.dot(normal, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], np.float32)
    first = np.cross(reference, normal)
    first /= max(float(np.linalg.norm(first)), 1e-6)
    second = np.cross(normal, first)
    return first, second


def forecast_target(env):
    """滚动复制目标、追捕机状态；不改变真实环境，也不读取未来真值。

    九条确定性假设对应名义势场和截面内八个机动方向。横向偏置仍受目标的
    加速度、jerk、速度和障碍约束，因此只是设计场景，非校准概率。
    """
    horizon = max(1, int(round(env.future_prediction_steps * env.decision_dt / env.physics_dt)))
    dt = env.future_prediction_steps * env.decision_dt / horizon
    hypotheses = ((0.0, 0.0), (0.6, 0.0), (-0.6, 0.0),
                  (0.0, 0.6), (0.0, -0.6),
                  (0.42, 0.42), (0.42, -0.42),
                  (-0.42, 0.42), (-0.42, -0.42))
    paths = []
    velocities = []
    p0 = env.target_position.copy()
    v0 = env.target_velocity.copy()
    a0 = env.target_acceleration.copy()
    pursuer_pos0 = env.uav_positions.copy()
    pursuer_vel0 = env.uav_velocities.copy()
    pursuer_acc0 = env.uav_accelerations.copy()
    axis = v0 if np.linalg.norm(v0) > 1e-5 else p0 - np.mean(pursuer_pos0, axis=0)
    if np.linalg.norm(axis) < 1e-5:
        axis = np.array([1.0, 0.0, 0.0], np.float32)
    lateral1, lateral2 = _plane_basis(axis)
    # 未知追捕策略采用零加速度指令外推一次，供所有目标机动场景共用。
    pursuer_paths = [pursuer_pos0.copy()]
    for _ in range(horizon):
        for i in range(env.num_agents):
            pp, vv, aa = advance_translation(
                pursuer_pos0[i], pursuer_vel0[i], pursuer_acc0[i],
                np.zeros(3), dt, env.acceleration_lag, env.max_accel,
                env.uav_jerk_max, env.uav_max_speed)
            unclipped = pp
            pp, _ = env._move_velocity_with_clip_3d(
                pursuer_pos0[i], pp - pursuer_pos0[i], env.uav_radius)
            if np.any(np.abs(pp - unclipped) > 1e-6):
                vv = (pp - pursuer_pos0[i]) / dt
            pursuer_pos0[i], pursuer_vel0[i], pursuer_acc0[i] = pp, vv, aa
        pursuer_paths.append(pursuer_pos0.copy())
    for gain1, gain2 in hypotheses:
        p, v, a = p0.copy(), v0.copy(), a0.copy()
        track = [p.copy()]
        velocity_track = [v.copy()]
        for k in range(horizon):
            command = env._evader_control(p, v, pursuer_paths[k], deterministic=True)
            command = limit_norm(command + env.target_accel *
                                 (gain1 * lateral1 + gain2 * lateral2),
                                 env.target_accel)
            np_, nv, na = advance_translation(
                p, v, a, command, dt, env.acceleration_lag,
                env.target_accel, env.target_jerk_max, env.target_max_speed)
            clipped, _ = env._move_velocity_with_clip_3d(p, np_ - p, env.target_radius)
            if np.any(np.abs(clipped - np_) > 1e-6):
                nv = (clipped - p) / dt
            p, v, a = clipped, nv.astype(np.float32), na
            track.append(p.copy())
            velocity_track.append(v.copy())
        paths.append(np.asarray(track, np.float32))
        velocities.append(np.asarray(velocity_track, np.float32))
    normal = velocities[0][-1]
    if np.linalg.norm(normal) < 1e-5:
        normal = v0 if np.linalg.norm(v0) > 1e-5 else axis
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    e1, e2 = _plane_basis(normal)
    weights = np.array([0.25] + [0.75 / 8.0] * 8, np.float32)
    section_points = (np.asarray(paths)[:, -1] - paths[0][-1]) @ np.stack([e1, e2], axis=1)
    mean = weights @ section_points
    centered = section_points - mean
    covariance = (centered * weights[:, None]).T @ centered
    return {"paths": np.asarray(paths), "velocities": np.asarray(velocities),
            "weights": weights, "section_points": section_points,
            "section_mean": mean, "section_covariance": covariance,
            "plane_origin": paths[0][-1].copy(), "plane_normal": normal,
            "plane_basis": np.stack([e1, e2], axis=1), "dt": dt}


def _candidate_track(env, index, endpoint, target_terminal_velocity, mode, steps, dt):
    """在完整三维动力学内滚动一种追捕控制原语；碰建筑的轨迹作废。"""
    p = env.uav_positions[index].copy()
    v = env.uav_velocities[index].copy()
    a = env.uav_accelerations[index].copy()
    track = [p.copy()]
    for k in range(steps):
        remaining = max((steps - k) * dt, dt)
        goal = endpoint + mode * target_terminal_velocity
        desired_v = limit_norm((goal - p) / remaining, env.uav_max_speed)
        distances = env._building_distances_3d(p)
        if len(distances) and float(np.min(distances)) < 25.0:
            building = env.buildings[int(np.argmin(distances))]
            if p[2] < building[4] + 15.0:
                closest = np.array([np.clip(p[0], building[0], building[1]),
                                    np.clip(p[1], building[2], building[3])])
                outward = p[:2] - closest
                if np.linalg.norm(outward) < 1e-5:
                    outward = p[:2] - np.array([(building[0] + building[1]) / 2,
                                                (building[2] + building[3]) / 2])
                outward /= max(float(np.linalg.norm(outward)), 1e-5)
                influence = (25.0 - float(np.min(distances))) / 25.0
                desired_v[:2] += influence * env.uav_max_speed * outward
                if mode != 0.0:
                    tangent = np.array([-outward[1], outward[0]]) * np.sign(mode)
                    desired_v[:2] += influence * env.uav_max_speed * tangent
                else:
                    desired_v[2] += influence * env.uav_max_speed
                desired_v = limit_norm(desired_v, env.uav_max_speed)
        command = limit_norm((desired_v - v) / env.acceleration_lag,
                             env.max_accel)
        np_, nv, na = advance_translation(
            p, v, a, command, dt, env.acceleration_lag,
            env.max_accel, env.uav_jerk_max, env.uav_max_speed)
        # 候选轨迹必须全程无碰撞；不能利用真实环境中的贴墙滑动。
        if env._prospective_collision_sources_velocity(p, np_ - p, env.uav_radius):
            return None
        p, v, a = np_, nv, na
        track.append(p.copy())
    return np.asarray(track)


def reach_hold_coverage(env, forecast):
    """每架机选择一条可飞轨迹，评价其对全部目标情景的联合覆盖。"""
    paths = forecast["paths"]
    dt = float(forecast["dt"])
    steps = paths.shape[1] - 1
    horizon = steps * dt
    distance = np.linalg.norm(env.uav_positions[:, None, :] - paths[:, -1][None, :, :], axis=-1)
    optimistic = env.uav_max_speed * horizon + env.catch_radius
    if float(np.min(distance)) > optimistic:
        return 0.0
    hold_steps = max(1, int(np.ceil(env.capture_hold_required / dt)))
    target_hold_paths = paths[:, -hold_steps-1:]
    candidates = [[] for _ in range(env.num_agents)]
    for i in range(env.num_agents):
        for j, path in enumerate(paths):
            if distance[i, j] > optimistic:
                continue
            endpoint = path[-1]
            terminal_v = forecast["velocities"][j, -1]
            for lead in (0.0, -0.5, 0.5):
                track = _candidate_track(env, i, endpoint, terminal_v,
                                         lead, steps, dt)
                if track is None:
                    continue
                # 同一追捕轨迹必须同时用于所有目标情景，不能逐情景重新选轨迹。
                errors = np.linalg.norm(
                    track[None, -hold_steps-1:] - target_hold_paths, axis=2)
                max_errors = np.max(errors, axis=1)
                scores = 1.0 / (1.0 + np.exp(np.clip(
                    (max_errors - env.catch_radius) / 3.0, -40, 40)))
                candidates[i].append(scores.astype(np.float32))
    selected = np.zeros((env.num_agents, len(paths)), np.float32)
    for i, options in enumerate(candidates):
        if not options:
            continue
        other_survival = np.prod(1.0 - selected, axis=0)
        selected[i] = max(options, key=lambda scores: float(np.dot(
            forecast["weights"], other_survival * scores)))
    union = 1.0 - np.prod(1.0 - selected, axis=0)
    return float(np.dot(forecast["weights"], union))


def coverage_preparation(env, forecast):
    """安全路线到达时间形成的联合覆盖准备度，不作为实际捕获概率。"""
    endpoints = forecast["paths"][:, -1]
    boxes = _inflated_buildings(env)
    goals = []
    for endpoint in endpoints:
        if g3.segment_los(endpoint, endpoint, boxes, env.z_min, env.z_max):
            goals.append([endpoint])
            continue
        # 目标贴近建筑时，允许追捕机到达其捕获邻域中的空域点。
        alternatives = []
        for distance in (2.0, 5.0, 10.0):
            for axis in np.eye(3):
                for sign in (-1.0, 1.0):
                    candidate = endpoint + sign * distance * axis
                    if (env.uav_radius < candidate[0] < env.map_size - env.uav_radius and
                            env.uav_radius < candidate[1] < env.map_size - env.uav_radius and
                            env.z_min < candidate[2] < env.z_max and
                            g3.segment_los(candidate, candidate, boxes,
                                           env.z_min, env.z_max)):
                        alternatives.append(candidate)
        goals.append(alternatives)
    ability = np.zeros((env.num_agents, len(endpoints)), np.float32)
    min_time = float("inf")
    for i in range(env.num_agents):
        for j, endpoint_options in enumerate(goals):
            arrival = min((estimate_safe_route_time(
                env, env.uav_positions[i], goal, env.uav_velocities[i], boxes)
                           for goal in endpoint_options), default=float("inf"))
            min_time = min(min_time, arrival)
            if np.isfinite(arrival):
                ability[i, j] = np.exp(-arrival / env.preparation_time_scale)
    union = 1.0 - np.prod(1.0 - ability, axis=0)
    return float(np.dot(forecast["weights"], union)), min_time
