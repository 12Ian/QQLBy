"""为覆盖准备度估计建筑约束下的三维安全路线与动力学时间。"""

import heapq
from functools import lru_cache

import numpy as np

from .Apollonius import geometry3d as g3
from .uav_kinematics_3d import advance_translation


@lru_cache(maxsize=16)
def _rest_to_distance_profile(max_speed, max_accel, lag, jerk_limit):
    """沿直线以受限加速度加速，缓存从静止出发的距离—时间曲线。"""
    if min(max_speed, max_accel, jerk_limit) <= 0.0:
        return np.array([0.0]), np.array([0.0])
    dt = 0.2
    p = np.zeros(3, np.float32)
    v = np.zeros(3, np.float32)
    a = np.zeros(3, np.float32)
    command = np.array([max_accel, 0.0, 0.0], np.float32)
    distances, times = [0.0], [0.0]
    while distances[-1] < 2000.0:
        p, v, a = advance_translation(
            p, v, a, command, dt, lag, max_accel, jerk_limit, max_speed)
        distances.append(float(p[0]))
        times.append(len(times) * dt)
    return np.asarray(distances), np.asarray(times)


def _motion_time(distance, first_direction, velocity, env, turns):
    distances, times = _rest_to_distance_profile(
        env.uav_max_speed, env.max_accel, env.acceleration_lag, env.uav_jerk_max)
    length = max(0.0, float(distance))
    if len(distances) == 1:
        return 0.0 if length == 0.0 else float("inf")
    index = int(np.searchsorted(distances, length))
    if index >= len(distances):
        travel = times[-1] + (length - distances[-1]) / env.uav_max_speed
    elif index == 0:
        travel = 0.0
    else:
        travel = float(np.interp(length, distances[index-1:index+1],
                                 times[index-1:index+1]))
    direction = np.asarray(first_direction, np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    speed = float(np.linalg.norm(velocity))
    along = float(np.dot(velocity, direction))
    alignment = max(0.0, speed - along) / max(env.max_accel, 1e-6)
    corner_delay = turns * (env.acceleration_lag +
                            env.uav_max_speed / max(2.0 * env.max_accel, 1e-6))
    return travel + alignment + corner_delay + env.capture_hold_required


def _inflated_buildings(env):
    boxes = np.asarray(env.buildings, np.float64).copy()
    if not len(boxes):
        return boxes.reshape(0, 5)
    clearance = env.uav_radius + 0.2
    boxes[:, 0] -= clearance
    boxes[:, 1] += clearance
    boxes[:, 2] -= clearance
    boxes[:, 3] += clearance
    boxes[:, 4] += clearance
    return boxes


def _planar_detour(start, goal, boxes, map_size):
    """在直接阻挡建筑的外角构图，寻找不越界的平面绕行路径。"""
    p, q = np.asarray(start[:2], np.float64), np.asarray(goal[:2], np.float64)
    blocking = [b for b in boxes if not g3.footprint_los(p, q, [b])]
    if not blocking:
        return None
    nodes = [p, q]
    for b in blocking[:6]:
        x0, x1, y0, y1 = b[:4]
        for x in (x0 - 3.0, x1 + 3.0):
            for y in (y0 - 3.0, y1 + 3.0):
                corner = np.array([x, y], np.float64)
                if (0.0 < x < map_size and 0.0 < y < map_size and
                        g3.footprint_los(corner, corner, boxes)):
                    nodes.append(corner)
    if len(nodes) == 2:
        return None
    best = [np.inf] * len(nodes)
    parent = [-1] * len(nodes)
    best[0] = 0.0
    queue = [(0.0, 0)]
    visibility = {}
    while queue:
        cost, u = heapq.heappop(queue)
        if cost > best[u]:
            continue
        if u == 1:
            route = []
            while u >= 0:
                route.append(nodes[u])
                u = parent[u]
            return route[::-1]
        for v in range(1, len(nodes)):
            if v == u:
                continue
            edge = (min(u, v), max(u, v))
            if edge not in visibility:
                visibility[edge] = g3.footprint_los(nodes[u], nodes[v], boxes)
            if not visibility[edge]:
                continue
            new_cost = cost + float(np.linalg.norm(nodes[v] - nodes[u]))
            if new_cost < best[v]:
                best[v], parent[v] = new_cost, u
                heapq.heappush(queue, (new_cost, v))
    return None


def estimate_safe_route_time(env, start, goal, velocity, boxes):
    """比较直达、平面绕行和楼顶越过；时间为动力学校准的保守近似。"""
    start = np.asarray(start, np.float64)
    goal = np.asarray(goal, np.float64)
    if not len(boxes) or g3.segment_los(start, goal, boxes, env.z_min, env.z_max):
        return _motion_time(np.linalg.norm(goal - start), goal - start,
                            velocity, env, 0)

    candidates = []
    route = _planar_detour(start, goal, boxes, env.map_size)
    if route is not None:
        lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        xy_length = float(np.sum(lengths))
        distance = np.hypot(xy_length, float(goal[2] - start[2]))
        first_z = start[2] + (goal[2] - start[2]) * lengths[0] / max(xy_length, 1e-9)
        first = np.array([route[1][0], route[1][1], first_z]) - start
        candidates.append(_motion_time(distance, first, velocity, env, len(route) - 2))

    crossing = [b for b in boxes if not g3.footprint_los(start, goal, [b])]
    if crossing:
        altitude = max(float(start[2]), float(goal[2]),
                       max(float(b[4]) for b in crossing) + 3.0)
        if altitude <= env.z_max - env.uav_radius:
            above_start = np.array([start[0], start[1], altitude])
            above_goal = np.array([goal[0], goal[1], altitude])
            points = (start, above_start, above_goal, goal)
            if all(g3.segment_los(a, b, boxes, env.z_min, env.z_max)
                   for a, b in zip(points[:-1], points[1:])):
                distance = sum(float(np.linalg.norm(b - a))
                               for a, b in zip(points[:-1], points[1:]))
                first = (above_start - start if altitude > start[2] + 1e-6
                         else above_goal - start)
                candidates.append(_motion_time(distance, first, velocity, env, 2))
    return min(candidates, default=float("inf"))
