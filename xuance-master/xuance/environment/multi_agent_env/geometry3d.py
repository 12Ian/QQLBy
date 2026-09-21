"""3D geometry helpers for the obstacle-aware UAV pursuit environment.

All obstacles are full-height vertical prisms: their footprint is an axis-aligned
box ``[xmin, xmax, ymin, ymax]`` extruded across the whole flight layer, so a 3D
line-of-sight reduces to the 2D footprint check on the (x, y) projection.
"""
import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` approximately-uniform unit vectors on S^2, shape (n, 3)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)            # polar angle in [0, pi]
    golden = np.pi * (1.0 + 5 ** 0.5)             # golden angle
    theta = golden * i                            # azimuth
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def solid_angle_per_point(n: int) -> float:
    """Equal-weight solid angle (steradians) carried by each Fibonacci point."""
    return 4.0 * np.pi / float(n)


def _slab(o, comp, lo, hi):
    """Per-direction entry/exit parameters for one axis slab [lo, hi]."""
    eps = 1e-9
    comp = np.where(np.abs(comp) < eps, eps, comp)
    t1 = (lo - o) / comp
    t2 = (hi - o) / comp
    return np.minimum(t1, t2), np.maximum(t1, t2)


def ray_free_distance(origin, dirs, map_size, z_min, z_max, buildings):
    """Distance from ``origin`` along each unit direction until it leaves the
    world box ``[0,map_size]^2 x [z_min,z_max]`` or first enters a building
    footprint. ``buildings`` is an (M,4) array of full-height xy AABBs
    ``[xmin,xmax,ymin,ymax]``. Returns (K,) float32 distances."""
    origin = np.asarray(origin, np.float64)
    d = np.asarray(dirs, np.float64)

    # world-box exit = nearest slab exit across x, y, z
    _, ex = _slab(origin[0], d[:, 0], 0.0, map_size)
    _, ey = _slab(origin[1], d[:, 1], 0.0, map_size)
    _, ez = _slab(origin[2], d[:, 2], z_min, z_max)
    best = np.maximum(np.minimum(np.minimum(ex, ey), ez), 0.0)

    if buildings is not None and len(buildings) > 0:
        b = np.asarray(buildings, np.float64)
        for m in range(b.shape[0]):
            en_x, ex_x = _slab(origin[0], d[:, 0], b[m, 0], b[m, 1])
            en_y, ex_y = _slab(origin[1], d[:, 1], b[m, 2], b[m, 3])
            t_enter = np.maximum(en_x, en_y)
            t_exit = np.minimum(ex_x, ex_y)
            hit = (t_exit >= 0) & (t_enter <= t_exit) & (t_enter > 0)
            best = np.where(hit, np.minimum(best, t_enter), best)
    return best.astype(np.float32)


def footprint_los(p1, p2, buildings):
    """True if the xy projection of segment p1->p2 hits no building footprint.
    Buildings are full-height prisms, so this is exact 3D LOS for them."""
    if buildings is None or len(buildings) == 0:
        return True
    p1 = np.asarray(p1, np.float64)
    p2 = np.asarray(p2, np.float64)
    D = p2[:2] - p1[:2]
    min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
    min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
    for b in buildings:
        xmin, xmax, ymin, ymax = b
        if max_x < xmin or min_x > xmax or max_y < ymin or min_y > ymax:
            continue
        t_enter, t_exit = 0.0, 1.0
        if D[0] == 0:
            if p1[0] < xmin or p1[0] > xmax:
                continue
        else:
            tx1 = (xmin - p1[0]) / D[0]
            tx2 = (xmax - p1[0]) / D[0]
            t_enter = max(t_enter, min(tx1, tx2))
            t_exit = min(t_exit, max(tx1, tx2))
        if D[1] == 0:
            if p1[1] < ymin or p1[1] > ymax:
                continue
        else:
            ty1 = (ymin - p1[1]) / D[1]
            ty2 = (ymax - p1[1]) / D[1]
            t_enter = max(t_enter, min(ty1, ty2))
            t_exit = min(t_exit, max(ty1, ty2))
        if t_enter <= t_exit and t_exit >= 0 and t_enter <= 1.0:
            return False
    return True


def radar_body_dirs():
    """16 unit dirs in the body frame: 8 horizontal ring + 4 up(+30 deg) +
    4 down(-30 deg). Returned shape (16, 3)."""
    out = []
    for k in range(8):
        a = 2 * np.pi * k / 8
        out.append([np.cos(a), np.sin(a), 0.0])
    c = np.cos(np.pi / 6)
    s = np.sin(np.pi / 6)
    for k in range(4):
        a = 2 * np.pi * k / 4
        out.append([c * np.cos(a), c * np.sin(a),  s])
        out.append([c * np.cos(a), c * np.sin(a), -s])
    return np.array(out[:16], np.float32)


def _rot_yaw_pitch(dirs, yaw, pitch):
    """Rotate body-frame dirs into world frame by yaw (about z) then pitch
    (about y)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], np.float64)
    return dirs @ (Rz @ Ry).T


def radar_distances(origin, yaw, pitch, radar_range, map_size, z_min, z_max,
                    buildings):
    """Normalized [0,1] obstacle distances along the 16 body-frame radar rays
    rotated by (yaw, pitch). 1.0 means nothing within ``radar_range``."""
    dirs = _rot_yaw_pitch(radar_body_dirs().astype(np.float64), yaw, pitch)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    d = ray_free_distance(origin, dirs, map_size, z_min, z_max, buildings)
    return np.clip(d / radar_range, 0.0, 1.0).astype(np.float32)


def footprint_los_batch(origin, points, buildings):
    """Vectorized footprint LOS from one origin to K points (xy projection).
    Returns bool[K]; True = unobstructed. Buildings are full-height xy AABBs.
    Matches the scalar ``footprint_los`` element-wise."""
    pts = np.asarray(points, np.float64)
    K = pts.shape[0]
    vis = np.ones(K, dtype=bool)
    if buildings is None or len(buildings) == 0:
        return vis
    o = np.asarray(origin, np.float64)
    dx = pts[:, 0] - o[0]
    dy = pts[:, 1] - o[1]
    eps = 1e-12
    dxs = np.where(np.abs(dx) < eps, eps, dx)
    dys = np.where(np.abs(dy) < eps, eps, dy)
    x_par = np.abs(dx) < eps
    y_par = np.abs(dy) < eps
    for b in buildings:
        xmin, xmax, ymin, ymax = b
        t_enter = np.zeros(K)
        t_exit = np.ones(K)
        # x slab
        tx1 = (xmin - o[0]) / dxs
        tx2 = (xmax - o[0]) / dxs
        tlo = np.minimum(tx1, tx2)
        thi = np.maximum(tx1, tx2)
        in_x = (o[0] >= xmin) and (o[0] <= xmax)
        t_enter = np.where(x_par, (0.0 if in_x else 1.0), np.maximum(t_enter, tlo))
        t_exit = np.where(x_par, (1.0 if in_x else 0.0), np.minimum(t_exit, thi))
        # y slab
        ty1 = (ymin - o[1]) / dys
        ty2 = (ymax - o[1]) / dys
        tlo = np.minimum(ty1, ty2)
        thi = np.maximum(ty1, ty2)
        in_y = (o[1] >= ymin) and (o[1] <= ymax)
        t_enter = np.where(y_par, t_enter if in_y else 1.0, np.maximum(t_enter, tlo))
        t_exit = np.where(y_par, t_exit if in_y else 0.0, np.minimum(t_exit, thi))
        hit = (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)
        vis &= ~hit
    return vis


# --- detour-aware travel distance -------------------------------------------------
#
# The visibility criterion was binary: if a building blocked a pursuer's line of sight to a
# probe point, that pursuer's arrival time went to infinity and its claim on the direction was
# dropped entirely. That is far too pessimistic. A pursuer sitting just behind a corner is not
# unable to cover a direction -- it needs a few extra seconds to come around, and whether that
# is enough to still beat the evader is exactly the comparison the escape criterion is supposed
# to make. Discarding it also leaves nobody assigned to that direction, so the team stops
# covering ground it could actually cover.
#
# Buildings are full-height prisms spanning the whole flight layer, so no route goes over them
# and the detour is planar. Paths bend only at footprint corners, so routing through the corners
# of whatever blocks the straight line recovers the true shortest path in the common cases: one
# leg around a single building (two corners), or a corner of each of two buildings. A wider
# search is possible but each extra hop costs an order of magnitude more queries, and the escape
# field already evaluates 200 directions x 12 probes x N pursuers per step.

def _corners(b, inflate):
    xmin, xmax, ymin, ymax = b
    return np.array([[xmin - inflate, ymin - inflate], [xmax + inflate, ymin - inflate],
                     [xmax + inflate, ymax + inflate], [xmin - inflate, ymax + inflate]], np.float64)


def detour_distance(p, q, buildings, inflate=3.0, max_corners=16):
    """Planar travel distance from p to q, routed around building footprints.

    Returns the straight-line distance when the line of sight is clear. Otherwise searches
    one- and two-corner detours through the corners of the blocking buildings and returns the
    shortest path found, or inf when no such route exists.

    inflate must stay above zero: a waypoint placed exactly on a footprint corner grazes the
    building, the line-of-sight test counts that as blocked, and every route is rejected. It
    doubles as the vehicle-radius clearance, so the default is a few metres.
    """
    p = np.asarray(p, np.float64)[:2]
    q = np.asarray(q, np.float64)[:2]
    if buildings is None or len(buildings) == 0:
        return float(np.linalg.norm(q - p))
    if footprint_los(p, q, buildings):
        return float(np.linalg.norm(q - p))

    # only the buildings that actually block the direct line can be the ones to route around
    blocking = [b for b in buildings if not footprint_los(p, q, [b])]
    if not blocking:
        return float(np.linalg.norm(q - p))
    cand = np.concatenate([_corners(b, inflate) for b in blocking])[:max_corners]

    best = np.inf
    reach_p = [c for c in cand if footprint_los(p, c, buildings)]
    reach_q = [c for c in cand if footprint_los(c, q, buildings)]
    for c in reach_p:                                   # one corner: p -> c -> q
        if any(np.allclose(c, d) for d in reach_q):
            best = min(best, float(np.linalg.norm(c - p) + np.linalg.norm(q - c)))
    if np.isfinite(best):
        return best
    for c1 in reach_p:                                  # two corners: p -> c1 -> c2 -> q
        for c2 in reach_q:
            if np.allclose(c1, c2) or not footprint_los(c1, c2, buildings):
                continue
            best = min(best, float(np.linalg.norm(c1 - p) + np.linalg.norm(c2 - c1)
                                   + np.linalg.norm(q - c2)))
    return best


def detour_distance_batch(origin, points, buildings, inflate=3.0):
    """detour_distance from one origin to K points. Straight-line where visible."""
    pts = np.asarray(points, np.float64)
    out = np.linalg.norm(pts[:, :2] - np.asarray(origin, np.float64)[None, :2], axis=1)
    if buildings is None or len(buildings) == 0:
        return out
    vis = footprint_los_batch(origin, pts, buildings)
    for k in np.where(~vis)[0]:
        out[k] = detour_distance(origin, pts[k], buildings, inflate)
    return out


# A precomputed corner-visibility graph was tried here and removed: it walks all 104 corners of
# the layout, while the direct search only looks at the corners of the few buildings that
# actually block the line, so it measured 3.7x SLOWER (660 us against 178 us). The useful
# reduction is not a faster detour query but issuing far fewer of them -- see the pruning in
# compute_escape_field_3d, which relies on straight-line distance being a lower bound on the
# detour distance.
