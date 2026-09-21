"""3D Apollonius escape solid-angle field (Euclidean criterion).

Generalizes the 2D planar escape field (``uav_pursuit_apollonius_obs_5.py``,
``_compute_apollonius_escape_field``) to the unit sphere. For each Fibonacci
direction from the target it classifies the direction as geometry-open /
dangerous / controlled by Euclidean time-reachability, and sums the solid angle
of the dangerous directions into the escape solid angle.

This increment uses straight-line pursuer distances. The visibility-constrained
variant (occluded pursuers' blockade voided) is a later increment.
"""
import numpy as np
from .geometry3d import solid_angle_per_point, footprint_los_batch, detour_distance_batch


def compute_escape_field_3d(target_pos, target_speed, uav_positions,
                            uav_speed_max, dirs, free_dist, catch_radius, r_f,
                            margin_ema=None, dangerous_prev=None,
                            probe_count=12, criterion_mode="euclidean",
                            buildings=None):
    """Classify each sphere direction by time-reachability.

    criterion_mode:
      - "euclidean": pursuer arrival time uses straight-line distance.
      - "visibility": a pursuer whose line-of-sight to a probe point is blocked
        by a building has its blockade of that point voided (arrival time -> inf).
      - "detour": a blocked pursuer keeps its claim but pays for flying around the
        obstruction. Voiding the claim outright treats a pursuer sitting one corner away as
        unable to cover a direction it could reach in a few extra seconds, and leaves that
        direction unassigned. Buildings span the full flight layer so the route is planar and
        bends at footprint corners; detour_distance searches one- and two-corner paths.

    Returns a dict with geometry_open / dangerous / controlled / best_owner /
    control_strength / escape_solid_angle / criterion_mode plus the updated
    margin_ema and dangerous_prev (EMA + hysteresis state the caller persists)."""
    target_pos = np.asarray(target_pos, np.float64)
    U = np.asarray(uav_positions, np.float64)
    vt = max(float(target_speed), 1e-6)
    vu = max(float(uav_speed_max), 1e-6)
    n = dirs.shape[0]

    geometry_open = free_dist > max(catch_radius * 1.2, r_f * 0.8)
    best_owner = np.full(n, -1, np.int32)
    best_margin = np.full(n, np.inf, np.float64)

    min_probe = max(catch_radius * 1.2, 1.0)
    max_probe_base = max(r_f * 1.8, catch_radius * 4.0)

    for k in range(n):
        if not geometry_open[k]:
            continue
        ray_limit = min(float(free_dist[k]), max_probe_base)
        if ray_limit <= min_probe:
            continue
        radii = np.linspace(min_probe, ray_limit, probe_count)
        pts = target_pos[None, :] + radii[:, None] * dirs[k][None, :]   # (P,3)
        t_target = radii / vt
        # (U,P) pursuer arrival times via straight-line distance
        dd = np.linalg.norm(pts[None, :, :] - U[:, None, :], axis=2) / vu
        if criterion_mode == "visibility" and buildings is not None and len(buildings) > 0:
            for ui in range(U.shape[0]):
                occluded = ~footprint_los_batch(U[ui], pts, buildings)
                if occluded.any():
                    dd[ui, occluded] = np.inf   # occluded pursuer's blockade voided
        elif criterion_mode == "detour" and buildings is not None and len(buildings) > 0:
            # Straight-line distance is a lower bound on the detour distance, so a pursuer that
            # already loses the race on the straight line loses it after routing around a
            # building too -- those entries need no detour computation. Only where the euclidean
            # comparison says the pursuer arrives first can the detour flip the verdict, so the
            # exact distance is spent solely on the entries that can change the answer. On the
            # 26-building layout this is what makes the criterion affordable: an unpruned pass
            # costs ~0.63 s per step, which is ~110 hours for a 5M-step run.
            for ui in range(U.shape[0]):
                occluded = ~footprint_los_batch(U[ui], pts, buildings)
                decisive = occluded & (dd[ui] < t_target)
                idx = np.where(decisive)[0]
                if len(idx):
                    dd[ui, idx] = detour_distance_batch(U[ui], pts[idx], buildings) / vu
        margins = dd - t_target[None, :]                # (U,P)
        per_uav_min = margins.min(axis=1)               # (U,)
        owner = int(np.argmin(per_uav_min))
        m = float(per_uav_min[owner])
        if m < best_margin[k]:
            best_margin[k] = m
            best_owner[k] = owner

    finite = np.where(np.isfinite(best_margin), best_margin, 10.0)
    if margin_ema is None or len(margin_ema) != n:
        ema = finite.copy()
    else:
        ema = 0.65 * np.asarray(margin_ema, np.float64) + 0.35 * finite
    if dangerous_prev is None or len(dangerous_prev) != n:
        prev = np.zeros(n, bool)
    else:
        prev = np.asarray(dangerous_prev, bool)

    # Hysteresis: keep a dangerous ray until clearly safe; require a stronger
    # margin to newly flag a safe ray as dangerous.
    time_dangerous = np.where(prev, ema > -0.10, ema > 0.20)
    dangerous = geometry_open & time_dangerous
    controlled = geometry_open & ~dangerous
    control_strength = 1.0 / (1.0 + np.exp(np.clip(1.25 * ema, -60.0, 60.0)))

    omega = solid_angle_per_point(n)
    escape_solid_angle = float(np.sum(dangerous) * omega)

    return {
        "geometry_open": geometry_open,
        "dangerous": dangerous,
        "controlled": controlled,
        "best_owner": best_owner,
        "control_strength": control_strength.astype(np.float32),
        "escape_solid_angle": escape_solid_angle,
        "criterion_mode": criterion_mode,
        "margin_ema": ema.astype(np.float32),
        "dangerous_prev": dangerous.copy(),
    }
