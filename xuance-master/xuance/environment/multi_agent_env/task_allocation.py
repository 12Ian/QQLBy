"""Balanced dynamic allocation of N pursuers to M (alive) targets for multi-target
cooperative pursuit. Each alive target receives ~N/M pursuers (coverage), assignment
minimises total pursuer->target distance, and a stickiness bonus discourages thrashing
by favouring each pursuer's previous (still-alive) target."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_pursuers_to_targets(P, T, prev_assign, alive, sticky=30.0, base_cost=None):
    """
    Args:
        P (array N,3): pursuer positions.
        T (array M,3): target positions.
        prev_assign (array N,): previous assignment (target index per pursuer, -1 if none).
        alive (array M,): bool mask of still-active targets.
        sticky (float): cost reduction for keeping a pursuer on its previous alive target.
    Returns:
        assign (array N,): index (into the original T) of the assigned target per pursuer.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    N = P.shape[0]
    alive = np.asarray(alive, dtype=bool)
    alive_idx = np.where(alive)[0]
    M = len(alive_idx)
    if M == 0:
        return np.full(N, -1, dtype=int)
    if M == 1:
        return np.full(N, int(alive_idx[0]), dtype=int)

    # balanced capacities over alive targets: sum(caps) == N
    base, rem = divmod(N, M)
    caps = [base + (1 if k < rem else 0) for k in range(M)]

    # base cost = pursuer->target distance, unless an alternative criterion is supplied.
    # Distance only asks who is nearest; base_cost lets the caller pass a capture-ability
    # criterion (see apollonius_coverage_cost) so a pursuer is sent where it can actually
    # seal escape directions rather than merely where it is close.
    if base_cost is not None:
        dist = np.asarray(base_cost, dtype=float).copy()        # (N, M_alive)
        # rescale so the sticky bonus keeps its intended magnitude against a different unit
        rng = float(np.ptp(dist))
        if rng > 1e-9:
            dist = dist / rng * 200.0
    else:
        dist = np.linalg.norm(P[:, None, :] - T[None, alive_idx, :], axis=2)  # (N, M)
    prev = np.asarray(prev_assign, dtype=int)
    for a, tj in enumerate(alive_idx):
        dist[prev == tj, a] -= sticky

    # expand each target into `caps` identical slots, then solve a 1-1 assignment
    slot_target = np.repeat(np.arange(M), caps)            # (N,) slot -> alive-column
    cost = dist[:, slot_target]                            # (N, N)
    rows, cols = linear_sum_assignment(cost)
    assign = np.empty(N, dtype=int)
    assign[rows] = alive_idx[slot_target[cols]]
    return assign

def apollonius_coverage_cost(P, T, target_speeds, uav_speed_max, dirs, free_dists,
                             catch_radius, r_f_list, alive):
    """Per (pursuer, target) capture-ability, for Apollonius-based allocation.

    The distance-based assignment is myopic: it asks who is NEAREST, not who can actually
    seal the target's escape. A pursuer close to a target but sitting on the wrong side
    contributes nothing to shrinking the escape solid angle, yet distance sends it there.

    This scores each pair by how many of the target's DANGEROUS escape directions that one
    pursuer could dominate on its own -- i.e. the directions where it reaches the Apollonius
    probe point no later than the evader. Returns a cost (negated coverage) so the same
    Hungarian solver can consume it.

    Returns (N, M_alive) cost array, lower is better.
    """
    P = np.asarray(P, float)
    T = np.asarray(T, float)
    alive_idx = np.where(np.asarray(alive, dtype=bool))[0]
    N, Ma = P.shape[0], len(alive_idx)
    cost = np.zeros((N, Ma), dtype=float)
    if Ma == 0:
        return cost
    probe_count = 8
    for a, m in enumerate(alive_idx):
        tpos = T[m]
        vt = max(float(target_speeds[m]), 1e-6)
        vu = max(float(uav_speed_max), 1e-6)
        free = np.asarray(free_dists[m], float)
        r_f = float(r_f_list[m])
        geo_open = free > max(catch_radius * 1.2, r_f * 0.8)
        open_k = np.where(geo_open)[0]
        if len(open_k) == 0:
            continue
        min_probe = max(catch_radius * 1.2, 1.0)
        max_probe = max(r_f * 1.8, catch_radius * 4.0)
        # coverage[i] = number of open directions pursuer i can reach before the evader does
        cover = np.zeros(N, dtype=float)
        for k in open_k:
            lim = min(float(free[k]), max_probe)
            if lim <= min_probe:
                continue
            radii = np.linspace(min_probe, lim, probe_count)
            pts = tpos[None, :] + radii[:, None] * dirs[k][None, :]      # (P,3)
            t_evader = radii / vt                                        # (P,)
            t_pursuer = np.linalg.norm(pts[None, :, :] - P[:, None, :], axis=2) / vu
            # a pursuer dominates this direction if it beats the evader to ANY probe point
            cover += np.any(t_pursuer <= t_evader[None, :], axis=1).astype(float)
        cost[:, a] = -cover
    return cost
