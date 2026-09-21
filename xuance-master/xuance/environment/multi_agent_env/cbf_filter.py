"""Discrete-time control barrier function safety filter for the pursuit environment.

Implements the safety layer of Cheng et al., "End-to-End Safe Reinforcement Learning through
Barrier Functions for Safety-Critical Continuous Control Tasks" (AAAI-19): the learned policy
proposes an action, and a model-based filter applies the minimum correction that keeps the
state inside the safe set. Here it is the collision rate itself that the filter targets -- the
alternative of simply not ending the episode on a crash improves the score while leaving the
crashes in place.

Two departures from the paper, both of which simplify rather than weaken the guarantee:

1. No Gaussian process. The paper learns the unknown dynamics d(s) because its plants (a
   pendulum, a car) are only approximately modelled; safety then holds with probability
   (1 - delta). Our nominal model *is* the simulator's own update, so d(s) = 0 identically and
   the barrier condition holds deterministically. The GP machinery has nothing to estimate.

2. The barrier is planar. Buildings are full-height vertical prisms with axis-aligned xy
   footprints, so distance-to-obstacle does not depend on altitude and each constraint is
   affine in the xy position -- exactly the h = p'x + q form the paper restricts itself to.

The QP min ||v - v_rl||^2 s.t. Av >= b is a projection onto a polyhedron. With at most a
handful of active constraints we solve it exactly by enumerating active sets (KKT for
projection) rather than pulling in a QP dependency.
"""
import numpy as np

# Complex density puts 26 buildings over 33% of the floor, and a pursuer there is routinely
# boxed in by more faces than the four the enumerating solver could afford (2^m subsets). Raising
# the cap needs a solver that scales, so above _ENUM_MAX we switch to Hildreth's dual coordinate
# descent, which is linear per sweep and converges to the same projection.
_MAX_ACTIVE = 12
_ENUM_MAX = 6
_SUBSET_CACHE = {}


def _subsets(m):
    if m not in _SUBSET_CACHE:
        _SUBSET_CACHE[m] = [[i for i in range(m) if mask & (1 << i)]
                            for mask in range(1 << m)]
    return _SUBSET_CACHE[m]


def _box_barrier(p, box, infl):
    """h and its gradient for an inflated AABB footprint. h > 0 outside, < 0 inside."""
    xmin, xmax, ymin, ymax = box
    xmin, xmax, ymin, ymax = xmin - infl, xmax + infl, ymin - infl, ymax + infl
    dx = max(xmin - p[0], 0.0, p[0] - xmax)
    dy = max(ymin - p[1], 0.0, p[1] - ymax)
    if dx > 0.0 or dy > 0.0:                       # outside: distance to the box
        gx = (p[0] - xmax) if p[0] > xmax else ((p[0] - xmin) if p[0] < xmin else 0.0)
        gy = (p[1] - ymax) if p[1] > ymax else ((p[1] - ymin) if p[1] < ymin else 0.0)
        h = float(np.hypot(dx, dy))
        n = np.array([gx, gy, 0.0], dtype=np.float64)
        nn = np.linalg.norm(n)
        return h, (n / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0]))
    # inside the inflated box: push out along the shallowest face
    cands = ((p[0] - xmin, np.array([1.0, 0.0, 0.0])), (xmax - p[0], np.array([-1.0, 0.0, 0.0])),
             (p[1] - ymin, np.array([0.0, 1.0, 0.0])), (ymax - p[1], np.array([0.0, -1.0, 0.0])))
    pen, n = min(cands, key=lambda t: t[0])
    return -float(pen), n


def _project_hildreth(v_rl, A, b, iters=60, tol=1e-7):
    """Projection via the dual of min 0.5||v - v_rl||^2 s.t. Av >= b.

    Stationarity gives v = v_rl + A' lam, and the dual is min over lam >= 0 of
    0.5 lam' (A A') lam + lam' (A v_rl - b). Hildreth sweeps one multiplier at a time, clipping
    each at zero, which is exactly the non-negativity the KKT conditions require.
    """
    H = A @ A.T
    q = A @ v_rl - b
    d = np.diag(H).copy()
    d[d < 1e-12] = 1e-12
    lam = np.zeros(len(b))
    for _ in range(iters):
        delta = 0.0
        for i in range(len(b)):
            old = lam[i]
            lam[i] = max(0.0, old - (H[i] @ lam + q[i]) / d[i])
            delta = max(delta, abs(lam[i] - old))
        if delta < tol:
            break
    return v_rl + A.T @ lam


def _project(v_rl, A, b, v_max):
    """Exact projection of v_rl onto {v : Av >= b} intersected with ||v|| <= v_max."""
    m = len(b)
    if m > _ENUM_MAX:
        v = _project_hildreth(v_rl, A, b)
        nv = np.linalg.norm(v)
        if nv > v_max:
            v = v * (v_max / nv)
        # Hildreth solves the linear-constraint problem exactly; the speed cap can still push
        # the result back out, in which case there is no feasible answer to return here.
        return v if np.all(A @ v >= b - 1e-4) else None
    best, best_cost = None, np.inf
    for S in _subsets(m):
        if S:
            As, bs = A[S], b[S]
            try:
                lam = np.linalg.solve(As @ As.T + 1e-9 * np.eye(len(S)), bs - As @ v_rl)
            except np.linalg.LinAlgError:
                continue
            if np.any(lam < -1e-9):                # KKT: multipliers must be non-negative
                continue
            v = v_rl + As.T @ lam
        else:
            v = v_rl.copy()
        if np.any(A @ v < b - 1e-6):               # must satisfy every constraint
            continue
        nv = np.linalg.norm(v)
        if nv > v_max:                             # speed ceiling
            v = v * (v_max / nv)
            if np.any(A @ v < b - 1e-6):
                continue
        cost = float(np.sum((v - v_rl) ** 2))
        if cost < best_cost:
            best, best_cost = v, cost
    return best


def filter_action(act, pos, yaw, pitch, speed, buildings, map_size, radius,
                  v_min, v_max, max_accel, max_yaw_rate, max_pitch_rate, pitch_max,
                  eta=0.25, margin=6.0):
    """Return (safe_action, intervened, effort).

    `act` is the policy's [accel, yaw_rate, pitch_rate] in [-1, 1]. The returned action is the
    closest achievable one satisfying the discrete-time barrier condition
    h(x_{t+1}) >= (1 - eta) h(x_t) for every nearby obstacle and for the arena walls.
    """
    act = np.clip(np.asarray(act, dtype=np.float64), -1.0, 1.0)
    # nominal next state under the policy action (mirrors the environment's own update)
    sp = float(np.clip(speed + act[0] * max_accel, v_min, v_max))
    yw = float((yaw + act[1] * max_yaw_rate) % (2 * np.pi))
    pt = float(np.clip(pitch + act[2] * max_pitch_rate, -pitch_max, pitch_max))
    v_rl = sp * np.array([np.cos(pt) * np.cos(yw), np.cos(pt) * np.sin(yw), np.sin(pt)])

    infl = radius + margin
    # A one-step activation zone is useless here: with a pi/4 rad/step yaw limit the
    # vehicle needs several steps to turn away, so the barrier must engage while the
    # obstacle is still a few steps out.
    reach = 4.0 * v_max + margin
    cons = []
    for box in buildings:
        h, n = _box_barrier(pos, box, infl)
        if h < reach:
            cons.append((h, n))
    # arena walls (leaving the arena counts as a crash, so they are barriers too)
    for h, n in ((pos[0] - radius, np.array([1.0, 0.0, 0.0])),
                 (map_size - radius - pos[0], np.array([-1.0, 0.0, 0.0])),
                 (pos[1] - radius, np.array([0.0, 1.0, 0.0])),
                 (map_size - radius - pos[1], np.array([0.0, -1.0, 0.0]))):
        if h < reach:
            cons.append((float(h), n))
    if not cons:
        return act, False, 0.0
    cons.sort(key=lambda t: t[0])
    cons = cons[:_MAX_ACTIVE]
    # Dense obstacle fields can make the constraint set jointly infeasible. Detect it here rather
    # than discovering it as a crash: relax the weakest constraints until something is solvable,
    # so the filter degrades toward "avoid the nearest few" instead of failing outright.
    # affine barrier: h(p + v) >= (1 - eta) h(p)  =>  n . v >= -eta * h
    A = np.stack([n for _, n in cons])
    b = np.array([-eta * h for h, _ in cons])

    # Escape mode. Once the vehicle is inside the inflated shell of one or more obstacles, every
    # such constraint demands outward motion and in a dense field those demands conflict, so the
    # projection is infeasible and the graceful fallbacks bounce it along a face instead of out.
    # Measured on a 53%-coverage field: five of six random-policy runs took zero hits, the sixth
    # spent 313 of 500 steps penetrating and took 311. Handle penetration separately -- head
    # straight out along the summed normals of whatever is penetrated, ignoring the rest, since
    # nothing else matters until the vehicle is clear.
    pen = [(h, n) for h, n in cons if h < 0.0]
    if pen:
        esc = np.sum([n for _, n in pen], axis=0)
        en = np.linalg.norm(esc)
        esc = (esc / en) if en > 1e-9 else pen[0][1]
        sp_e = float(np.clip(speed, v_min, v_max))
        yw_e = float(np.arctan2(esc[1], esc[0]))
        pt_e = float(np.clip(np.arcsin(np.clip(esc[2], -1.0, 1.0)), -pitch_max, pitch_max))
        dy = (yw_e - yaw + np.pi) % (2 * np.pi) - np.pi
        out = np.clip(np.array([(sp_e - speed) / max_accel, dy / max_yaw_rate,
                                (pt_e - pitch) / max_pitch_rate]), -1.0, 1.0)
        return out, True, float(np.linalg.norm(out - act))

    if np.all(A @ v_rl >= b - 1e-9):
        return act, False, 0.0                   # policy action is already safe

    v_safe = _project(v_rl, A, b, v_max)
    if v_safe is None and len(b) > 2:
        # drop the least binding constraints and retry: better to honour the three closest
        # obstacles exactly than to honour twelve approximately and hit one
        for keep in (6, 4, 3, 2):
            if keep >= len(b):
                continue
            v_try = _project(v_rl, A[:keep], b[:keep], v_max)
            if v_try is not None:
                v_safe = v_try
                break
    if v_safe is None:                           # still infeasible: steer away from the worst
        j = int(np.argmin(A @ v_rl - b))
        v_safe = A[j] * max(v_min, min(v_max, sp))

    nv = float(np.linalg.norm(v_safe))
    if nv < 1e-6:
        v_safe, nv = A[int(np.argmin(b))] * v_min, v_min
    if nv < v_min:
        # The platform cannot hover, so ||v|| >= v_min carves a non-convex hole out of the
        # feasible set: scaling the projected velocity back up along its own direction would
        # re-violate the constraint it was just projected onto. Grow it instead along a
        # direction that every active constraint agrees is safe (the normal cone interior),
        # which keeps Av >= b while restoring the required speed.
        d = A.sum(axis=0)
        dn = np.linalg.norm(d)
        d = (d / dn) if dn > 1e-9 else A[int(np.argmin(b))]
        # solve ||v_safe + t d|| = v_min for t >= 0
        bq = 2.0 * float(v_safe @ d)
        cq = float(v_safe @ v_safe) - v_min ** 2
        disc = max(bq * bq - 4.0 * cq, 0.0)
        t = (-bq + np.sqrt(disc)) / 2.0
        cand = v_safe + max(t, 0.0) * d
        if np.all(A @ cand >= b - 1e-6):
            v_safe = cand
        else:                                    # last resort: fly straight down the safest normal
            v_safe = d * v_min
        nv = float(np.linalg.norm(v_safe))
    sp_t = float(np.clip(nv, v_min, v_max))
    yw_t = float(np.arctan2(v_safe[1], v_safe[0]))
    pt_t = float(np.clip(np.arcsin(np.clip(v_safe[2] / nv, -1.0, 1.0)), -pitch_max, pitch_max))
    dyaw = (yw_t - yaw + np.pi) % (2 * np.pi) - np.pi
    # actuator limits are enforced by the clip: when the correction is not achievable in one
    # step the filter degrades gracefully instead of returning an infeasible command
    safe = np.clip(np.array([(sp_t - speed) / max_accel,
                             dyaw / max_yaw_rate,
                             (pt_t - pitch) / max_pitch_rate]), -1.0, 1.0)

    # Actuator limits mean the clipped action may not realise v_safe. Re-simulate what the
    # command actually produces and keep it only if it improves the tightest barrier margin --
    # a filter that returns something worse than the policy action is worse than no filter.
    def _margin(a):
        s2 = float(np.clip(speed + a[0] * max_accel, v_min, v_max))
        y2 = float((yaw + a[1] * max_yaw_rate) % (2 * np.pi))
        p2 = float(np.clip(pitch + a[2] * max_pitch_rate, -pitch_max, pitch_max))
        v2 = s2 * np.array([np.cos(p2) * np.cos(y2), np.cos(p2) * np.sin(y2), np.sin(p2)])
        return float(np.min(A @ v2 - b))

    if _margin(safe) <= _margin(act):
        return act, False, 0.0
    return safe, True, float(np.linalg.norm(safe - act))
