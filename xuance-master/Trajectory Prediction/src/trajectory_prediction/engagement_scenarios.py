"""Intercept-engagement scenario simulation for the military port.

Implements the dataset generator described in the paper (Sections II/IV):
  * Eq. (1)-(2) point-mass kinematics for target T and defender M, propagated
    with a trapezoidal (velocity-preserving) update at sampling interval dt.
  * Engagement descriptors r, rdot, eL, eLdot, V_L and the target FOV
    visibility indicator (half-angle theta_FOV measured off the velocity axis).
  * Evasion trigger r <= (2/3) r0; anti-LOS evasion command
    a_eva = -eL (I - eT eT^T) saturated to a_eva_max = min(a_bar, aT_max)
    and held constant within short primitive segments (Section IV-B).
  * NATOPS-style maneuver primitives (coordinated turn / climb / dive and
    coupled composites) concatenated with bounded durations and overloads.
  * Table IV parameter sampling: r0 in [4,8] km, q_phi in [-180,180] deg,
    q_theta in [-30,30] deg, dh0 in [-1,1] km, V_T in [250,350] m/s,
    V_M in [380,450] m/s, a_phi0/a_theta0 in [-88.3, 88.3] m/s^2.
  * Window labels: nine-class semantic intention (dominant primitive) and
    eleven-level normal-overload quantization (mean window overload).

Output schema matches civilian_scenarios.simulate_scenes so that
data.load_tracks / split_tracks / Windows work unchanged; the defender track
plays the role of the single context entity (context_entities=1). Extra
military fields (overload_labels, engagement_states, evasion_mask) are stored
alongside and ignored by the civilian loader until the military config
explicitly requests them.
"""
from pathlib import Path

import numpy as np

from .data import fingerprint

# Nine-class semantic intention space (paper Section IV-A); index order aligns
# with data.CLASS_NAMES (straight/left/right/up/down/left-up/right-up/
# left-down/right-down) which maps onto the paper's {no maneuver, left turn,
# right turn, climb, dive, left-climb, right-climb, left-dive, right-dive}.
SEMANTIC_CLASSES = 9
# Class -> (yaw sign, pitch sign) in the velocity frame.  Sign convention is
# the one used by semantic_label(): +yaw = left turn, -yaw = right turn,
# +pitch = climb, -pitch = dive, 0 = below the straight-flight threshold.
MANEUVER_CLASS_SIGNS = {
    0: (0., 0.),    # no maneuver / straight
    1: (1., 0.),    # left turn
    2: (-1., 0.),   # right turn
    3: (0., 1.),    # climb
    4: (0., -1.),   # dive
    5: (1., 1.),    # left-climb
    6: (-1., 1.),   # right-climb
    7: (1., -1.),   # left-dive
    8: (-1., -1.),  # right-dive
}
# Acceleration magnitude below which a primitive counts as "no maneuver".
STRAIGHT_THRESHOLD_MPS2 = 10.0
# Eleven ordered normal-overload levels in g (paper IV-A second experiment).
OVERLOAD_LEVELS = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.])
G0 = 9.81  # m/s^2

# Engagement-state descriptor layout (per time step, stored in
# engagement_states): [r/1e3, rdot/100, eL(3), eLdot(3), visible]
# Eq. (2) defines V_L = V_M e_M . e_L - V_T e_T . e_L as well; with the LOS
# pointing from the defender toward the target it is exactly -rdot, so rdot
# already carries that information and V_L is not stored a second time (the
# edge descriptor phi_edg does carry it, see ENGAGEMENT_EDGE_DIM).
ENGAGEMENT_DIM = 9


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _rotation_axis(axis, angle):
    """Rodrigues rotation matrix around unit axis."""
    axis = _unit(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def los_state(rT, vT, rM, vM):
    """Eq. (2) relative kinematics: r, rdot, eL, eLdot and V_L.

    Convention: LOS points from the defender toward the target; r is the
    scalar separation, rdot its rate (negative when closing).
    """
    relative = rT - rM
    r = float(np.linalg.norm(relative))
    eL = relative / max(r, 1e-9)
    v_rel = vT - vM
    rdot = float(np.dot(v_rel, eL))
    eLdot = np.cross(np.cross(eL, v_rel), eL) / max(r, 1e-9)
    V_L = float(np.dot(vM, eL) - np.dot(vT, eL))
    return r, rdot, eL, eLdot, V_L


def fov_visible(vT, eL, half_angle_deg=60.):
    """Target-fixed FOV: the defender is visible iff the LOS-to-defender
    direction lies within theta_FOV of the velocity axis (theta_FOV=60 deg).
    Note the descriptor uses -eL (direction target -> defender)."""
    if np.linalg.norm(vT) < 1e-9:
        return True
    cosine = float(np.dot(_unit(vT), -eL))
    return bool(cosine >= np.cos(np.radians(half_angle_deg)))


def _saturate(command, limit):
    norm = np.linalg.norm(command)
    return command * (limit / norm) if norm > limit else command


def _primitive_overloads(rng, overload_limit):
    """Legacy unconstrained primitive sampler (kept for civilian regression).

    Sampling (yaw_share, signs) freely makes the "no maneuver" class
    unreachable (both components almost surely exceed the straight-flight
    threshold), so the paper-facing generator uses
    :func:`sample_semantic_class` + :func:`_primitive_from_class` instead.
    """
    duration = rng.uniform(2.0, 6.0)
    total = overload_limit * rng.uniform(0.35, 0.95)
    yaw_share = rng.uniform(0., 1.)
    yaw = rng.choice([-1., 1.]) * total * yaw_share
    pitch = rng.choice([-1., 1.]) * total * np.sqrt(max(0., 1. - yaw_share ** 2))
    return duration, yaw, pitch


def sample_semantic_class(rng, class_balance=None):
    """Draw one of the nine semantic intention classes.

    Section IV-A states the maneuver classes are sampled with *balanced
    coverage*; ``class_balance`` (length-9 weights) lets a caller skew the mix
    (e.g. to give straight flight a larger share), and defaults to uniform.
    """
    if class_balance is None:
        return int(rng.integers(0, SEMANTIC_CLASSES))
    weights = np.asarray(class_balance, dtype=float)
    if weights.shape != (SEMANTIC_CLASSES,) or weights.min() < 0 or weights.sum() <= 0:
        raise ValueError("class_balance must be nine nonnegative weights summing to > 0")
    return int(rng.choice(SEMANTIC_CLASSES, p=weights / weights.sum()))


def _primitive_from_class(rng, cls, overload_limit, threshold=STRAIGHT_THRESHOLD_MPS2,
                          duration_range=(2.0, 6.0)):
    """Sample one NATOPS-style primitive constrained to semantic class ``cls``.

    Returns ``(duration_s, yaw_overload, pitch_overload)`` expressed in the
    velocity frame, so the downstream :func:`semantic_label` recomputes exactly
    ``cls`` for the whole primitive.  Composite classes (5-8) split the total
    overload on both axes with a 25-65 deg share so that neither component
    falls below the straight-flight threshold.  ``duration_range`` bounds the
    segment duration (Section IV-B "segment duration ... sampled within
    predefined bounds"; the paper gives no numbers, default 2-6 s).
    """
    if cls not in MANEUVER_CLASS_SIGNS:
        raise ValueError(f"unknown semantic class {cls}")
    duration = rng.uniform(*duration_range)
    yaw_sign, pitch_sign = MANEUVER_CLASS_SIGNS[int(cls)]
    if yaw_sign == 0. and pitch_sign == 0.:
        return duration, 0., 0.
    total = overload_limit * rng.uniform(0.35, 0.95)
    if yaw_sign == 0. or pitch_sign == 0.:
        return duration, yaw_sign * total, pitch_sign * total
    share = np.radians(rng.uniform(25., 65.))
    return duration, yaw_sign * total * np.cos(share), pitch_sign * total * np.sin(share)


def _frame_from_velocity(vT):
    """Velocity frame: eT plus the two normals used by the maneuver commands.

    ``lateral`` lies in the horizontal plane and is perpendicular to the
    velocity (positive = left turn); ``vertical`` lies in the vertical plane
    through the velocity and is perpendicular to it (positive = climb).  The
    previous definition swapped the two, which silently relabelled "left
    turn" as a climb command.
    """
    eT = _unit(vT)
    up = np.array([0., 0., 1.])
    lateral = np.cross(up, eT)
    if np.linalg.norm(lateral) < 1e-6:                 # vertical flight
        lateral = np.cross(np.array([1., 0., 0.]), eT)
    lateral = _unit(lateral)
    vertical = np.cross(eT, lateral)
    return eT, lateral, vertical


def _command_from_overloads(vT, a_yaw, a_pitch):
    _, horizontal, vertical = _frame_from_velocity(vT)
    return a_yaw * horizontal + a_pitch * vertical


def semantic_label(aT, vT, threshold=STRAIGHT_THRESHOLD_MPS2):
    """Dominant-primitive semantic label from the acceleration command:
    (yaw in {-1,0,1}, pitch in {-1,0,1}) -> nine classes."""
    _, horizontal, vertical = _frame_from_velocity(vT)
    a_yaw = float(np.dot(aT, horizontal))
    a_pitch = float(np.dot(aT, vertical))
    turning = 1 if a_yaw > threshold else (-1 if a_yaw < -threshold else 0)
    vertical_sign = 1 if a_pitch > threshold else (-1 if a_pitch < -threshold else 0)
    mapping = {(0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, 1): 3, (0, -1): 4,
               (1, 1): 5, (-1, 1): 6, (1, -1): 7, (-1, -1): 8}
    return mapping[(turning, vertical_sign)]


def normal_overload(aT, vT):
    """Eq. (4): normal overload magnitude an = ||VT x aT|| / ||VT||."""
    return float(np.linalg.norm(np.cross(vT, aT)) / max(np.linalg.norm(vT), 1e-9))


def signed_normal_overload(aT, vT):
    """Signed normal overload for the eleven-level intention task (IV-A).

    Eq. (4) is a magnitude and therefore nonnegative, yet the second IV-A
    experiment quantizes into I = {-10, -8, ..., 0, ..., +10} g, which
    requires a sign.  Convention adopted here: the normal acceleration
    n = ((VT x aT) x VT) / ||VT||^2 is decomposed on the velocity frame
    (horizontal = turn axis, vertical = climb axis); the sign of the
    *dominant* component (largest magnitude) is attached to the magnitude of
    n.  Left turn / climb -> positive, right turn / dive -> negative.
    """
    speed = float(np.linalg.norm(vT))
    if speed < 1e-9:
        return 0.
    cross = np.cross(vT, aT)
    normal = np.cross(cross, vT) / (speed ** 2)
    magnitude = float(np.linalg.norm(normal))
    if magnitude < 1e-9:
        return 0.
    _, lateral, vertical = _frame_from_velocity(vT)
    turn = float(np.dot(normal, lateral))
    climb = float(np.dot(normal, vertical))
    dominant = climb if abs(climb) > abs(turn) else turn
    return magnitude if dominant >= 0. else -magnitude


def overload_label(window_overloads, levels=OVERLOAD_LEVELS, censored_ends=False):
    """Eleven-level quantization of the window-mean normal overload.

    The paper defines the window label by locating the mean overload a_bar_n in
    the interval (I_m - 0.5, I_m + 0.5] g, with boundary ties mapped to the
    larger label.  With the level spacing of 2 g that rule leaves 1 g-wide gaps
    between neighbouring bins (e.g. 1.0 g < a_bar_n <= 1.5 g belongs to no
    interval); those means are assigned to the nearest level and counted by
    :func:`overload_gap_report` so the convention stays visible.

    ``censored_ends=True`` instead widens the two extreme bins to
    (-inf, I_min + 1.5] and [I_max - 1.5, +inf); this is needed if the caller
    wants all eleven rows populated, because aT_max = 9 g makes the strict
    (9.5, 10.5] bin physically unreachable.
    """
    if len(window_overloads) == 0:
        raise ValueError("overload label needs a nonempty window")
    levels = np.asarray(levels, dtype=float)
    mean_g = float(np.mean(window_overloads)) / G0
    half = np.full(levels.shape, 0.5)
    if censored_ends and levels.size > 1:
        half[0] = half[-1] = 1.5
    offset = mean_g - levels
    inside = (offset > -half) & (offset <= half)
    if inside.any():
        return int(np.flatnonzero(inside)[-1])        # ties -> larger bin
    return int(np.argmin(np.abs(offset)))


def overload_gap_report(window_means, levels=OVERLOAD_LEVELS):
    """Diagnostics for the paper's bin definition: count window-mean overloads
    that fall into the 1 g gaps between neighbouring (I_m - 0.5, I_m + 0.5]
    intervals.  ``window_means`` is a flat sequence of per-window means (m/s^2)."""
    levels = np.asarray(levels, dtype=float)
    means = np.asarray(window_means, dtype=float).reshape(-1) / G0
    if means.size == 0:
        return dict(means=0, in_gap=0, gap_fraction=0.)
    offset = means[:, None] - levels[None]
    inside = ((offset > -0.5) & (offset <= 0.5)).any(1)
    return dict(means=int(means.size), in_gap=int((~inside).sum()),
                gap_fraction=float((~inside).mean()))


def _defender_command(rT, vT, rM, vM, aM_max):
    """Pure-pursuit interceptor guidance: accelerate toward the target's
    current position, saturated at the defender overload limit."""
    direction = _unit(rT - rM)
    desired = direction * np.linalg.norm(vM)
    command = (desired - vM) / max(0.5, 1.0)  # first-order velocity alignment
    return _saturate(command, aM_max)


def _evade_command(eL, vT, a_bar, aT_max, eps=1e-6):
    """Anti-LOS evasion command (Section IV-B): project -eL onto the plane
    normal to the target velocity, then saturate to min(a_bar, aT_max)."""
    eT = _unit(vT)
    projected = -eL - eT * np.dot(-eL, eT)     # (I - eT eT^T)(-eL)
    limit = min(a_bar, aT_max)
    norm = np.linalg.norm(projected)
    if norm <= eps:
        return np.zeros(3)
    return projected * (limit / max(norm, eps))


# Table VI: initial conditions of the four composite maneuver scenarios used
# by Fig. 14. (a_pitch, a_yaw) are given in the velocity frame; the paper also
# tabulates them as a_theta,0 / a_phi,0, kept here verbatim for cross-checking.
FIGURE14_SCENARIOS = (
    dict(name="Left-Climb", a_theta_0=39., a_phi_0=-58., r0_km=5.3,
         e_l0=(0.98, -0.15, 0.09), yaw_sign=1., pitch_sign=1.),
    dict(name="Left-Dive", a_theta_0=-39., a_phi_0=-58., r0_km=5.3,
         e_l0=(0.98, -0.15, -0.09), yaw_sign=1., pitch_sign=-1.),
    dict(name="Right-Climb", a_theta_0=39., a_phi_0=58., r0_km=5.3,
         e_l0=(0.98, 0.15, 0.09), yaw_sign=-1., pitch_sign=1.),
    dict(name="Right-Dive", a_theta_0=-39., a_phi_0=58., r0_km=5.3,
         e_l0=(0.98, 0.15, -0.09), yaw_sign=-1., pitch_sign=-1.),
)
# Fig. 14 protocol constants: To=8 s, Tp=6 s, VT=300 m/s, VM=360 m/s.
FIGURE14_OBSERVATION_S = 8.
FIGURE14_PREDICTION_S = 6.
FIGURE14_VT = 300.
FIGURE14_VM = 360.


def simulate_engagement_batch(path, conditions, steps=140, dt=.1, seed=0,
                             aT_max=9. * G0, aM_max=15. * G0, fov_half_angle=60.,
                             evasion_fraction=2. / 3., overload_window_steps=20):
    """Deterministic engagements with explicit initial conditions (Table VI).

    Each condition is a mapping with ``name``, ``e_l0`` (initial LOS unit
    vector, defender -> target), ``r0_m``, ``vT``, ``vM`` and the initial
    maneuver command ``a_yaw``/``a_pitch`` in the velocity frame.  The target
    holds that composite maneuver until the evasion trigger, then switches to
    the anti-LOS command, exactly as in :func:`simulate_engagements`.
    """
    if not conditions:
        raise ValueError("conditions must contain at least one scenario")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    records = {name: [] for name in ("track_id", "scene_id", "scenario", "time", "states",
                                     "labels", "context_states", "context_mask",
                                     "overload_labels", "engagement_states", "evasion_mask",
                                     "overload_series")}
    diagnostics = []
    for condition in conditions:
        name = str(condition["name"])
        eL0 = _unit(np.asarray(condition["e_l0"], dtype=float))
        r0 = float(condition.get("r0_m", 5300.))
        vT0 = float(condition.get("vT", FIGURE14_VT))
        vM0 = float(condition.get("vM", FIGURE14_VM))
        prim_yaw = float(condition.get("a_yaw", 0.))
        prim_pitch = float(condition.get("a_pitch", 0.))
        rT = np.zeros(3)
        rM = rT - eL0 * r0
        heading = np.radians(float(condition.get("heading_deg", 0.)))
        climb = np.radians(float(condition.get("climb_deg", 0.)))
        vT = vT0 * _unit(np.array([np.cos(heading) * np.cos(climb),
                                   np.sin(heading) * np.cos(climb), np.sin(climb)]))
        vM = vM0 * eL0
        r0_actual = float(np.linalg.norm(rT - rM))
        states_T, states_M = np.zeros((steps, 6)), np.zeros((steps, 6))
        engagement = np.zeros((steps, ENGAGEMENT_DIM))
        semantic = np.zeros(steps, dtype=np.int64)
        overload_bin = np.zeros(steps, dtype=np.int64)
        evade = np.zeros(steps, dtype=bool)
        overload_series = np.zeros(steps)
        window = max(1, overload_window_steps)
        evasion_on, evasion_started = False, -1
        for step in range(steps):
            r, rdot, eL, eLdot, V_L = los_state(rT, vT, rM, vM)
            visible = fov_visible(vT, eL, fov_half_angle)
            states_T[step, :3], states_T[step, 3:] = rT, vT
            states_M[step, :3], states_M[step, 3:] = rM, vM
            engagement[step] = [r / 1e3, rdot / 100., *eL, *eLdot, float(visible)]
            if not evasion_on and r <= evasion_fraction * r0_actual:
                evasion_on, evasion_started = True, step
            aT_cmd = (_evade_command(eL, vT, aT_max, aT_max) if evasion_on
                      else _command_from_overloads(vT, prim_yaw, prim_pitch))
            aM_cmd = _defender_command(rT, vT, rM, vM, aM_max)
            semantic[step] = semantic_label(aT_cmd, vT)
            overload_series[step] = signed_normal_overload(aT_cmd, vT)
            overload_bin[step] = overload_label(overload_series[max(0, step - window + 1):step + 1])
            evade[step] = evasion_on
            vT_next = vT + aT_cmd * dt
            vM_next = vM + aM_cmd * dt
            vT_next = _unit(vT_next) * vT0 if np.linalg.norm(vT_next) > 1e-9 else vT
            vM_next = _unit(vM_next) * vM0 if np.linalg.norm(vM_next) > 1e-9 else vM
            rT = rT + dt * (vT + vT_next) / 2.
            rM = rM + dt * (vM + vM_next) / 2.
            vT, vM = vT_next, vM_next
        separation = np.linalg.norm(states_T[:, :3] - states_M[:, :3], axis=1)
        conditions_echo = {key: (list(value) if isinstance(value, (tuple, list, np.ndarray)) else value)
                           for key, value in condition.items()}
        diagnostics.append(dict(scene_id=name, scenario=name, r0_m=r0, r0_actual_m=r0_actual,
                                vT0_mps=vT0, vM0_mps=vM0, a_yaw_mps2=prim_yaw, a_pitch_mps2=prim_pitch,
                                evasion_step=evasion_started, min_separation_m=float(separation.min()),
                                final_separation_m=float(separation[-1]),
                                visible_fraction=float(engagement[:, -1].mean()),
                                conditions=conditions_echo))
        records["track_id"].extend([f"{name}_agent_0"] * steps)
        records["scene_id"].extend([name] * steps)
        records["scenario"].extend([name] * steps)
        records["time"].extend(np.arange(steps) * dt)
        records["states"].extend(states_T)
        records["labels"].extend(semantic)
        records["context_states"].extend(states_M[:, None])
        records["context_mask"].extend((rng.random((steps, 1)) >= 0.).astype(bool))
        records["overload_labels"].extend(overload_bin)
        records["engagement_states"].extend(engagement)
        records["evasion_mask"].extend(evade[:, None])
        records["overload_series"].extend(overload_series)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, values in records.items():
        if key in ("states", "context_states", "engagement_states"):
            arrays[key] = np.asarray(values, dtype=np.float32)
        elif key == "time":
            arrays[key] = np.asarray(values, dtype=np.float64)
        elif key == "overload_series":
            arrays[key] = np.asarray(values, dtype=np.float32)
        elif key in ("labels", "overload_labels"):
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key in ("context_mask", "evasion_mask"):
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=str)
    np.savez_compressed(path, **arrays)
    return dict(path=str(path.resolve()), seed=seed, scenes=len(conditions),
                tracks=len(conditions), steps=steps, dt=dt, sample_stride=1, internal_dt=dt,
                overload_window_steps=overload_window_steps, sha256=fingerprint(path),
                domain="military_intercept_engagement_table_vi", context_entities=1,
                max_speed_mps=450., max_overload_g=aT_max / G0, defender_overload_g=aM_max / G0,
                fov_half_angle_deg=fov_half_angle, evasion_fraction=evasion_fraction,
                observation_seconds=steps * dt, diagnostics=diagnostics,
                table_vi=[{k: (list(v) if isinstance(v, (tuple, list)) else v)
                           for k, v in condition.items()} for condition in conditions])


def simulate_engagements(path, scenes=60, steps=200, dt=.1, seed=42,
                         r0_range=(4000., 8000.), q_phi_range=(-180., 180.),
                         q_theta_range=(-30., 30.), dh0_range=(-1000., 1000.),
                         vT_range=(250., 350.), vM_range=(380., 450.),
                         aT_max=9. * G0, aM_max=15. * G0,
                         vT_fixed=None, vM_fixed=None, fov_half_angle=60.,
                         evasion_fraction=2. / 3., a_bar_eva=None,
                         measurement_dropout=0., initial_fov_coverage=1.0,
                         sample_stride=1, overload_window_steps=20,
                         class_balance=None, overload_sign=True,
                         evasion_overload_range=(0.5, 1.0),
                         intercept_radius_m=50., miss_grow_steps=30,
                         censored_overload_ends=False,
                         primitive_duration_range=(2.0, 6.0),
                         evasion_segment_seconds=4.0):
    """Simulate target-defender intercept engagements (Table IV/V protocol).

    The defender closes with pure-pursuit guidance saturated at aM_max; the
    target flies concatenated NATOPS primitives until the evasion trigger
    r <= evasion_fraction * r0, then switches to the anti-LOS evasion
    command. Speeds are held constant in magnitude (overload acts only in
    the velocity-normal plane, matching coordinated-turn assumptions).

    The kinematics are always propagated on a fine internal grid,
    dt_internal = dt / sample_stride, and every sample_stride-th state is
    stored (decimation D4): one fine simulation serves both IV-A
    (dt=0.5 s, stride 5) and IV-B (dt=0.1 s, stride 1) protocols.

    Maneuver primitives are drawn *per semantic class* (``class_balance``,
    default uniform over the nine classes so that "no maneuver" is reachable
    and the IV-A confusion matrix has all nine rows) and are held constant in
    the velocity frame, so a primitive keeps one label for its whole duration.
    ``overload_sign=True`` records the signed normal overload, which is what
    the eleven-level {-10 g ... +10 g} intention task quantizes.

    Section IV-B: "Segment duration and the nominal overload level a_bar_eva
    are sampled within predefined bounds ... The evasion maneuver is generated
    by holding a_eva(t) within each short primitive segment and updating it at
    segment transitions."  Here a_eva = min{a_bar_eva, aT_max} is the *scalar*
    magnitude: it is held for ``evasion_segment_steps`` internal steps (C6
    resamples it from ``evasion_overload_range`` at each transition), while
    the direction template a_tilde_eva(t) = -(I - e_T e_T^T) e_L(t) remains a
    pointwise function of the live LOS geometry, exactly as the (t)-indexed
    formula prescribes.  (A direction-freezing variant was A/B-tested on
    2026-10-14 and reverted -- audit 10.10.)

    C9 intercept termination: a complete engagement trajectory segment ends at
    interception (r <= ``intercept_radius_m``) or after a miss is certain (the
    separation has been growing for ``miss_grow_steps`` consecutive internal
    steps after its minimum).  Post-termination rows keep the frozen terminal
    state so the fixed-length window protocol stays intact; the termination
    step and reason are recorded in the diagnostics.
    """
    if not isinstance(scenes, int) or scenes < 3 or not isinstance(steps, int) or steps < 8:
        raise ValueError("Need at least three independent scenes and eight steps")
    if not isinstance(sample_stride, int) or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    if not isinstance(overload_window_steps, int) or overload_window_steps < 1:
        raise ValueError("overload_window_steps must be a positive integer")
    primitive_duration_range = (float(primitive_duration_range[0]), float(primitive_duration_range[1]))
    if not (primitive_duration_range[0] > 0 and primitive_duration_range[0] <= primitive_duration_range[1]):
        raise ValueError("primitive_duration_range needs two seconds 0 < lo <= hi")
    if not (evasion_segment_seconds > 0):
        raise ValueError("evasion_segment_seconds must be positive")
    if not np.isfinite(dt) or dt <= 0 or not 0 <= measurement_dropout < 1:
        raise ValueError("Invalid sampling interval or measurement dropout")
    if a_bar_eva is None:
        a_bar_eva = aT_max
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    internal_steps = steps * sample_stride          # fine-grid propagation steps
    dt_int = dt / sample_stride                     # internal integration interval
    keep = np.arange(0, internal_steps, sample_stride)   # stored (decimated) rows
    records = {name: [] for name in ("track_id", "scene_id", "scenario", "time", "states",
                                     "labels", "context_states", "context_mask",
                                     "overload_labels", "engagement_states", "evasion_mask",
                                     "overload_series")}
    diagnostics = []
    for scene in range(scenes):
        # ---- A1: Table IV parameter sampling --------------------------------
        r0 = float(rng.uniform(*r0_range))
        q_phi = np.radians(rng.uniform(*q_phi_range))
        q_theta = np.radians(rng.uniform(*q_theta_range))
        dh0 = float(rng.uniform(*dh0_range))
        vT0 = float(vT_fixed) if vT_fixed else rng.uniform(*vT_range)
        vM0 = float(vM_fixed) if vM_fixed else rng.uniform(*vM_range)
        # ---- A2: instantiate engagement geometry ----------------------------
        rT = np.zeros(3)
        horizontal = np.array([np.cos(q_phi), np.sin(q_phi), 0.])
        eL0 = _unit(horizontal * np.cos(q_theta) + np.array([0., 0., np.sin(q_theta)]))
        rM = rT - eL0 * r0
        rM = rM + np.array([0., 0., -dh0])           # altitude offset
        heading = float(rng.uniform(-np.pi, np.pi))
        climb = np.radians(rng.uniform(-10., 10.))
        vT = vT0 * _unit(np.array([np.cos(heading) * np.cos(climb),
                                   np.sin(heading) * np.cos(climb), np.sin(climb)]))
        vM = vM0 * eL0                          # defender pursues along the LOS
        if rng.random() < initial_fov_coverage:
            # rotate target heading toward the incoming defender direction
            # (-eL0) so the defender starts inside the visible sector (IV-B)
            axis = np.cross(-eL0, vT)
            if np.linalg.norm(axis) > 1e-6:
                vT = _rotation_axis(axis, rng.uniform(0., np.radians(50.))) @ vT
        # Paper IV-B: r0 is the *actual* initial distance between target and
        # defender (the dh0 altitude offset changes it from the sampled range).
        r0_actual = float(np.linalg.norm(rT - rM))
        # ---- A3/A4: propagate with kinematics and evasion --------------------
        states_T = np.zeros((internal_steps, 6))
        states_M = np.zeros((internal_steps, 6))
        engagement = np.zeros((internal_steps, ENGAGEMENT_DIM))
        semantic = np.zeros(internal_steps, dtype=np.int64)
        overload_bin = np.zeros(internal_steps, dtype=np.int64)
        evade = np.zeros(internal_steps, dtype=bool)
        overload_series = np.zeros(internal_steps)
        overload_window = max(1, overload_window_steps * sample_stride)
        primitive_left, a_prim = 0.0, np.zeros(3)
        prim_yaw, prim_pitch = 0., 0.
        evasion_on, evasion_started_step = False, -1
        # C6: per-segment nominal evasion overload, sampled within predefined
        # bounds (Section IV-B) and updated at segment transitions.
        evasion_segment_steps = max(1, int(round(evasion_segment_seconds / dt_int)))
        evasion_segment_left = 0
        a_bar_segment = a_bar_eva if a_bar_eva is not None else aT_max
        # C9: intercept termination bookkeeping.
        terminated_step, termination_reason = -1, "ran_to_horizon"
        separation_prev = r0_actual
        growing_run = 0
        for step in range(internal_steps):
            r, rdot, eL, eLdot, V_L = los_state(rT, vT, rM, vM)
            visible = fov_visible(vT, eL, fov_half_angle)
            states_T[step, :3], states_T[step, 3:] = rT, vT
            states_M[step, :3], states_M[step, 3:] = rM, vM
            engagement[step] = [r / 1e3, rdot / 100., eL[0], eL[1], eL[2],
                                eLdot[0], eLdot[1], eLdot[2], float(visible)]
            # evasion trigger (Section IV-B): r <= 2/3 * actual initial range
            if not evasion_on and r <= evasion_fraction * r0_actual:
                evasion_on, evasion_started_step = True, step
            # target acceleration command
            if evasion_on:
                # Section IV-B: the *scalar* evasion magnitude a_eva =
                # min{a_bar_eva, aT_max} is held within each short primitive
                # segment and updated at segment transitions (C6 bounded
                # resampling), while the direction template
                # a_tilde_eva(t) = -(I - e_T e_T^T) e_L(t) is a pointwise
                # function of the live engagement geometry (the paper indexes
                # it with (t) and derives the lateral/vertical components from
                # eL(t), eT(t) directly).  A 2026-10-14 A/B test of freezing
                # the direction per segment (audit 10.10 v1) produced a
                # memorization signature on the semantic task (train CE 0.07
                # vs validation CE rising past 3.6) and was reverted.
                if evasion_segment_left <= 0.:
                    a_bar_segment = aT_max * rng.uniform(*evasion_overload_range)
                    evasion_segment_left = evasion_segment_steps
                evasion_segment_left -= dt_int
                aT_cmd = _evade_command(eL, vT, a_bar_segment, aT_max)
            else:
                if primitive_left <= 0.:
                    cls = sample_semantic_class(rng, class_balance)
                    duration, prim_yaw, prim_pitch = _primitive_from_class(
                        rng, cls, aT_max, duration_range=primitive_duration_range)
                    primitive_left = duration
                primitive_left -= dt_int
                # coordinated-turn semantics: the primitive holds a constant
                # (yaw, pitch) overload in the *current* velocity frame, which
                # also keeps the recomputed semantic label stable per primitive
                a_prim = _command_from_overloads(vT, prim_yaw, prim_pitch)
                aT_cmd = a_prim
            aM_cmd = _defender_command(rT, vT, rM, vM, aM_max)
            # labels from the commanded acceleration (window semantics come
            # from the label aggregation in the dataset, per-step here)
            semantic[step] = semantic_label(aT_cmd, vT)
            overload_series[step] = (signed_normal_overload(aT_cmd, vT) if overload_sign
                                     else normal_overload(aT_cmd, vT))
            # Eq. IV-A statistic: trailing window of one observation window
            overload_bin[step] = overload_label(overload_series[max(0, step - overload_window + 1):step + 1],
                                                censored_ends=censored_overload_ends)
            evade[step] = evasion_on
            # C9: engagement termination (intercept or certain miss).
            if terminated_step < 0:
                if r <= intercept_radius_m:
                    terminated_step, termination_reason = step, "intercepted"
                else:
                    growing_run = growing_run + 1 if r > separation_prev else 0
                    if evasion_on and growing_run >= miss_grow_steps:
                        terminated_step, termination_reason = step, "missed"
                separation_prev = r
            if terminated_step == step and step + 1 < internal_steps:
                # freeze both aircraft at the terminal state; the remaining
                # rows keep constant states/descriptors (no post-intercept
                # crossing artifacts in the training distribution).
                states_T[step + 1:] = states_T[step]
                states_M[step + 1:] = states_M[step]
                engagement[step + 1:] = engagement[step]
                semantic[step + 1:] = semantic[step]
                overload_series[step + 1:] = overload_series[step]
                overload_bin[step + 1:] = overload_bin[step]
                evade[step + 1:] = evade[step]
                break
            # trapezoidal velocity update: overload changes direction only
            vT_next = vT + aT_cmd * dt_int
            vM_next = vM + aM_cmd * dt_int
            # keep speeds constant (normal-plane commands): rescale to nominal
            vT_next = _unit(vT_next) * vT0 if np.linalg.norm(vT_next) > 1e-9 else vT
            vM_next = _unit(vM_next) * vM0 if np.linalg.norm(vM_next) > 1e-9 else vM
            rT = rT + dt_int * (vT + vT_next) / 2.
            rM = rM + dt_int * (vM + vM_next) / 2.
            vT, vM = vT_next, vM_next
        # ---- A5/A6: record per-agent tracks (target focal, defender context) --
        scene_id = f"scene_{scene:06d}"
        separation = np.linalg.norm(states_T[:, :3] - states_M[:, :3], axis=1)
        diagnostics.append(dict(scene_id=scene_id, scenario="engagement",
                                r0_m=r0, r0_actual_m=r0_actual,
                                q_phi0_deg=float(np.degrees(q_phi)),
                                q_theta0_deg=float(np.degrees(q_theta)), dh0_m=dh0,
                                vT0_mps=vT0, vM0_mps=vM0,
                                evasion_step=evasion_started_step,
                                terminated_step=terminated_step,
                                termination_reason=termination_reason,
                                min_separation_m=float(separation.min()),
                                final_separation_m=float(separation[-1]),
                                visible_fraction=float(engagement[:, -1].mean())))
        availability = (rng.random((internal_steps, 1)) >= measurement_dropout)[keep]
        records["track_id"].extend([f"{scene_id}_agent_0"] * len(keep))
        records["scene_id"].extend([scene_id] * len(keep))
        records["scenario"].extend(["engagement"] * len(keep))
        records["time"].extend(np.arange(len(keep)) * dt)
        records["states"].extend(states_T[keep])
        records["labels"].extend(semantic[keep])
        records["context_states"].extend(states_M[keep][:, None])
        records["context_mask"].extend(availability)
        records["overload_labels"].extend(overload_bin[keep])
        records["engagement_states"].extend(engagement[keep])
        records["evasion_mask"].extend(evade[keep, None])
        # F1: keep the raw signed normal-overload series so the eleven-level
        # window label can be recomputed as the paper defines it (window mean
        # of a_n, one quantization per observation window) instead of majority
        # voting over trailing-window bins.
        records["overload_series"].extend(overload_series[keep])
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, values in records.items():
        if name in ("states", "context_states", "engagement_states"):
            arrays[name] = np.asarray(values, dtype=np.float32)
        elif name == "time":
            arrays[name] = np.asarray(values, dtype=np.float64)
        elif name == "overload_series":
            arrays[name] = np.asarray(values, dtype=np.float32)
        elif name in ("labels", "overload_labels"):
            arrays[name] = np.asarray(values, dtype=np.int64)
        elif name in ("context_mask", "evasion_mask"):
            arrays[name] = np.asarray(values, dtype=bool)
        else:
            arrays[name] = np.asarray(values, dtype=str)
    # F1: persist the bin convention so the window-label recomputation in
    # data.window_intention_label uses the same extremes the generator used.
    arrays["overload_censored"] = np.asarray(bool(censored_overload_ends))
    np.savez_compressed(path, **arrays)
    semantic_counts = np.bincount(arrays["labels"], minlength=SEMANTIC_CLASSES).tolist()
    overload_counts = np.bincount(arrays["overload_labels"],
                                  minlength=len(OVERLOAD_LEVELS)).tolist()
    return dict(path=str(path.resolve()), seed=seed, scenes=scenes, tracks=scenes,
                steps=steps, dt=dt, sample_stride=sample_stride, internal_dt=dt_int,
                overload_window_steps=overload_window_steps, sha256=fingerprint(path),
                domain="military_intercept_engagement", context_entities=1,
                measurement_dropout=measurement_dropout, max_speed_mps=450.,
                max_overload_g=aT_max / G0, defender_overload_g=aM_max / G0,
                fov_half_angle_deg=fov_half_angle, evasion_fraction=evasion_fraction,
                evasion_overload_range=[float(evasion_overload_range[0]),
                                        float(evasion_overload_range[1])],
                censored_overload_ends=bool(censored_overload_ends),
                primitive_duration_range_s=[primitive_duration_range[0],
                                            primitive_duration_range[1]],
                evasion_segment_seconds=float(evasion_segment_seconds),
                intercept_radius_m=float(intercept_radius_m),
                miss_grow_steps=int(miss_grow_steps),
                intercepts=sum(1 for d in diagnostics if d["termination_reason"] == "intercepted"),
                misses=sum(1 for d in diagnostics if d["termination_reason"] == "missed"),
                horizon_completes=sum(1 for d in diagnostics
                                      if d["termination_reason"] == "ran_to_horizon"),
                table_iv=dict(r0_range_km=[r0_range[0] / 1e3, r0_range[1] / 1e3],
                              q_phi_range_deg=list(q_phi_range), q_theta_range_deg=list(q_theta_range),
                              dh0_range_km=[dh0_range[0] / 1e3, dh0_range[1] / 1e3],
                              vT_range_mps=list(vT_range), vM_range_mps=list(vM_range),
                              a_cmd_range_mps2=[-aT_max, aT_max]),
                diagnostics=diagnostics,
                # IV-A reproducibility guards: both label spaces must be
                # populated, otherwise the reported confusion matrices cannot
                # be reproduced (class 0 empty; negative overload bins unused).
                semantic_class_counts=semantic_counts,
                semantic_classes_used=int(np.count_nonzero(semantic_counts)),
                overload_level_counts=overload_counts,
                overload_levels_used=int(np.count_nonzero(overload_counts)),
                overload_sign=bool(overload_sign),
                class_balance=None if class_balance is None else [float(w) for w in class_balance],
                prediction_inputs="measured target/defender histories and engagement descriptors; no future states",
                safety_claim="Simulated intercept benchmark; separations are recorded, not a guaranteed deconfliction controller")
