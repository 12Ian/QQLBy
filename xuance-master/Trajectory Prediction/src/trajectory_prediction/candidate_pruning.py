"""Candidate future-branch expansion and interactive distribution cropping.

Paper Eqs. (22)-(25) and Table I: the interaction distribution cropper prunes
the set of candidate future states before message passing. Engineering
decision D1 (see MILITARY_PORT_PLAN.md) resolves the tension between Eq. (15)
(history nodes) and Table I (candidate kinematic states) with an anchored
candidate construction:

  * the 2O historical nodes stay the graph backbone;
  * A candidate branches are attached to the end of the target's observation
    window. Each candidate is one constant-speed coordinated turn whose
    heading change theta is sampled by a near-uniform anchor direction, so the
    candidate displacement is the turn chord d = 2 (v T / theta) sin(theta/2)
    along the chord bisector (an arc of constant lateral acceleration);
  * every candidate j is gated by three hard constraints
        m(j) = Cesc(j) * Cenv(j) * Cene(j)                       (Eq. 30)
    - Cesc  escape consistency: over the sampled arc the time-averaged LOS
            range rate must satisfy (1/T) sum_k rdot_k >= rdot_min
            (Eq. 22/27); it is bypassed (Cesc = 1) when the defender is
            outside the target FOV or unmeasured, where the engagement term
            is undefined (Table I step 4).
    - Cenv  environment: EVERY arc point stays inside the free region /
            altitude band, g(y_k) in F for all k (Eq. 23/28; simplified
            terrain per decision D2).
    - Cene  energy: the time-averaged acceleration increment of the turn,
            v * theta^2 / T^2, stays within the target overload limit
            (Eq. 24/29).  The instantaneous turn load v*theta/T is not a
            hard gate here (fidelity A-05): it drives the energy teacher
            (Eq. 45) and the soft crop cost instead.
  * the soft crop exp(-Lambda_crop * P_ij) of Eq. (25) is applied as a
    destination penalty on the attention logits, scaled by the same
    crop_progress schedule the demo already uses for progressive cropping
    ("cropping strength grows with training iterations").

Pruned candidates are removed from the graph entirely: they receive no
message, contribute no teacher term, and are masked out of the decoder memory.
"""
import torch

from .data import rotate_xy

G0 = 9.81


def anchor_directions(count, device=None):
    """A candidate direction set [A, 3] in the heading-aligned canonical frame.

    Direction 0 is always the canonical +z axis, i.e. the straight-ahead
    (no-maneuver) branch, so the candidate set never collapses; the remaining
    A-1 directions are a near-uniform Fibonacci sphere covering turns (and, at
    long horizons, reversals) that the energy constraint can prune.
    """
    if count < 1:
        raise ValueError("Need at least one anchor direction")
    straight = torch.tensor([[0., 0., 1.]], device=device, dtype=torch.float32)
    if count == 1:
        return straight
    index = torch.arange(count - 1, device=device, dtype=torch.float64)
    z = 1. - 2. * (index + .5) / (count - 1)
    radius = torch.sqrt(torch.clamp(1. - z * z, min=0.))
    angle = torch.pi * (3. - 5. ** .5) * index
    sphere = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), z), -1).float()
    return torch.cat((straight, sphere), 0)


def _velocity_basis(velocity):
    """Orthonormal (e1, e2, e3 = heading) frame per sample; [B,3] each."""
    e3 = velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    up = e3.new_tensor([0., 0., 1.]).expand_as(e3)
    e1 = up - e3 * (up * e3).sum(-1, keepdim=True)
    fallback = e3.new_tensor([1., 0., 0.]).expand_as(e3)
    degenerate = e1.norm(dim=-1) < 1e-6
    e1 = torch.where(degenerate[..., None], fallback - e3 * (fallback * e3).sum(-1, keepdim=True), e1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    e2 = torch.linalg.cross(e3, e1)
    return e1, e2, e3


def _last_measured(context, context_mask):
    """Defender state at its last available measurement; [B,6] and [B] flag."""
    steps = context_mask.shape[-1]
    has = context_mask.any(-1)                                     # [B,N]
    from_end = context_mask.flip(-1).to(torch.int64).argmax(-1)    # [B,N]
    index = (steps - 1 - from_end).clamp_min(0)
    picked = torch.gather(context, 2, index[..., None, None].expand(-1, -1, 1, 6)).squeeze(2)
    weights = has.to(context.dtype)[..., None]
    merged = (picked * weights).sum(1) / weights.sum(1).clamp_min(1.)
    return merged, has.any(-1)


def _constant_turn_arc(position, velocity, world_direction, horizon, samples):
    """Sample a constant-speed coordinated-turn arc in the world frame.

    The velocity rotates from the current heading e3 to the anchor-derived unit
    direction ``world_direction`` about the turn axis n = normalize(e3 x world)
    through the total angle theta, at the constant rate omega = theta / horizon.
    Sampling the arc at fractions s = 1/S .. 1 lets the cropper evaluate the
    constraints over the WHOLE candidate trajectory as Table I requires
    (Eq. (27) time-averaged range rate, Eq. (28) forall-k environment), rather
    than only at its endpoint.

    Returns positions [B,A,S,3], velocities [B,A,S,3], theta [B,A].
    """
    speed = velocity.norm(dim=-1).clamp_min(1e-6)                  # [B]
    e3 = velocity / speed[:, None]                                 # [B,3]
    axis = torch.linalg.cross(e3[:, None], world_direction, dim=-1)
    n = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)     # [B,A,3]
    n_cross_e3 = torch.linalg.cross(n, e3[:, None].expand_as(n), dim=-1)
    theta = (world_direction * e3[:, None]).sum(-1).clamp(-1., 1.).acos()   # [B,A]
    grid = torch.linspace(1. / samples, 1., samples, device=position.device,
                          dtype=position.dtype)                    # [S]
    phi = theta[:, :, None] * grid[None, None, :]                  # [B,A,S]
    cos_p, sin_p = torch.cos(phi), torch.sin(phi)
    velocities = speed[:, None, None, None] * (
        e3[:, None, None] * cos_p[..., None]
        + n_cross_e3[:, :, None] * sin_p[..., None])               # [B,A,S,3]
    # Closed-form integral of the rotating velocity: A(phi)=sin(phi)/theta,
    # B(phi)=(1-cos(phi))/theta, with the straight-flight limit A->s, B->0.
    safe_theta = theta.clamp_min(1e-8)
    a_coef = torch.where(theta[..., None] > 1e-6,
                         torch.sin(phi) / safe_theta[..., None], grid.expand_as(phi))
    b_coef = torch.where(theta[..., None] > 1e-6,
                         (1. - torch.cos(phi)) / safe_theta[..., None],
                         torch.zeros_like(phi))
    displacement = (speed * horizon)[:, None, None, None] * (
        e3[:, None, None] * a_coef[..., None]
        + n_cross_e3[:, :, None] * b_coef[..., None])              # [B,A,S,3]
    positions = position[:, None, None] + displacement
    return positions, velocities, theta


def candidate_branch(observed, context, context_mask, cfg, anchors):
    """Expand the A candidate branches and evaluate the Table I constraints.

    Returns a dict with the candidate block padded to O rows so it can be
    concatenated with the historical entities:
      relative  [B,O,6] candidate states in the focal relative frame
      absolute  [B,O,6] the same states in world coordinates (teachers)
      keep      [B,O]   hard mask m(j) = Cesc * Cenv * Cene
      cost       [B,O]   normalized pruning cost P_ij (soft crop)
    """
    batch, steps = observed.shape[:2]
    device, dtype = observed.device, observed.dtype
    count = anchors.shape[0]
    if count > steps:
        raise ValueError(f"candidate_anchors ({count}) cannot exceed observation_steps ({steps})")
    position, velocity = observed[:, -1, :3], observed[:, -1, 3:]
    speed = velocity.norm(dim=-1).clamp_min(1e-6)
    horizon = max(cfg.prediction_steps, 1) * cfg.dt
    a_max = cfg.engagement_overload_g * G0
    e1, e2, e3 = _velocity_basis(velocity)
    # Anchor directions expressed in the current heading frame.
    world = (anchors[None, :, 0, None] * e1[:, None]
             + anchors[None, :, 1, None] * e2[:, None]
             + anchors[None, :, 2, None] * e3[:, None])             # [B,A,3]
    world = world / world.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # Sample the whole constant-turn arc so Eq. (27)/(28) see every step k.
    arc_pos, arc_vel, theta = _constant_turn_arc(
        position, velocity, world, horizon, max(int(cfg.candidate_arc_samples), 1))

    # ---- Cene (Eq. 24/29): time-averaged acceleration increment ------------
    # (1/T) * integral ||a_{T,k} - a_{T,k-1}|| <= a_T,max.  For the constant
    # -rate coordinated turn that defines a candidate branch the acceleration
    # vector rotates at omega = theta / T while keeping |a| = v * theta / T, so
    # the mean increment per unit time is a_req * omega = v * theta^2 / T^2.
    # Eq. (24)/(29) gates exactly this increment; the instantaneous turn load
    # a_req = v*theta/T is NOT a hard cropper condition - it drives the energy
    # teacher (Eq. 45) and the soft crop cost P_ij below (see fidelity A-05).
    a_req = speed[:, None] * theta / horizon                        # Eq. 45
    mean_increment = a_req * (theta / horizon)                      # Eq. 24
    energy = mean_increment <= a_max * (1. + 1e-6)

    # Endpoint state (s = 1) is the candidate node feature; it is the arc chord.
    ahead = arc_pos[:, :, -1]                                       # [B,A,3]
    candidate_velocity = arc_vel[:, :, -1]                          # [B,A,3]
    displacement = ahead - position[:, None]

    # ---- Cesc (Eq. 22/27): mean LOS range rate over the sampling window ----
    # cesc(j) = 1[ (1/T) sum_k rdot_k >= rdot_min ].  The defender's future
    # control is unknown, so it is propagated at constant velocity from its
    # last measurement (the same zero-acceleration predictor used elsewhere).
    escape = torch.ones(batch, count, dtype=torch.bool, device=device)
    if context is not None:
        defender, measured = _last_measured(context, context_mask)
        p_m0, v_m0 = defender[:, :3], defender[:, 3:]
        sample_count = arc_pos.shape[2]
        t_grid = torch.linspace(horizon / sample_count, horizon, sample_count,
                                device=device, dtype=dtype)         # [S]
        defender_pos = p_m0[:, None, None] + v_m0[:, None, None] * t_grid[None, None, :, None]
        los = arc_pos - defender_pos                                # [B,A,S,3]
        separation = los.norm(dim=-1).clamp_min(1e-3)
        e_l = los / separation[..., None]
        rdot_steps = ((arc_vel - v_m0[:, None, None]) * e_l).sum(-1)   # [B,A,S]
        mean_rdot = rdot_steps.mean(-1)                             # Eq. (27)
        engaged = mean_rdot >= cfg.escape_rate_min_mps
        # Table I step 4: with the defender outside the FOV (or unmeasured) the
        # engagement term is undefined and Cesc collapses to 1.  Visibility is
        # decided at the current instant from the target's present geometry.
        toward = p_m0 - position
        cosine = (velocity * toward).sum(-1) / (velocity.norm(dim=-1) * toward.norm(dim=-1).clamp_min(1e-6))
        visible = (cosine >= float(torch.cos(torch.deg2rad(torch.tensor(
            float(cfg.fov_half_angle_deg)))).to(dtype))) & measured
        escape = torch.where(visible[:, None], engaged, torch.ones_like(engaged))

    # ---- Cenv (Eq. 23/28): environment/terrain at EVERY arc step k ---------
    if getattr(cfg, "terrain_enabled", False):
        low, high = cfg.terrain_altitude_min_m, cfg.terrain_altitude_max_m
        environment = ((arc_pos[..., 2] >= low) & (arc_pos[..., 2] <= high)).all(-1)
    else:
        environment = torch.ones(batch, count, dtype=torch.bool, device=device)

    keep = energy & escape & environment
    # Soft-crop cost P_ij (Eq. 31 soft penalty rho_ij): overload utilisation
    # a_req / a_max, the same normalised indicator the energy teacher (Eq. 47)
    # uses; lambda_crop scales it on the attention logits in the model.
    cost = (a_req / a_max).clamp(0., 4.)                            # P_ij
    # Safety net (D1): the cropper never empties the branch set. If every
    # candidate violates a hard constraint (e.g. a head-on, faster defender
    # keeps all branches closing), the cheapest branch survives so the block
    # degenerates to the constant-velocity candidate instead of disappearing.
    cheapest = cost[:, :count].argmin(-1, keepdim=True)
    keep = torch.where(keep.any(-1, keepdim=True), keep, keep.scatter(1, cheapest, True))

    def pad(values, fill):
        block = values.new_zeros(batch, steps, *values.shape[2:]) if fill == 0 else \
            torch.full((batch, steps, *values.shape[2:]), fill, dtype=values.dtype, device=device)
        block[:, :count] = values
        return block

    relative_position = displacement.clone()
    relative_velocity = candidate_velocity.clone()
    if cfg.relative_frame == "yaw":
        heading = torch.atan2(observed[:, -1, 4], observed[:, -1, 3])
        relative_position = rotate_xy(relative_position, heading)
        relative_velocity = rotate_xy(relative_velocity, heading)
    relative = torch.cat((relative_position, relative_velocity), -1)
    absolute = torch.cat((ahead, candidate_velocity), -1)
    return {"relative": pad(relative, 0.).to(dtype),
            "absolute": pad(absolute, 0.).to(dtype),
            "keep": pad(keep, False),
            "cost": pad(cost, 0.).to(dtype),
            # Pre-safety-net factors of Eq. (30), exposed for independent
            # auditing/testing of the three constraints (keep = all three, then
            # the never-empty safety net may revive the single cheapest branch).
            "escape": pad(escape, False),
            "environment": pad(environment, False),
            "energy": pad(energy, False),
            "count": count}
