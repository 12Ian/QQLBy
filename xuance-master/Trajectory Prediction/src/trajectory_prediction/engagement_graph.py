"""Engagement-descriptor edges for the military interaction graph.

Implements the edge feature mapping phi_edg of Eq. (15): every edge carries
an engagement descriptor composed of relative separation, LOS-relative speed
and its temporal variation, LOS direction and its temporal variation, and a
FOV visibility indicator. Three edge types are produced:

  * temporal edges  - consecutive nodes along the same entity timeline
                      (chronological consistency, civilian 7-dim geometry);
  * spatial edges   - target/defender nodes at the same time index
                      (instantaneous engagement coupling);
  * countermeasure  - cross-entity edges flagged by engagement activity
    (adversarial)    (visibility/closing) that carry the intercept-
                      engagement and evasion constraints used later by the
                      interactive distribution cropper (Eq. 22-25).

Edge layout (21 dims): [0:7] civilian geometry (normalized relative
position/velocity/time-lag); [7:18] engagement descriptor (r/r_ref, rdot/v_ref,
V_L/v_ref, dV_L/dt/a_ref, eL xyz, eLdot/omega_ref xyz, FOV visibility);
[18:21] edge-type one-hot (temporal, spatial, countermeasure).

Node layout matches model.py with context_entities=1: the first O nodes are
the target, the following O nodes the defender, and (when the cropper of
decision D1 is enabled) a trailing O-node block of candidate branches.  The
candidate block is NOT a third time-indexed entity: its rows index alternative
future branches, so per decision D1 (G=1) every surviving branch converges on
the observation-END nodes (terminal target and terminal defender) and carries
no temporal or same-time-index spatial edges.  All history quantities are
derived on the fly from the measured histories (no future information).
"""
import torch


ENGAGEMENT_EDGE_DIM = 21  # 7 civilian geometry + 11 engagement descriptor + 3 type one-hot
# The 11 engagement dims are the components that Section III-B assigns to
# phi_edg: relative separation, LOS-relative speed and its temporal variation,
# LOS direction and its temporal variation, and a visibility indicator --
# [r, rdot, V_L, V_Ldot, eL(3), eLdot(3), visible].


def engagement_edges(features, valid, cfg, progress=1.):
    """Build engagement-descriptor edges for the multi-entity interaction graph.

    features: [B, k*O, 6] physical (relative-frame) states; entity blocks of O
              nodes each (target, defender, and optionally the candidate
              branch block added by the cropper, decision D1).
    valid:    [B, k*O] boolean node availability (measurement / crop mask).
    Returns (edges [B, k*O, k*O, 21], allowed [B, k*O, k*O]).
    """
    batch, nodes, _ = features.shape
    length = cfg.observation_steps
    if nodes % length or nodes // length < 2:
        raise ValueError(f"Engagement graph expects whole entity blocks of O={length} nodes, got {nodes}")
    device = features.device
    index = torch.arange(nodes, device=device)
    time, entity = index % length, index // length
    same = entity[:, None] == entity[None]
    lag_steps = (time[:, None] - time[None, :]).abs()

    position = features[:, :, None, :3] - features[:, None, :, :3]   # p_i - p_j
    velocity = features[:, :, None, 3:] - features[:, None, :, 3:]   # v_i - v_j

    # ---- civilian geometry block (7 dims, scale as in interaction_graph) ----
    speed_ref = cfg.max_speed_mps or 300.
    span = speed_ref * max(length - 1, 1) * cfg.dt
    geo = torch.cat((position / span, velocity / speed_ref,
                     (lag_steps * cfg.dt / (max(length - 1, 1) * cfg.dt)).to(features.dtype)
                     [None, :, :, None].expand(batch, -1, -1, -1)), -1)

    # ---- engagement descriptor block (9 dims) -------------------------------
    separation = position.norm(dim=-1).clamp_min(1e-3)               # r_ij
    eL = position / separation[..., None]                            # LOS dir j->i
    rdot = (velocity * eL).sum(-1)                                   # LOS rate
    # eLdot = [eL x v_rel] x eL / r   (component of relative velocity
    # perpendicular to the LOS, divided by the separation)
    cross = torch.cross(eL, velocity, dim=-1)
    eLdot = torch.cross(cross, eL, dim=-1) / separation[..., None]
    # visibility: target node i sees the defender node j iff the direction
    # toward j lies within theta_FOV of the target velocity at node i
    v_i = features[:, :, None, 3:]                                  # [B,N,1,3]
    toward = -eL                                                    # i -> j
    cosine = (v_i * toward).sum(-1) / v_i.norm(dim=-1).clamp_min(1e-6)
    half_angle = float(getattr(cfg, "fov_half_angle_deg", 60.))
    visible = (cosine >= float(torch.cos(torch.deg2rad(torch.tensor(
        half_angle))))).to(features.dtype)
    r_ref = getattr(cfg, "engagement_range_ref_m", 8000.)
    v_ref = speed_ref
    omega_ref = v_ref / r_ref
    # ---- Eq. (2) V_L and its temporal variation ----------------------------
    # phi_edg also carries "LOS-relative speed and its temporal variation".
    # Both are properties of the target-defender *pairing* (the same time index
    # of entity blocks 0 and 1), not of an arbitrary node pair, so they are
    # evaluated on that pairing and broadcast over the destination axis of each
    # source node.  V_L = V_M e_M . e_L - V_T e_T . e_L, i.e. the negative LOS
    # rate for the LOS pointing from the defender toward the target.
    entities = nodes // length
    hist = entity < 2                                              # target/defender only
    hist_pair = (hist[:, None] & hist[None, :])[None]              # [1,N,N]
    v_l = torch.zeros(batch, nodes, device=device, dtype=features.dtype)
    if entities >= 2:
        pair = features[:, :length] - features[:, length:2 * length]
        pair_sep = pair[..., :3].norm(dim=-1).clamp_min(1e-3)
        pair_eL = pair[..., :3] / pair_sep[..., None]
        v_l_pair = -(pair[..., 3:] * pair_eL).sum(-1)          # Eq. (2) V_L
        v_l[:, :length] = v_l_pair
        v_l[:, length:2 * length] = v_l_pair                   # shared by both sides
    if entities >= 3:
        # Candidate branches are evaluated at the current engagement instant,
        # so they inherit the terminal (current) pairing V_L; its future time
        # derivative is unknown and stays zero.
        v_l[:, 2 * length:] = v_l_pair[:, -1:].to(v_l.dtype)
    # Backward difference inside every entity block gives dV_L/dt.
    v_l_dot = torch.zeros_like(v_l)
    blocks = v_l.view(batch, entities, length)
    blocks_dot = v_l_dot.view(batch, entities, length)
    blocks_dot[:, :, 1:] = (blocks[:, :, 1:] - blocks[:, :, :-1]) / cfg.dt
    a_ref = getattr(cfg, "engagement_overload_g", 9.0) * 9.81
    engagement = torch.cat((separation[..., None] / r_ref,
                            rdot[..., None] / v_ref,
                            v_l[:, :, None, None].expand(batch, nodes, nodes, 1) / v_ref,
                            v_l_dot[:, :, None, None].expand(batch, nodes, nodes, 1) / a_ref,
                            eL, eLdot / omega_ref,
                            visible[..., None]), -1)

    # ---- edge types (one-hot 3) ---------------------------------------------
    # Generic time/space/adversarial edges are defined only among the two
    # historical entities.  Candidate rows index alternative futures (not time),
    # so they are excluded from time-indexed typing and attached separately.
    temporal = (hist_pair & same[None].expand(batch, -1, -1)
                & (lag_steps <= cfg.temporal_radius)[None] & (lag_steps > 0)[None])
    spatial = hist_pair & (~same)[None].expand(batch, -1, -1) & (lag_steps == 0)[None]
    # countermeasure/adversarial: cross-entity pairs with engagement activity
    # (defender within the engagement range and either visible or closing),
    # any time offset inside the interaction window; strictly ordered after
    # temporal/spatial so the three type flags stay mutually exclusive
    active = (separation <= r_ref) & ((visible > .5) | (rdot < 0))
    counter_hist = (hist_pair & (~same)[None].expand(batch, -1, -1)
                    & (lag_steps <= cfg.interaction_time_radius)[None]
                    & (lag_steps > 0)[None] & active)
    # Decision D1 (G=1): every surviving candidate branch is linked (both ways)
    # to the observation-END target and defender nodes only.  This keeps the
    # futures attached to the state they branch from, instead of mapping branch
    # index a onto observation time a (which would wire them to the oldest past
    # whenever A < O).
    candidate_link = torch.zeros(batch, nodes, nodes, dtype=torch.bool, device=device)
    if entities >= 3:
        cand_idx = torch.nonzero(entity == 2, as_tuple=False).flatten()
        for terminal in (length - 1, 2 * length - 1):
            candidate_link[:, terminal, cand_idx] = True
            candidate_link[:, cand_idx, terminal] = True
    countermeasure = counter_hist | candidate_link
    edge_type = torch.stack((temporal, spatial, countermeasure.to(temporal.dtype)), -1)

    edges = torch.cat((geo, engagement, edge_type), -1)

    # ---- allowed connectivity -----------------------------------------------
    speed_cap = cfg.max_speed_mps
    plausible = features[..., 3:].norm(dim=-1) <= speed_cap if speed_cap else None
    allowed = temporal | spatial | counter_hist
    if getattr(cfg, "overload_reachability", False):
        # IV-B: "The interactive graph dynamically reconstructs itself and
        # imposes hard blocking on links deemed unreachable."  A cross-entity
        # link (i, j) is unreachable when rotating the relative velocity onto
        # the LOS-of-pair direction within the available time lag would need a
        # normal overload beyond the target limit aT_max: a_req = |v_rel| *
        # theta / lag, with theta the angle between v_rel and the pair LOS.
        # Same-entity temporal links are always kinematically consistent, and
        # lag = 0 links carry no rotation demand, so both stay unblocked; the
        # candidate branches already pass the Eq. (24) energy gate and are exempt.
        overload = getattr(cfg, "engagement_overload_g", 9.0) * 9.81
        lag_seconds = (lag_steps * cfg.dt).clamp_min(cfg.dt).to(features.dtype)
        speed_rel = velocity.norm(dim=-1)                                # [B,N,N]
        cosine = (velocity * eL).sum(-1) / (speed_rel * separation).clamp_min(1e-6)
        theta = cosine.clamp(-1 + 1e-6, 1 - 1e-6).acos()
        a_req = speed_rel * theta / lag_seconds
        unreachable = (hist_pair & (~same)[None].expand(batch, -1, -1)
                       & (lag_steps > 0)[None] & (a_req > overload))
        allowed = allowed & ~unreachable
    if getattr(cfg, "fov_attention_mask", False):
        # Paper IV-B: "the FOV mask is enabled during interaction".  When the
        # defender is outside the target FOV the cross-entity (engagement)
        # coupling is removed, so the predictor degenerates to plain
        # extrapolation; only in-entity temporal edges survive.  The target's
        # own candidate futures are not defender sightings and stay exempt.
        allowed = allowed & (same[None].expand(batch, -1, -1) | (visible > .5) | candidate_link)
    allowed = allowed & valid[:, :, None] & valid[:, None]
    if plausible is not None:
        allowed = allowed & plausible[:, None]
    if entities >= 3:
        # Re-add the D1 candidate-terminal links after the history-only filters,
        # still dropping cropped/padding branches through the validity mask.
        link = candidate_link & valid[:, :, None] & valid[:, None]
        if plausible is not None:
            link = link & plausible[:, None] & plausible[:, :, None]
        allowed = allowed | link
    allowed = allowed | torch.eye(nodes, dtype=torch.bool, device=device)[None]
    return edges, allowed
