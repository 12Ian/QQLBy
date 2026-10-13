"""Attention-level military teacher supervision (paper Eqs. 42-53).

Three factor-specific teacher score matrices over node pairs (i, j) of the
2O-node engagement graph, replacing the civilian smoothness/consistency/
locality teachers when cfg.military_teachers is enabled:

  H_E energy   (Eqs. 44-47): required normal acceleration a_req for the
               velocity transition i -> j, braking energy loss c_r, and the
               jerk magnitude, aggregated by a smooth LogSumExp of the
               normalized squared indicators (the most critical violation
               dominates). Higher score = energy-saving smooth transition.
  H_R risk     (Eqs. 48-49): per-node threat level aggregated from relative
               separation r, LOS rate rdot and the maneuverable reachable
               domain P_T,t via LSE, then aggregated again over the future
               window tau in [t, t+H] with the weighted log-sum-exp of
               Eq. (49), so the most dangerous moment inside the horizon
               dominates. Higher score = more threatened node, which attracts
               attention (the paper's HR collects threat information).
  H_T timeliness (Eqs. 50-53): corridor deviation (radial/normal) gated
               against terminal convergence by mu_t = sigmoid(kappa (t -
               T*)/T): early in the window attention biases corridor
               correction, near the end it biases terminal convergence.

All scores are computed under torch.no_grad from measured histories only
(Remark 4: teachers supervise training only, never inference inputs).
"""
import torch


def _pairwise_energy(observed, cfg, entity_steps):
    """Eqs. 42-47: energy teacher score s_E(i, j); higher = smoother."""
    batch, nodes, _ = observed.shape
    velocity = observed[..., 3:]
    speed = velocity.norm(dim=-1).clamp_min(1e-6)
    dt = cfg.dt
    time = torch.arange(nodes, device=observed.device) % entity_steps
    lag = (time[:, None] - time[None, :]).abs().clamp_min(1).float() * dt

    # a_req: normal acceleration needed to rotate v_i onto the direction of
    # the i->j transition (Eq. 45, arc-based approximation)
    cosine = (velocity[:, :, None] * velocity[:, None]).sum(-1) / (speed[:, :, None] * speed[:, None])
    angle = cosine.clamp(-1 + 1e-6, 1 - 1e-6).acos()
    a_req = speed[:, :, None] * angle / lag

    # c_r: braking (rearward) component of the required acceleration (Eq. 46)
    unit = velocity / speed[..., None]
    delta = (velocity[:, None] - velocity[:, :, None]) / lag[..., None]
    braking = -(delta * unit[:, :, None]).sum(-1)
    c_r = braking.clamp_min(0)

    # jerk: difference of the nodes' own historical accelerations (Eq. 44)
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
    acceleration = torch.cat((acceleration[:, :1], acceleration), 1)
    jerk = (acceleration[:, None] - acceleration[:, :, None]).norm(dim=-1)

    scale = cfg.engagement_overload_g * 9.81
    kappa = cfg.teacher_kappa
    indicators = torch.stack(((a_req / scale) ** 2,
                              (c_r / scale) ** 2,
                              (jerk / (scale / dt)) ** 2), -1)
    # s_E = -(1/kappa) log sum exp(kappa * indicator)  (Eq. 47)
    score = -torch.logsumexp(kappa * indicators, dim=-1) / kappa
    return score


def _threat_levels(observed, cfg, entity_steps):
    """Eqs. 48: per-node threat level from r, |rdot| and reachable domain."""
    batch, nodes, _ = observed.shape
    length = entity_steps
    position, velocity = observed[..., :3], observed[..., 3:]
    # partner of node k at the same time index (target <-> defender)
    partner = (torch.arange(nodes, device=observed.device) + length) % (2 * length)
    relative = position - position[:, partner]
    separation = relative.norm(dim=-1).clamp_min(1e-3)
    eL = relative / separation[..., None]
    # escape rate: positive rdot means the separation is growing (Eq. 48);
    # a closing target (rdot < 0) adds no break-through risk from its own side
    rdot = ((velocity - velocity[:, partner]) * eL).sum(-1).clamp_min(0)
    # reachable domain radius within the prediction horizon (Eq. 48 P_T,t):
    # speed * Tp + 0.5 * a_max * Tp^2, capped for normalization stability
    tp = cfg.prediction_steps * cfg.dt
    a_max = cfg.engagement_overload_g * 9.81
    reachable = (velocity.norm(dim=-1) * tp + .5 * a_max * tp ** 2).clamp_min(1e-3)

    r_ref = cfg.engagement_range_ref_m
    v_ref = cfg.max_speed_mps or 300.
    kappa = cfg.teacher_kappa
    factors = torch.stack((separation / r_ref,
                           rdot / v_ref,
                           reachable / r_ref), -1)
    threat = torch.logsumexp(kappa * factors, dim=-1) / kappa
    return threat


def _risk_score(observed, cfg, entity_steps):
    """Eq. 49: cross-temporal risk teacher; higher = more threatened.

        s^(R)_ij = (1/lambda_R) log sum_{tau=t}^{t+H} w_{j,tau} exp(lambda_R r_{j,tau})

    i.e. the threat levels r_{j,tau} of the *destination entity* j are
    aggregated by a weighted LogSumExp over the prediction window that starts
    at the source node's time index t.  The windowed LSE emphasises the most
    dangerous moment inside the horizon instead of averaging it away.
    """
    threat = _threat_levels(observed, cfg, entity_steps)          # [B, N]
    batch, nodes = threat.shape
    length = entity_steps
    entities = max(nodes // length, 1)
    device = observed.device
    # r_{j,tau}: threat of entity e_j at time tau  ->  [B, E, L]
    threat_entity = threat[:, :entities * length].view(batch, entities, length)

    # temporal window tau in [t, t + H] for every source time index t; the
    # horizon H is the prediction horizon expressed in sampling steps.
    steps = torch.arange(length, device=device)
    horizon_steps = int(min(max(cfg.prediction_steps, 1), max(length - 1, 1)))
    t = steps.float()
    inside = (steps[None, :] >= steps[:, None]) & (steps[None, :] <= steps[:, None] + horizon_steps)
    lag = (steps[None, :].float() - t[:, None]).clamp_min(0) * cfg.dt
    horizon_seconds = max(cfg.prediction_steps, 1) * cfg.dt
    # w_{j,tau}: cross-temporal aggregation weight (recent moments dominate).
    weight = torch.where(inside, torch.exp(-lag / horizon_seconds), torch.zeros((), device=device))
    log_weight = torch.log(weight.clamp_min(1e-12))

    lam = max(float(cfg.teacher_risk_lambda), 1e-6)
    # [B, E, L(t), L(tau)] -> log-sum-exp over tau
    terms = lam * threat_entity[:, :, None, :] + log_weight[None, None, :, :]
    windowed = torch.logsumexp(terms, dim=-1) / lam                  # [B, E, L]

    # s^(R)_ij with i = (e_i, t_i) and j = (e_j, .): depends on e_j and t_i.
    entity_of = (torch.arange(nodes, device=device) // length) % entities
    time_of = torch.arange(nodes, device=device) % length
    score = windowed[:, entity_of, :][:, :, time_of]                 # [B, j, i]
    return score.transpose(1, 2)                                     # [B, i, j]


def corridor_deviation(distance, half_width):
    """Eqs. (50)-(51): delta = (dist - b) / b, negative inside the corridor."""
    return (distance - half_width) / half_width


def _timeliness_score(observed, cfg, entity_steps):
    """Eqs. 50-53: corridor deviation gated toward terminal convergence."""
    batch, nodes, _ = observed.shape
    length = entity_steps
    position, velocity = observed[..., :3], observed[..., 3:]
    anchor = position[:, length - 1]                                # window end
    heading = velocity[:, length - 1]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    offset = position - anchor[:, None]                              # [B, N, 3]
    along = (offset * heading[:, None]).sum(-1)
    radial = (offset - along[..., None] * heading[:, None]).norm(dim=-1)
    normal = (offset - along[..., None] * heading[:, None])[..., 2].abs()
    # Corridor half-widths (Eqs. 50-51): the paper does not tabulate b_r/b_n,
    # so the default is 20% of the distance flown over the prediction horizon.
    tp = cfg.prediction_steps * cfg.dt
    travel = (velocity[:, :length].norm(dim=-1).mean(-1) * tp).clamp_min(1.0)   # [B]
    if cfg.teacher_corridor_radial_m:
        b_r = torch.full((batch, 1), float(cfg.teacher_corridor_radial_m),
                         dtype=observed.dtype, device=observed.device)
    else:
        b_r = (0.2 * travel)[:, None]
    if cfg.teacher_corridor_normal_m:
        b_n = torch.full((batch, 1), float(cfg.teacher_corridor_normal_m),
                         dtype=observed.dtype, device=observed.device)
    else:
        b_n = b_r
    delta_r = corridor_deviation(radial, b_r)
    delta_n = corridor_deviation(normal, b_n)

    # terminal convergence: distance to the extrapolated mission point and
    # remaining-time mismatch (Eq. 52 second term)
    speed_end = velocity[:, length - 1].norm(dim=-1).clamp_min(1e-6)
    target_point = anchor + heading * (speed_end * tp)[:, None]
    terminal_pos = (position - target_point[:, None]).norm(dim=-1) / (speed_end * tp)[:, None].clamp_min(1e-3)
    time = torch.arange(nodes, device=observed.device) % length
    remaining = (tp - time * cfg.dt).abs() / tp
    remaining = remaining[None].expand(batch, -1)
    kappa = cfg.teacher_kappa
    # No clamp here: Eqs. (50)-(52) feed the signed deviations straight into
    # the LSE, so a node inside the corridor scores a negative penalty.
    corridor = torch.logsumexp(kappa * torch.stack((delta_r, delta_n), -1), -1) / kappa
    terminal = torch.logsumexp(kappa * torch.stack(
        (terminal_pos, remaining), -1), -1) / kappa

    # gating mu_t (Eq. 53): early window -> corridor focus; late -> terminal.
    # The attended node j supplies the score, and its own time index gates it,
    # so the [B,1,N] term broadcasts over the source axis to [B,N,N].
    progress = (time.float() / max(length - 1, 1))                  # [N]
    mu = torch.sigmoid(kappa * (2 * progress - 1))                   # [N]
    gated = (1 - mu)[None, :] * corridor[:, None, :] + mu[None, :] * terminal[:, None, :]
    return -gated.expand(batch, nodes, nodes)


def military_teacher_scores(observed, cfg, entity_steps):
    """Return the (E, R, T) teacher score matrices [B, N, N]."""
    with torch.no_grad():
        energy = _pairwise_energy(observed, cfg, entity_steps)
        risk = _risk_score(observed, cfg, entity_steps)
        timeliness = _timeliness_score(observed, cfg, entity_steps)
    return [energy, risk, timeliness]
