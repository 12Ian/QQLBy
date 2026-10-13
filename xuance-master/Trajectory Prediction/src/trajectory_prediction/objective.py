import torch
from torch import nn
from torch.nn import functional as F


def displacement_errors(prediction, truth):
    """Per-sample ADE/FDE: Euclidean distances in metres, not squared MSE."""
    error = torch.linalg.vector_norm(prediction - truth, dim=-1)
    return error.mean(-1), error[:, -1]


def base_components(output, batch, cfg):
    prediction, observed = output["trajectory"], batch["observed"]
    if batch["future"].shape[1] == 0:
        # IV-A: classification only, there is no prediction horizon to score.
        ade = fde = torch.zeros(len(observed), device=observed.device, dtype=observed.dtype)
    else:
        ade, fde = displacement_errors(prediction, batch["future"])
    known = batch["label"] >= 0
    ce = (F.cross_entropy(output["logits"][known], batch["label"][known])
          if known.any() else output["logits"].sum() * 0)
    if prediction.shape[1] == 0:
        smoothness = torch.zeros((), device=observed.device, dtype=observed.dtype)
    else:
        position = torch.cat((observed[:, -1:, :3], prediction), dim=1)
        velocity = position.diff(dim=1) / cfg.dt
        velocity = torch.cat((observed[:, -1:, 3:], velocity), dim=1)
        acceleration = velocity.diff(dim=1) / cfg.dt
        smoothness = acceleration.square().mean()
    if cfg.stage == "intention":
        loss = ce
    else:
        loss = cfg.ade_weight * ade.mean() + cfg.fde_weight * fde.mean()
        loss = loss + cfg.smoothness_weight * smoothness
        if cfg.stage == "joint":
            loss = loss + cfg.intention_weight * ce
    return {"loss": loss, "ade_m": ade.mean(), "fde_m": fde.mean(),
            "cross_entropy": ce, "acceleration_mse": smoothness}


def _edge_subsets(nodes, entity_steps, cfg, device):
    """Eq. (36): the three disjoint edge subsets E_k of the interaction graph.

    Section III-B defines temporal edges (consecutive nodes of the same entity
    timeline), spatial edges (target and defender nodes at the same time index)
    and countermeasure edges (cross-entity nodes inside the interaction
    window).  Eq. (36) normalises pi^(k) over E_k(i) only, through the
    indicator 1(i -> j in E_k(i)) in front of the denominator sum.  Definition 1
    states that each head group is supervised "using its corresponding teacher
    score and its corresponding edge subset".  The paper does not tabulate the
    group -> subset mapping, so it is fixed here as H_E -> temporal (own-motion
    smoothness), H_R -> spatial (same-instant engagement geometry) and
    H_T -> countermeasure (decision D8).
    """
    index = torch.arange(nodes, device=device)
    entity = index // entity_steps
    time = index % entity_steps
    same = entity[:, None] == entity[None]
    lag = (time[:, None] - time[None, :]).abs()
    temporal = same & (lag > 0) & (lag <= cfg.temporal_radius)
    spatial = (~same) & (lag == 0)
    countermeasure = (~same) & (lag > 0) & (lag <= cfg.interaction_time_radius)
    return [temporal, spatial, countermeasure]


def attention_terms(attention, observed, cfg, node_valid=None, entity_steps=None):
    """Three factor teacher groups for attention supervision.

    military_teachers=False (civilian demo): smoothness, velocity consistency,
    time locality - signals that depend only on observed histories.
    military_teachers=True (paper Eqs. 42-53): energy E, risk R, timeliness T.
    """
    zero = observed.sum() * 0
    if not attention or not cfg.attention_supervision:
        return [zero, zero, zero], zero
    # entity_steps stays None for single-entity (civilian) callers; the military
    # teachers still need an explicit per-entity length, so fall back there only.
    if getattr(cfg, "military_teachers", False):
        from .military_teachers import military_teacher_scores
        teachers = military_teacher_scores(observed, cfg, entity_steps or observed.shape[1])
        # Eq. (36): the single temperature division happens in the loop below.
        # Dividing here as well would square it (tau^2 = 0.01 instead of 0.1).
    else:
        with torch.no_grad():
            velocity = observed[..., 3:].float()
            acceleration = velocity.diff(dim=1) / cfg.dt
            acceleration = torch.cat((acceleration[:, :1], acceleration), dim=1)
            # Per-entity reshaping needs both the entity length and the node mask
            # (single-entity civilian callers pass node_valid=None).
            if entity_steps is not None and node_valid is not None:
                shaped = velocity.view(len(velocity), -1, entity_steps, 3)
                availability = node_valid.view(len(velocity), -1, entity_steps)
                differences = shaped.diff(dim=2) / cfg.dt
                differences = differences * (availability[:, :, 1:] & availability[:, :, :-1])[..., None]
                acceleration = torch.cat((differences[:, :, :1], differences), 2).flatten(1, 2)
            acceleration_scale = cfg.max_acceleration_mps2 or 1.0
            speed_scale = cfg.max_speed_mps or 6.0
            smoothness = torch.cdist(acceleration, acceleration).square() / acceleration_scale ** 2
            consistency = torch.cdist(velocity, velocity).square() / speed_scale ** 2
            index = torch.arange(observed.shape[1], device=observed.device).float()
            if entity_steps is not None:
                index = index.remainder(entity_steps)
            locality = (index[:, None] - index[None]).abs() / max(1, (entity_steps or len(index)) - 1)
            teachers = [-smoothness, -consistency, -locality[None].expand(len(observed), -1, -1)]
    groups = [zero, zero, zero]
    separation = zero
    paper_regularizers = getattr(cfg, "teacher_regularizers", "legacy") == "paper"
    # Eq. (36) edge subsets: only meaningful once the graph carries several
    # entity blocks, i.e. when the caller supplies an explicit per-entity length.
    subsets = None
    if getattr(cfg, "teacher_edge_subsets", False) and entity_steps and entity_steps > 0 \
            and attention and attention[0].shape[-1] % entity_steps == 0 \
            and attention[0].shape[-1] // entity_steps >= 2:
        subsets = _edge_subsets(attention[0].shape[-1], entity_steps, cfg, observed.device)
    # Jitter (Eq. 39) compares adjacent time steps, so its row mask is the
    # intersection of consecutive availability flags rather than the node mask.
    valid_rows = None
    if node_valid is not None and node_valid.shape[-1] > 1:
        valid_rows = node_valid[:, :-1] & node_valid[:, 1:]

    def _time_sum(values, availability):
        """Eq. (37): sum over the temporal (node) axis, mean over batch/heads.

        L_attn_k = sum_t [ (1/|H_k|) sum_{h_k} DKL(...) + rho_k dH_k + zeta_k dbeta_k ],
        so the node axis is *summed*; only the head axis (|H_k|) and the batch
        are averaged.  Averaging over nodes as well would silently cancel the
        paper's sum over t.
        """
        if availability is None:
            return values.sum(-1).mean()
        mask = availability[:, None] if values.dim() > availability.dim() else availability
        return (values * mask.to(values.dtype)).sum(-1).mean()

    def _row_mean(values, availability):
        if availability is None:
            return values.mean()
        return (values * availability).sum() / availability.sum().clamp_min(1)

    for weights in attention:
        weights = weights.float()
        width = weights.shape[1] // 3
        for group in range(3):
            student = weights[:, group * width:(group + 1) * width]
            valid = student.detach() > 0
            if subsets is not None:
                # Eq. (36): restrict the teacher support to E_k(i).  Rows whose
                # subset is empty (e.g. no countermeasure neighbour at that
                # index) would leave pi^(k) undefined, so they fall back to the
                # unrestricted student support instead of producing NaNs.
                restricted = valid & subsets[group][None, None]
                fallback = ~restricted.any(-1, keepdim=True)
                valid = torch.where(fallback, valid, restricted)
            scores = teachers[group][:, None].expand_as(student) / cfg.teacher_temperature
            teacher = scores.masked_fill(~valid, -torch.inf).softmax(-1).detach()
            log_student = student.clamp_min(1e-12).log()
            def row_mean(values, availability=node_valid):
                # Eq. (37) reduction: sum over t, mean over batch and |H_k|.
                return _time_sum(values, availability)
            kl = row_mean((teacher * (teacher.clamp_min(1e-12).log() - log_student)).sum(-1))
            if paper_regularizers:
                # Eqs. (38)/(40): entropy of beta_bar, the distribution averaged
                # over the heads of the group, with a two-sided floor penalty.
                beta = student.mean(1)
                beta = beta / beta.sum(-1, keepdim=True).clamp_min(1e-12)
                entropy = -(beta * beta.clamp_min(1e-12).log()).sum(-1)
                floor = cfg.entropy_floor_fraction * valid.any(1).sum(-1).float().clamp_min(1).log()
                collapse = row_mean((floor - entropy).abs())
                # Eq. (39): jitter compares adjacent time steps of that mean.
                step = (beta[:, :-1] - beta[:, 1:]).abs().sum(-1)
                jitter = _time_sum(step, valid_rows)
            else:
                entropy = -(student * log_student).sum(-1)
                floor = cfg.entropy_floor_fraction * valid.sum(-1).float().clamp_min(1).log()
                collapse = row_mean((floor - entropy).clamp_min(0))
                previous, current = student[..., :-1, :-1], student[..., 1:, 1:]
                jitter_valid = None
                if entity_steps is not None:
                    entities = observed.shape[1] // entity_steps
                    matrix = student.view(len(observed), width, entities, entity_steps, entities, entity_steps)
                    previous = matrix[:, :, :, :-1, :, :-1].reshape(len(observed), width, -1, entities * (entity_steps - 1))
                    current = matrix[:, :, :, 1:, :, 1:].reshape_as(previous)
                    shaped_valid = node_valid.view(len(observed), entities, entity_steps)
                    jitter_valid = (shaped_valid[:, :, :-1] & shaped_valid[:, :, 1:]).flatten(1, 2)
                previous = previous / previous.sum(-1, keepdim=True).clamp_min(1e-12)
                current = current / current.sum(-1, keepdim=True).clamp_min(1e-12)
                jitter = row_mean((previous - current).abs().sum(-1), jitter_valid)
            groups[group] = groups[group] + kl + cfg.entropy_weight * collapse + cfg.jitter_weight * jitter
        separation_weights = weights if node_valid is None else weights * node_valid[:, None, :, None]
        flat = F.normalize(separation_weights.flatten(2), dim=-1, eps=1e-12)
        cosine = flat @ flat.transpose(1, 2)
        assignment = torch.arange(weights.shape[1], device=weights.device) // width
        separate = assignment[:, None] != assignment[None]
        # Eq. (41): eta * sum_k sum_{h_k in H_k} sum_{h_k' not in H_k} cos^2(...).
        # The triple sum is an ordered sum over all cross-group head pairs
        # (no normalisation), averaged over the batch afterwards.
        separation = separation + cosine[:, separate].square().sum(-1).mean()
    return [value / len(attention) for value in groups], separation / len(attention)


class TrajectoryObjective(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.log_variance = nn.Parameter(torch.zeros(4), requires_grad=cfg.loss_balance == "uncertainty")

    def forward(self, output, batch):
        cfg = self.cfg
        components = base_components(output, batch, cfg)
        groups, separation = attention_terms(output["attention"], output.get("graph_observed", batch["observed"]), cfg,
                                             output.get("node_valid"), output.get("entity_steps"))
        if cfg.stage != "intention":
            geometry = cfg.ade_weight * components["ade_m"] + cfg.fde_weight * components["fde_m"]
            if cfg.loss_balance == "uncertainty":
                # Eq. (55): L_tra = L_attn + 1/(2 sigma_sup^2) (w1 ADE + w2 FDE).
                # The supervised geometric term is *weighted* by its own
                # homoscedastic scale but carries no log-sigma regulariser
                # (that term belongs to Eq. 54, the attention groups only).
                # Index 3 of log_variance is sigma_sup; indices 0-2 are the
                # three attention groups H_E/H_R/H_T.
                factors = ([cfg.attention_weight * value for value in groups]
                           if cfg.attention_supervision else [])
                variances = self.log_variance[:len(factors)].clamp(-6, 6)
                supervision = self.log_variance[3].clamp(-6, 6)
                loss = (0.5 * torch.exp(-supervision) * geometry
                        + sum(0.5 * torch.exp(-variance) * factor + 0.5 * variance
                              for variance, factor in zip(variances, factors)))
                loss = loss + cfg.smoothness_weight * components["acceleration_mse"]
                if cfg.stage == "joint":
                    loss = loss + cfg.intention_weight * components["cross_entropy"]
            else:
                loss = components["loss"] + cfg.attention_weight * sum(groups)
            components["loss"] = loss + cfg.separation_weight * separation
        components.update(attention_smoothness=groups[0], attention_velocity=groups[1],
                          attention_temporal=groups[2], attention_separation=separation)
        return components


def loss_components(output, batch, cfg):
    # Stateless helper for numerical tests; training owns a persistent objective.
    return TrajectoryObjective(cfg).to(batch["observed"].device)(output, batch)
