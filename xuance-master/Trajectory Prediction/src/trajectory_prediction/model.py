"""Generic BiLSTM intention recognition and temporal graph forecasting."""

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_pruning import anchor_directions, candidate_branch
from .data import relative_features, relative_context, rotate_xy
from .engagement_graph import ENGAGEMENT_EDGE_DIM, engagement_edges

# Per-step engagement descriptors stored by the military generator:
# [r/1e3, rdot/100, eL(3), eLdot(3), visible] (engagement_scenarios.ENGAGEMENT_DIM).
ENGAGEMENT_STATE_DIM = 9
from .history_graph import history_edges
from .interaction_graph import interaction_edges


class IntentionEncoder(nn.Module):
    def __init__(self, cfg, static_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(6, cfg.lstm_hidden, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(2 * cfg.lstm_hidden)
        dropout = cfg.dropout if cfg.intention_dropout is None else cfg.intention_dropout
        self.project = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(2 * cfg.lstm_hidden, cfg.projection_dim),
                                     nn.ReLU(), nn.Dropout(dropout))
        self.static = (nn.Sequential(nn.Linear(static_dim, cfg.projection_dim), nn.ReLU())
                       if cfg.static_state_branch else None)
        # Eq. (14): the classification head consumes z = [h_t; h_s] directly -
        # no extra fusion layer before it (W_h maps the splice to width m).
        latent = 2 * cfg.projection_dim if cfg.static_state_branch else cfg.projection_dim
        self.classifier = nn.Sequential(nn.Linear(latent, cfg.projection_dim),
                                        nn.ReLU(), nn.Linear(cfg.projection_dim, cfg.intention_classes))

    def forward(self, features, static_features=None):
        _, (hidden, _) = self.lstm(features)
        # Both directions summarize the entire observed window (no future data).
        # Eq. (12): h_t = ReLU(W_t BiLSTM(x) + b_t).
        context = self.project(self.norm(torch.cat((hidden[-2], hidden[-1]), dim=-1)))
        if self.static is not None:
            # Eq. (13)/(14): h_s = ReLU(W_s s_M + b_s), z = [h_t; h_s].
            context = torch.cat((context, self.static(
                features[:, -1] if static_features is None else static_features)), dim=-1)
        # The same z seeds the graph nodes through Eq. (16) h^(0) = [g; [h_t; h_s; p]].
        return self.classifier(context), context


class TemporalGraphAttention(nn.Module):
    def __init__(self, dimension, heads, radius, dropout, head_dim=0, top_k=0, temperature=1.0,
                 edge_features=False, edge_dim=7, node_context_dim=0):
        super().__init__()
        self.heads, self.width, self.radius = heads, head_dim or dimension // heads, radius
        self.top_k, self.temperature = top_k, temperature
        self.projection = nn.Linear(dimension, heads * self.width, bias=False)
        self.source = nn.Parameter(torch.empty(heads, self.width))
        self.destination = nn.Parameter(torch.empty(heads, self.width))
        self.edge_gate = nn.Linear(2 * dimension, heads)
        self.node_gate = nn.Linear(dimension, heads)
        self.edge_context = nn.Linear(edge_dim, heads, bias=False) if edge_features else None
        # Eq. (20): the nodewise soft mask is m_j = sigmoid(f_n(h_j, u)), so the
        # node gate takes the LOS engagement descriptor u as an explicit input,
        # just as the edge gate f_e(h_i, h_j, u) of Eq. (19) does.  Absent on the
        # civilian graph, where the paper defines no engagement descriptor.
        self.node_context = nn.Linear(node_context_dim, heads, bias=False) if node_context_dim else None
        self.output = nn.Linear(heads * self.width, dimension)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.destination)

    def forward(self, nodes, feasible=None, edges=None, reachable=None, crop_progress=1.0, allowed=None,
                penalty=None, node_u=None):
        batch, length, dim = nodes.shape
        projected = self.projection(nodes).view(batch, length, self.heads, self.width).transpose(1, 2)
        source = (projected * self.source[None, :, None]).sum(-1)
        destination = (projected * self.destination[None, :, None]).sum(-1)
        scores = F.leaky_relu(source.unsqueeze(-1) + destination.unsqueeze(-2), 0.2)
        # log(sigmoid) evaluates equation-like multiplicative gates stably.
        # Factor the linear concatenation to avoid allocating [B,T,T,2D].
        left = F.linear(nodes, self.edge_gate.weight[:, :dim], self.edge_gate.bias)
        right = F.linear(nodes, self.edge_gate.weight[:, dim:])
        edge_logits = left.unsqueeze(2) + right.unsqueeze(1)
        if self.edge_context is not None:
            # Eq. (19): f_e(h_i, h_j, u) - the edge gate also reads the pairwise
            # engagement descriptor u carried by phi_edg.
            edge_logits = edge_logits + self.edge_context(edges)
        scores = scores / self.temperature + crop_progress * F.logsigmoid(edge_logits).permute(0, 3, 1, 2)
        node_logits = self.node_gate(nodes)
        if self.node_context is not None and node_u is not None:
            # Eq. (20): f_n(h_j, u) - the node gate additionally reads the
            # per-node LOS engagement descriptor u before attention normalization.
            node_logits = node_logits + self.node_context(node_u.to(node_logits.dtype))
        scores = scores + crop_progress * F.logsigmoid(node_logits).transpose(1, 2).unsqueeze(-2)
        if penalty is not None:
            # Eq. (25) soft crop: alpha_ij ~ exp(-Lambda_crop * P_ij), applied to
            # the destination axis and ramped in by the same crop schedule.
            scores = scores - crop_progress * penalty[:, None, None, :].to(scores.dtype)
        index = torch.arange(length, device=nodes.device)
        neighbors = (index[:, None] - index[None, :]).abs() <= self.radius
        neighbors = neighbors[None, None].expand(batch, self.heads, -1, -1)
        if allowed is not None:
            neighbors = allowed[:, None].expand(batch, self.heads, -1, -1)
        if reachable is not None:
            neighbors = neighbors & reachable[:, None]
        if feasible is not None:
            neighbors = neighbors & feasible[:, None, None, :]
            # Keep self messages if all physically plausible neighbors disappear.
            neighbors = neighbors | torch.eye(length, dtype=torch.bool, device=nodes.device)[None, None]
        scores = scores.masked_fill(~neighbors, -torch.inf)
        top_k = self.top_k
        if top_k:
            import math
            top_k = math.ceil(length - (length - min(top_k, length)) * crop_progress)
        if top_k and top_k < length:
            # Keep the self edge plus up to K-1 learned neighbors.
            diagonal = torch.eye(length, dtype=torch.bool, device=nodes.device)[None, None]
            ranking = scores.masked_fill(diagonal, torch.inf)
            selected = ranking.topk(top_k, dim=-1).indices
            keep = torch.zeros_like(neighbors).scatter_(-1, selected, True)
            scores = scores.masked_fill(~keep, -torch.inf)
        # Self edges guarantee at least one valid neighbor, including radius=0.
        attention = torch.softmax(scores, dim=-1)
        aggregated = torch.matmul(attention, projected).transpose(1, 2).reshape(batch, length, self.heads * self.width)
        return self.norm(nodes + self.dropout(self.output(F.elu(aggregated)))), attention


class Forecaster(nn.Module):
    supports_intention = True
    def __init__(self, cfg, normalizer=None):
        super().__init__()
        self.cfg = cfg
        normalizer = normalizer or {"mean": torch.zeros(6), "std": torch.ones(6)}
        self.register_buffer("feature_mean", normalizer["mean"].clone())
        self.register_buffer("feature_std", normalizer["std"].clone())
        if cfg.context_entities:
            # One embedding per entity block: target, peers, and (when cropping is
            # enabled) the candidate branch block.
            self.entity_embedding = nn.Parameter(torch.randn(
                1 + cfg.context_entities + (1 if cfg.candidate_pruning else 0), cfg.graph_dim) * .02)
        if cfg.candidate_pruning:
            self.register_buffer("candidate_anchors", anchor_directions(cfg.candidate_anchors))
        use_engagement_edges = bool(cfg.context_entities and cfg.engagement_edges)
        # Descriptors enter the intention query (static branch) and every
        # temporal node feature, not just the pairwise edges (Eq. 15, 42-53).
        self.engagement_state = use_engagement_edges
        self.engagement_norm = nn.LayerNorm(ENGAGEMENT_STATE_DIM) if use_engagement_edges else None
        # Eq. (16): h^(0) = [g^(0); [h_t; h_s; p_hat]], so the intention splice
        # contributes 2m channels whenever the static state branch is enabled.
        self.intention_context_dim = cfg.projection_dim * (2 if cfg.static_state_branch else 1)
        self.node_embedding = nn.Linear(6 + self.intention_context_dim + cfg.intention_classes
                                        + (ENGAGEMENT_STATE_DIM if use_engagement_edges else 0),
                                        cfg.graph_dim)
        self.intention = IntentionEncoder(cfg, static_dim=6 + (ENGAGEMENT_STATE_DIM if use_engagement_edges else 0))
        self.graph = nn.ModuleList([TemporalGraphAttention(cfg.graph_dim, cfg.heads,
                                                          cfg.temporal_radius, cfg.dropout,
                                                          cfg.graph_head_dim, cfg.graph_top_k,
                                                          cfg.attention_temperature,
                                                          cfg.historical_edge_features or use_engagement_edges,
                                                          edge_dim=(ENGAGEMENT_EDGE_DIM if use_engagement_edges else 7),
                                                          node_context_dim=(ENGAGEMENT_STATE_DIM if use_engagement_edges else 0))
                                    for _ in range(cfg.graph_layers)])
        if cfg.crop_warmup_epochs:
            self.register_buffer("crop_progress", torch.tensor(0.0))
        self.time_embedding = nn.Parameter(torch.empty(cfg.prediction_steps, cfg.graph_dim))
        nn.init.normal_(self.time_embedding, std=0.02)
        if cfg.decoder_type == "parallel":
            self.causal_attention = nn.MultiheadAttention(cfg.graph_dim, cfg.heads, cfg.dropout, batch_first=True)
        else:
            self.history_embedding = nn.Parameter(torch.empty(cfg.observation_steps, cfg.graph_dim))
            nn.init.normal_(self.history_embedding, std=0.02)
            self.temporal_memory = (nn.TransformerEncoder(nn.TransformerEncoderLayer(
                cfg.graph_dim, cfg.temporal_heads, 4 * cfg.graph_dim, cfg.dropout, batch_first=True),
                cfg.temporal_layers, enable_nested_tensor=False) if cfg.temporal_layers else nn.Identity())
            self.gru_input = nn.Linear(cfg.graph_dim + 3, cfg.graph_dim)
            self.gru = nn.GRUCell(cfg.graph_dim, cfg.graph_dim)
            self.cross_attention = nn.MultiheadAttention(cfg.graph_dim, cfg.temporal_heads,
                                                         cfg.dropout, batch_first=True)
        self.decoder_norm = nn.LayerNorm(cfg.graph_dim)
        # Eq. (32) writes the decoder generically as an MLP, but IV-B pins the
        # instantiated model: "The linear regression is employed for the output
        # head."  A single linear map is that case, so it is the default; set
        # decoder_head="mlp" for the generic Eq. (32) reading.
        if cfg.decoder_head == "linear":
            self.decoder = nn.Sequential(nn.Linear(cfg.graph_dim, 3))
        else:
            self.decoder = nn.Sequential(nn.Linear(cfg.graph_dim, cfg.graph_dim), nn.ReLU(),
                                         nn.Linear(cfg.graph_dim, 3))
        # Start at the interpretable constant-velocity baseline.  The paper
        # specifies no initialisation, so PyTorch defaults are the literal
        # reading; the zero init is an engineering choice that decouples the
        # first optimisation steps from the graph parameters.  Flip
        # decoder_zero_init=False to disable it (audit E1).
        if getattr(cfg, "decoder_zero_init", True):
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)
        if cfg.freeze_intention:
            for parameter in self.intention.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_intention:
            self.intention.eval()
        return self

    def set_crop_progress(self, value):
        if self.cfg.crop_warmup_epochs:
            self.crop_progress.fill_(min(1.0, max(0.0, value)))

    def forward(self, observed, context=None, context_mask=None, engagement=None):
        if observed.ndim != 3 or observed.shape[1:] != (self.cfg.observation_steps, 6):
            raise ValueError(f"Expected [batch,{self.cfg.observation_steps},6] observed state")
        if engagement is not None:
            if engagement.shape[:2] != observed.shape[:2] or engagement.shape[-1] != ENGAGEMENT_STATE_DIM:
                raise ValueError(f"Expected engagement descriptors [batch,{self.cfg.observation_steps},{ENGAGEMENT_STATE_DIM}]")
            engagement = self.engagement_norm(engagement.to(observed.dtype))
        elif self.engagement_state:
            # Rolling inference without stored descriptors: neutral input, the
            # geometry still reaches the model through the engagement edges.
            engagement = torch.zeros(len(observed), self.cfg.observation_steps,
                                     ENGAGEMENT_STATE_DIM, dtype=observed.dtype, device=observed.device)
        physical_features = relative_features(observed, self.cfg.relative_frame)
        features = (physical_features - self.feature_mean) / self.feature_std
        peer_features, static_features = None, None
        node_valid = torch.ones(observed.shape[:2], dtype=torch.bool, device=observed.device)
        graph_observed = observed
        candidate = None
        if self.cfg.context_entities:
            static_features = features.new_zeros(len(observed), 6)
            if self.cfg.use_interaction_context:
                expected = (len(observed), self.cfg.context_entities, self.cfg.observation_steps, 6)
                if context is None or tuple(context.shape) != expected:
                    raise ValueError(f"Need synchronized context with shape {expected}")
                if context_mask is None:
                    context_mask = torch.ones(expected[:-1], dtype=torch.bool, device=observed.device)
                if tuple(context_mask.shape) != expected[:-1] or context_mask.dtype != torch.bool:
                    raise ValueError("Context availability must be a boolean [B,N,O] mask")
                measured = context.masked_fill(~context_mask[..., None], 0)
                physical_peer = relative_context(observed, measured, self.cfg.relative_frame).masked_fill(~context_mask[..., None], 0)
                peer_features = ((physical_peer - self.feature_mean) / self.feature_std).masked_fill(~context_mask[..., None], 0)
                weights = context_mask[:, :, -1:].to(features.dtype)
                static_features = (peer_features[:, :, -1] * weights).sum(1) / weights.sum(1).clamp_min(1)
                physical_features = torch.cat((physical_features, physical_peer.flatten(1, 2)), 1)
                graph_observed = torch.cat((observed, measured.flatten(1, 2)), 1)
                node_valid = torch.cat((node_valid, context_mask.flatten(1, 2)), 1)
                # ---- candidate future branches (Table I, decision D1) -------
                if self.cfg.candidate_pruning:
                    candidate = candidate_branch(observed, context, context_mask, self.cfg,
                                                 self.candidate_anchors)
                    physical_features = torch.cat((physical_features, candidate["relative"]), 1)
                    graph_observed = torch.cat((graph_observed, candidate["absolute"]), 1)
                    # Cropped candidates leave the graph: no edges, no teacher term.
                    node_valid = torch.cat((node_valid, candidate["keep"]), 1)
        if engagement is not None:
            # Q_int: the current engagement geometry (r, rdot, LOS rates,
            # visibility) conditions intention recognition alongside the state.
            tail = static_features if static_features is not None else features[:, -1]
            static_features = torch.cat((tail, engagement[:, -1]), dim=-1)
        logits, intention_context = self.intention(features, static_features)
        probabilities = logits.softmax(-1)
        if self.cfg.stage == "intention":
            return {"logits": logits, "probabilities": probabilities,
                    "trajectory": constant_velocity(observed, self.cfg.prediction_steps, self.cfg.dt),
                    "attention": []}
        if not self.cfg.use_intention:
            intention_context = torch.zeros_like(intention_context)
            probabilities = torch.full_like(probabilities, 1 / self.cfg.intention_classes)
        if peer_features is not None:
            features = torch.cat((features, peer_features.flatten(1, 2)), 1)
        if candidate is not None:
            candidate_features = ((candidate["relative"] - self.feature_mean) / self.feature_std
                                  ).masked_fill(~candidate["keep"][..., None], 0)
            features = torch.cat((features, candidate_features), 1)
        length = features.shape[1]
        if engagement is not None:
            # Q_tra: descriptors are a property of the engagement (shared by
            # both aircraft), so they are broadcast over every entity block.
            features = torch.cat((features, engagement.repeat(1, length // self.cfg.observation_steps, 1)), dim=-1)
        nodes = self.node_embedding(torch.cat((features,
                    intention_context[:, None].expand(-1, length, -1),
                    probabilities[:, None].expand(-1, length, -1)), dim=-1))
        if self.cfg.context_entities:
            nodes = nodes + self.entity_embedding[:length // self.cfg.observation_steps].repeat_interleave(self.cfg.observation_steps, 0)[None]
        attention = []
        progress = float(self.crop_progress) if self.cfg.crop_warmup_epochs else 1.0
        allowed = None
        edges, reachable = (history_edges(physical_features, self.cfg, progress)
                            if not self.cfg.context_entities and (self.cfg.historical_edge_features or self.cfg.historical_reachability) else (None, None))
        if self.cfg.context_entities:
            if self.cfg.engagement_edges:
                edges, allowed = engagement_edges(physical_features, node_valid, self.cfg, progress)
            else:
                edges, allowed = interaction_edges(physical_features, node_valid, self.cfg, progress)
            reachable = None
        feasible = None
        penalty = None
        if self.cfg.max_speed_mps and not self.cfg.context_entities:
            feasible = torch.linalg.vector_norm(observed[..., 3:], dim=-1) <= self.cfg.max_speed_mps
        if candidate is not None:
            history_length = length - self.cfg.observation_steps
            history = (torch.ones(len(observed), history_length, dtype=torch.bool, device=observed.device)
                       if feasible is None else feasible)
            # m(j) gates attention to the candidate; P_ij drives the soft crop.
            feasible = torch.cat((history, candidate["keep"]), 1)
            penalty = torch.cat((torch.zeros(len(observed), history_length, dtype=features.dtype,
                                             device=observed.device),
                                 candidate["cost"] * self.cfg.candidate_crop_lambda), 1)
        # Per-node engagement descriptor u feeding the Eq. (20) node gate.  It is
        # a property of the engagement at each time step and is shared by the
        # target and defender nodes; candidate branches are evaluated "now", so
        # they all receive the terminal (current) descriptor.
        node_u = None
        if engagement is not None and self.cfg.context_entities:
            obs_steps = self.cfg.observation_steps
            node_blocks = [engagement, engagement]
            if candidate is not None:
                node_blocks.append(engagement[:, -1:].expand(-1, obs_steps, -1))
            node_u = torch.cat(node_blocks, 1)
        for layer in self.graph:
            nodes, weights = layer(nodes, feasible, edges, reachable, progress, allowed, penalty,
                                   node_u=node_u)
            attention.append(weights)
        velocity = physical_features[:, self.cfg.observation_steps - 1, 3:]
        if self.cfg.decoder_type == "parallel":
            future = nodes.mean(1)[:, None] + self.time_embedding[None]
            causal = torch.ones(self.cfg.prediction_steps, self.cfg.prediction_steps,
                                dtype=torch.bool, device=observed.device).triu(1)
            message, _ = self.causal_attention(future, future, future, attn_mask=causal, need_weights=False)
            delta = self.decoder(self.decoder_norm(future + message)) * self.cfg.dt
            increments = velocity[:, None] * self.cfg.dt + delta
        else:
            position_embedding = self.history_embedding.repeat(length // self.cfg.observation_steps, 1)
            padding = ~node_valid if self.cfg.context_entities else None
            temporal_input = nodes + position_embedding[None]
            memory = (self.temporal_memory(temporal_input, src_key_padding_mask=padding)
                      if self.cfg.temporal_layers else self.temporal_memory(temporal_input))
            hidden = memory[:, :self.cfg.observation_steps].mean(1)
            previous = velocity * self.cfg.dt
            steps = []
            for step in range(self.cfg.prediction_steps):
                token = torch.cat((previous, self.time_embedding[step][None].expand(len(nodes), -1)), dim=-1)
                hidden = self.gru(self.gru_input(token), hidden)
                message, _ = self.cross_attention(hidden[:, None], memory, memory, key_padding_mask=padding, need_weights=False)
                previous = velocity * self.cfg.dt + self.decoder(self.decoder_norm(hidden + message[:, 0])) * self.cfg.dt
                steps.append(previous)
            increments = torch.stack(steps, dim=1)
        increments = bound_increments(increments, velocity, self.cfg)
        if self.cfg.relative_frame == "yaw":
            increments = rotate_xy(increments, torch.atan2(observed[:, -1, 4], observed[:, -1, 3]), inverse=True)
        trajectory = observed[:, -1:, :3] + increments.cumsum(dim=1)
        result = {"trajectory": trajectory, "logits": logits,
                  "probabilities": probabilities, "attention": attention}
        if self.cfg.context_entities:
            result.update(graph_observed=graph_observed, node_valid=node_valid, entity_steps=self.cfg.observation_steps)
        return result


def bound_increments(increments, velocity, cfg):
    """Optional civilian speed/acceleration projection, with differentiable norms."""
    if not cfg.max_speed_mps and not cfg.max_acceleration_mps2:
        return increments
    if cfg.max_speed_mps and cfg.max_acceleration_mps2 and bool(
            (torch.linalg.vector_norm(velocity.detach(), dim=-1) > cfg.max_speed_mps + 1e-5).any()):
        raise ValueError("Last observed speed exceeds the configured speed bound; adjust the civilian bounds")
    steps = []
    previous = velocity
    for increment in increments.unbind(dim=1):
        desired = increment / cfg.dt
        if cfg.max_acceleration_mps2:
            change = desired - previous
            factor = (cfg.max_acceleration_mps2 * cfg.dt / torch.linalg.vector_norm(change, dim=-1,
                       keepdim=True).clamp_min(1e-8)).clamp_max(1)
            desired = previous + change * factor
        if cfg.max_speed_mps:
            factor = (cfg.max_speed_mps / torch.linalg.vector_norm(desired, dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
            desired = desired * factor
        steps.append(desired * cfg.dt)
        previous = desired
    return torch.stack(steps, dim=1)


def build_forecaster(cfg, normalizer=None):
    if cfg.model_type == "graph":
        return Forecaster(cfg, normalizer)
    from .baselines import SequenceForecaster
    return SequenceForecaster(cfg, normalizer)


def constant_velocity(observed, horizon, dt):
    time = torch.arange(1, horizon + 1, device=observed.device, dtype=observed.dtype) * dt
    return observed[:, -1:, :3] + time[None, :, None] * observed[:, -1:, 3:]
