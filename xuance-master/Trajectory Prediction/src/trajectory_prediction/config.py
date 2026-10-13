from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass
class Config:
    observation_steps: int = 20
    prediction_steps: int = 12
    stride: int = 5
    dt: float = 0.5
    intention_classes: int = 9
    lstm_hidden: int = 128
    projection_dim: int = 64
    graph_dim: int = 96
    heads: int = 3
    graph_layers: int = 2
    temporal_radius: int = 2
    dropout: float = 0.2
    intention_dropout: float | None = None
    batch_size: int = 256
    # Memory guard for GPUs smaller than the paper's hardware: the effective
    # batch stays batch_size (the paper pins 256 for IV-A and IV-B), while
    # each optimizer step is computed as gradient accumulation over equal
    # sub-batches of this many samples.  The model carries no BatchNorm-style
    # batch statistics, so accumulated gradients equal the full-batch
    # gradient.  0 = disabled (single full-batch forward, default, and the
    # historically verified path).
    micro_batch_size: int = 0
    epochs: int = 70
    learning_rate: float = 0.001
    lr_step: int = 30
    lr_gamma: float = 0.5
    gradient_clip: float = 1.0
    ade_weight: float = 1.0
    fde_weight: float = 0.5
    intention_weight: float = 0.2
    smoothness_weight: float = 0.001
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    seed: int = 42
    split_seed: int = 42
    device: str = "auto"
    workers: int = 0
    cpu_threads: int = 2
    stage: str = "joint"
    freeze_intention: bool = False
    model_type: str = "graph"
    relative_frame: str = "world"
    static_state_branch: bool = False
    use_intention: bool = True
    decoder_type: str = "parallel"
    # "linear" = IV-B ("The linear regression is employed for the output head");
    # "mlp" = the generic Eq. (32) decoder y_k = MLP(h_k^L).
    decoder_head: str = "linear"
    # True (default, engineering choice) zero-initialises the decoder output
    # layer so training starts at the constant-velocity baseline; False uses
    # PyTorch defaults, which is the literal reading because the paper
    # specifies no initialisation (audit E1).
    decoder_zero_init: bool = True
    graph_head_dim: int = 0
    graph_top_k: int = 0
    attention_temperature: float = 1.0
    temporal_layers: int = 0
    temporal_heads: int = 4
    baseline_width: int = 256
    baseline_layers: int = 2
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    schedule: str = "step"
    warmup_epochs: int = 5
    schedule_epochs: int = 0
    min_lr_fraction: float = 0.01
    early_stopping_patience: int = 0
    # Paper IV-B: stop when neither ADE nor FDE improves by >= 1 m over the
    # patience window. 0.001 m (the old default) effectively never triggers.
    early_stopping_min_delta_m: float = 1.0
    selection_metric: str = "ade"
    loss_balance: str = "fixed"
    attention_supervision: bool = False
    attention_weight: float = 0.1
    separation_weight: float = 0.1
    entropy_weight: float = 0.01
    jitter_weight: float = 0.01
    entropy_floor_fraction: float = 0.3
    teacher_temperature: float = 0.1
    # "paper" evaluates Eqs. 38-41 literally (entropy of the group-averaged
    # head distribution, jitter between adjacent steps of that mean, two-sided
    # entropy floor); "legacy" keeps the earlier per-head approximation.
    teacher_regularizers: str = "legacy"
    # Eq. (36) normalises each teacher distribution pi^(k) over its own edge
    # subset E_k(i) (the indicator 1(i->j in E_k(i)) in the denominator), and
    # Definition 1 assigns each head group "its corresponding teacher score and
    # its corresponding edge subset".  The three disjoint subsets of Section
    # III-B are mapped H_E -> temporal, H_R -> spatial, H_T -> countermeasure
    # (the paper does not tabulate the mapping itself, decision D8).  Off by
    # default: single-entity civilian graphs have no spatial/countermeasure
    # edges, so every group would fall back to the unrestricted mask anyway.
    teacher_edge_subsets: bool = False
    max_speed_mps: float = 0.0
    max_acceleration_mps2: float = 0.0
    mixed_precision: bool = False
    historical_edge_features: bool = False
    historical_reachability: bool = False
    reachability_position_tolerance_m: float = 0.25
    reachability_velocity_tolerance_mps: float = 0.05
    crop_warmup_epochs: int = 0
    early_stopping_mode: str = "weighted"
    context_entities: int = 0
    use_interaction_context: bool = True
    interaction_radius_m: float = 6.0
    interaction_time_radius: int = 0
    # ---- military intercept-engagement extensions (paper Sections III/IV) ----
    engagement_edges: bool = False        # use engagement-descriptor edges (Eq. 15)
    fov_half_angle_deg: float = 60.0      # target FOV half-angle for visibility
    fov_attention_mask: bool = False      # drop cross-entity edges when invisible (IV-B)
    overload_reachability: bool = False   # A7: hard-block links needing > aT_max (IV-B;
                                          # off by default so civilian configs and
                                          # pre-A7 checkpoints keep validating)
    engagement_range_ref_m: float = 8000.0  # normalization range for r descriptors
    engagement_overload_g: float = 9.0    # target overload limit aT_max in g
    defender_overload_g: float = 15.0     # defender overload limit aM_max in g
    intention_task: str = "semantic9"     # "semantic9" | "overload11"
    holdout_test: bool = True             # False = paper IV-B 80/20 (no test split)
    # ---- military teacher supervision (Eqs. 42-53) ---------------------------
    military_teachers: bool = False       # replace civilian teachers by E/R/T
    teacher_kappa: float = 1.0            # LSE sharpness (kappa_E/kappa_R/kappa_T)
    teacher_risk_lambda: float = 1.0      # risk assessment coefficient lambda_R
    teacher_corridor_radial_m: float = 0.0   # radial corridor half-width (0 = auto)
    teacher_corridor_normal_m: float = 0.0   # normal corridor half-width (0 = auto)
    # ---- candidate branch cropping (Table I, Eqs. 22-25; decision D1) -------
    candidate_pruning: bool = False       # attach A candidate branches and crop them
    candidate_anchors: int = 16           # A: candidate directions (padded to O)
    escape_rate_min_mps: float = 0.0      # rdot_min of Cesc (Eq. 22)
    terrain_enabled: bool = False         # Cenv simplified terrain (decision D2)
    terrain_altitude_min_m: float = -2000.0
    terrain_altitude_max_m: float = 20000.0
    candidate_crop_lambda: float = 1.0    # Lambda_crop scale of the soft crop
    candidate_arc_samples: int = 12       # arc points for Eq. (27) mean rdot / Eq. (28) forall-k
    # ---- IV-A class-balanced window sampling (paper: "balanced coverage") ---
    balance_intention_classes: bool = False  # equalise windows per intention class
    balance_samples_per_class: int = 0       # 0 = target the per-class median

    def validate(self):
        if self.intention_dropout is not None and not (math.isfinite(self.intention_dropout) and 0 <= self.intention_dropout < 1):
            raise ValueError("intention_dropout must be None or in [0,1)")
        for name in ("dt", "learning_rate", "lr_gamma", "gradient_clip", "dropout", "ade_weight",
                     "fde_weight", "intention_weight", "smoothness_weight", "train_fraction", "validation_fraction",
                     "weight_decay", "attention_temperature", "min_lr_fraction", "early_stopping_min_delta_m",
                     "attention_weight", "separation_weight", "entropy_weight", "jitter_weight",
                     "entropy_floor_fraction", "teacher_temperature", "max_speed_mps", "max_acceleration_mps2"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("observation_steps", "stride", "intention_classes",
                     "lstm_hidden", "projection_dim", "graph_dim", "heads", "graph_layers",
                     "batch_size", "epochs", "lr_step", "cpu_threads", "temporal_heads",
                     "baseline_width", "baseline_layers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        # IV-A is a pure classification experiment with no prediction horizon:
        # prediction_steps=0 is allowed there (and only there).
        if self.prediction_steps < 0 or (self.prediction_steps and self.prediction_steps < 2):
            raise ValueError("Prediction horizon must be zero (IV-A) or at least two steps")
        if self.prediction_steps == 0 and self.stage != "intention":
            raise ValueError("prediction_steps=0 is only valid for the intention stage")
        if self.observation_steps < 2:
            raise ValueError("Observation window needs at least two steps")
        if self.graph_head_dim == 0 and self.graph_dim % self.heads:
            raise ValueError("graph_dim must be divisible by heads")
        if self.decoder_type == "parallel" and self.graph_dim % self.heads:
            raise ValueError("Parallel decoder requires graph_dim divisible by heads")
        if self.micro_batch_size and (not isinstance(self.micro_batch_size, int)
                                      or self.micro_batch_size < 1 or self.batch_size % self.micro_batch_size):
            raise ValueError("micro_batch_size must be zero or a positive divisor of batch_size")
        if self.micro_batch_size and self.batch_size // self.micro_batch_size < 1:
            raise ValueError("micro_batch_size cannot exceed batch_size")
        if self.temporal_radius < 0 or self.workers < 0:
            raise ValueError("temporal_radius and workers cannot be negative")
        if self.dt <= 0 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("dt, learning_rate, gradient_clip must be positive")
        if not 0 <= self.dropout < 1 or not 0 < self.lr_gamma <= 1:
            raise ValueError("Invalid dropout or learning-rate decay")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1):
            raise ValueError("Split fractions must leave nonempty train and validation partitions")
        if self.holdout_test and self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Split fractions must leave nonempty train/validation/test partitions")
        if not self.holdout_test and self.train_fraction + self.validation_fraction > 1:
            raise ValueError("Split fractions cannot exceed 1")
        if self.stage not in {"joint", "intention", "forecast"}:
            raise ValueError("stage must be joint, intention, or forecast")
        if self.freeze_intention and self.stage != "forecast":
            raise ValueError("freeze_intention only applies to forecast training")
        for name in ("ade_weight", "fde_weight", "intention_weight", "smoothness_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.ade_weight + self.fde_weight <= 0:
            raise ValueError("At least one trajectory loss weight must be positive")
        for name in ("graph_head_dim", "graph_top_k", "temporal_layers", "warmup_epochs", "schedule_epochs",
                     "early_stopping_patience", "crop_warmup_epochs", "context_entities", "interaction_time_radius"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.decoder_type not in {"parallel", "autoregressive_gru"}:
            raise ValueError("Unknown decoder_type")
        if self.decoder_head not in {"linear", "mlp"}:
            raise ValueError("decoder_head must be linear (IV-B) or mlp (Eq. 32)")
        if not math.isfinite(self.interaction_radius_m) or self.interaction_radius_m <= 0:
            raise ValueError("interaction_radius_m must be finite and positive")
        if self.context_entities and (not self.static_state_branch or self.decoder_type != "autoregressive_gru"):
            raise ValueError("Multi-entity histories require the static branch and autoregressive decoder")
        if self.relative_frame not in {"world", "yaw"}:
            raise ValueError("relative_frame must be world or yaw")
        if self.model_type not in {"graph", "lstm", "gru", "tcn", "transformer"}:
            raise ValueError("Unknown model_type")
        if self.model_type != "graph" and self.stage != "forecast":
            raise ValueError("Sequence baselines support forecast training only")
        if self.optimizer not in {"adam", "adamw"} or self.schedule not in {"step", "warmup_cosine"}:
            raise ValueError("Unknown optimizer or schedule")
        if self.loss_balance not in {"fixed", "uncertainty"} or self.selection_metric not in {"ade", "ade_fde"}:
            raise ValueError("Unknown loss balancing or selection metric")
        if self.early_stopping_mode not in {"weighted", "both"}:
            raise ValueError("early_stopping_mode must be weighted or both")
        if self.teacher_regularizers not in {"legacy", "paper"}:
            raise ValueError("teacher_regularizers must be legacy or paper")
        if self.attention_supervision and self.heads % 3:
            # The attention heads are split into three teacher groups (E/R/T in
            # the military variant), so the head count must divide by three;
            # the paper's H = 6 gives two heads per group.
            raise ValueError("attention_supervision needs heads divisible by 3 (paper uses H=6)")
        if self.engagement_edges and not self.context_entities:
            raise ValueError("Engagement edges require the defender context (context_entities>=1)")
        if self.overload_reachability and not self.engagement_edges:
            raise ValueError("overload_reachability hard blocking requires engagement_edges")
        if self.candidate_pruning:
            if not self.context_entities:
                raise ValueError("Candidate branch cropping requires the defender context")
            if not 1 <= self.candidate_anchors <= self.observation_steps:
                raise ValueError("candidate_anchors must be in [1, observation_steps]")
            if not isinstance(self.candidate_arc_samples, int) or self.candidate_arc_samples < 1:
                raise ValueError("candidate_arc_samples must be a positive integer")
            if self.terrain_enabled and self.terrain_altitude_min_m >= self.terrain_altitude_max_m:
                raise ValueError("Terrain altitude band must be a nonempty interval")
            if not math.isfinite(self.escape_rate_min_mps) or self.candidate_crop_lambda < 0:
                raise ValueError("Invalid escape-rate threshold or crop strength")
        if not math.isfinite(self.fov_half_angle_deg) or not 0 < self.fov_half_angle_deg <= 90:
            raise ValueError("fov_half_angle_deg must be in (0, 90]")
        if not math.isfinite(self.engagement_range_ref_m) or self.engagement_range_ref_m <= 0:
            raise ValueError("engagement_range_ref_m must be finite and positive")
        if self.engagement_overload_g <= 0 or self.defender_overload_g <= 0:
            raise ValueError("Engagement overload limits must be positive")
        if self.intention_task not in {"semantic9", "overload11"}:
            raise ValueError("intention_task must be semantic9 or overload11")
        expected = {"semantic9": 9, "overload11": 11}[self.intention_task]
        if self.intention_classes != expected:
            raise ValueError(f"intention_task '{self.intention_task}' requires intention_classes={expected}")
        if self.military_teachers and not self.attention_supervision:
            raise ValueError("military_teachers requires attention_supervision")
        for name in ("teacher_kappa", "teacher_risk_lambda",
                     "teacher_corridor_radial_m", "teacher_corridor_normal_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("reachability_position_tolerance_m", "reachability_velocity_tolerance_mps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.historical_reachability and not (self.max_speed_mps > 0 and self.max_acceleration_mps2 > 0):
            raise ValueError("Historical reachability requires positive civilian speed and acceleration bounds")
        if self.attention_supervision and (self.heads % 3 or self.model_type != "graph"):
            raise ValueError("Civilian attention supervision needs three equally sized disjoint head groups")
        if (self.decoder_type == "autoregressive_gru" or self.temporal_layers) and self.graph_dim % self.temporal_heads:
            raise ValueError("graph_dim must be divisible by temporal_heads")
        if self.model_type == "transformer" and self.baseline_width % self.temporal_heads:
            raise ValueError("baseline_width must be divisible by temporal_heads")
        if self.attention_temperature <= 0 or self.teacher_temperature <= 0:
            raise ValueError("Attention temperatures must be positive")
        if not 0 <= self.entropy_floor_fraction <= 1 or not 0 <= self.min_lr_fraction <= 1:
            raise ValueError("Invalid entropy floor or learning-rate floor")
        for name in ("weight_decay", "early_stopping_min_delta_m", "attention_weight", "separation_weight",
                     "entropy_weight", "jitter_weight", "max_speed_mps", "max_acceleration_mps2"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path=None):
        values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**values).validate()
