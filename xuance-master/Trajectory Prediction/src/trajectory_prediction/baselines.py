"""History-only LSTM, GRU, TCN and Transformer forecasting baselines."""
import torch
from torch import nn
from torch.nn import functional as F

from .data import relative_features, rotate_xy
from .model import bound_increments


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.layers = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=dilation) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        residual = value
        for layer in self.layers:
            value = self.dropout(F.relu(layer(F.pad(value, (self.padding, 0)))))
        return value + residual


class SequenceForecaster(nn.Module):
    supports_intention = False

    def __init__(self, cfg, normalizer=None):
        super().__init__()
        self.cfg = cfg
        norm = normalizer or {"mean": torch.zeros(6), "std": torch.ones(6)}
        self.register_buffer("feature_mean", norm["mean"].clone())
        self.register_buffer("feature_std", norm["std"].clone())
        width = cfg.baseline_width
        dropout = cfg.dropout if cfg.baseline_layers > 1 else 0
        self.input = nn.Linear(6, width)
        if cfg.model_type in {"lstm", "gru"}:
            rnn = nn.LSTM if cfg.model_type == "lstm" else nn.GRU
            self.encoder = rnn(width, width, cfg.baseline_layers, dropout=dropout, batch_first=True)
            self.decoder = rnn(width, width, cfg.baseline_layers, dropout=dropout, batch_first=True)
            self.feedback = nn.Linear(3, width)
            self.output = nn.Linear(width, 3)
        elif cfg.model_type == "tcn":
            self.encoder = nn.Sequential(*[CausalBlock(width, d, cfg.dropout) for d in (1, 2, 4, 8)])
            self.output = nn.Linear(width, cfg.prediction_steps * 3)
        else:
            self.history_position = nn.Parameter(torch.randn(cfg.observation_steps, width) * 0.02)
            self.future_position = nn.Parameter(torch.randn(cfg.prediction_steps, width) * 0.02)
            self.transformer = nn.Transformer(width, cfg.temporal_heads, cfg.baseline_layers,
                                               cfg.baseline_layers, 4 * width, cfg.dropout, batch_first=True)
            self.output = nn.Linear(width, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, observed, context=None, context_mask=None, engagement=None):
        # Baselines ignore the engagement descriptors; the argument keeps the
        # call signature identical to the graph model (paper Section IV-E).
        if observed.ndim != 3 or observed.shape[1:] != (self.cfg.observation_steps, 6):
            raise ValueError("Baseline observation shape does not match configuration")
        physical = relative_features(observed, self.cfg.relative_frame)
        features = (physical - self.feature_mean) / self.feature_std
        embedded = self.input(features)
        velocity = physical[:, -1, 3:]
        if self.cfg.model_type in {"lstm", "gru"}:
            _, state = self.encoder(embedded)
            previous = velocity * self.cfg.dt
            residuals = []
            for _ in range(self.cfg.prediction_steps):
                decoded, state = self.decoder(self.feedback(previous)[:, None], state)
                delta = self.output(decoded[:, 0]) * self.cfg.dt
                previous = velocity * self.cfg.dt + delta
                residuals.append(delta)
            delta = torch.stack(residuals, dim=1)
        elif self.cfg.model_type == "tcn":
            memory = self.encoder(embedded.transpose(1, 2))[:, :, -1]
            delta = self.output(memory).view(len(observed), self.cfg.prediction_steps, 3) * self.cfg.dt
        else:
            target = self.future_position[None].expand(len(observed), -1, -1)
            causal = torch.ones(self.cfg.prediction_steps, self.cfg.prediction_steps,
                                dtype=torch.bool, device=observed.device).triu(1)
            decoded = self.transformer(embedded + self.history_position[None], target, tgt_mask=causal)
            delta = self.output(decoded) * self.cfg.dt
        increments = bound_increments(velocity[:, None] * self.cfg.dt + delta, velocity, self.cfg)
        if self.cfg.relative_frame == "yaw":
            increments = rotate_xy(increments, torch.atan2(observed[:, -1, 4], observed[:, -1, 3]), inverse=True)
        prediction = observed[:, -1:, :3] + increments.cumsum(1)
        logits = observed.new_zeros(len(observed), self.cfg.intention_classes)
        return {"trajectory": prediction, "logits": logits, "probabilities": logits.softmax(-1), "attention": []}
