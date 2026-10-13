"""Prediction-only API for observed single-track histories."""

import numpy as np
import torch

from .config import Config
from .engine import choose_device, load_checkpoint
from .model import build_forecaster


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        saved = load_checkpoint(checkpoint)
        self.cfg = Config(**saved["config"]).validate()
        if self.cfg.stage == "intention":
            raise ValueError("Use a forecast/joint checkpoint for trajectory prediction")
        self.device = choose_device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(self.cfg.cpu_threads)
        self.model = build_forecaster(self.cfg, saved["normalizer"]).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, history, context=None, context_mask=None, engagement=None):
        history = np.asarray(history, dtype=np.float32)
        expected = (self.cfg.observation_steps, 6)
        if history.ndim != 2 or history.shape[1] != 6 or len(history) < expected[0]:
            raise ValueError(f"Need at least {expected[0]} observed rows with six state columns")
        history = history[-expected[0]:]
        if not np.isfinite(history).all():
            raise ValueError("History contains nonfinite values")
        context_tensor, mask_tensor = None, None
        if context is not None:
            context = np.asarray(context, dtype=np.float32)
            if context.ndim != 3 or context.shape[0] != self.cfg.context_entities or context.shape[2] != 6 or context.shape[1] < expected[0]:
                raise ValueError("Peer history must be [configured N,at least O,6]")
            mask = np.ones(context.shape[:2], dtype=bool) if context_mask is None else np.asarray(context_mask)
            if mask.shape != context.shape[:2] or not np.isin(mask, [0, 1]).all():
                raise ValueError("Peer mask must match [N,T] and contain boolean/0/1 values")
            context, mask = context[:, -expected[0]:], mask[:, -expected[0]:].astype(bool)
            if not np.isfinite(context[mask]).all():
                raise ValueError("Available peer measurements must be finite")
            context_tensor = torch.from_numpy(np.where(mask[..., None], context, 0))[None].to(self.device)
            mask_tensor = torch.from_numpy(mask)[None].to(self.device)
        elif context_mask is not None:
            raise ValueError("Peer mask requires peer history")
        engagement_tensor = None
        if engagement is not None:
            engagement = np.asarray(engagement, dtype=np.float32)
            if engagement.ndim != 2 or engagement.shape[1] != 9 or engagement.shape[0] < expected[0]:
                raise ValueError("Engagement descriptors must be [at least O,9]")
            if not np.isfinite(engagement).all():
                raise ValueError("Engagement descriptors must be finite")
            engagement_tensor = torch.from_numpy(engagement[-expected[0]:])[None].to(self.device)
        output = self.model(torch.from_numpy(history).unsqueeze(0).to(self.device), context=context_tensor,
                            context_mask=mask_tensor, engagement=engagement_tensor)
        return {"trajectory": output["trajectory"][0].cpu().numpy(),
                "intention_probabilities": (output["probabilities"][0].cpu().numpy()
                    if self.model.supports_intention and self.cfg.use_intention else None)}
