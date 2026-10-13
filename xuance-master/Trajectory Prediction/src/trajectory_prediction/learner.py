"""Algorithm-specific optimization shared by source CLI and the Xuance learner."""
import math
import torch

from .objective import TrajectoryObjective


class TrajectoryLearner:
    def __init__(self, cfg, policy, device, steps_per_epoch=1):
        self.cfg, self.policy, self.device = cfg, policy, device
        self.steps_per_epoch = steps_per_epoch
        self.objective = TrajectoryObjective(cfg).to(device)
        parameters = [p for p in list(policy.parameters()) + list(self.objective.parameters()) if p.requires_grad]
        optimizer = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        self.optimizer = optimizer(parameters, lr=cfg.learning_rate,
                                   weight_decay=cfg.weight_decay if cfg.optimizer == "adamw" else 0)
        if cfg.schedule == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.lr_step, cfg.lr_gamma)
        else:
            horizon = cfg.schedule_epochs or cfg.epochs
            if cfg.warmup_epochs >= horizon:
                raise ValueError("warmup_epochs must be smaller than the cosine schedule horizon")

            def factor(epoch):
                if epoch < cfg.warmup_epochs:
                    return (epoch + 1) / max(1, cfg.warmup_epochs)
                progress = min(1.0, (epoch - cfg.warmup_epochs) / max(1, horizon - cfg.warmup_epochs))
                return cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * (1 + math.cos(math.pi * progress)) / 2

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
        # Gradient accumulation (audit round 4): the effective batch stays
        # cfg.batch_size, which the paper pins to 256; only the per-forward
        # sub-batch shrinks so activations fit small-VRAM GPUs.  Because the
        # model carries no BatchNorm-style batch statistics, averaging the
        # sub-batch gradients reproduces the full-batch gradient exactly
        # (up to floating-point summation order).
        micro = int(getattr(cfg, "micro_batch_size", 0) or 0)
        self.accum_steps = max(1, cfg.batch_size // micro) if 0 < micro <= cfg.batch_size else 1
        self._micro_index = 0
        self._group_pending = 0
        self._group_samples = 0
        self.iterations = 0

    def compute(self, batch):
        with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            output = self.policy(batch["observed"], context=batch.get("context"), context_mask=batch.get("context_mask"),
                                 engagement=batch.get("engagement"))
            losses = self.objective(output, batch)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("Nonfinite objective; check data, units, dt and learning rate")
        return output, losses

    def update(self, batch):
        if self.accum_steps <= 1:
            return self._update_once(batch)
        return self._update_accumulated(batch)

    def _update_once(self, batch):
        if self.cfg.crop_warmup_epochs and hasattr(self.policy, "set_crop_progress"):
            self.policy.set_crop_progress((self.iterations + 1) / (self.steps_per_epoch * self.cfg.crop_warmup_epochs))
        self.optimizer.zero_grad(set_to_none=True)
        output, losses = self.compute(batch)
        self.scaler.scale(losses["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        parameters = list(self.policy.parameters()) + list(self.objective.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            self.objective.log_variance.clamp_(-6, 6)
        self.iterations += 1
        return output, losses

    def _update_accumulated(self, batch):
        if not self._group_pending:
            self.optimizer.zero_grad(set_to_none=True)
            self._group_samples = 0
        if self.cfg.crop_warmup_epochs and hasattr(self.policy, "set_crop_progress"):
            self.policy.set_crop_progress((self.iterations + 1) / (self.steps_per_epoch * self.cfg.crop_warmup_epochs))
        output, losses = self.compute(batch)
        # Backprop the sample-weighted SUM of the micro-batch loss; the group
        # mean is restored by dividing the accumulated gradients by the
        # group's total sample count.  This reproduces the gradient of one
        # full cfg.batch_size forward EXACTLY for any group composition,
        # including a trailing partial group at the epoch boundary (which
        # then matches the partial last batch a plain batch_size DataLoader
        # would produce).
        count = len(batch["observed"])
        self.scaler.scale(losses["loss"] * count).backward()
        self._group_samples += count
        self._micro_index += 1
        if self._group_samples < self.cfg.batch_size:
            self._group_pending = 1
            return output, losses  # defer the optimizer step
        self._step_group()
        return output, losses

    def _step_group(self):
        parameters = list(self.policy.parameters()) + list(self.objective.parameters())
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(self._group_samples)
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            self.objective.log_variance.clamp_(-6, 6)
        self._group_pending = 0
        self.iterations += 1

    def flush(self):
        """Step a pending partial group (call at the end of a training epoch)."""
        if self.accum_steps > 1 and self._group_pending:
            self._step_group()

    def state_dict(self):
        return {"criterion": self.objective.state_dict(), "scaler": self.scaler.state_dict(),
                "learner_iterations": self.iterations, "learner_micro_index": self._micro_index,
                "learner_group_pending": self._group_pending}

    def restore(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "criterion" in checkpoint:
            self.objective.load_state_dict(checkpoint["criterion"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.iterations = checkpoint.get("learner_iterations", 0)
        self._micro_index = checkpoint.get("learner_micro_index", 0)
        self._group_pending = checkpoint.get("learner_group_pending", 0)
