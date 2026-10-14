"""Performance gate for the fixed-map UAV baseline curriculum."""
from dataclasses import dataclass
import math


@dataclass
class PerformanceCurriculum:
    level: int = 0
    max_level: int = 4
    min_steps: int = 100000
    min_episodes: int = 100
    success_threshold: float = 0.8
    collision_threshold: float = 0.1
    required_passes: int = 2
    level_start_step: int = 0
    passes: int = 0
    last_eval_step: int = -1

    def __post_init__(self):
        if (not 0 <= self.level <= self.max_level or self.min_steps < 0 or
                self.min_episodes < 1 or self.required_passes < 1 or
                not 0 <= self.success_threshold <= 1 or not 0 <= self.collision_threshold <= 1):
            raise ValueError("Invalid performance curriculum gate")

    def observe(self, step, metrics):
        """Return the next level after two qualifying evaluations; never force it."""
        step = int(step)
        if step <= self.last_eval_step:
            return None
        self.last_eval_step = step
        if self.level >= self.max_level or step - self.level_start_step < self.min_steps:
            self.passes = 0
            return None
        safe_rate = float(metrics["safe_success_rate"])
        collision_rate = float(metrics["collision_rate"])
        if (not math.isfinite(safe_rate) or not math.isfinite(collision_rate) or
                not 0 <= safe_rate <= 1 or not 0 <= collision_rate <= 1):
            raise ValueError("Evaluation rates must be finite probabilities")
        passed = (int(metrics["episodes"]) >= self.min_episodes and
                  safe_rate >= self.success_threshold and collision_rate <= self.collision_threshold)
        self.passes = self.passes + 1 if passed else 0
        if self.passes < self.required_passes:
            return None
        self.level += 1
        self.level_start_step = step
        self.passes = 0
        return self.level
