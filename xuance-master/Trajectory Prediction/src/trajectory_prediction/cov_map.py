"""Read the existing xuance cov map and check candidate paths against it.

The cov environment uses a 1000 m square and rectangular building prisms
[xmin, xmax, ymin, ymax, top]. This adapter preserves that geometry, including
the target radius and altitude limits. It does not rescale paper trajectories.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class CovMap:
    map_size: float
    z_min: float
    z_max: float
    buildings: np.ndarray
    radius: float = 0.5
    source: str = "environment snapshot"
    mode: str = "snapshot"

    def __post_init__(self):
        blocks = np.asarray(self.buildings, dtype=np.float32)
        if blocks.size == 0:
            blocks = np.zeros((0, 5), dtype=np.float32)
        if blocks.ndim != 2 or blocks.shape[1] != 5:
            raise ValueError("Cov buildings must be [N,5] prisms")
        if (self.map_size <= 0 or self.z_min >= self.z_max or self.radius < 0 or
                not np.all(np.isfinite(blocks)) or
                np.any(blocks[:, 0] >= blocks[:, 1]) or
                np.any(blocks[:, 2] >= blocks[:, 3]) or
                np.any(blocks[:, :4] < 0) or
                np.any(blocks[:, :4] > self.map_size) or
                np.any(blocks[:, 4] < self.z_min) or
                np.any(blocks[:, 4] > self.z_max)):
            raise ValueError("Invalid cov map bounds or building prisms")
        object.__setattr__(self, "buildings", blocks)

    @classmethod
    def from_cov_config(cls, config_path: Path, buildings_dir: Path) -> "CovMap":
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("randomize_building_layout") or config.get("randomize_density"):
            raise ValueError("Cov layout is randomized; snapshot env.buildings after reset")
        mode = str(config.get("building_mode", "medium"))
        map_path = buildings_dir / f"buildings_{mode}.json"
        if mode == "empty":
            blocks = []
        else:
            blocks = json.loads(map_path.read_text(encoding="utf-8"))
        blocks = np.asarray(blocks, dtype=np.float32)
        if blocks.size and blocks.ndim == 2 and blocks.shape[1] == 4:
            tops = np.full((len(blocks), 1), float(config.get("z_max", 350.0)))
            blocks = np.concatenate((blocks, tops), axis=1)
        return cls(map_size=1000.0, z_min=float(config.get("z_min", 10.0)),
                   z_max=float(config.get("z_max", 350.0)), buildings=blocks,
                   radius=0.5, source=str(map_path.resolve()), mode=mode)

    @classmethod
    def from_env(cls, env) -> "CovMap":
        """Snapshot a randomized or fixed map after the cov environment resets."""
        return cls(map_size=float(env.map_size), z_min=float(env.z_min),
                   z_max=float(env.z_max), buildings=np.asarray(env.buildings),
                   radius=float(env.target_radius),
                   source="UAVPursuitCoverage3DEnv.buildings", mode=str(env.building_mode))

    def describe(self) -> dict:
        return {"source": self.source, "mode": self.mode,
                "map_size_m": self.map_size, "z_min_m": self.z_min,
                "z_max_m": self.z_max, "radius_m": self.radius,
                "buildings": len(self.buildings),
                "building_sha256": hashlib.sha256(self.buildings.tobytes()).hexdigest()}

    def classify_candidates(self, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return [B,K] collision flags, using each physics-sampled point."""
        if candidates.ndim != 4 or candidates.shape[-1] != 3:
            raise ValueError("Expected candidate positions [B,K,T,3]")
        x, y, z = candidates.unbind(dim=-1)
        r = self.radius
        outside = ((x < r) | (x > self.map_size - r) |
                   (y < r) | (y > self.map_size - r) |
                   (z < self.z_min) | (z > self.z_max)).any(dim=-1)
        building_hit = torch.zeros_like(outside)
        for xmin, xmax, ymin, ymax, top in self.buildings:
            collision = ((x > float(xmin) - r) & (x < float(xmax) + r) &
                         (y > float(ymin) - r) & (y < float(ymax) + r) &
                         (z <= float(top) + r)).any(dim=-1)
            building_hit |= collision
        return {"boundary": outside, "building": building_hit,
                "valid": ~(outside | building_hit)}

    def candidate_valid(self, candidates: torch.Tensor) -> torch.Tensor:
        return self.classify_candidates(candidates)["valid"]

    def assert_starts_inside(self, target: torch.Tensor, defender: torch.Tensor) -> None:
        """Reject paper-scale states before map masking can silently discard all paths."""
        positions = torch.stack((target, defender), dim=1)
        if not bool(self.classify_candidates(positions[:, :, None, :])["valid"].all()):
            raise ValueError("Observed states are outside the cov map or inside a building")
