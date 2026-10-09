"""Cov prism geometry and map-conditioned candidate masking."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from trajectory_prediction.cov_map import CovMap
from trajectory_prediction.prediction import TrajectoryPredictor


class CovMapChecks(unittest.TestCase):
    def test_prisms_boundary_and_height(self):
        cov_map = CovMap(1000.0, 50.0, 350.0,
                         np.array([[200, 250, 200, 250, 120]], np.float32))
        paths = torch.tensor([[[[190.0, 225.0, 100.0], [210.0, 225.0, 100.0]],
                               [[190.0, 225.0, 160.0], [210.0, 225.0, 160.0]],
                               [[999.8, 225.0, 100.0], [1001.0, 225.0, 100.0]]]])
        result = cov_map.classify_candidates(paths)
        self.assertEqual(result["building"].tolist(), [[True, False, False]])
        self.assertEqual(result["boundary"].tolist(), [[False, False, True]])
        self.assertEqual(result["valid"].tolist(), [[False, True, False]])

    def test_config_loader_and_paper_scale_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = root / "buildings"
            maps.mkdir()
            (maps / "buildings_open.json").write_text(
                json.dumps([[200, 250, 200, 250, 120]]))
            config = root / "cov.yaml"
            config.write_text("building_mode: open\nz_min: 50\nz_max: 350\n"
                              "randomize_building_layout: false\n")
            cov_map = CovMap.from_cov_config(config, maps)
            self.assertEqual(cov_map.mode, "open")
            self.assertEqual(len(cov_map.buildings), 1)
            snapshot = CovMap.from_env(SimpleNamespace(
                map_size=1000.0, z_min=50.0, z_max=350.0,
                buildings=cov_map.buildings, target_radius=0.5,
                building_mode="open"))
            np.testing.assert_array_equal(snapshot.buildings, cov_map.buildings)
            with self.assertRaisesRegex(ValueError, "outside the cov map"):
                cov_map.assert_starts_inside(torch.tensor([[0.0, 0.0, 6000.0]]),
                                             torch.tensor([[0.0, 0.0, 6000.0]]))
            config.write_text(config.read_text() + "randomize_density: [open, medium]\n")
            with self.assertRaisesRegex(ValueError, "snapshot"):
                CovMap.from_cov_config(config, maps)

    def test_predictor_uses_cov_mask(self):
        cov_map = CovMap(1000.0, 50.0, 350.0,
                         np.array([[200, 250, 200, 250, 120]], np.float32))
        target = torch.zeros((1, 80, 6))
        defender = torch.zeros_like(target)
        target[:, :, :3] = torch.tensor([191.0, 225.0, 119.0])
        defender[:, :, :3] = torch.tensor([191.0, 180.0, 119.0])
        target[:, :, 3] = 3.0
        model = TrajectoryPredictor(candidate_acceleration_g=1 / 9.81,
                                    min_altitude_m=50.0).eval()
        with torch.no_grad():
            result = model(target, defender, torch.full((1, 9), 1 / 9),
                           cov_map=cov_map)
        self.assertFalse(bool(result["candidate_valid"][0, 0]))
        self.assertGreater(int(result["candidate_valid"].sum()), 0)
        torch.testing.assert_close(result["candidate_weights"].sum(dim=1),
                                   torch.ones(1), atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
