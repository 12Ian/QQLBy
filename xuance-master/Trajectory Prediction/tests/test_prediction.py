"""Shape, feasibility, and gradient checks for the prediction pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.data import prediction_windows
from trajectory_prediction.intention import IntentionBiLSTM
from trajectory_prediction.online import forecast_from_prefix
from trajectory_prediction.prediction import TrajectoryPredictor, primitive_candidates
from trajectory_prediction.simulator import ScenarioConfig, simulate_track


class PredictionChecks(unittest.TestCase):
    def test_candidates_and_model(self):
        position = torch.tensor([[0.0, 0.0, 6000.0]])
        velocity = torch.tensor([[300.0, 0.0, 0.0]])
        candidates = primitive_candidates(position, velocity)
        self.assertEqual(tuple(candidates.shape), (1, 9, 60, 3))
        torch.testing.assert_close(candidates[0, 0, -1],
                                   torch.tensor([1800.0, 0.0, 6000.0]), atol=1e-2, rtol=0)
        target = torch.zeros((1, 80, 6))
        defender = torch.zeros_like(target)
        target[:, :, 2] = 6000.0
        defender[:, :, 0] = 5000.0
        defender[:, :, 2] = 6000.0
        target[:, :, 3] = 300.0
        defender[:, :, 3] = -400.0
        model = TrajectoryPredictor()
        result = model(target, defender, torch.full((1, 9), 1 / 9))
        self.assertEqual(tuple(result["future_xyz"].shape), (1, 60, 3))
        self.assertEqual(int(result["candidate_kept"].sum()), 8)
        self.assertTrue(bool(result["candidate_kept"][0, 0]))
        torch.testing.assert_close(result["candidate_weights"].sum(dim=1),
                                   torch.ones(1), atol=1e-5, rtol=0)
        result["future_xyz"].square().mean().backward()
        self.assertIsNotNone(model.delta_correction.weight.grad)

    def test_intention_and_prediction_share_only_observed_prefix(self):
        track = simulate_track(ScenarioConfig(duration=16.0), seed=10)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "track_000000.npz"
            np.savez_compressed(path, **track)
            window = next(prediction_windows([path], intention_seconds=10.0))
            self.assertEqual(window["intention_target_history"].shape, (20, 6))
            self.assertEqual(window["target_history"].shape, (80, 6))
            np.testing.assert_array_equal(window["intention_target_history"][-1],
                                          window["target_history"][-1])
            np.testing.assert_array_equal(window["future_xyz"][0],
                                          track["target"][96, :3])

    def test_forecast_prefix_is_independent_of_future_truth(self):
        track = simulate_track(ScenarioConfig(duration=16.0), seed=11)
        altered = track["target"].copy()
        altered[96:, :3] += 100000.0
        scaler = {"target_mean": np.zeros(6, np.float32),
                  "target_std": np.ones(6, np.float32),
                  "defender_mean": np.zeros(6, np.float32),
                  "defender_std": np.ones(6, np.float32)}
        intention = IntentionBiLSTM().eval()
        predictor = TrajectoryPredictor().eval()
        device = torch.device("cpu")
        result = forecast_from_prefix(track["target"][:96], track["defender"][:96],
                                      intention, scaler, predictor, device)
        changed = forecast_from_prefix(altered[:96], track["defender"][:96],
                                       intention, scaler, predictor, device)
        np.testing.assert_array_equal(result["future_xyz"], changed["future_xyz"])


if __name__ == "__main__":
    unittest.main()
