"""Teacher scores must stay finite and train only from observed geometry."""

import unittest
from pathlib import Path

import torch
import yaml

from trajectory_prediction.prediction import TrajectoryPredictor
from trajectory_prediction.teachers import TeacherObjective, teacher_scores


class TeacherChecks(unittest.TestCase):
    def test_scores_and_factor_head_gradients(self):
        config = yaml.safe_load(Path("configs/prediction.yaml").read_text())
        target = torch.zeros(2, 80, 6)
        defender = torch.zeros_like(target)
        target[:, :, 2] = 6000.0
        defender[:, :, 0] = 5000.0
        defender[:, :, 2] = 6000.0
        target[:, :, 3] = 300.0
        defender[:, :, 3] = -400.0
        model = TrajectoryPredictor(**config["model"])
        output = model(target, defender, torch.full((2, 9), 1 / 9))
        self.assertTrue(bool(torch.isfinite(output["escape_rate_mps"]).all()))
        self.assertTrue(bool(((output["engagement_gate"] >= 0) &
                              (output["engagement_gate"] <= 1)).all()))
        scores = teacher_scores(output["candidate_xyz"], target, defender,
                                config["teacher"])
        self.assertEqual(tuple(scores.shape), (2, 3, 9))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertGreater(float(scores[0, 0, 0]), float(scores[0, 0, 1]))
        self.assertEqual(tuple(output["candidate_head_weights"].shape), (2, 6, 9))
        torch.testing.assert_close(output["candidate_head_weights"].sum(-1),
                                   torch.ones(2, 6), atol=1e-5, rtol=0)
        objective = TeacherObjective(config["teacher"], 0.4, 0.6)
        loss, parts = objective(output, target, defender, output["candidate_xyz"][:, 0])
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("risk_kl", parts)
        loss.backward()
        self.assertIsNotNone(model.candidate_heads.weight.grad)
        self.assertGreater(float(model.candidate_heads.weight.grad.abs().sum()), 0)
        self.assertIsNotNone(objective.log_sigma.grad)


if __name__ == "__main__":
    unittest.main()
