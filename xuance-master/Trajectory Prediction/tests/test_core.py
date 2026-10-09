"""Checks for physics and data boundaries that would invalidate evaluation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory_prediction.data import (prediction_windows, split_prediction_tracks,
                                        split_tracks)
from trajectory_prediction.simulator import G, ScenarioConfig, simulate_track


class CoreChecks(unittest.TestCase):
    def test_state_uses_only_completed_acceleration(self):
        track = simulate_track(ScenarioConfig(duration=15.0), seed=7)
        target = track["target"]
        speed = np.linalg.norm(target[:, 3:6], axis=1)
        acceleration = np.linalg.norm(target[:, 6:9], axis=1)
        self.assertLess(float(np.max(np.abs(speed - speed[0]))), 1e-3)
        self.assertLessEqual(float(acceleration.max()), 9 * G + 1e-3)
        self.assertEqual(float(acceleration[0]), 0.0)
        self.assertEqual(int(track["intent"][0]), 0)
        # Direction at t+1 is determined by the acceleration stored at t+1.
        for step in (0, 25, 100):
            previous = target[step, 3:6].astype(np.float64)
            applied = target[step + 1, 6:9].astype(np.float64)
            expected = previous / np.linalg.norm(previous) + applied * 0.1 / np.linalg.norm(previous)
            expected *= np.linalg.norm(previous) / np.linalg.norm(expected)
            np.testing.assert_allclose(target[step + 1, 3:6], expected, atol=1e-4)

    def test_fov_config_and_disjoint_split(self):
        visible = [simulate_track(ScenarioConfig(duration=0.1), seed=i)["fov"][0]
                   for i in range(12)]
        hidden = [simulate_track(ScenarioConfig(duration=0.1, initial_visible_fraction=0.0),
                                 seed=i)["fov"][0] for i in range(12)]
        self.assertTrue(all(visible))
        self.assertFalse(any(hidden))
        groups = split_tracks([Path(f"track_{i}.npz") for i in range(3)], seed=7)
        self.assertEqual({name: len(group) for name, group in groups.items()},
                         {"train": 1, "val": 1, "test": 1})
        self.assertEqual(len(set(sum(groups.values(), []))), 3)
        prediction_groups = split_prediction_tracks(
            [Path(f"track_{i}.npz") for i in range(10)], seed=7)
        self.assertEqual([len(prediction_groups[name]) for name in ("train", "eval")],
                         [8, 2])

    def test_prediction_window_ends_before_future(self):
        track = simulate_track(ScenarioConfig(duration=15.0), seed=9)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "track_000000.npz"
            np.savez_compressed(path, **track)
            window = next(prediction_windows([path]))
            np.testing.assert_array_equal(window["target_history"][-1, :3],
                                          track["target"][79, :3])
            np.testing.assert_array_equal(window["target_acceleration"],
                                          track["target"][79, 6:9])
            np.testing.assert_array_equal(window["future_xyz"][0],
                                          track["target"][80, :3])
            self.assertEqual(window["target_history"].shape, (80, 6))
            self.assertEqual(window["future_xyz"].shape, (60, 3))


if __name__ == "__main__":
    unittest.main()
