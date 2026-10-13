"""Held-out metrics clustered by independent scenes, or tracks for uncoupled data."""
import csv
from pathlib import Path

import numpy as np
import torch

from .data import CLASS_NAMES
from .model import constant_velocity


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EvaluationDetails:
    def __init__(self, cfg):
        self.cfg, self.rows, self.errors, self.cv_errors = cfg, [], [], []
        self.speed_max, self.acceleration_max = 0.0, 0.0
        self.speed_violations, self.acceleration_violations, self.steps = 0, 0, 0

    def add(self, output, batch):
        cfg = self.cfg
        if batch["future"].shape[1] == 0:
            return                      # IV-A: no horizon, nothing to score
        prediction = output["trajectory"].float()
        error = torch.linalg.vector_norm(prediction - batch["future"], dim=-1).cpu().numpy()
        baseline = constant_velocity(batch["observed"], cfg.prediction_steps, cfg.dt)
        cv = torch.linalg.vector_norm(baseline - batch["future"], dim=-1).cpu().numpy()
        self.errors.append(error)
        self.cv_errors.append(cv)
        for i, tid in enumerate(batch["track_id"]):
            near_peer = False
            if "context" in batch:
                distances = (batch["context"][i, ..., :3] - batch["observed"][i, None, :, :3]).norm(dim=-1)
                near_peer = bool(((distances <= cfg.interaction_radius_m) & batch["context_mask"][i]).any())
            self.rows.append(dict(track_id=tid, last_observed_time_s=float(batch["last_time"][i]),
                                  scene_id=batch.get("scene_id", batch["track_id"])[i],
                                  scenario=batch.get("scenario", ["independent"] * len(batch["track_id"]))[i],
                                  observed_nearby_peer=near_peer,
                                  ade_m=float(error[i].mean()), fde_m=float(error[i, -1]),
                                  cv_ade_m=float(cv[i].mean()), cv_fde_m=float(cv[i, -1]),
                                  observed_motion_class=int(batch["label"][i]) if "label" in batch else -1))
        velocity = torch.cat((batch["observed"][:, -1:, :3], prediction), 1).diff(dim=1) / cfg.dt
        acceleration = torch.cat((batch["observed"][:, -1:, 3:], velocity), 1).diff(dim=1) / cfg.dt
        speeds = torch.linalg.vector_norm(velocity, dim=-1)
        accelerations = torch.linalg.vector_norm(acceleration, dim=-1)
        self.speed_max = max(self.speed_max, float(speeds.max()))
        self.acceleration_max = max(self.acceleration_max, float(accelerations.max()))
        if cfg.max_speed_mps:
            self.speed_violations += int((speeds > cfg.max_speed_mps + 1e-3).sum())
        if cfg.max_acceleration_mps2:
            self.acceleration_violations += int((accelerations > cfg.max_acceleration_mps2 + 1e-3).sum())
        self.steps += speeds.numel()

    @staticmethod
    def classification_metrics(result):
        """Turn the confusion matrix into the IV-A reporting quantities.

        Adds macro F1, balanced accuracy, per-class precision/recall/F1 and the
        majority-class baseline (the accuracy floor a constant predictor
        reaches), which is what the paper's 88% / 94% statistics are measured
        against.
        """
        if "confusion_matrix" not in result:
            return result
        matrix = np.asarray(result["confusion_matrix"], dtype=float)
        support, predicted = matrix.sum(1), matrix.sum(0)
        recall = np.divide(matrix.diagonal(), support, out=np.zeros_like(support), where=support > 0)
        precision = np.divide(matrix.diagonal(), predicted, out=np.zeros_like(support), where=predicted > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(support),
                       where=precision + recall > 0)
        result["intention_macro_f1"] = float(f1.mean()) if support.sum() else None
        result["intention_balanced_accuracy"] = float(recall[support > 0].mean()) if support.sum() else None
        result["intention_majority_baseline"] = (float(support.max() / support.sum())
                                                if support.sum() else None)
        result["intention_classes_present"] = int(np.count_nonzero(support))
        result["confusion_matrix_row_normalized"] = np.divide(matrix, support[:, None],
            out=np.zeros_like(matrix), where=support[:, None] > 0).tolist()
        result["intention_per_class"] = [dict(class_id=i,
            name=CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i), support=int(support[i]),
            precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
            for i in range(len(support))]
        return result

    def finish(self, result, output_path=None):
        if not self.rows:
            # IV-A classification run: there is no prediction horizon to score,
            # but the classification metrics of the confusion matrix (macro F1,
            # per-class recall, majority-class baseline) must still be reported.
            self.classification_metrics(result)
            if output_path and "intention_per_class" in result:
                write_csv(Path(output_path).with_suffix("").with_name(
                    Path(output_path).with_suffix("").name + "_classes.csv"),
                    result["intention_per_class"])
            return result
        error, cv = np.concatenate(self.errors), np.concatenate(self.cv_errors)
        ids = np.asarray([row["track_id"] for row in self.rows])
        track_rows, sums, counts = [], [], []
        for tid in sorted(set(ids)):
            values = np.asarray([[self.rows[i][key] for key in ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m")]
                                 for i in np.flatnonzero(ids == tid)])
            means = values.mean(0)
            track_rows.append(dict(track_id=tid, windows=len(values), **dict(zip(
                ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m"), means.tolist()))))
            sums.append(values.sum(0))
            counts.append(len(values))
        coupled = any(row["scene_id"] != row["track_id"] for row in self.rows)
        scene_rows = []
        if coupled:
            sums, counts = [], []
            for scene in sorted({row["scene_id"] for row in self.rows}):
                selected = [row for row in self.rows if row["scene_id"] == scene]
                values = np.asarray([[row[key] for key in ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m")] for row in selected])
                sums.append(values.sum(0)); counts.append(len(values))
                scene_rows.append(dict(scene_id=scene, scenario=selected[0]["scenario"], windows=len(values),
                                       **dict(zip(("ade_m", "fde_m", "cv_ade_m", "cv_fde_m"), values.mean(0).tolist()))))
        sums, counts = np.asarray(sums), np.asarray(counts)
        # Preserve the window-weighted estimator; resample whole independent groups.
        rng = np.random.default_rng(self.cfg.split_seed)
        indices = rng.integers(len(counts), size=(1000, len(counts)))
        draws = sums[indices].sum(1) / counts[indices].sum(1)[:, None]
        bounds = np.quantile(draws, [0.025, 0.975], axis=0)
        result["scene_cluster_bootstrap_95ci" if coupled else "track_cluster_bootstrap_95ci"] = dict(zip(
            ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m"), bounds.T.tolist()))
        result["bootstrap"] = {"replicates": 1000, "seed": self.cfg.split_seed,
                               "unit": "independent scene" if coupled else "independent track",
                               "clusters": len(counts), "tracks": len(track_rows), "estimator": "window weighted mean"}
        if coupled:
            scenario_rows = []
            for scenario in sorted({row["scenario"] for row in self.rows}):
                selected = [row for row in self.rows if row["scenario"] == scenario]
                scenario_rows.append(dict(scenario=scenario, windows=len(selected), scenes=len({row["scene_id"] for row in selected}),
                                          **{key: float(np.mean([row[key] for row in selected])) for key in ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m")}))
            result["error_by_scenario"] = scenario_rows
            interaction_rows = []
            for nearby in (False, True):
                selected = [row for row in self.rows if row["observed_nearby_peer"] == nearby]
                if selected:
                    interaction_rows.append(dict(observed_nearby_peer=nearby, windows=len(selected),
                        scenes=len({row["scene_id"] for row in selected}),
                        **{key: float(np.mean([row[key] for row in selected])) for key in ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m")}))
            result["error_by_observed_interaction"] = interaction_rows
        result["track_mean_ade_m"] = float(np.mean([r["ade_m"] for r in track_rows]))
        result["track_mean_fde_m"] = float(np.mean([r["fde_m"] for r in track_rows]))
        horizon = [dict(step=i + 1, horizon_s=(i + 1) * self.cfg.dt,
                        displacement_m=float(error[:, i].mean()), cv_displacement_m=float(cv[:, i].mean()))
                   for i in range(error.shape[1])]
        result["error_by_horizon"] = horizon
        motion_rows = []
        for label in sorted({row["observed_motion_class"] for row in self.rows}):
            selected = [row for row in self.rows if row["observed_motion_class"] == label]
            motion_rows.append(dict(class_id=label, name=CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "unknown",
                windows=len(selected), tracks=len({row["track_id"] for row in selected}),
                **{key: float(np.mean([row[key] for row in selected])) for key in ("ade_m", "fde_m", "cv_ade_m", "cv_fde_m")}))
        result["error_by_observed_motion_class"] = motion_rows
        result["displacement_rmse_m"] = float(np.sqrt(np.mean(error ** 2)))
        result["ade_improvement_over_cv_percent"] = (100 * (1 - result["ade_m"] / result["cv_ade_m"])
                                                    if result["cv_ade_m"] > 1e-12 else None)
        result["physical_diagnostics"] = dict(max_speed_mps=self.speed_max, max_acceleration_mps2=self.acceleration_max,
            speed_violation_fraction=self.speed_violations / self.steps if self.cfg.max_speed_mps else None,
            acceleration_violation_fraction=self.acceleration_violations / self.steps if self.cfg.max_acceleration_mps2 else None)
        self.classification_metrics(result)
        if output_path:
            prefix = Path(output_path).with_suffix("")
            artifacts = {}
            for suffix, rows in (("windows", self.rows), ("tracks", track_rows), ("horizon", horizon),
                                 ("classes", result["intention_per_class"])):
                path = prefix.with_name(prefix.name + "_" + suffix + ".csv")
                write_csv(path, rows)
                artifacts[suffix] = str(path.resolve())
            result["artifacts"] = artifacts
            path = prefix.with_name(prefix.name + "_motion.csv")
            write_csv(path, motion_rows)
            result["artifacts"]["motion"] = str(path.resolve())
            if coupled:
                for suffix, rows in (("scenes", scene_rows), ("scenarios", scenario_rows), ("interaction", interaction_rows)):
                    path = prefix.with_name(prefix.name + "_" + suffix + ".csv")
                    write_csv(path, rows)
                    result["artifacts"][suffix] = str(path.resolve())
        return result
