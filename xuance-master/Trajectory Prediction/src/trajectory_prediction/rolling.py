"""Rolling civilian forecasts from measured states, never fed-back predictions."""
from collections import deque
import csv
from pathlib import Path
import time

import numpy as np
import torch

from .data import fingerprint, load_tracks, validate_split
from .engine import load_checkpoint, write_json
from .inference import Predictor
from .metrics import EvaluationDetails
from .model import constant_velocity
from .objective import displacement_errors


class RollingPredictor:
    def __init__(self, checkpoint, device="auto", refresh_steps=1):
        if not isinstance(refresh_steps, int) or refresh_steps < 1:
            raise ValueError("refresh_steps must be a positive integer")
        self.predictor = checkpoint if isinstance(checkpoint, Predictor) else Predictor(checkpoint, device)
        self.cfg, self.refresh_steps = self.predictor.cfg, refresh_steps
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.cfg.observation_steps)
        self.context_history = deque(maxlen=self.cfg.observation_steps)
        self.context_availability = deque(maxlen=self.cfg.observation_steps)
        self.engagement_history = deque(maxlen=self.cfg.observation_steps)
        self.last_time, self.received, self.forecasts = None, 0, 0

    def push(self, timestamp, state, context=None, context_mask=None, engagement=None):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all() or not np.isfinite(timestamp):
            raise ValueError("Need a finite timestamp and six finite measured state values")
        timestamp = float(timestamp)
        if self.last_time is not None and not np.isclose(timestamp - self.last_time, self.cfg.dt, rtol=1e-5, atol=1e-6):
            raise ValueError("Measurements must have strictly increasing, regular dt timestamps")
        if self.cfg.context_entities:
            if context is None:
                raise ValueError("Provide synchronized peer measurements; use a false mask when unavailable")
            context = np.asarray(context, dtype=np.float32)
            mask = np.ones(self.cfg.context_entities, dtype=bool) if context_mask is None else np.asarray(context_mask)
            if context.shape != (self.cfg.context_entities, 6) or mask.shape != (self.cfg.context_entities,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("Current peer states/masks must match [N,6]/[N]")
            mask = mask.astype(bool)
            if not np.isfinite(context[mask]).all():
                raise ValueError("Available peer measurements must be finite")
            self.context_history.append(np.where(mask[:, None], context, 0).copy())
            self.context_availability.append(mask.copy())
        elif context is not None or context_mask is not None:
            raise ValueError("This checkpoint has no context entities")
        if engagement is not None:
            engagement = np.asarray(engagement, dtype=np.float32)
            if engagement.shape != (9,) or not np.isfinite(engagement).all():
                raise ValueError("Engagement descriptors must be nine finite values")
            self.engagement_history.append(engagement.copy())
        self.history.append(state.copy())
        self.last_time, self.received = timestamp, self.received + 1
        if len(self.history) < self.cfg.observation_steps or (self.received - self.cfg.observation_steps) % self.refresh_steps:
            return None
        started = time.perf_counter()
        prediction = self.predictor.predict(np.stack(self.history),
            np.stack(self.context_history, axis=1) if self.cfg.context_entities else None,
            np.stack(self.context_availability, axis=1) if self.cfg.context_entities else None,
            np.stack(self.engagement_history) if len(self.engagement_history) == self.cfg.observation_steps else None)
        self.forecasts += 1
        prediction.update(last_time_s=timestamp,
                          future_time_s=timestamp + np.arange(1, self.cfg.prediction_steps + 1) * self.cfg.dt,
                          inference_seconds=time.perf_counter() - started, forecast_index=self.forecasts)
        return prediction


def rolling_evaluate(data, checkpoint, output, split="test", device="auto", refresh_steps=1):
    saved = load_checkpoint(checkpoint)
    predictor = Predictor(checkpoint, device)
    cfg = predictor.cfg
    if fingerprint(data) != saved["dataset_sha256"]:
        raise ValueError("Rolling held-out evaluation requires the checkpoint's original dataset")
    tracks = load_tracks(data, cfg.dt, cfg.observation_steps + cfg.prediction_steps, cfg.intention_classes)
    validate_split(tracks, saved["splits"], bool(getattr(cfg, "holdout_test", True)))
    from .data import Windows, window_intention_label
    # IV-B keeps no test partition: rolling evaluation falls back to the held-out
    # validation set, mirroring engine.evaluate.
    if not saved["splits"].get(split):
        split = "validation"
    Windows(tracks, saved["splits"][split], cfg)  # Validate context protocol before mutating output.
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Rolling output directory must be empty")
    details = EvaluationDetails(cfg)
    confusion = torch.zeros(cfg.intention_classes, cfg.intention_classes, dtype=torch.int64)
    latency, labeled, correct = [], 0, 0
    total_ade, total_fde, total_cv_ade, total_cv_fde, samples = 0., 0., 0., 0., 0
    stream = RollingPredictor(predictor, refresh_steps=refresh_steps)
    for track in tracks:
        if track.track_id not in saved["splits"][split]:
            continue
        stream.reset()
        # Only complete future windows are scored; truth is accessed AFTER prediction.
        for index in range(len(track.states) - cfg.prediction_steps):
            predicted = stream.push(track.time[index], track.states[index],
                                    track.context[index] if cfg.context_entities else None,
                                    track.context_mask[index] if cfg.context_entities else None,
                                    track.engagement[index] if track.engagement is not None and cfg.engagement_edges else None)
            if predicted is None:
                continue
            observed = track.states[index - cfg.observation_steps + 1:index + 1]
            truth = track.states[index + 1:index + cfg.prediction_steps + 1, :3]
            label = window_intention_label(track, index - cfg.observation_steps + 1, index + 1,
                                           cfg.intention_classes, cfg.intention_task)
            batch = dict(observed=torch.from_numpy(observed)[None], future=torch.from_numpy(truth)[None],
                         track_id=[track.track_id], last_time=torch.tensor([track.time[index]], dtype=torch.float64),
                         scene_id=[track.scene_id or track.track_id], scenario=[track.scenario],
                         label=torch.tensor([label]))
            if cfg.context_entities:
                batch.update(context=torch.from_numpy(track.context[index - cfg.observation_steps + 1:index + 1].transpose(1, 0, 2).copy())[None],
                             context_mask=torch.from_numpy(track.context_mask[index - cfg.observation_steps + 1:index + 1].T.copy())[None])
            prediction = torch.from_numpy(predicted["trajectory"])[None]
            details.add(dict(trajectory=prediction), batch)
            ade, fde = displacement_errors(prediction, batch["future"])
            cv = constant_velocity(batch["observed"], cfg.prediction_steps, cfg.dt)
            cv_ade, cv_fde = displacement_errors(cv, batch["future"])
            total_ade += float(ade[0]); total_fde += float(fde[0])
            total_cv_ade += float(cv_ade[0]); total_cv_fde += float(cv_fde[0])
            if label >= 0 and predicted["intention_probabilities"] is not None:
                guess = int(predicted["intention_probabilities"].argmax())
                confusion[label, guess] += 1
                labeled += 1
                correct += int(label == guess)
            positions = []
            for step, (position, target) in enumerate(zip(predicted["trajectory"], truth)):
                positions.append(dict(track_id=track.track_id, last_observed_time_s=track.time[index],
                    horizon_step=step + 1, time_s=predicted["future_time_s"][step],
                    x=position[0], y=position[1], z=position[2], truth_x=target[0], truth_y=target[1], truth_z=target[2]))
            output.mkdir(parents=True, exist_ok=True)
            with (output / "rolling_predictions.csv").open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(positions[0]))
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerows(positions)
            latency.append(predicted["inference_seconds"])
            samples += 1
    result = dict(ade_m=total_ade / samples, fde_m=total_fde / samples,
                  cv_ade_m=total_cv_ade / samples, cv_fde_m=total_cv_fde / samples,
                  samples=samples, labeled_samples=labeled, intention_accuracy=correct / labeled if labeled else None,
                  confusion_matrix=confusion.tolist(), split=split, epoch=saved["epoch"],
                  dataset_sha256=saved["dataset_sha256"], checkpoint=str(Path(checkpoint).resolve()),
                  protocol=dict(mode="rolling measured history", refresh_steps=refresh_steps,
                                refresh_interval_s=refresh_steps * cfg.dt, observation_steps=cfg.observation_steps,
                                prediction_steps=cfg.prediction_steps, dt_s=cfg.dt, future_inputs_used=False),
                  latency_seconds=dict(mean=float(np.mean(latency)), median=float(np.median(latency)), p95=float(np.quantile(latency, .95))))
    details.finish(result, output / "rolling_metrics.json")
    result["protocol"]["context_entities"] = cfg.context_entities
    result["artifacts"]["predictions"] = str(output / "rolling_predictions.csv")
    write_json(output / "rolling_metrics.json", result)
    return result
