"""Reproducible CPU/CUDA training, validation, resume, and held-out evaluation."""

import csv
import hashlib
import json
import platform
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import Windows, fingerprint, fit_normalizer, load_tracks, split_tracks, validate_split
from .model import build_forecaster, constant_velocity
from .objective import displacement_errors, TrajectoryObjective
from .learner import TrajectoryLearner
from .metrics import EvaluationDetails


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def choose_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu or install CUDA PyTorch")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
    return device


def configure_runtime(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(device):
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "code_sha256": source_fingerprint(),
            "xuance": getattr(sys.modules.get("xuance"), "__version__", None)}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    Config(**checkpoint["config"]).validate()
    return checkpoint


def atomic_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_loader(dataset, cfg, device, training=False, generator=None):
    # micro_batch_size shrinks only the per-forward batch; the effective
    # optimization batch stays cfg.batch_size via gradient accumulation in
    # TrajectoryLearner (paper pins batch 256).
    return DataLoader(dataset, batch_size=cfg.micro_batch_size or cfg.batch_size, shuffle=training,
                      num_workers=cfg.workers, pin_memory=device.type == "cuda",
                      generator=generator, persistent_workers=False)


def run_epoch(model, loader, cfg, device, optimizer=None, include_confusion=False, learner=None, details=None):
    training = optimizer is not None
    model.train(training)
    totals = {k: 0.0 for k in ("loss", "ade_m", "fde_m", "cross_entropy", "acceleration_mse",
                                "attention_smoothness", "attention_velocity", "attention_temporal", "attention_separation",
                                "cv_ade_m", "cv_fde_m")}
    samples, labeled, correct = 0, 0, 0
    phases = {name: [0.0, 0] for name in ("pre_evasion_ade_m", "pre_evasion_fde_m",
                                          "evasion_ade_m", "evasion_fde_m")}
    confusion = torch.zeros(cfg.intention_classes, cfg.intention_classes, dtype=torch.int64)
    objective = learner.objective if learner else TrajectoryObjective(cfg).to(device)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = to_device(batch, device)
            if training and learner is not None:
                output, losses = learner.update(batch)
            elif learner is not None:
                output, losses = learner.compute(batch)
            else:
                output = model(batch["observed"], context=batch.get("context"), context_mask=batch.get("context_mask"),
                               engagement=batch.get("engagement"))
                losses = objective(output, batch)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError("Nonfinite objective; check data, units, dt and learning rate")
            if training and learner is None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
            count = len(batch["observed"])
            if details is not None:
                details.add(output, batch)
            for name, value in losses.items():
                if name != "cross_entropy":
                    totals[name] += float(value.detach()) * count
            if batch["future"].shape[1]:
                baseline = constant_velocity(batch["observed"], cfg.prediction_steps, cfg.dt)
                cv_ade, cv_fde = displacement_errors(baseline, batch["future"])
                totals["cv_ade_m"] += float(cv_ade.sum())
                totals["cv_fde_m"] += float(cv_fde.sum())
            if "evaded" in batch and batch["future"].shape[1]:
                # Military reporting: errors before and after the evasion trigger.
                ade, fde = displacement_errors(output["trajectory"].detach(), batch["future"])
                active = batch["evaded"].bool()
                for name, values, mask in (("pre_evasion_ade_m", ade, ~active),
                                           ("pre_evasion_fde_m", fde, ~active),
                                           ("evasion_ade_m", ade, active),
                                           ("evasion_fde_m", fde, active)):
                    phases[name][0] += float(values[mask].sum())
                    phases[name][1] += int(mask.sum())
            known = (batch["label"] >= 0) & bool(model.supports_intention and (cfg.use_intention or cfg.stage == "intention"))
            totals["cross_entropy"] += float(losses["cross_entropy"].detach()) * int(known.sum())
            guesses = output["logits"].argmax(-1)
            labeled += int(known.sum())
            correct += int((guesses[known] == batch["label"][known]).sum())
            if include_confusion and known.any():
                encoded = (batch["label"][known] * cfg.intention_classes + guesses[known]).cpu()
                confusion += torch.bincount(encoded, minlength=cfg.intention_classes ** 2).reshape_as(confusion)
            samples += count
        # Gradient accumulation: step the trailing partial group so every
        # epoch ends with a fully applied optimizer step, matching the
        # partial last batch a plain batch_size DataLoader would take.
        if training and learner is not None and getattr(learner, "accum_steps", 1) > 1:
            learner.flush()
    result = {name: value / samples for name, value in totals.items()}
    result["cross_entropy"] = totals["cross_entropy"] / labeled if labeled else 0.0
    result.update(samples=samples, labeled_samples=labeled,
                  intention_accuracy=correct / labeled if labeled else None)
    if include_confusion:
        result["confusion_matrix"] = confusion.tolist()
    if any(count for _, count in phases.values()):
        result["evasion_phases"] = {
            "pre_evasion": dict(ade_m=phases["pre_evasion_ade_m"][0] / max(phases["pre_evasion_ade_m"][1], 1),
                                fde_m=phases["pre_evasion_fde_m"][0] / max(phases["pre_evasion_fde_m"][1], 1),
                                samples=phases["pre_evasion_ade_m"][1]),
            "evasion": dict(ade_m=phases["evasion_ade_m"][0] / max(phases["evasion_ade_m"][1], 1),
                            fde_m=phases["evasion_fde_m"][0] / max(phases["evasion_fde_m"][1], 1),
                            samples=phases["evasion_ade_m"][1])}
    return result


def train(data_path, output_path, cfg, resume=None, intention_checkpoint=None,
          policy_factory=build_forecaster, learner_factory=TrajectoryLearner, integration=None):
    output_path, data_path = Path(output_path).resolve(), Path(data_path).resolve()
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        if Path(resume).resolve().parent != output_path:
            raise ValueError("Resume in the checkpoint's existing output directory to preserve logs")
        if intention_checkpoint:
            raise ValueError("Cannot both resume and import an intention checkpoint")
        previous = Config(**checkpoint["config"])
        if cfg.schedule_epochs == 0:
            cfg.schedule_epochs = previous.schedule_epochs
        # Only runtime settings and the requested final epoch count may change on resume.
        permitted = {"epochs", "device", "workers", "cpu_threads"}
        if any(getattr(previous, k) != getattr(cfg, k) for k in cfg.to_dict() if k not in permitted):
            raise ValueError("Resume configuration differs in architecture, data, loss, or optimizer settings")
        if cfg.epochs <= checkpoint["epoch"]:
            raise ValueError("--epochs must exceed the completed checkpoint epoch")
    elif output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new path or --resume last.pt")
    if cfg.freeze_intention and not (intention_checkpoint or checkpoint):
        raise ValueError("Frozen intention encoder requires a pretrained checkpoint")
    configure_runtime(cfg)
    if cfg.schedule == "warmup_cosine" and cfg.schedule_epochs == 0:
        cfg.schedule_epochs = cfg.epochs
    device = choose_device(cfg.device)
    digest = fingerprint(data_path)
    tracks = load_tracks(data_path, cfg.dt, cfg.observation_steps + cfg.prediction_steps, cfg.intention_classes)
    holdout = bool(getattr(cfg, "holdout_test", True))
    splits = checkpoint["splits"] if checkpoint else split_tracks(tracks, cfg.train_fraction,
                                                                 cfg.validation_fraction, cfg.split_seed,
                                                                 holdout)
    if checkpoint and checkpoint["dataset_sha256"] != digest:
        raise ValueError("Resume dataset differs from the checkpoint dataset")
    validate_split(tracks, splits, holdout)
    # IV-B protocol keeps no test partition; Windows refuse empty id lists.
    datasets = {name: Windows(tracks, ids, cfg) for name, ids in splits.items() if name != "test" or ids}
    if cfg.stage in {"joint", "intention"} and datasets["train"].labeled_count() == 0:
        raise ValueError("Intention training needs observed-window labels; use --stage forecast for unlabeled data")
    normalizer = checkpoint["normalizer"] if checkpoint else fit_normalizer(datasets["train"])
    model = policy_factory(cfg, normalizer).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if intention_checkpoint:
        pretrained = load_checkpoint(intention_checkpoint)
        pre_config = Config(**pretrained["config"])
        for key in ("lstm_hidden", "projection_dim", "intention_classes", "observation_steps", "dt",
                    "relative_frame", "static_state_branch", "context_entities", "intention_task",
                    "engagement_edges"):
            # engagement_edges decides whether the static branch splices the
            # 9-dim engagement descriptors (A2): a mismatch is an incompatible
            # architecture, not a loadable state_dict.
            if getattr(pre_config, key) != getattr(cfg, key):
                raise ValueError(f"Intention checkpoint has incompatible {key}")
        if pretrained["dataset_sha256"] != digest or pretrained["splits"] != splits:
            raise ValueError("Intention checkpoint must use the same dataset and track partitions")
        model.intention.load_state_dict({key.removeprefix("intention."): value
                                        for key, value in pretrained["model"].items() if key.startswith("intention.")})
        # Transfer the exact train-fitted feature statistics with the classifier.
        for key, value in pretrained["normalizer"].items():
            normalizer[key] = value
        model.feature_mean.copy_(normalizer["mean"].to(device))
        model.feature_std.copy_(normalizer["std"].to(device))
    learner = learner_factory(cfg, model, device, steps_per_epoch=(len(datasets["train"]) + cfg.batch_size - 1) // cfg.batch_size)
    optimizer, scheduler = learner.optimizer, learner.scheduler
    generator = torch.Generator().manual_seed(cfg.seed)
    start, best = 1, float("inf")
    if checkpoint:
        learner.restore(checkpoint)
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_validation_score"]
    loaders = {name: make_loader(dataset, cfg, device, name == "train", generator if name == "train" else None)
               for name, dataset in datasets.items() if name != "test"}
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "config.json", cfg.to_dict())
    write_json(output_path / ("resume_environment.json" if checkpoint else "environment.json"), environment(device))
    write_json(output_path / "manifest.json", {"dataset": str(data_path), "dataset_sha256": digest,
               "splits": splits, "window_counts": {k: len(v) for k, v in datasets.items()},
               "scene_counts": {k: len({t.scene_id or t.track_id for t in tracks if t.track_id in ids}) for k, ids in splits.items()},
               "normalizer": {k: v.tolist() for k, v in normalizer.items()},
               "selection": "validation cross entropy" if cfg.stage == "intention" else "validation " + cfg.selection_metric + " (m)",
               "selection_metric": cfg.selection_metric, "integration": integration,
               "parameters": sum(p.numel() for p in model.parameters()),
               "intention_checkpoint": str(Path(intention_checkpoint).resolve()) if intention_checkpoint else None})
    history = output_path / "history.csv"
    if checkpoint and history.exists():
        with history.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows and int(rows[-1]["epoch"]) != checkpoint["epoch"]:
            raise ValueError("History and checkpoint epochs disagree; resume from last.pt")
    # An extra validation-loader iteration would consume global RNG on resume.
    initial = checkpoint["validation"] if checkpoint else run_epoch(model, loaders["validation"], cfg, device, learner=learner)
    if not checkpoint:
        write_json(output_path / "initial_validation.json", initial)
    metric_names = ["loss", "ade_m", "fde_m", "cross_entropy", "acceleration_mse", "attention_smoothness",
                    "attention_velocity", "attention_temporal", "attention_separation", "cv_ade_m", "cv_fde_m",
                    "samples", "labeled_samples", "intention_accuracy"]
    fields = ["epoch", "phase", "learning_rate", "seconds"] + metric_names
    if checkpoint and history.exists():
        with history.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            old_fields, rows = reader.fieldnames, list(reader)
        if old_fields != fields:
            with history.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, restval="", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
    stale = checkpoint.get("stale_epochs", 0) if checkpoint else 0
    significant_best = checkpoint.get("significant_best", best) if checkpoint else float("inf")
    stop_anchor = checkpoint.get("early_stop_anchor", [float("inf"), float("inf")]) if checkpoint else [float("inf"), float("inf")]
    started = time.perf_counter()
    for epoch in range(start, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        rate = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, loaders["train"], cfg, device, optimizer, learner=learner)
        validation = run_epoch(model, loaders["validation"], cfg, device, learner=learner)
        seconds = time.perf_counter() - epoch_started
        with history.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if stream.tell() == 0:
                writer.writeheader()
            for phase, values in (("train", training), ("validation", validation)):
                # History stays flat: nested reports (e.g. evasion phases) are
                # written to the metrics JSON, not to the per-epoch CSV.
                flat = {k: v for k, v in values.items() if k in fields}
                writer.writerow(dict(epoch=epoch, phase=phase, learning_rate=rate, seconds=seconds, **flat))
        score = (validation["cross_entropy"] if cfg.stage == "intention" else validation["ade_m"]
                 if cfg.selection_metric == "ade" else cfg.ade_weight * validation["ade_m"] + cfg.fde_weight * validation["fde_m"])
        improved = score < best
        if improved:
            best = score
        threshold = cfg.early_stopping_min_delta_m if cfg.stage != "intention" else 0.0
        significant = score < significant_best - threshold
        if cfg.early_stopping_mode == "both" and cfg.stage != "intention":
            significant = all(validation[k] < anchor - threshold for k, anchor in zip(("ade_m", "fde_m"), stop_anchor))
        if significant:
            significant_best, stale = score, 0
            stop_anchor = [validation["ade_m"], validation["fde_m"]]
        else:
            stale += 1
        scheduler.step()
        saved = {"format_version": 1, "config": cfg.to_dict(), "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "normalizer": normalizer,
                 "splits": splits, "dataset_sha256": digest, "best_validation_score": best,
                 "stale_epochs": stale, "significant_best": significant_best,
                 "early_stop_anchor": stop_anchor,
                 "validation": validation, "loader_rng": generator.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "environment": environment(device), **learner.state_dict()}
        atomic_checkpoint(output_path / "last.pt", saved)
        if improved:
            atomic_checkpoint(output_path / "best.pt", saved)
        print(json.dumps({"epoch": epoch, "train_loss": training["loss"], "validation": validation,
                          "seconds": round(seconds, 3)}, allow_nan=False), flush=True)
        if cfg.early_stopping_patience and stale >= cfg.early_stopping_patience:
            break
    result = {"completed_epoch": epoch, "requested_epochs": cfg.epochs,
              "early_stopped": epoch < cfg.epochs, "best_validation_score": best,
              "elapsed_seconds": time.perf_counter() - started, "output": str(output_path),
              "stage": cfg.stage, "test_evaluated": False}
    write_json(output_path / "training_summary.json", result)
    return result


def evaluate(data_path, checkpoint_path, split="test", device_name="auto", output_path=None,
             policy_factory=build_forecaster, learner_factory=TrajectoryLearner):
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = Config(**checkpoint["config"]).validate()
    configure_runtime(cfg)
    device = choose_device(device_name)
    if fingerprint(data_path) != checkpoint["dataset_sha256"]:
        raise ValueError("Evaluation requires the original dataset; fingerprint mismatch")
    tracks = load_tracks(data_path, cfg.dt, cfg.observation_steps + cfg.prediction_steps, cfg.intention_classes)
    holdout = bool(getattr(cfg, "holdout_test", True))
    validate_split(tracks, checkpoint["splits"], holdout)
    # IV-B has no test partition: reporting falls back to the held-out validation set.
    if not checkpoint["splits"].get(split):
        split = "validation"
    dataset = Windows(tracks, checkpoint["splits"][split], cfg)
    model = policy_factory(cfg, checkpoint["normalizer"]).to(device)
    model.load_state_dict(checkpoint["model"])
    learner = learner_factory(cfg, model, device)
    if "criterion" in checkpoint:
        learner.objective.load_state_dict(checkpoint["criterion"])
    details = EvaluationDetails(cfg)
    result = run_epoch(model, make_loader(dataset, cfg, device), cfg, device, include_confusion=True, learner=learner, details=details)
    details.finish(result, output_path)
    result.update(split=split, epoch=checkpoint["epoch"], checkpoint=str(Path(checkpoint_path).resolve()),
                  dataset_sha256=checkpoint["dataset_sha256"], stage=cfg.stage,
                  protocol=dict(units="metres and seconds", dt_s=cfg.dt,
                                observation_steps=cfg.observation_steps, prediction_steps=cfg.prediction_steps,
                                forecast_horizon_s=cfg.prediction_steps * cfg.dt, stride=cfg.stride,
                                model_type=cfg.model_type, relative_frame=cfg.relative_frame))
    result["protocol"].update(context_entities=cfg.context_entities,
                               context_enabled=cfg.context_entities > 0 and cfg.use_interaction_context and cfg.model_type == "graph",
                               split_unit="independent scene" if cfg.context_entities else "independent track")
    if output_path:
        write_json(output_path, result)
    return result
