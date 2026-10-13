"""Reproducible two-stage and controlled baseline experiment workflows."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .engine import train, evaluate, load_checkpoint, write_json
from .metrics import write_csv
from .xuance_bridge import train_native, evaluate_native


def train_backend(backend, *args, **kwargs):
    return (train_native if backend == "xuance" else train)(*args, **kwargs)


def evaluate_backend(backend, data, checkpoint, split="test", device="auto", output=None):
    if backend == "xuance":
        from .config import Config
        return evaluate_native(data, checkpoint, Config(**load_checkpoint(checkpoint)["config"]), split, device, output)
    return evaluate(data, checkpoint, split, device, output)


def stage_train(data, output, cfg, backend, allow_resume=False, intention=None):
    output = Path(output)
    last = output / "last.pt"
    if allow_resume and last.exists():
        saved = load_checkpoint(last)
        # A resumed pipeline keeps its original experiment settings.
        permitted = {"device", "workers", "cpu_threads", "epochs"}
        expected = cfg.to_dict()
        if expected["schedule_epochs"] == 0:
            expected["schedule_epochs"] = saved["config"].get("schedule_epochs", 0)
        if any(value != saved["config"].get(key, value) for key, value in expected.items() if key not in permitted):
            raise ValueError("Resumed workflow configuration differs from saved experiment")
        from .data import fingerprint
        if saved["dataset_sha256"] != fingerprint(data):
            raise ValueError("Resumed workflow dataset differs")
        summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        if saved["epoch"] >= cfg.epochs or (summary.get("early_stopped") and summary["requested_epochs"] == cfg.epochs):
            return summary
        return train_backend(backend, data, output, cfg, resume=last)
    return train_backend(backend, data, output, cfg, intention_checkpoint=intention)


def pipeline(data, output, cfg, intention_epochs=70, backend="xuance", resume=False):
    if cfg.model_type != "graph" or not cfg.use_intention:
        raise ValueError("Two-stage intention conditioning requires the graph model")
    output = Path(output).resolve()
    classifier = classifier_config(cfg, intention_epochs)
    forecast = replace(cfg, stage="forecast", freeze_intention=True)
    result = {"backend": backend, "data": str(Path(data).resolve()), "same_observation_protocol": True,
              "intention_training": stage_train(data, output / "intention", classifier.validate(), backend, resume)}
    write_json(output / "pipeline.json", result)
    result["forecast_training"] = stage_train(data, output / "forecast", forecast.validate(), backend, resume,
                                             output / "intention/best.pt")
    result["intention_test"] = evaluate_backend(backend, data, output / "intention/best.pt",
                                               device=cfg.device, output=output / "intention/test_metrics.json")
    result["forecast_test"] = evaluate_backend(backend, data, output / "forecast/best.pt",
                                              device=cfg.device, output=output / "forecast/test_metrics.json")
    result["prediction_checkpoint"] = str(output / "forecast/best.pt")
    write_json(output / "pipeline.json", result)
    return result


def classifier_config(cfg, epochs):
    # IV-A is the BiLSTM intention-recognition stage (Section III-A, Eqs. 12-14).
    # The engagement teacher scores supervise the *graph attention* heads of the
    # trajectory stage (Definition 1, Eqs. 36-53), so they are switched off here;
    # keeping them on with attention_supervision=False trips the config check and
    # would otherwise leak IV-B supervision into the intention stage.
    return replace(cfg, model_type="graph", stage="intention", use_intention=True, freeze_intention=False, epochs=epochs,
                   optimizer="adam", schedule="step", schedule_epochs=0, loss_balance="fixed",
                   attention_supervision=False, military_teachers=False, teacher_edge_subsets=False,
                   selection_metric="ade", early_stopping_mode="weighted")


def compare(data, output, cfg, models, seeds, backend="xuance", resume=False,
            graph_training="two_stage", intention_epochs=70):
    valid = {"graph", "lstm", "gru", "tcn", "transformer", "graph_no_intention", "graph_no_attention_supervision", "graph_no_both",
             "graph_no_context", "graph_no_intention_context", "graph_no_engagement", "graph_no_intention_engagement"}
    engagement_ablations = {"graph_no_engagement", "graph_no_intention_engagement"}
    if not models or set(models) - valid or not seeds:
        raise ValueError(f"Models must be drawn from {sorted(valid)} and seeds cannot be empty")
    if len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate models or seeds")
    if any(model in engagement_ablations for model in models) and not (
            cfg.engagement_edges or cfg.military_teachers or cfg.candidate_pruning):
        raise ValueError("Engagement ablations need engagement edges, military teachers or candidate cropping enabled")
    if not cfg.context_entities and any(model in {"graph_no_context", "graph_no_intention_context"} for model in models):
        raise ValueError("Interaction ablations need context_entities>0")
    if not cfg.use_interaction_context and any(model in {"graph_no_context", "graph_no_intention_context"} for model in models):
        raise ValueError("The full model must enable context for interaction ablations")
    output = Path(output).resolve()
    if graph_training not in {"joint", "two_stage"}:
        raise ValueError("graph_training must be joint or two_stage")
    if any(model in {"graph_no_intention", "graph_no_attention_supervision", "graph_no_both"} for model in models):
        if not cfg.attention_supervision:
            raise ValueError("Ablation comparisons require attention_supervision=true in the full configuration")
    pretrained = {}
    if graph_training == "two_stage" and any(model.startswith("graph") for model in models):
        # The E axis changes the IntentionEncoder input width (the A2 fix splices
        # the 9-dim engagement descriptors into the static branch, Eqs. 12-14),
        # and importing engagement-trained weights into the E-off quadrants
        # would both mismatch dimensions and leak engagement information into
        # the ablation it is supposed to remove.  Train one classifier per
        # (model seed, E-axis setting); the E-on variant keeps the legacy
        # _intention/seed_<s> path so --resume reuses earlier checkpoints.
        graph_models = [model for model in models if model.startswith("graph")]
        for seed in seeds:
            for engaged in sorted({model not in engagement_ablations for model in graph_models}):
                path = output / "_intention" / (f"seed_{seed}" if engaged else f"plain_seed_{seed}")
                # The E-off classifier mirrors the E-off quadrants: no engagement
                # descriptors anywhere (edges, A7 blocking included).
                variant = classifier_config(replace(cfg, seed=seed,
                                                    engagement_edges=cfg.engagement_edges and engaged,
                                                    overload_reachability=cfg.overload_reachability and engaged),
                                            intention_epochs)
                stage_train(data, path, variant.validate(), backend, resume)
                pretrained[(seed, engaged)] = path / "best.pt"
    rows, partitions = [], None
    for model in models:
        for seed in seeds:
            graph = model.startswith("graph")
            # Military I/E quadrants: "engagement" covers the engagement-descriptor
            # edges, the candidate cropper and the E/R/T teachers; without it the
            # defender is still a context entity but modelled as a generic
            # neighbour under the civilian teachers.
            engagement = model not in engagement_ablations
            settings = replace(cfg, model_type="graph" if graph else model, seed=seed,
                stage="joint" if graph else "forecast", freeze_intention=False,
                attention_supervision=cfg.attention_supervision if graph else False,
                use_intention=model not in {"graph_no_intention", "graph_no_both",
                                           "graph_no_intention_context", "graph_no_intention_engagement"},
                use_interaction_context=cfg.use_interaction_context and model not in {"graph_no_context", "graph_no_intention_context"},
                engagement_edges=cfg.engagement_edges and engagement,
                # A7 hard blocking belongs to the engagement-edge pathway, so
                # the E-off ablation quadrant disables it with the edges.
                overload_reachability=cfg.overload_reachability and engagement,
                candidate_pruning=cfg.candidate_pruning and engagement,
                military_teachers=cfg.military_teachers and engagement)
            if graph and graph_training == "two_stage":
                settings.stage, settings.freeze_intention = "forecast", True
            if not settings.use_intention:
                settings.stage = "forecast"
            if model in {"graph_no_attention_supervision", "graph_no_both"}:
                settings.attention_supervision = False
            path = output / model / f"seed_{seed}"
            stage_train(data, path, settings.validate(), backend, resume,
                        pretrained.get((seed, model not in engagement_ablations)) if graph else None)
            saved = load_checkpoint(path / "best.pt")
            if partitions is None:
                partitions = saved["splits"]
            elif partitions != saved["splits"]:
                raise RuntimeError("Comparison partitions differ")
            metrics = evaluate_backend(backend, data, path / "best.pt", device=cfg.device, output=path / "test_metrics.json")
            rows.append(dict(model=model, seed=seed, best_epoch=saved["epoch"], ade_m=metrics["ade_m"],
                             fde_m=metrics["fde_m"], cv_ade_m=metrics["cv_ade_m"], cv_fde_m=metrics["cv_fde_m"],
                             intention_accuracy=metrics["intention_accuracy"],
                             intention_enabled=graph and settings.use_intention,
                             civilian_teachers_enabled=graph and settings.attention_supervision,
                             interaction_context_enabled=graph and cfg.context_entities > 0 and settings.use_interaction_context,
                             intention_pretraining_epochs=intention_epochs if graph and graph_training == "two_stage" else 0,
                             engagement_edges_enabled=graph and settings.engagement_edges,
                             candidate_pruning_enabled=graph and settings.candidate_pruning,
                             military_teachers_enabled=graph and settings.military_teachers,
                             parameters=json.loads((path / "manifest.json").read_text(encoding="utf-8"))["parameters"]))
            write_csv(output / "comparison_runs.csv", rows)
    summaries = []
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        summary = {"model": model, "runs": len(selected)}
        for metric in ("ade_m", "fde_m"):
            values = [row[metric] for row in selected]
            summary[metric + "_mean"] = float(np.mean(values))
            summary[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summaries.append(summary)
    write_csv(output / "comparison_summary.csv", summaries)
    result = dict(backend=backend, graph_training=graph_training, intention_epochs=intention_epochs, context_entities=cfg.context_entities,
                  split_seed=cfg.split_seed, partitions=partitions, runs=rows, summary=summaries,
                  selection="validation only; test is evaluated after selection",
                  protocol="identical focal histories, scene partitions, normalization and forecast budget; graph variants share one training-only classifier per seed; baselines use focal histories only; I/C ablations retain civilian teachers; engagement ablations drop the engagement edges, candidate cropper and E/R/T teachers and fall back to the civilian interaction graph; pretraining budget recorded separately")
    write_json(output / "comparison.json", result)
    return result
