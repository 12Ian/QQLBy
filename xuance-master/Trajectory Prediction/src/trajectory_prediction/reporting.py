"""Figures from saved held-out evidence; no smoothing or best-case selection."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .engine import load_checkpoint, write_json
from .data import CLASS_NAMES, Windows, load_tracks, fingerprint
from .config import Config
from .inference import Predictor


LABELS = {"graph": "GAT +I +T", "graph_no_intention": "GAT -I +T",
          "graph_no_attention_supervision": "GAT +I -T", "graph_no_both": "GAT -I -T",
          "lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#7B61A8", "#555555"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 1)), (0, (7, 2))]


def report(comparison_path, output, data=None, device="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    root, output = Path(comparison_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Report output must be empty")
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    labels_map = dict(LABELS)
    if comparison.get("context_entities", 0):
        labels_map.update(graph="GAT +I +C +T", graph_no_intention="GAT -I +C +T",
                          graph_no_attention_supervision="GAT +I +C -T", graph_no_both="GAT -I +C -T",
                          graph_no_context="GAT +I -C +T", graph_no_intention_context="GAT -I -C +T")
    models = [row["model"] for row in comparison["summary"]]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300})
    sources, artifacts, evidence = set(), {}, {}
    sources.add(root / "comparison.json")
    output.mkdir(parents=True, exist_ok=True)

    def export(fig, name):
        files = []
        # Include projected 3D axis labels explicitly; their bounds can be omitted
        # by Axes3D tight-layout calculations near the right edge of the canvas.
        fig.canvas.draw()
        extra = [axis.label for ax in fig.axes if hasattr(ax, "zaxis")
                 for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]
        extra.extend(fig.legends)
        for extension in ("svg", "pdf", "png"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, bbox_inches="tight", bbox_extra_artists=extra or None, pad_inches=.15)
            files.append(str(path))
        plt.close(fig)
        artifacts[name] = files

    results = {}
    dataset_sha256 = None
    for model in models:
        rows = [row for row in comparison["runs"] if row["model"] == model]
        loaded = []
        for row in rows:
            path = root / model / f"seed_{row['seed']}" / "test_metrics.json"
            sources.add(path)
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
        if any(item["dataset_sha256"] != loaded[0]["dataset_sha256"] for item in loaded):
            raise ValueError("Figure inputs differ in dataset fingerprint")
        dataset_sha256 = dataset_sha256 or loaded[0]["dataset_sha256"]
        if loaded[0]["dataset_sha256"] != dataset_sha256:
            raise ValueError("Models in the comparison used different datasets")
        results[model] = loaded
    for monochrome in (False, True):
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        curves = {}
        for i, model in enumerate(models):
            loaded = results[model]
            time = np.asarray([row["horizon_s"] for row in loaded[0]["error_by_horizon"]])
            for item in loaded:
                if [row["horizon_s"] for row in item["error_by_horizon"]] != time.tolist():
                    raise ValueError("Comparison horizon grids differ")
            values = np.asarray([[row["displacement_m"] for row in item["error_by_horizon"]] for item in loaded])
            mean = values.mean(0)
            color = str(.1 + .65 * i / max(1, len(models) - 1)) if monochrome else COLORS[i % len(COLORS)]
            ax.plot(time, mean, label=labels_map.get(model, model), color=color, linestyle=STYLES[i % len(STYLES)], linewidth=1.6)
            if len(values) > 1:
                sd = values.std(0, ddof=1)
                ax.fill_between(time, mean - sd, mean + sd, color=color, alpha=.09)
            curves[model] = dict(horizon_s=time.tolist(), mean_m=mean.tolist(),
                                  std_m=values.std(0, ddof=1).tolist() if len(values) > 1 else None, seeds=len(values))
        cv = results[models[0]][0]["error_by_horizon"]
        ax.plot([row["horizon_s"] for row in cv], [row["cv_displacement_m"] for row in cv],
                color="black", linestyle=(0, (9, 3)), linewidth=1.3, label="Constant velocity")
        ax.set(xlabel="Forecast horizon (s)", ylabel="Mean Euclidean displacement error (m)",
               title="Held-out civilian trajectories; bands = SD across initialization seeds")
        ax.grid(alpha=.2)
        ax.legend(ncols=3, fontsize=8)
        export(fig, "horizon_error_grayscale" if monochrome else "horizon_error")
        evidence["horizon_error"] = curves
    evidence["constant_velocity_horizon"] = cv
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=True)
    for metric, ax in zip(("ade_m", "fde_m"), axes):
        values = [row[metric + "_mean"] for row in comparison["summary"]]
        deviations = [row[metric + "_std"] or 0 for row in comparison["summary"]]
        ax.barh(np.arange(len(models)), values, xerr=deviations, capsize=3, color=COLORS[:len(models)], alpha=.85)
        ax.set_yticks(np.arange(len(models)), [labels_map.get(model, model) for model in models])
        ax.set_xlabel(metric.removesuffix("_m").upper() + " (m)")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Validation-selected checkpoints; mean ± SD across initialization seeds")
    export(fig, "ade_fde_comparison")
    evidence["ade_fde"] = comparison["summary"]
    first = comparison["runs"][0]
    representative = root / first["model"] / f"seed_{first['seed']}"
    history_file = representative / "history.csv"
    sources.add(history_file)
    with history_file.open(newline="", encoding="utf-8") as file:
        history = list(csv.DictReader(file))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for phase, color, style in (("train", COLORS[0], "-"), ("validation", COLORS[1], "--")):
        rows = [row for row in history if row["phase"] == phase]
        axes[0].plot([int(row["epoch"]) for row in rows], [float(row["loss"]) for row in rows], label=phase, color=color, linestyle=style)
        for metric, metric_style in (("ade_m", "-"), ("fde_m", "--")):
            axes[1].plot([int(row["epoch"]) for row in rows], [float(row[metric]) for row in rows],
                         label=f"{phase} {metric[:3].upper()}", color=color, linestyle=metric_style)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Total training objective")
    axes[1].set_ylabel("Euclidean displacement error (m)")
    fig.suptitle(f"{labels_map.get(first['model'], first['model'])}, seed {first['seed']}; raw epoch values")
    export(fig, "training_curves")
    evidence["training_curves"] = dict(model=first["model"], seed=first["seed"], raw_rows=history)
    classifier_history = root / "_intention" / f"seed_{first['seed']}" / "history.csv"
    if classifier_history.exists():
        sources.add(classifier_history)
        with classifier_history.open(newline="", encoding="utf-8") as file:
            classifier_rows = list(csv.DictReader(file))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for phase, color, style in (("train", COLORS[0], "-"), ("validation", COLORS[1], "--")):
            rows = [row for row in classifier_rows if row["phase"] == phase]
            epochs = [int(row["epoch"]) for row in rows]
            axes[0].plot(epochs, [float(row["cross_entropy"]) for row in rows], label=phase, color=color, linestyle=style)
            axes[1].plot(epochs, [float(row["intention_accuracy"]) for row in rows], label=phase, color=color, linestyle=style)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Classification cross-entropy")
        axes[1].set_ylabel("Observed-window classification accuracy")
        axes[1].set_ylim(0, 1)
        fig.suptitle(f"Independent intention pretraining, seed {first['seed']}; raw epoch values")
        export(fig, "intention_learning_curves")
        evidence["intention_learning_curves"] = dict(seed=first["seed"], raw_rows=classifier_rows)
    classified = next(((model, item) for model in models for item in results[model]
                       if item["labeled_samples"] > 0), None)
    if classified:
        model, item = classified
        matrix = item["confusion_matrix_row_normalized"]
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        count = len(matrix)
        names = CLASS_NAMES[:count] if count <= 9 else [str(i) for i in range(count)]
        ax.set_xticks(np.arange(count), names, rotation=45, ha="right")
        ax.set_yticks(np.arange(count), names)
        ax.set(xlabel="Predicted observed-window class", ylabel="Observed-window reference class",
               title=f"Row-normalized held-out confusion matrix: {labels_map.get(model, model)}")
        fig.colorbar(image, ax=ax, label="Fraction within reference class")
        export(fig, "intention_confusion")
        evidence["confusion_matrix_row_normalized"] = matrix
        evidence["confusion_source_checkpoint"] = item["checkpoint"]
    if data:
        checkpoint = load_checkpoint(representative / "best.pt")
        cfg = Config(**checkpoint["config"])
        if fingerprint(data) != checkpoint["dataset_sha256"]:
            raise ValueError("Trajectory figure data differs from training data")
        tracks = load_tracks(data, cfg.dt, cfg.observation_steps + cfg.prediction_steps, cfg.intention_classes)
        windows = Windows(tracks, checkpoint["splits"]["test"], cfg)
        selected = {}
        for i in range(len(windows)):
            sample = windows[i]
            label = int(sample["label"])
            key = sample["scenario"] if cfg.context_entities else label
            if (cfg.context_entities or label in (5, 6, 7, 8)) and key not in selected and len(selected) < 6:
                selected[key] = i
        if not selected:
            selected = {int(windows[0]["label"]): 0}
        predictors = {model: Predictor(root / model / f"seed_{next(row['seed'] for row in comparison['runs'] if row['model'] == model)}" / "best.pt", device) for model in models}
        for model in models:
            sources.add(Path(results[model][0]["checkpoint"]))
        panels = max(2, int(np.ceil(len(selected) / 2)))
        fig = plt.figure(figsize=(12, 4.5 * panels), layout="constrained")
        case_evidence = []
        for panel, (label, index) in enumerate(sorted(selected.items()), 1):
            ax = fig.add_subplot(panels, 2, panel, projection="3d")
            sample = windows[index]
            observed, truth = sample["observed"].numpy(), sample["future"].numpy()
            ax.plot(*observed[:, :3].T, color="#999999", label="Observed", linewidth=2)
            future = np.concatenate((observed[-1:, :3], truth))
            ax.plot(*future.T, color="black", label="Ground truth", linewidth=2)
            all_positions = [observed[:, :3], truth]
            context = sample["context"].numpy() if cfg.context_entities else None
            context_mask = sample["context_mask"].numpy() if cfg.context_entities else None
            if context is not None:
                for peer, mask in zip(context, context_mask):
                    peer_position = peer[mask, :3]
                    if len(peer_position):
                        all_positions.append(peer_position)
                        ax.plot(*peer_position.T, color="#7B61A8", linestyle=":", linewidth=1.4, label="Measured peer history")
            forecasts = {}
            for i, (model, predictor) in enumerate(predictors.items()):
                forecast = predictor.predict(observed, context, context_mask)["trajectory"]
                values = np.concatenate((observed[-1:, :3], forecast))
                all_positions.append(values)
                forecasts[model] = forecast.tolist()
                ax.plot(*values.T, color=COLORS[i % len(COLORS)], linestyle=STYLES[i % len(STYLES)],
                        linewidth=1.2, label=labels_map.get(model, model))
            combined = np.concatenate(all_positions)
            center = (combined.max(0) + combined.min(0)) / 2
            radius = max(1., float(np.ptp(combined, axis=0).max()) / 2) * 1.05
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            title = str(label) if cfg.context_entities else CLASS_NAMES[label] if 0 <= label < 9 else "unknown class"
            ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=title)
            case_evidence.append(dict(track_id=sample["track_id"], last_observed_time_s=float(sample["last_time"]),
                                     class_id=int(sample["label"]), scene_id=sample["scene_id"], scenario=sample["scenario"],
                                     observed=observed[:, :3].tolist(), truth=truth.tolist(), forecasts=forecasts,
                                     peer_observed=context.tolist() if context is not None else None,
                                     peer_availability=context_mask.tolist() if context_mask is not None else None))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.02), ncols=5, fontsize=8)
        export(fig, "civilian_interaction_cases" if cfg.context_entities else "civilian_3d_cases")
        evidence["cases"] = case_evidence
        sources.add(Path(data).resolve())
    result = dict(artifacts=artifacts, matplotlib=matplotlib.__version__, graph_training=comparison.get("graph_training", "legacy joint"),
        aggregation="Window-weighted held-out metric within seed; arithmetic mean and sample SD across initialization seeds.",
        case_selection="First sorted held-out window per scenario (up to six) for interaction data, or per observed composite-motion class; never selected by prediction error.",
        transformations="No smoothing, interpolation, filtering or removal of failed/contradictory results.",
        factors="I=intention; C=measured civilian peer context; T=civilian historical teachers; these are not the paper's specialized factors.",
        sources=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(sources)],
        evidence=evidence)
    write_json(output / "report_values.json", result)
    markdown = "# Civilian trajectory experiment report\n\n" + result["aggregation"] + "\n\n" + result["case_selection"] + "\n\n" + result["factors"] + "\n\n"
    markdown += "Synthetic short CPU runs validate code paths, not paper performance or scientific superiority.\n\n"
    for name, files in artifacts.items():
        markdown += f"## {name}\n\n![{name}]({Path(files[-1]).name})\n\n"
    (output / "REPORT.md").write_text(markdown, encoding="utf-8")
    return {"output": str(output), "figures": list(artifacts), "data_manifest": str(output / "report_values.json")}
