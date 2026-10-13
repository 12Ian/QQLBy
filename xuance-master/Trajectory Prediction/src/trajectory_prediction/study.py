"""Independent civilian sample libraries crossed with model initialization seeds."""
import json
from pathlib import Path

import numpy as np

from .data import generate_tracks, fingerprint
from .engine import write_json
from .metrics import write_csv
from .workflows import compare


def study(output, cfg, models, data_seeds, model_seeds, tracks=180, steps=240,
          intention_epochs=70, backend="xuance", resume=False, figures=False, data_mode="independent"):
    if len(set(data_seeds)) != len(data_seeds) or not data_seeds:
        raise ValueError("Data seeds must be nonempty and unique")
    if steps < cfg.observation_steps + cfg.prediction_steps or tracks < 3:
        raise ValueError("Sample library dimensions are too small for the configured windows")
    if data_mode not in {"independent", "civilian_interaction", "engagement"}:
        raise ValueError("Unknown data mode")
    if data_mode == "civilian_interaction" and cfg.context_entities != 1:
        raise ValueError("Civilian pair simulation requires context_entities=1")
    if data_mode == "engagement" and cfg.context_entities != 1:
        raise ValueError("Intercept-engagement simulation requires context_entities=1")
    if data_mode == "independent" and cfg.context_entities:
        raise ValueError("Independent generation has no peer histories; use civilian_interaction")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Study output is not empty; use --resume")
    rows, comparisons = [], []
    for data_seed in data_seeds:
        data = output / "data" / f"seed_{data_seed}.npz"
        metadata = data.with_suffix(".metadata.json")
        if resume and data.exists():
            provenance = json.loads(metadata.read_text(encoding="utf-8"))
            count = provenance.get("scenes") if data_mode in {"civilian_interaction", "engagement"} else provenance["tracks"]
            if (provenance["seed"], count, provenance["steps"], provenance["dt"]) != (data_seed, tracks, steps, cfg.dt):
                raise ValueError("Resumed sample library protocol differs")
            if provenance["sha256"] != fingerprint(data):
                raise ValueError("Resumed sample library was changed")
        else:
            if data_mode == "civilian_interaction":
                from .civilian_scenarios import simulate_scenes
                provenance = simulate_scenes(data, tracks, steps, cfg.dt, data_seed)
            elif data_mode == "engagement":
                from .engagement_scenarios import simulate_engagements
                provenance = simulate_engagements(data, tracks, steps, cfg.dt, data_seed)
            else:
                provenance = generate_tracks(data, tracks, steps, cfg.dt, data_seed)
            write_json(metadata, provenance)
        path = output / f"data_seed_{data_seed}"
        result = compare(data, path, cfg, models, model_seeds, backend, resume,
                         graph_training="two_stage", intention_epochs=intention_epochs)
        comparisons.append(str(path / "comparison.json"))
        for row in result["runs"]:
            rows.append(dict(data_seed=data_seed, **row))
        write_csv(output / "study_runs.csv", rows)
        if figures:
            from .reporting import report
            figure_dir = path / "figures"
            if not (resume and (figure_dir / "report_values.json").exists()):
                report(path, figure_dir, data, cfg.device)
    summaries = []
    for model in models:
        summary = dict(model=model, independent_libraries=len(data_seeds), initialization_seeds=len(model_seeds))
        for metric in ("ade_m", "fde_m"):
            values = np.asarray([[next(r[metric] for r in rows if r["model"] == model and r["data_seed"] == data_seed and r["seed"] == seed)
                                  for seed in model_seeds] for data_seed in data_seeds])
            means = values.mean(1)
            summary[metric + "_mean"] = float(means.mean())
            summary[metric + "_between_library_std"] = float(means.std(ddof=1)) if len(means) > 1 else None
            rng = np.random.default_rng(cfg.split_seed)
            library_indices = rng.integers(len(data_seeds), size=(1000, len(data_seeds)))
            seed_indices = rng.integers(len(model_seeds), size=(1000, len(data_seeds), len(model_seeds)))
            draws = values[library_indices[:, :, None], seed_indices].mean(axis=(1, 2))
            low, high = np.quantile(draws, [.025, .975])
            summary[metric + "_hierarchical_95ci_low"] = float(low)
            summary[metric + "_hierarchical_95ci_high"] = float(high)
        summaries.append(summary)
    write_csv(output / "study_summary.csv", summaries)
    paired = data_mode in {"civilian_interaction", "engagement"}
    domain = {"engagement": "military target/defender intercept engagements",
              "civilian_interaction": "independent low-speed civilian scenes"}.get(
                  data_mode, "independent low-speed civilian trajectories")
    result = dict(domain=domain,
                  data_mode=data_mode, data_seeds=data_seeds,
                  model_seeds=model_seeds, tracks_per_library=2 * tracks if paired else tracks,
                  scenes_per_library=tracks if paired else None, steps_per_track=steps,
                  config=cfg.to_dict(), comparisons=comparisons, summary=summaries,
                  uncertainty="1000 hierarchical bootstrap resamples of library seeds and initialization seeds; within-library "
                              + ("scene" if paired else "track")
                              + " cluster intervals are reported separately",
                  original_paper_scenarios_reproduced=False)
    write_json(output / "study.json", result)
    return result
