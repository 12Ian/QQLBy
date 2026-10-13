import argparse
import csv
import json
from pathlib import Path
import sys

from .config import Config
from .data import CLASS_NAMES, OVERLOAD_NAMES, generate_tracks, load_tracks
from .engine import environment, evaluate, load_checkpoint, train, write_json, choose_device
from .inference import Predictor
from .workflows import train_backend, evaluate_backend, pipeline, compare


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Civilian supervised 3D forecasting with optional measured neighbors")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate independent low-speed civilian tracks")
    generate.add_argument("--output", required=True)
    generate.add_argument("--tracks", type=int, default=180)
    generate.add_argument("--steps", type=int, default=120)
    generate.add_argument("--dt", type=float, default=0.5)
    generate.add_argument("--seed", type=int, default=42)
    simulation = commands.add_parser("simulate", help="Low-speed civilian fixed routes, crossing/merge/right-of-way")
    simulation.add_argument("--output", required=True)
    simulation.add_argument("--scenes", type=int, default=60)
    simulation.add_argument("--steps", type=int, default=80)
    simulation.add_argument("--dt", type=float, default=.5)
    simulation.add_argument("--seed", type=int, default=42)
    simulation.add_argument("--measurement-dropout", type=float, default=0.)
    engagement = commands.add_parser("simulate-engagement",
                                     help="Military target/defender intercept engagements (Table IV/V)")
    engagement.add_argument("--output", required=True)
    engagement.add_argument("--scenes", type=int, default=60)
    engagement.add_argument("--steps", type=int, default=200)
    engagement.add_argument("--dt", type=float, default=.1)
    engagement.add_argument("--sample-stride", type=int, default=1,
                            help="Decimation of the dt/stride internal grid (5 gives IV-A dt=0.5 s)")
    engagement.add_argument("--seed", type=int, default=42)
    engagement.add_argument("--measurement-dropout", type=float, default=0.)
    engagement.add_argument("--aT-max-g", type=float, default=9.)
    engagement.add_argument("--aM-max-g", type=float, default=15.)
    engagement.add_argument("--fov-deg", type=float, default=60.)
    engagement.add_argument("--overload-window-steps", type=int, default=20)
    engagement.add_argument("--initial-fov-coverage", type=float, default=1.0,
                            help="Fraction of scenes starting with the defender inside the target FOV "
                                 "(paper IV-A uses 0.5 so both visible and invisible starts exist)")
    engagement.add_argument("--target-windows", type=int, default=0,
                            help="Grow --scenes until this many sliding windows exist (IV-A/IV-B use 5e4)")
    engagement.add_argument("--observation-steps", type=int, default=0)
    engagement.add_argument("--prediction-steps", type=int, default=0)
    engagement.add_argument("--window-stride", type=int, default=1)
    engagement.add_argument("--class-balance", default="",
                            help="Comma-separated nine weights for the semantic maneuver classes")
    engagement.add_argument("--evasion-overload-range", default="0.5,1.0",
                            help="Per-segment nominal evasion overload as fractions of aT-max "
                                 "(paper IV-B bounded sampling; use 1,1 to force saturation)")
    engagement.add_argument("--censored-overload-ends", action="store_true", default=False,
                            help="Widen the two extreme overload bins to (-inf, -8.5] and [8.5, +inf) "
                                 "so the +/-10 g levels become reachable (9 g saturation makes the "
                                 "strict (9.5, 10.5] bin empty; use for the eleven-level task)")
    engagement.add_argument("--primitive-duration-range", default="2.0,6.0",
                            help="Bounds (seconds) of the maneuver-primitive segment duration "
                                 "(Section IV-B 'sampled within predefined bounds'; the paper "
                                 "gives no numbers, default 2-6 s)")
    engagement.add_argument("--evasion-segment-seconds", type=float, default=4.0,
                            help="Duration of one evasion-maneuver segment before the nominal "
                                 "overload is resampled (Section IV-B; default 4 s)")
    engagement.add_argument("--intercept-radius-m", type=float, default=50.,
                            help="C9 termination: intercept when r falls below this radius")
    engagement.add_argument("--miss-grow-steps", type=int, default=30,
                            help="C9 termination: certain miss after this many consecutive "
                                 "growing-separation internal steps (0 disables miss termination)")
    training = commands.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--config")
    training.add_argument("--output", required=True)
    training.add_argument("--device")
    training.add_argument("--epochs", type=int)
    training.add_argument("--stage", choices=["joint", "intention", "forecast"])
    training.add_argument("--resume")
    training.add_argument("--intention-checkpoint")
    training.add_argument("--freeze-intention", action="store_true", default=None)
    training.add_argument("--backend", choices=["xuance", "standalone"], default="xuance")
    training.add_argument("--model", choices=["graph", "lstm", "gru", "tcn", "transformer"])
    for name in ("pipeline", "compare"):
        workflow = commands.add_parser(name)
        workflow.add_argument("--data", required=True)
        workflow.add_argument("--config", required=True)
        workflow.add_argument("--output", required=True)
        workflow.add_argument("--backend", choices=["xuance", "standalone"], default="xuance")
        workflow.add_argument("--device")
        workflow.add_argument("--epochs", type=int)
        workflow.add_argument("--resume", action="store_true")
        if name == "pipeline":
            workflow.add_argument("--intention-epochs", type=int, default=70)
        else:
            workflow.add_argument("--models", default="graph,lstm,gru,tcn,transformer")
            workflow.add_argument("--seeds", default="42,43,44")
            workflow.add_argument("--graph-training", choices=["two_stage", "joint"], default="two_stage")
            workflow.add_argument("--intention-epochs", type=int, default=70)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--output")
    evaluation.add_argument("--backend", choices=["xuance", "standalone"], default="xuance")
    rolling = commands.add_parser("rolling-evaluate")
    rolling.add_argument("--data", required=True)
    rolling.add_argument("--checkpoint", required=True)
    rolling.add_argument("--output", required=True)
    rolling.add_argument("--split", choices=["validation", "test"], default="test")
    rolling.add_argument("--device", default="auto")
    rolling.add_argument("--refresh-steps", type=int, default=1)
    figures = commands.add_parser("report")
    figures.add_argument("--comparison", required=True)
    figures.add_argument("--output", required=True)
    figures.add_argument("--data")
    figures.add_argument("--device", default="cpu")
    experiments = commands.add_parser("study")
    experiments.add_argument("--config", required=True)
    experiments.add_argument("--output", required=True)
    experiments.add_argument("--device")
    experiments.add_argument("--epochs", type=int)
    experiments.add_argument("--intention-epochs", type=int, default=70)
    experiments.add_argument("--data-seeds", default="101,102,103")
    experiments.add_argument("--model-seeds", default="42,43,44")
    experiments.add_argument("--models", default="graph,lstm,gru,tcn,transformer,graph_no_intention,graph_no_attention_supervision,graph_no_both")
    experiments.add_argument("--tracks", type=int, default=180)
    experiments.add_argument("--scenes", type=int, default=60)
    experiments.add_argument("--data-mode", choices=["independent", "civilian_interaction", "engagement"],
                             default="independent")
    experiments.add_argument("--steps", type=int, default=240)
    experiments.add_argument("--backend", choices=["xuance", "standalone"], default="xuance")
    experiments.add_argument("--resume", action="store_true")
    experiments.add_argument("--figures", action="store_true")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--track-id", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="auto")
    check = commands.add_parser("check")
    check.add_argument("--device", default="auto")
    check.add_argument("--backend", choices=["xuance", "standalone"], default="xuance")
    check.add_argument("--config")
    check.add_argument("--data")
    check.add_argument("--backward", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_tracks(args.output, args.tracks, args.steps, args.dt, args.seed)
        write_json(Path(args.output).with_suffix(".metadata.json"), result)
    elif args.command == "simulate":
        from .civilian_scenarios import simulate_scenes
        result = simulate_scenes(args.output, args.scenes, args.steps, args.dt, args.seed, args.measurement_dropout)
        write_json(Path(args.output).with_suffix(".metadata.json"), result)
    elif args.command == "simulate-engagement":
        from .engagement_scenarios import G0, simulate_engagements
        scenes = args.scenes
        if args.target_windows:
            total = args.observation_steps + args.prediction_steps
            if args.observation_steps < 2 or total > args.steps:
                raise ValueError("--target-windows needs --observation-steps and steps >= O+P")
            per_scene = (args.steps - total) // max(1, args.window_stride) + 1
            if per_scene < 1:
                raise ValueError("No complete windows in one scene; increase --steps")
            scenes = max(scenes, -(-args.target_windows // per_scene))
        balance = [float(value) for value in args.class_balance.split(",")] if args.class_balance else None
        evasion_range = tuple(float(v) for v in args.evasion_overload_range.split(","))
        if (len(evasion_range) != 2 or not all(v == v and abs(v) != float("inf") for v in evasion_range)
                or not 0 < evasion_range[0] <= evasion_range[1]):
            from argparse import ArgumentTypeError
            raise ArgumentTypeError("--evasion-overload-range needs two fractions 0 < lo <= hi")
        duration_range = tuple(float(v) for v in args.primitive_duration_range.split(","))
        if (len(duration_range) != 2 or not all(v == v and abs(v) != float("inf") for v in duration_range)
                or not 0 < duration_range[0] <= duration_range[1]):
            from argparse import ArgumentTypeError
            raise ArgumentTypeError("--primitive-duration-range needs two seconds 0 < lo <= hi")
        if not (args.evasion_segment_seconds == args.evasion_segment_seconds
                and args.evasion_segment_seconds > 0):
            from argparse import ArgumentTypeError
            raise ArgumentTypeError("--evasion-segment-seconds must be positive")
        if not (args.intercept_radius_m == args.intercept_radius_m
                and args.intercept_radius_m > 0):
            from argparse import ArgumentTypeError
            raise ArgumentTypeError("--intercept-radius-m must be positive")
        if args.miss_grow_steps < 0:
            from argparse import ArgumentTypeError
            raise ArgumentTypeError("--miss-grow-steps cannot be negative")
        result = simulate_engagements(args.output, scenes=scenes, steps=args.steps, dt=args.dt,
                                      seed=args.seed, aT_max=args.aT_max_g * G0, aM_max=args.aM_max_g * G0,
                                      fov_half_angle=args.fov_deg,
                                      measurement_dropout=args.measurement_dropout,
                                      sample_stride=args.sample_stride,
                                      overload_window_steps=args.overload_window_steps,
                                      initial_fov_coverage=args.initial_fov_coverage,
                                      class_balance=balance,
                                      evasion_overload_range=evasion_range,
                                      intercept_radius_m=args.intercept_radius_m,
                                      miss_grow_steps=(args.miss_grow_steps
                                                       if args.miss_grow_steps else 10 ** 9),
                                      censored_overload_ends=args.censored_overload_ends,
                                      primitive_duration_range=duration_range,
                                      evasion_segment_seconds=args.evasion_segment_seconds)
        if args.target_windows:
            result["target_windows"] = args.target_windows
            result["windows_per_scene"] = per_scene
            result["window_protocol"] = dict(observation_steps=args.observation_steps,
                                             prediction_steps=args.prediction_steps,
                                             stride=args.window_stride)
        write_json(Path(args.output).with_suffix(".metadata.json"), result)
    elif args.command == "train":
        cfg = (Config(**load_checkpoint(args.resume)["config"]) if args.resume and not args.config
               else Config.read(args.config))
        for name in ("device", "epochs", "stage", "freeze_intention"):
            if getattr(args, name) is not None:
                setattr(cfg, name, getattr(args, name))
        if args.model:
            cfg.model_type = args.model
            if args.model != "graph":
                cfg.stage, cfg.attention_supervision, cfg.freeze_intention = "forecast", False, False
        result = train_backend(args.backend, args.data, args.output, cfg.validate(), args.resume, args.intention_checkpoint)
    elif args.command in {"pipeline", "compare"}:
        cfg = Config.read(args.config)
        if args.device:
            cfg.device = args.device
        if args.epochs:
            cfg.epochs = args.epochs
        cfg.validate()
        if args.command == "pipeline":
            result = pipeline(args.data, args.output, cfg, args.intention_epochs, args.backend, args.resume)
        else:
            result = compare(args.data, args.output, cfg, args.models.split(","),
                             [int(seed) for seed in args.seeds.split(",")], args.backend, args.resume,
                             args.graph_training, args.intention_epochs)
    elif args.command == "evaluate":
        result = evaluate_backend(args.backend, args.data, args.checkpoint, args.split, args.device, args.output)
    elif args.command == "rolling-evaluate":
        from .rolling import rolling_evaluate
        result = rolling_evaluate(args.data, args.checkpoint, args.output, args.split, args.device, args.refresh_steps)
    elif args.command == "report":
        from .reporting import report
        result = report(args.comparison, args.output, args.data, args.device)
    elif args.command == "study":
        from .study import study
        cfg = Config.read(args.config)
        if args.device:
            cfg.device = args.device
        if args.epochs:
            cfg.epochs = args.epochs
        result = study(args.output, cfg.validate(), args.models.split(","),
                       [int(v) for v in args.data_seeds.split(",")], [int(v) for v in args.model_seeds.split(",")],
                       args.scenes if args.data_mode in {"civilian_interaction", "engagement"} else args.tracks,
                       args.steps, args.intention_epochs, args.backend, args.resume, args.figures, args.data_mode)
    elif args.command == "predict":
        predictor = Predictor(args.checkpoint, args.device)
        tracks = load_tracks(args.data, predictor.cfg.dt, predictor.cfg.observation_steps,
                             predictor.cfg.intention_classes)
        matches = [t for t in tracks if t.track_id == args.track_id]
        if not matches:
            raise ValueError(f"Unknown track_id: {args.track_id}")
        track = matches[0]
        predicted = predictor.predict(track.states, track.context.transpose(1, 0, 2) if predictor.cfg.context_entities else None,
                                      track.context_mask.T if predictor.cfg.context_entities else None,
                                      track.engagement if track.engagement is not None and predictor.cfg.engagement_edges else None)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to replace prediction: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", "x", "y", "z"])
            for step, position in enumerate(predicted["trajectory"], 1):
                writer.writerow([track.time[-1] + step * predictor.cfg.dt, *position.tolist()])
        probabilities = (predicted["intention_probabilities"].tolist()
                         if predicted["intention_probabilities"] is not None else None)
        result = {"output": str(output.resolve()), "track_id": track.track_id,
                  "intention_probabilities": probabilities,
                  "classes": (CLASS_NAMES if len(probabilities) == 9 else
                              OVERLOAD_NAMES if len(probabilities) == 11 else list(range(len(probabilities)))) if probabilities else None}
        write_json(output.with_suffix(".json"), result)
    else:
        if args.backend == "xuance":
            from .xuance_bridge import framework
            framework()
            from xuance.extensions.trajectory_prediction import register
            register()
        result = environment(choose_device(args.device))
        if args.config:
            import torch
            from .model import build_forecaster
            from .learner import TrajectoryLearner
            from .data import Windows, split_tracks, fit_normalizer
            from .engine import configure_runtime
            cfg = Config.read(args.config)
            cfg.device = args.device
            configure_runtime(cfg)
            device = choose_device(args.device)
            if args.data:
                tracks = load_tracks(args.data, cfg.dt, cfg.observation_steps + cfg.prediction_steps, cfg.intention_classes)
                splits = split_tracks(tracks, cfg.train_fraction, cfg.validation_fraction, cfg.split_seed,
                                      cfg.holdout_test)
                dataset = Windows(tracks, splits["train"], cfg)
                normalizer = fit_normalizer(dataset)
                batch = {key: torch.stack([dataset[i][key] for i in range(min(2, len(dataset)))]).to(device)
                         for key in (("observed", "future", "label", "context", "context_mask") if cfg.context_entities else ("observed", "future", "label"))}
            else:
                normalizer = None
                observed = torch.zeros(2, cfg.observation_steps, 6, device=device)
                observed[..., 3] = 2
                observed[..., 0] = torch.arange(cfg.observation_steps, device=device) * cfg.dt * 2
                batch = dict(observed=observed, future=observed[:, -1:, :3].expand(-1, cfg.prediction_steps, -1).clone(),
                             label=torch.zeros(2, dtype=torch.long, device=device))
                if cfg.context_entities:
                    peers = observed[:, None].expand(-1, cfg.context_entities, -1, -1).clone()
                    peers[..., 1] += 2
                    batch.update(context=peers, context_mask=torch.ones(peers.shape[:-1], dtype=torch.bool, device=device))
            if args.backend == "xuance":
                from xuance.torch import REGISTRY_Policy, REGISTRY_Learners
                policy_factory, learner_factory = REGISTRY_Policy["CivilianTrajectory"], REGISTRY_Learners["CivilianTrajectoryLearner"]
            else:
                policy_factory, learner_factory = build_forecaster, TrajectoryLearner
            policy = policy_factory(cfg, normalizer).to(device)
            learner = learner_factory(cfg, policy, device)
            output, losses = learner.update(batch) if args.backward else learner.compute(batch)
            result.update(architecture=cfg.to_dict(), parameters=sum(p.numel() for p in policy.parameters()),
                          output_shape=list(output["trajectory"].shape), finite_loss=float(losses["loss"].detach()),
                          backward_verified=args.backward, learner=type(learner).__name__)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
