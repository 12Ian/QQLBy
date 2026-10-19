"""Run one entity-attention/LSTM arm and compare fixed held-out scenes."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
import hashlib
import json
import os
import time
import numpy as np
import torch
from xuance import get_runner
from uav_lstm_prediction.core import write_json, sha256
from uav_lstm_prediction.fixed_eval import evaluate_fixed


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_frozen(run):
    for filename in ("frozen_inputs_sha256.json", "frozen_source_sha256.json"):
        manifest = json.loads((run / filename).read_text())
        for name, digest in manifest.items():
            path = Path(name) if Path(name).is_absolute() else run / "code" / name
            if sha256(path) != digest:
                raise ValueError("Frozen input/source changed: " + str(path))


def compare(run, folder, reference):
    rows, comparisons = [], []
    for checkpoint, levels in (("best", range(1, 6)), ("final", [5])):
        for level in levels:
            filename = f"{checkpoint}_level_{level}.json"
            old, new = (json.loads((p / "test" / filename).read_text()) for p in (reference, folder))
            fingerprints = lambda r: [(v["case"], v["seed"], v["initial_sha256"]) for v in r["records"]]
            if fingerprints(old) != fingerprints(new):
                raise AssertionError("Held-out initial scenes differ from reference")
            for method, result in (("mlp_lstm_reference", old), ("entity_attention_lstm", new)):
                rows.append(dict(method=method, checkpoint=checkpoint, level=level, **result["metrics"]))
            delta = np.array([int(a["safe_success"]) - int(b["safe_success"])
                              for a, b in zip(new["records"], old["records"])])
            rng = np.random.default_rng(710190 + level)
            bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1) * 100
            comparisons.append(dict(checkpoint=checkpoint, level=level, episodes=len(delta),
                safe_capture_gain_percentage_points=float(100 * delta.mean()),
                paired_scene_bootstrap_95_ci_percentage_points=np.quantile(bootstrap, [.025, .975]).tolist(),
                paired_initial_scenes_verified=True))
    write_json(run / "comparison.json", dict(rows=rows, comparisons=comparisons,
        reference=str(reference), episode_length=600, training_seed=1,
        initialization_confound="Reference warm-started from exp020; entity policy starts randomly.",
        uncertainty_scope="Scene bootstrap only; single training seed does not estimate training-seed uncertainty."))
    with (run / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(run, smoke=False):
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    job = protocol["jobs"][0]
    assert len(protocol["jobs"]) == 1 and job["seed"] == 1
    folder = run / "preflight/smoke" if smoke else Path(job["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "job_status.json"
    torch.set_num_threads(2)
    kwargs = dict(device="cuda:0")
    if smoke:
        kwargs.update(running_steps=4096, start_training=256, buffer_size=4096, batch_size=64,
            eval_interval=4096, test_episode=4, baseline_checkpoint_interval=2048,
            log_dir=str(folder / "logs"), model_dir=str(folder / "models"), result_dir=str(folder / "results"),
            capture_eval_workers=4)
    verify_frozen(run)
    started = time.time()
    runner = get_runner("matd3", "uav_pursuit_coverage_3d", "entity_attention_lstm_seed_1",
                        config_path=job["config"], parser_args=Namespace(**kwargs))
    runner.manage_resources = False
    agent, env = runner.agent, runner.agent.train_envs
    assert runner.config.episode_length == 600 and runner.config.curriculum_start_level == 4
    assert runner.config.baseline_init_checkpoint is None
    predictor = env.predictor.model
    frozen_before = model_hash(predictor)
    try:
        write_json(status, dict(stage="training", pid=os.getpid(), seed=1,
            total_steps=4096 if smoke else job["environment_steps"], started_unix=started,
            policy="UAV_EntityAttention_MATD3_Policy", initialization="random_policy_frozen_pretrained_lstm"))
        runner.run(mode="benchmark", running_steps=4096 if smoke else job["environment_steps"],
            eval_interval=4096 if smoke else 200000, test_episodes=4 if smoke else 100,
            benchmark_result_path=str(folder / "results"))
        summary = json.loads((folder / "results/baseline_summary.json").read_text())
        assert summary["environment_steps"] == (4096 if smoke else job["environment_steps"])
        assert model_hash(predictor) == frozen_before and not predictor.training
        assert all(not p.requires_grad and p.grad is None for p in predictor.parameters())
        assert sha256(runner.config.uav_prediction_checkpoint) == runner.config.uav_prediction_sha256
        assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
        assert all(torch.isfinite(p.grad).all() for p in agent.policy.parameters() if p.grad is not None)
        finals = list(Path(agent.model_dir_save).rglob("final_train_model.pth"))
        assert len(finals) == 1
        final, best = finals[0], folder / "results/best_model/best_model.pth"
        assert best.is_file()
        models = dict(best=str(best), final=str(final), initial=str(final.parent / "initial_random_model.pth"))
        write_json(folder / "model_paths.json", models)
        env.close()
        if not smoke:
            write_json(status, dict(stage="heldout_testing", pid=os.getpid(), training_steps=job["environment_steps"]))
            for level in range(5):
                cases = [dict(case=i, seed=631000000 + level * 1000 + i, level=level) for i in range(200)]
                evaluate_fixed(job["config"], best, cases, folder / "test" / f"best_level_{level + 1}.json", workers=8)
            cases = [dict(case=i, seed=631004000 + i, level=4) for i in range(200)]
            evaluate_fixed(job["config"], final, cases, folder / "test/final_level_5.json", workers=8)
            compare(run, folder, Path(protocol["reference_job"]))
        verify_frozen(run)
        write_json(status, dict(stage="complete", smoke=smoke, seed=1,
            training_steps=summary["environment_steps"], learner_updates=agent.learner.iterations,
            elapsed_seconds=time.time() - started, predictor_frozen=True, models=models,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except Exception as error:
        write_json(status, dict(stage="failed", smoke=smoke, error=repr(error), elapsed_seconds=time.time() - started))
        raise
    finally:
        env.close()
        agent.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.run, args.smoke)
