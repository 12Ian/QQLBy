"""Run a unique short CUDA integration test with the actual exp009 architecture."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from xuance import get_runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="exp009_gpu_smoke_v1")
    args = parser.parse_args()
    result = ROOT / "outputs/results/matd3" / ("baseline_" + args.name)
    if result.exists():
        raise FileExistsError(result)
    torch.set_num_threads(1)
    overrides = argparse.Namespace(env_id="baseline_" + args.name, seed=1, env_seed=1,
        device="cuda:0", parallels=2, running_steps=2048, eval_interval=1024,
        test_episode=2, baseline_eval_parallels=2, episode_length=32,
        buffer_size=4096, batch_size=32, start_training=32, noise_decay_steps=1024,
        baseline_init_checkpoint=None, baseline_init_actor=None,
        curriculum_min_steps_per_level=100000, baseline_final_eval_interval=2048)
    runner = get_runner("matd3", "uav_pursuit_coverage_3d", "baseline_" + args.name,
                        str(ROOT / "experiments/baseline_exp009.yaml"), overrides)
    initial = {key: value.clone() for key, value in runner.agent.policy.state_dict().items()}
    try:
        runner.run(mode="benchmark")
        final = runner.agent.policy.state_dict()
        changed = {group: any(not torch.equal(initial[key], final[key]) for key in initial
                             if key.startswith(group + ".")) for group in ("actor", "critic_A", "critic_B")}
        assert all(changed.values()) and all(torch.isfinite(value).all() for value in final.values())
        assert not runner.agent.policy.use_graph_module and not runner.agent.policy.use_obstacle_gat
        assert next(value for key, value in final.items()
                    if key.startswith("actor.") and key.endswith("weight")).shape[1] == 83
        updates = list(csv.DictReader((result / "training_updates.csv").open()))
        episodes = list(csv.DictReader((result / "training_episodes.csv").open()))
        assert episodes and all(float(row["r_progress"]) == 0. and row["capture_type"] != "direct" for row in episodes)
        assert all(row["r_purs"] != "" and row["r_form"] != "" for row in episodes)
        directory = Path(runner.agent.log_dir)
        assert not any(path.is_dir() for path in directory.iterdir())
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        tags = events.Tags()["scalars"]
        assert all(np.isfinite(event.value) for tag in tags for event in events.Scalars(tag))
        for agent in ("uav_0", "uav_1", "uav_2", "uav_3"):
            for reward in ("r_purs", "r_form", "r_progress"):
                assert f"Train-Rewards/{agent}/{reward}" in tags
        report = dict(status="passed", environment_steps=2048, architecture_changed=changed,
                      critic_updates=int(updates[-1]["updates"]), actor_updates=int(updates[-1]["actor_updates"]),
                      root_scalar_tags=len(tags), all_scalars_finite=True, gat_enabled=False,
                      experiment_name=args.name, note="Integration test, not evidence of policy performance or convergence")
        output = ROOT / "outputs/reviews" / args.name
        output.mkdir(parents=True, exist_ok=False)
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GPU_AUDIT", json.dumps(report), flush=True)
    finally:
        runner.agent.train_envs.close()
        runner.agent.writer.close()


if __name__ == "__main__":
    main()
