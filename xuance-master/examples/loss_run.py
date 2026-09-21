"""Dedicated instrumented training to capture actor/critic LOSS curves.
The framework does not log learner losses to TensorBoard, so we monkey-patch
learner.update() to record loss_actor/loss_critic per update into a CSV.
Runs the validated full-method config @medium (shorter horizon is enough to show
loss convergence). Plot with plot_loss.py."""
import argparse
import csv
import os
import numpy as np
from xuance import get_runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lossrun_full")
    ap.add_argument("--steps", type=int, default=1500000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--out", default="p3_results/loss_full.csv")
    a = ap.parse_args()

    env_id = f"apollonius_3d_{a.name}"
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_3d",
                           env_id=env_id, device="cuda:0")
    p.num_agents = a.num_agents
    p.seed = a.seed
    p.parallels = 16
    p.running_steps = a.steps
    p.curriculum_enabled = False
    p.building_mode = "medium"
    p.target_max_speed = 6.0
    p.closure_weight = 8.0
    p.w_pos = 0.6
    p.w_gap = 0.8
    p.eval_interval = a.steps        # only one eval at the end -> minimal overhead
    p.test_episode = 5

    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_3d",
                        env_id=env_id, parser_args=p)
    agent = runner.agent
    learner = agent.learner
    orig_update = learner.update

    rows = []

    def patched(*args, **kw):
        info = orig_update(*args, **kw)
        la = [float(v) for k, v in info.items() if "loss_actor" in str(k)]
        lc = [float(v) for k, v in info.items() if "loss_critic" in str(k)]
        step = int(getattr(agent, "current_step", len(rows)))
        rows.append((step,
                     float(np.mean(la)) if la else float("nan"),
                     float(np.mean(lc)) if lc else float("nan")))
        return info

    learner.update = patched
    runner.run(mode="benchmark")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss_actor", "loss_critic"])
        w.writerows(rows)
    print("saved", a.out, "rows", len(rows))


if __name__ == "__main__":
    main()
