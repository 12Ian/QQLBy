"""Launch a named benchmark training run with config overrides into an isolated
env_id (distinct results/models/logs dirs). Guard with __main__."""
import argparse
import os
from xuance import get_runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--algo", default="maddpg")   # maddpg | mappo | masac | matd3 ...
    ap.add_argument("--env", default="uav_pursuit_apollonius_3d")  # ..._multitarget_3d for Ch4
    ap.add_argument("--num-targets", type=int, default=None)       # Ch4 multi-target
    ap.add_argument("--no-dynamic-alloc", action="store_true")     # Ch4 allocation ablation
    ap.add_argument("--apollonius-alloc", action="store_true")     # Ch4: Apollonius-based allocation
    ap.add_argument("--criterion", default="euclidean", choices=["euclidean", "visibility", "detour"])
    ap.add_argument("--use-obstacle-gat", action="store_true")
    ap.add_argument("--use-graph-module", action="store_true")
    ap.add_argument("--reward-disable", default="")     # comma-separated
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--building-mode", default=None)     # None -> curriculum default
    ap.add_argument("--randomize-density", default="")   # comma-separated densities (domain randomization)
    ap.add_argument("--strict-capture", action="store_true",
                    help="score a target captured only on physical reach, matching Ch3")
    ap.add_argument("--evader-center-pull", type=float, default=None,
                    help="APF pull toward airspace centre; keeps captures off the walls")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10000000)
    ap.add_argument("--parallels", type=int, default=16)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--test-episode", type=int, default=None)
    ap.add_argument("--target-max-speed", type=float, default=None)
    ap.add_argument("--target-min-speed", type=float, default=None)
    ap.add_argument("--closure-weight", type=float, default=None)
    ap.add_argument("--pincer-weight", type=float, default=None)  # 2nd-nearest closure (faster-evader)
    ap.add_argument("--close-k", type=int, default=None)          # layered encirclement: inner closers
    ap.add_argument("--obs-avoid-weight", type=float, default=None)  # earlier obstacle avoidance
    ap.add_argument("--no-guide-collapse", action="store_true")  # ablation: disable closure design
    ap.add_argument("--w-pos", type=float, default=None)
    ap.add_argument("--w-gap", type=float, default=None)
    ap.add_argument("--w-finish", type=float, default=None)
    ap.add_argument("--load-from", default=None)  # pretrained .pth for curriculum warmup (e.g. empty->medium)
    ap.add_argument("--surround-spawn", action="store_true")  # pursuers spawn AROUND the evader
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--separation-weight", type=float, default=None,
                    help="early teammate-separation shaping weight (N=4 stability profile)")
    ap.add_argument("--separation-distance", type=float, default=None,
                    help="distance in metres at which teammate separation shaping starts")
    ap.add_argument("--soft-collisions", action="store_true",
                    help="penalize obstacle hits but keep the training episode alive")
    ap.add_argument("--start-noise", type=float, default=None)
    ap.add_argument("--end-noise", type=float, default=None)
    ap.add_argument("--policy-only-load", action="store_true",
                    help="load policy/targets from a checkpoint but keep fresh optimizer state")
    ap.add_argument("--cbf-safety", action="store_true",
                    help="control-barrier-function safety filter on the policy action (Cheng+ AAAI-19)")
    ap.add_argument("--cbf-eta", type=float, default=None)
    ap.add_argument("--cbf-margin", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    env_id = f"apollonius_3d_{args.name}"
    p = argparse.Namespace(algo=args.algo, env=args.env,
                           env_id=env_id, device=args.device)
    if args.num_targets is not None:
        p.num_targets = args.num_targets
    if args.apollonius_alloc:
        p.apollonius_alloc = True
    if args.no_dynamic_alloc:
        p.dynamic_alloc = False
    p.criterion_mode = args.criterion
    p.use_obstacle_gat = args.use_obstacle_gat
    p.use_graph_module = args.use_graph_module
    if args.strict_capture:
        p.strict_capture = True
    if args.evader_center_pull is not None:
        p.evader_center_pull = args.evader_center_pull
    p.num_agents = args.num_agents
    p.seed = args.seed
    p.parallels = args.parallels
    p.running_steps = args.steps
    if args.eval_interval is not None:
        p.eval_interval = args.eval_interval
    if args.test_episode is not None:
        p.test_episode = args.test_episode
    if args.reward_disable:
        p.reward_disable = [s for s in args.reward_disable.split(",") if s]
    if args.building_mode:
        p.curriculum_enabled = False
        p.building_mode = args.building_mode
    if args.randomize_density:
        p.curriculum_enabled = False
        p.randomize_density = [s for s in args.randomize_density.split(",") if s]
    if args.target_max_speed is not None:
        p.target_max_speed = args.target_max_speed
    if args.target_min_speed is not None:
        p.target_min_speed = args.target_min_speed
    if args.closure_weight is not None:
        p.closure_weight = args.closure_weight
    if args.pincer_weight is not None:
        p.pincer_weight = args.pincer_weight
    if args.close_k is not None:
        p.close_k = args.close_k
    if args.obs_avoid_weight is not None:
        p.obs_avoid_weight = args.obs_avoid_weight
    if args.no_guide_collapse:
        p.guide_collapse = False
    if args.w_pos is not None:
        p.w_pos = args.w_pos
    if args.w_gap is not None:
        p.w_gap = args.w_gap
    if args.w_finish is not None:
        p.w_finish = args.w_finish
    if args.surround_spawn:
        p.surround_spawn = True
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    if args.separation_weight is not None:
        p.separation_weight = args.separation_weight
    if args.separation_distance is not None:
        p.separation_distance = args.separation_distance
    if args.cbf_safety:
        p.cbf_enabled = True
    if args.cbf_eta is not None:
        p.cbf_eta = args.cbf_eta
    if args.cbf_margin is not None:
        p.cbf_margin = args.cbf_margin
    if args.soft_collisions:
        p.terminate_on_collision = False
    if args.start_noise is not None:
        p.start_noise = args.start_noise
    if args.end_noise is not None:
        p.end_noise = args.end_noise
    runner = get_runner(algo=args.algo, env=args.env,
                        env_id=env_id, parser_args=p)
    if args.load_from:  # curriculum warmup: initialise from a pretrained policy
        if args.policy_only_load:
            if not os.path.isfile(args.load_from):
                raise ValueError("--policy-only-load requires --load-from to be a checkpoint file")
            import torch
            checkpoint = torch.load(args.load_from, map_location=args.device, weights_only=True)
            runner.agent.policy.load_state_dict(checkpoint["policy"], strict=False)
            print(f"[run_experiment] policy-only warm-started from {args.load_from}")
        else:
            runner.agent.load_model(args.load_from)
            print(f"[run_experiment] warm-started from {args.load_from}")
    runner.run(mode='benchmark')


if __name__ == "__main__":
    main()
