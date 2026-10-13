"""按指定参数启动独立的多智能体追捕训练任务。"""
import argparse
import os
from xuance import get_runner


def main():
    ap = argparse.ArgumentParser(
        description="启动一组新的多智能体无人机追捕训练，并将结果保存到独立目录。")
    ap.add_argument("--name", required=True,
                    help="本次实验名称；用于区分日志、模型和结果目录（必填）")
    ap.add_argument("--algo", default="maddpg",
                    help="强化学习算法，例如 maddpg、mappo、masac、matd3（默认：maddpg）")
    ap.add_argument("--env", default="uav_pursuit_coverage_3d",
                    help="环境名称；单目标默认 coverage_3d，可指定多目标环境")
    ap.add_argument("--num-targets", type=int, default=None,
                    help="多目标环境中的逃逸机数量（默认：使用算法配置）")
    ap.add_argument("--no-dynamic-alloc", action="store_true",
                    help="关闭多目标任务中的动态目标分配")
    ap.add_argument("--apollonius-alloc", action="store_true",
                    help="启用基于 Apollonius 几何的目标分配")
    ap.add_argument("--criterion", default="euclidean", choices=["euclidean", "visibility", "detour"],
                    help="追捕距离判据：euclidean 欧氏距离、visibility 可见路径、detour 绕障距离（默认：euclidean）")
    ap.add_argument("--use-obstacle-gat", action="store_true",
                    help="启用建筑物注意力网络（默认：关闭）")
    ap.add_argument("--obstacle-gat-k", type=int, default=None,
                    help="提供给建筑物注意力网络的最近建筑物数量（默认：使用算法配置）")
    ap.add_argument("--use-graph-module", action="store_true",
                    help="启用追捕机之间的图通信网络（默认：关闭）")
    ap.add_argument("--reward-disable", default="",
                    help="要关闭的奖励项名称，多个名称用英文逗号分隔（默认：不关闭）")
    ap.add_argument("--num-agents", type=int, default=6,
                    help="追捕无人机数量（默认：6）")
    ap.add_argument("--building-mode", default=None,
                    help="建筑环境类型；指定后关闭课程切换（默认：由课程配置决定）")
    ap.add_argument("--randomize-density", default="",
                    help="建筑密度随机化选项，多个值用英文逗号分隔；指定后关闭课程切换")
    ap.add_argument("--strict-capture", action="store_true",
                    help="仅在无人机实际到达目标时判定捕获，使用严格物理捕获标准")
    ap.add_argument("--evader-center-pull", type=float, default=None,
                    help="逃逸机朝空域中心移动的引导强度（默认：使用环境配置）")
    ap.add_argument("--seed", type=int, default=1,
                    help="随机种子（默认：1）")
    ap.add_argument("--steps", type=int, default=10000000,
                    help="训练总环境步数（默认：10000000）")
    ap.add_argument("--parallels", type=int, default=16,
                    help="并行运行的环境数量（默认：16）")
    ap.add_argument("--eval-interval", type=int, default=100000,
                    help="每训练多少步进行一次测试（默认：使用算法配置）")
    ap.add_argument("--test-episode", type=int, default=100,
                    help="每次测试运行的回合数（默认：使用算法配置）")
    ap.add_argument("--target-max-speed", type=float, default=None,
                    help="逃逸机最大速度，单位 m/s（默认：使用环境配置）")
    ap.add_argument("--target-min-speed", type=float, default=None,
                    help="逃逸机最小速度，单位 m/s（默认：使用环境配置）")
    ap.add_argument("--obs-avoid-weight", type=float, default=None,
                    help="障碍物规避奖励权重（默认：使用环境配置）")
    ap.add_argument("--w-cov", type=float, default=None, help="未来持续覆盖权重")
    ap.add_argument("--w-prep", dest="w_prep", type=float,
                    default=None, help="安全路线覆盖准备度权重")
    ap.add_argument("--w-hold", type=float, default=None, help="真实持续捕获进展权重")
    ap.add_argument("--w-safe", type=float, default=None, help="安全惩罚权重")
    ap.add_argument("--w-terminal", type=float, default=None, help="真实捕获终端奖励")
    ap.add_argument("--load-from", default=None,
                    help="用于继续训练或预热的模型检查点 .pth 路径（默认：不加载）")
    ap.add_argument("--surround-spawn", action="store_true",
                    help="让追捕机在逃逸机周围出生（默认：关闭）")
    ap.add_argument("--spawn-radius", type=float, default=None,
                    help="环绕出生时的初始半径，单位 m（默认：使用环境配置）")
    ap.add_argument("--separation-weight", type=float, default=None,
                    help="追捕机之间的分散奖励权重（默认：使用环境配置）")
    ap.add_argument("--separation-distance", type=float, default=None,
                    help="开始计算追捕机分散奖励的距离，单位 m（默认：使用环境配置）")
    ap.add_argument("--soft-collisions", action="store_true",
                    help="碰撞建筑物时施加惩罚，但继续当前回合（默认：关闭）")
    ap.add_argument("--start-noise", type=float, default=None,
                    help="训练初期的探索噪声强度（默认：使用算法配置）")
    ap.add_argument("--end-noise", type=float, default=None,
                    help="训练结束阶段的探索噪声强度（默认：使用算法配置）")
    ap.add_argument("--policy-only-load", action="store_true",
                    help="仅加载策略网络参数并重新初始化优化器；需同时指定 --load-from")
    ap.add_argument("--device", default="cuda:0",
                    help="模型运行设备，例如 cuda:0 或 cpu（默认：cuda:0）")
    args = ap.parse_args()

    env_id = f"coverage_3d_{args.name}" if args.env == "uav_pursuit_coverage_3d" else f"apollonius_3d_{args.name}"
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
    if args.obstacle_gat_k is not None:
        p.obstacle_gat_k = args.obstacle_gat_k
        p.nearest_building_k = args.obstacle_gat_k
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
    if args.obs_avoid_weight is not None:
        p.obs_avoid_weight = args.obs_avoid_weight
    for weight_name in ("w_cov", "w_prep", "w_hold", "w_safe", "w_terminal"):
        value = getattr(args, weight_name)
        if value is not None:
            setattr(p, weight_name, value)
    if args.surround_spawn:
        p.surround_spawn = True
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    if args.separation_weight is not None:
        p.separation_weight = args.separation_weight
    if args.separation_distance is not None:
        p.separation_distance = args.separation_distance
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
