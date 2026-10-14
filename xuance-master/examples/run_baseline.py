"""Run the fixed-city pursuit baseline with performance-based curriculum."""
import argparse
from datetime import datetime
from pathlib import Path
import yaml
from xuance import get_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--parallels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--start-level", type=int, choices=range(5), default=0)
    parser.add_argument("--config", type=Path, help="Baseline YAML for a numbered parameter experiment")
    parser.add_argument("--smoke", action="store_true", help="Short CPU/GPU integration test with small networks")
    args = parser.parse_args()
    if min(args.steps, args.parallels, args.eval_interval, args.test_episodes) < 1:
        parser.error("Steps, parallels, evaluation interval and episode count must be positive")
    env_id = f"baseline_{args.name}"
    root = Path(__file__).resolve().parents[1]
    overrides = argparse.Namespace(
        env_id=env_id, device=args.device, seed=args.seed, env_seed=args.seed,
        parallels=args.parallels, running_steps=args.steps, eval_interval=args.eval_interval,
        test_episode=args.test_episodes, curriculum_start_level=args.start_level)
    if args.smoke:
        overrides.parallels = 2
        overrides.running_steps = 1024
        overrides.eval_interval = 512
        overrides.test_episode = 2
        overrides.episode_length = 32
        overrides.actor_hidden_size = [64, 64]
        overrides.critic_hidden_size = [64, 64]
        overrides.batch_size = 32
        overrides.buffer_size = 4096
        overrides.start_training = 32
        overrides.training_frequency = 1
    config_path = (args.config.resolve() if args.config is not None else
                   root / "xuance/configs/maddpg/uav_pursuit_baseline_3d.yaml")
    if not config_path.is_file():
        parser.error(f"Configuration does not exist: {config_path}")
    method = yaml.safe_load(config_path.read_text())["agent"].lower()
    if method not in ("maddpg", "matd3"):
        parser.error("The pursuit baseline supports MADDPG or MATD3")
    if (root / "outputs/results" / method / env_id).exists():
        parser.error("Experiment name already exists; choose a new --name")
    runner = get_runner(method, "uav_pursuit_coverage_3d", env_id,
                        config_path=str(config_path), parser_args=overrides)
    if args.smoke:
        runner.agent.save_model(model_name="smoke_initial_model.pth")
    runner.run(mode="benchmark")


if __name__ == "__main__":
    main()
