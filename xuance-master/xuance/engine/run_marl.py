import os
import json
import time
import csv
import numpy as np
from copy import deepcopy
from argparse import Namespace
from datetime import datetime
from xuance.common import Optional, create_directory
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, make_envs
from xuance.engine import RunnerBase


def initialize_baseline_policy(agent, config):
    """Load actor-only or full policy weights; keep this run's fresh optimizers."""
    actor_path = getattr(config, "baseline_init_actor", None)
    full_path = getattr(config, "baseline_init_checkpoint", None)
    if actor_path and full_path:
        raise ValueError("Actor and full checkpoint initialization are exclusive")
    path = full_path or actor_path
    if not path:
        return None
    import hashlib
    import torch
    from pathlib import Path

    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    current = agent.policy.state_dict()
    prefixes = ("actor.", "target_actor.")
    selected = {key: value for key, value in checkpoint["policy"].items()
                if full_path or key.startswith(prefixes)}
    expected = set(current) if full_path else {key for key in current if key.startswith(prefixes)}
    if set(selected) != expected or any(selected[key].shape != current[key].shape for key in selected):
        raise ValueError("Baseline initialization has incompatible parameters")
    if any(not torch.isfinite(value).all().item() for value in selected.values()):
        raise ValueError("Baseline initialization has non-finite parameters")
    current.update(selected)
    agent.policy.load_state_dict(current)
    return dict(kind="full_policy" if full_path else "actor_only", path=str(Path(path).resolve()),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), tensors=len(selected),
                optimizer_restored=False)


class RunnerMARL(RunnerBase):
    def __init__(self,
                 config: Namespace,
                 envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                 agent=None,
                 manage_resources: bool = None):
        # Store configuration
        self.config = config
        self.env_id = self.config.env_id
        
        super(RunnerMARL, self).__init__(self.config, envs, agent, manage_resources)

        # Build agent if not injected externally
        if getattr(self.config, 'dl_toolbox', 'torch'):
            from xuance.torch.agents import REGISTRY_Agents
            from xuance.torch.utils import collect_device_info
        elif getattr(self.config, 'dl_toolbox', 'tensorflow'):
            from xuance.tensorflow.agents import REGISTRY_Agents
            from xuance.tensorflow.utils import collect_device_info
        elif getattr(self.config, 'dl_toolbox', 'mindspore'):
            from xuance.mindspore.agents import REGISTRY_Agents
            from xuance.mindspore.utils import collect_device_info
        else:
            raise NotImplementedError
        self.collect_device_info = collect_device_info
        self.agent = REGISTRY_Agents[self.config.agent](self.config, self.envs) if agent is None else agent

        # Distributed training setup (rank-aware behavior)
        if self.agent.distributed_training:
            self.rank = int(os.environ['RANK'])

    def _run_train(self, **kwargs):
        n_train_steps = max(1, self.config.running_steps // self.n_envs)
        self.agent.train(n_train_steps)
        self.rprint("Finish training.")
        self.agent.save_model(model_name="final_train_model.pth")

    def _run_test(self, **kwargs):
        config_test = deepcopy(self.config)
        config_test.parallels = kwargs.get("n_envs", 1)
        config_test.render = kwargs.get('render', True)
        config_test.render_mode = kwargs.get('render_mode', getattr(self.config, 'render_mode', 'human'))
        model_path = kwargs.get('model_path', self.agent.model_dir_load)
        test_episodes = kwargs.get('test_episodes', self.config.test_episode)
        test_envs = make_envs(config_test)

        if self.rank == 0:
            self.agent.load_model(model_path)
            scores = self.agent.test(test_episodes=test_episodes, test_envs=test_envs, close_envs=True)
            print("\n---------------------Testing Results--------------------")
            print("Test Episode Scores: ", scores)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Best Score: %.2f" % max(scores))
            print("Finish testing.")

    def _run_benchmark(self, **kwargs):
        if getattr(self.config, "reward_mode", "coverage") == "baseline":
            return self._run_baseline_benchmark(**kwargs)
        running_steps = kwargs.get('running_steps', self.config.running_steps)
        eval_interval = kwargs.get('eval_interval', self.config.eval_interval)
        test_episodes = kwargs.get('test_episodes', self.config.test_episode)
        benchmark_result_path = kwargs.get('benchmark_result_path', self.config.result_dir)
        best_model_path = os.path.join(os.getcwd(), benchmark_result_path, "best_model")
        # Prepare directory for storing benchmark results.
        benchmark_result_path = os.path.join(os.getcwd(), benchmark_result_path)
        create_directory(benchmark_result_path)
        # Create test_scores.csv file to store testing scores.
        test_scores_csv = os.path.join(benchmark_result_path, "test_scores.csv")
        learning_curve_csv = os.path.join(benchmark_result_path, "learning_curve.csv")
        if self.rank == 0:
            with open(test_scores_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["step"] + [f"return_episode_{i}" for i in range(test_episodes)]
                writer.writerow(header)
            with open(learning_curve_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "avg_return"])

        meta_data = self.agent.meta_data
        meta_data['benchmark'] = {
            "running_steps": running_steps,
            "eval_interval": eval_interval,
            "test_episodes": test_episodes,
        }
        meta_data["system_info"] = self.collect_device_info(rank=int(getattr(self, "rank", 0)), agent=self.agent)
        config_dict = vars(self.agent.config).copy()
        config_dict.pop("observation_space", None)
        config_dict.pop("action_space", None)

        # Prepare testing environments.
        config_test = deepcopy(self.config)
        config_test.parallels = 1  # config_test.test_episode
        test_envs = make_envs(config_test)

        train_steps = max(1, running_steps // self.n_envs)
        eval_interval = max(1, eval_interval // self.n_envs)
        num_epoch = train_steps // eval_interval

        # Start benchmarking...
        start_time = time.time()
        start_time_iso = datetime.now().astimezone().isoformat()
        best_model_time_iso = start_time_iso
        if self.rank == 0:
            test_scores = self.agent.test(test_episodes=test_episodes, test_envs=test_envs, close_envs=False)
            with open(test_scores_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.agent.current_step)] + test_scores)
            with open(learning_curve_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.agent.current_step), np.mean(test_scores)])
        else:
            test_scores = 0.0
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agent.current_step}
        curriculum_stage = int(getattr(self.config, "curriculum_start_level", 0))
        for i_epoch in range(num_epoch):
            if bool(getattr(self.config, "curriculum_auto_advance_by_steps", False)):
                interval = int(getattr(self.config, "curriculum_min_steps_per_level", 300000))
                max_level = int(getattr(self.config, "curriculum_max_level", 4))
                if interval <= 0:
                    raise ValueError("curriculum_min_steps_per_level must be positive")
                next_stage = min(max_level, int(self.agent.current_step) // interval)
                if next_stage != curriculum_stage:
                    requested = self.agent.train_envs.env_method(
                        "schedule_curriculum_level", next_stage)
                    self.rprint(f"[curriculum] step={self.agent.current_step} "
                                f"level {curriculum_stage} -> {next_stage}; "
                                f"apply on reset in {len(requested)} environments")
                    curriculum_stage = next_stage
            self.rprint("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agent.train(train_steps=eval_interval)
            if self.rank == 0:
                test_scores = self.agent.test(test_episodes=test_episodes,
                                              test_envs=test_envs,
                                              close_envs=False)
                with open(test_scores_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(self.agent.current_step)] + test_scores)
                with open(learning_curve_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(self.agent.current_step), np.mean(test_scores)])

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": self.agent.current_step}
                    # save best model
                    self.agent.save_model(model_name="best_model.pth", model_path=best_model_path)
                    best_model_time_iso = datetime.now().astimezone().isoformat()

        # End benchmarking.
        # Save best model information.
        end_time = time.time()
        end_time_iso = datetime.now().astimezone().isoformat()
        timestamps = {
            "start_time": start_time_iso,
            "best_model_time": best_model_time_iso,
            "end_time": end_time_iso,
            "elapsed_seconds": round(end_time - start_time, 3)
        }
        meta_data["timestamps"] = timestamps
        if self.rank == 0:
            with open(os.path.join(benchmark_result_path, "meta_data.json"), "w", encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            with open(os.path.join(benchmark_result_path, "config.json"), "w", encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            with open(os.path.join(benchmark_result_path, "best_model_info.json"), "w", encoding='utf-8') as f:
                json.dump(best_scores_info, f, indent=2, ensure_ascii=False)

        self.agent.save_model(model_name="final_train_model.pth")
        test_envs.close()
        self.rprint("Best Model Score: %.2f, std=%.2f. "
                    "Best Step: %d" % (best_scores_info["mean"], best_scores_info["std"],
                                       best_scores_info["step"]))

    def _run_baseline_benchmark(self, **kwargs):
        from .uav_curriculum import PerformanceCurriculum
        from .uav_baseline_logging import BaselineTrainingLog

        if self.agent.distributed_training:
            raise ValueError("The UAV baseline benchmark uses a single learner")
        if bool(getattr(self.config, "curriculum_auto_advance_by_steps", False)):
            raise ValueError("Baseline curriculum must advance by evaluation performance")
        running_steps = int(kwargs.get("running_steps", self.config.running_steps))
        interval = int(kwargs.get("eval_interval", self.config.eval_interval))
        episodes = int(kwargs.get("test_episodes", self.config.test_episode))
        final_eval_interval = int(getattr(self.config, "baseline_final_eval_interval", interval))
        if min(running_steps, interval, episodes) < 1:
            raise ValueError("Training and evaluation budgets must be positive")
        if final_eval_interval <= 0:
            raise ValueError("Final task evaluation interval must be positive")
        noise_decay_steps = int(getattr(self.config, "noise_decay_steps", running_steps))
        if noise_decay_steps <= 0:
            raise ValueError("Noise decay steps must be positive")
        if self.agent.noise_scale is not None:
            self.agent.delta_noise = (self.agent.start_noise - self.agent.end_noise) / noise_decay_steps
        initialization = initialize_baseline_policy(self.agent, self.config)
        if initialization:
            self.rprint(f"[policy initialization] {initialization['kind']}: {initialization['path']}; "
                        f"loaded {initialization['tensors']} tensors")
        if bool(getattr(self.config, "baseline_save_initial_model", False)):
            self.agent.save_model(model_name="initial_random_model.pth")
        result_dir = os.path.abspath(kwargs.get("benchmark_result_path", self.config.result_dir))
        create_directory(result_dir)
        if initialization:
            with open(os.path.join(result_dir, "initialization_audit.json"), "w", encoding="utf-8") as stream:
                json.dump(initialization, stream, indent=2)
        best_dir = os.path.join(result_dir, "best_model")
        create_directory(best_dir)
        final_level = len(self.config.curriculum_schedule) - 1
        gate = PerformanceCurriculum(
            level=int(getattr(self.config, "curriculum_start_level", 0)), max_level=final_level,
            min_steps=int(self.config.curriculum_min_steps_per_level), min_episodes=episodes,
            success_threshold=float(getattr(self.config, "curriculum_success_threshold", 0.8)),
            collision_threshold=float(getattr(self.config, "curriculum_collision_threshold", 0.1)),
            required_passes=int(getattr(self.config, "curriculum_required_passes", 2)),
            level_start_step=int(self.agent.current_step))
        config_test = deepcopy(self.config)
        config_test.parallels = min(episodes, int(getattr(self.config, "baseline_eval_parallels", 1)))
        if config_test.parallels < 1:
            raise ValueError("Evaluation parallel count must be positive")
        config_test.seed = int(self.config.seed) + 10000
        config_test.env_seed = int(self.config.env_seed) + 10000
        if bool(getattr(self.config, "baseline_evaluate_full_episode", False)):
            # Evaluation retains the original 400-step collision accounting.
            config_test.terminate_on_boundary_collision = False
            config_test.terminate_on_collision = False
            config_test.terminate_on_building_collision = False
        test_envs = make_envs(config_test)
        fields = ["step", "phase", "level", "episodes", "success_rate", "safe_success_rate",
                  "collision_rate", "building_collision_rate", "boundary_collision_rate",
                  "uav_collision_rate", "mean_episode_steps", "mean_return"]
        history_path = os.path.join(result_dir, "curriculum_eval.csv")
        with open(history_path, "w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
        best_final = None
        best_stages = {}
        transitions = []
        start_step = int(self.agent.current_step)
        started = time.time()
        config_dict = vars(self.agent.config).copy()
        for key in ("observation_space", "action_space", "state_space"):
            config_dict.pop(key, None)
        config_dict["evaluation_env_seed"] = config_test.env_seed
        with open(os.path.join(result_dir, "config.json"), "w", encoding="utf-8") as stream:
            json.dump(config_dict, stream, indent=2, ensure_ascii=False, default=str)
        monitor = BaselineTrainingLog(result_dir, self.agent.log_infos)
        self.agent.baseline_monitor = monitor

        def evaluate(level, phase):
            # Reuse a separate deterministic evaluation seed stream for comparisons.
            test_envs.env_method("seed")
            scores = self.agent.run_episodes(n_episodes=episodes, run_envs=test_envs,
                                            test_mode=True, close_envs=False, curriculum_level=level)
            metrics = dict(self.agent.last_test_metrics)
            metrics.update(step=int(self.agent.current_step), phase=phase, level=level,
                           mean_return=float(np.mean(scores)))
            with open(history_path, "a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(metrics)
            self.agent.log_infos({f"Baseline-Eval/{phase}/Level-{level + 1}/{name}": float(metrics[name])
                                  for name in ("success_rate", "safe_success_rate", "collision_rate",
                                               "building_collision_rate", "boundary_collision_rate")},
                                 self.agent.current_step)
            self.rprint(f"[baseline eval] step={metrics['step']} level={level} {phase}: "
                        f"success={metrics['success_rate']:.3f} "
                        f"safe_success={metrics['safe_success_rate']:.3f} "
                        f"collision={metrics['collision_rate']:.3f}")
            return metrics

        def checkpoint_key(metrics):
            return (metrics["safe_success_rate"], -metrics["collision_rate"], metrics["mean_return"])

        try:
            while True:
                current = evaluate(gate.level, "current")
                if bool(getattr(self.config, "baseline_evaluate_previous_levels", False)):
                    for previous_level in range(gate.level):
                        evaluate(previous_level, "retention")
                previous = best_stages.get(gate.level)
                if previous is None or checkpoint_key(current) > checkpoint_key(previous):
                    best_stages[gate.level] = current
                    self.agent.save_model(model_name=f"best_stage_{gate.level}.pth", model_path=best_dir)
                remaining = running_steps - (int(self.agent.current_step) - start_step)
                final = current if gate.level == final_level else None
                if final is None and ((int(self.agent.current_step) - start_step) % final_eval_interval == 0
                                      or remaining <= 0):
                    final = evaluate(final_level, "final_task")
                if final is not None and (best_final is None or checkpoint_key(final) > checkpoint_key(best_final)):
                    best_final = final
                    self.agent.save_model(model_name="best_model.pth", model_path=best_dir)
                next_level = (gate.observe(int(self.agent.current_step), current)
                              if remaining > 0 and self.config.curriculum_enabled else None)
                if next_level is not None:
                    # Training workers apply the requested map/radius/speed on their next reset.
                    self.agent.train_envs.env_method("schedule_curriculum_level", next_level)
                    transitions.append({"step": int(self.agent.current_step), "level": next_level})
                    self.agent.log_infos({"Curriculum/Requested-Level": next_level}, self.agent.current_step)
                    self.rprint(f"[curriculum] requested level {next_level}; apply on reset")
                if remaining <= 0:
                    break
                vector_steps = min(max(1, interval // self.n_envs), max(1, remaining // self.n_envs))
                self.agent.train(train_steps=vector_steps)
                monitor.flush_updates()
                self.agent.writer.flush() if not self.agent.use_wandb else None
                self.agent.save_model(model_name="latest_train_model.pth")
            self.agent.save_model(model_name="final_train_model.pth")
            summary = {
                "reward_mode": "baseline", "curriculum_level": gate.level,
                "transitions": transitions, "best_final_task": best_final,
                "best_by_stage": best_stages, "environment_steps": int(self.agent.current_step) - start_step,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            with open(os.path.join(result_dir, "baseline_summary.json"), "w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, ensure_ascii=False)
        finally:
            self.agent.baseline_monitor = None
            monitor.close()
            test_envs.close()
