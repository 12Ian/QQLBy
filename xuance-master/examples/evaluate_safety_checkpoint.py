"""Evaluate network proposals with/without filtering and retain safety totals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from xuance import get_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--env-seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--parallels', type=int, default=8)
    parser.add_argument('--level', type=int, choices=range(5), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--disable-filter', action='store_true')
    args = parser.parse_args()
    if min(args.episodes, args.parallels) < 1 or args.env_seed < 0 or args.output.exists():
        parser.error('Use positive counts, a nonnegative seed and a new output path')
    cfg = yaml.safe_load(args.config.read_text())
    name = f'safety_eval_{args.env_seed}_{args.level}_{os.getpid()}'
    overrides = argparse.Namespace(env_id=name, device=args.device, seed=1,
        env_seed=args.env_seed, parallels=min(args.episodes, args.parallels),
        test_episode=args.episodes, curriculum_start_level=args.level,
        test_curriculum_level=args.level, baseline_init_actor=None,
        safety_filter_enabled=not args.disable_filter, uav_collision_enabled=True,
        terminate_on_boundary_collision=True, terminate_on_building_collision=True,
        terminate_on_collision=True)
    runner = get_runner(cfg['agent'].lower(), 'uav_pursuit_coverage_3d', name,
                        str(args.config.resolve()), overrides)
    envs = runner.agent.train_envs
    records = []
    original_step = envs.step

    def collect(actions):
        result = original_step(actions)
        _, _, terminated, truncated, infos = result
        for index, info in enumerate(infos):
            if (all(terminated[index].values()) or truncated[index]) and len(records) < args.episodes:
                diagnostic = info['infos']
                collision = bool(info['is_collision']) or any(diagnostic[f'{kind}_contact_events_episode'] > 0
                                for kind in ('building', 'boundary', 'uav'))
                records.append(dict(success=bool(info['is_success']), collision=collision,
                    steps=info['episode_step'], timeout=bool(truncated[index]),
                    safety_checks=diagnostic['safety_checks_episode'],
                    safety_interventions=diagnostic['safety_interventions_episode'],
                    uncertified_substeps=diagnostic['safety_uncertified_substeps_episode']))
        return result

    envs.step = collect
    started = time.time()
    try:
        checkpoint = torch.load(args.checkpoint, map_location=runner.agent.device, weights_only=False)
        runner.agent.policy.load_state_dict(checkpoint['policy'])
        assert all(torch.isfinite(v).all() for v in checkpoint['policy'].values())
        runner.agent.policy.eval()
        envs.env_method('seed')
        scores = runner.agent.run_episodes(n_episodes=args.episodes, run_envs=envs,
            test_mode=True, close_envs=False, curriculum_level=args.level)
        metrics = dict(runner.agent.last_test_metrics)
        assert metrics['episodes'] == len(records) == args.episodes
        assert np.mean([r['collision'] for r in records]) == metrics['collision_rate']
        checks = sum(r['safety_checks'] for r in records)
        interventions = sum(r['safety_interventions'] for r in records)
        report = dict(checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            config=str(args.config.resolve()), env_seed=args.env_seed, level=args.level,
            exploration_noise=False, collision_termination=True,
            uav_collision_enabled=True, safety_filter_enabled=not args.disable_filter,
            metrics=metrics, mean_return=float(np.mean(scores)), episodes=records,
            safety_metrics=dict(checks=checks, interventions=interventions,
                intervention_fraction=interventions / checks if checks else None,
                uncertified_substeps=sum(r['uncertified_substeps'] for r in records)),
            elapsed_seconds=time.time() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(dict(output=str(args.output), metrics=metrics,
                              safety=report['safety_metrics'])), flush=True)
    finally:
        envs.close()
        runner.agent.finish()


if __name__ == '__main__':
    main()
