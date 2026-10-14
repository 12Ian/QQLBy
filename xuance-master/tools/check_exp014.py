"""Verify exp014 settings, 007 weight transfer and native CUDA updates."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from xuance import get_runner
from xuance.engine.run_marl import initialize_baseline_policy
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    name = 'exp014_gpu_check_v1'
    out = ROOT / 'outputs/reviews' / name
    assert not out.exists(), 'Do not overwrite the implementation check'
    configuration = ROOT / 'experiments/baseline_exp014.yaml'
    cfg = yaml.safe_load(configuration.read_text())
    parent = yaml.safe_load((ROOT / 'experiments/baseline_exp013.yaml').read_text())
    changed = {k for k in set(cfg) | set(parent) if cfg.get(k) != parent.get(k)}
    assert changed == {'env_id', 'running_steps', 'evader_sense',
                       'baseline_formation_weight', 'baseline_init_checkpoint'}
    assert cfg['running_steps'] == 6000000 and cfg['evader_sense'] == 200
    assert cfg['baseline_formation_weight'] == 0 and cfg['curriculum_start_level'] == 0
    checkpoint = Path(cfg['baseline_init_checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == '6c1f41773b47238f898d0f4243b092b8506d2f1ce3bba70e47cac3df0a7d5a32'
    reference = torch.load(checkpoint, map_location='cpu', weights_only=False)['policy']
    cfg['buildings_dir'] = str(ROOT / 'xuance/environment/multi_agent_env/buildings')
    environment = UAVPursuitCoverage3DEnv(Namespace(**cfg))
    try:
        environment.reset()
        assert environment.evader_sense == 200 and environment.baseline_formation_weight == 0
        assert all(space.shape == (79,) for space in environment.observation_space.values())
        for _ in range(4):
            environment.step({agent: np.zeros(3) for agent in environment.agents})
        assert all(parts['r_form'] == 0 for parts in environment.episode_sub_rewards.values())
    finally:
        environment.close()
    # Check overrides affect this short test only; keep the 512/512 model architecture.
    overrides = Namespace(env_id='baseline_' + name, seed=1, env_seed=1,
        device='cuda:0', parallels=2, running_steps=1024, eval_interval=1024,
        test_episode=2, baseline_eval_parallels=2, episode_length=32,
        buffer_size=4096, batch_size=32, start_training=32,
        baseline_final_eval_interval=1024)
    runner = get_runner('matd3', 'uav_pursuit_coverage_3d', 'baseline_' + name,
                        str(configuration), overrides)
    try:
        initialization = initialize_baseline_policy(runner.agent, runner.config)
        loaded = runner.agent.policy.state_dict()
        assert set(loaded) == set(reference)
        assert all(torch.equal(v.cpu(), reference[k]) for k, v in loaded.items())
        before = {k: v.clone() for k, v in loaded.items()}
        runner.run(mode='benchmark')
        after = runner.agent.policy.state_dict()
        updated = {prefix: any(not torch.equal(before[k], v) for k, v in after.items()
                              if k.startswith(prefix + '.'))
                   for prefix in ('actor', 'critic_A', 'critic_B')}
        assert all(updated.values()) and all(torch.isfinite(v).all() for v in after.values())
        maxima = {k: float(np.max(np.abs(v))) for k, v in runner.agent.memory.data['actions'].items()}
        assert max(maxima.values()) <= 1
        result = ROOT / 'outputs/results/matd3' / ('baseline_' + name)
        summary = json.loads((result / 'baseline_summary.json').read_text())
        assert summary['environment_steps'] == 1024
        report = dict(status='passed',transfers=1024,initialization=initialization,
            initial_tensors_match_007=True,model_tensors=len(reference),updated=updated,
            reward_formation_zero=True,evader_sense=200,observation_size=79,
            changed_config_keys=sorted(changed),bounded_replay=maxima,
            summary=summary,note='Implementation check; not a training performance result')
        out.mkdir(parents=True,exist_ok=False)
        (out / 'audit.json').write_text(json.dumps(report,indent=2) + '\n')
        print('EXP014_GPU_CHECK', json.dumps(report), flush=True)
    finally:
        runner.agent.train_envs.close()
        runner.agent.writer.close()


if __name__ == '__main__':
    main()
