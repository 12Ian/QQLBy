"""Read-only action/Q/return diagnosis using the actual baseline simulator."""
import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv


def rollout(config, checkpoint, seed, episodes, full=False):
    cfg = dict(config, env_seed=seed, curriculum_start_level=0, render=False,
               terminate_on_building_collision=not full, terminate_on_boundary_collision=not full)
    env = UAVPursuitCoverage3DEnv(Namespace(**cfg))
    n = env.num_agents
    ids = torch.eye(n)
    key = env.agents[0]
    d = env.observation_space[key].shape[0]
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['policy']
    actor = load_network(state, f'actor.{key}.model.', [d+n]+cfg['actor_hidden_size']+[3], True)
    critics = [load_network(state, f'critic_{letter}.{key}.model.', [(d+3)*n+n]+cfg['critic_hidden_size']+[1]) for letter in ('A','B')]
    rows = []
    all_actions = []
    for episode in range(episodes):
        obs, _ = env.reset()
        initial = env.uav_positions.copy()
        ret = np.zeros(n)
        discount = 1.
        trace = []
        first_hit = None
        any_hit = False
        for step in range(env.max_episode_steps):
            x = torch.tensor(np.stack([obs[a] for a in env.agents]))
            with torch.no_grad():
                actions = actor(torch.cat((x, ids), dim=1))
            if step == 0:
                joint_a = actions.detach().flatten().requires_grad_(True)
                joint = torch.cat((x.flatten(), joint_a)).repeat(n,1)
                values = [net(torch.cat((joint,ids),dim=1)).flatten() for net in critics]
                predicted = [v.detach().tolist() for v in values]
                gradient = torch.autograd.grad(values[0].mean(),joint_a)[0].reshape(n,3).tolist()
            a = actions.numpy()
            all_actions.append(a)
            before = dict(step=step, positions=env.uav_positions.tolist(), velocities=env.uav_velocities.tolist(),
                          accelerations=env.uav_accelerations.tolist(), target=env.target_position.tolist(), actions=a.tolist())
            obs,rewards,terminated,truncated,info = env.step({agent:a[i] for i,agent in enumerate(env.agents)})
            ret += discount*np.array([rewards[agent] for agent in env.agents])
            discount *= cfg['gamma']
            any_hit |= bool(info['is_collision'])
            if info['is_collision'] and first_hit is None:
                first_hit = dict(**before,sources=info['collision_sources'])
            if episode < 2:
                trace.append(dict(**before,rewards=rewards,collision=bool(info['is_collision'])))
            if all(terminated.values()) or truncated:
                break
        rows.append(dict(episode=episode,steps=step+1,initial=initial.tolist(),final=env.uav_positions.tolist(),
                         success=bool(info['is_success']),collision=any_hit,first_hit=first_hit,
                         initial_Q_A=predicted[0],initial_Q_B=predicted[1],initial_Q_action_gradient=gradient,
                         discounted_return=ret.tolist(),q_bias=(np.minimum(*predicted)-ret).tolist(),
                         bias_is_terminal_return=not bool(truncated), sub_rewards=env.episode_sub_rewards,
                         trace=trace))
    env.close()
    terminated = [row for row in rows if row['bias_is_terminal_return']]
    actions = np.concatenate(all_actions)
    return dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                seed=seed,collision_termination=not full,episodes=episodes,
                captures=sum(row['success'] for row in rows),safe_captures=sum(row['success'] and not row['collision'] for row in rows),
                collisions=sum(row['collision'] for row in rows),mean_steps=float(np.mean([row['steps'] for row in rows])),
                saturation_fraction=float(np.mean(np.abs(actions)>.95)),mean_actions=actions.mean(axis=0).tolist(),
                mean_initial_Q_min=float(np.mean([np.minimum(row['initial_Q_A'],row['initial_Q_B']) for row in rows])),
                mean_discounted_return=float(np.mean([row['discounted_return'] for row in rows])),
                mean_bias_terminal_only=float(np.mean([row['q_bias'] for row in terminated])) if terminated else None,
                terminal_episodes=len(terminated),rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cfg=json.loads((args.root/'outputs/results/matd3/baseline_exp009/config.json').read_text())
    sources={'initial':args.root/'outputs/experiments/exp009/initial_random_model.pth',
             'final':args.root/'outputs/reviews/exp009_final_v1/final009.pth',
             'best':args.root/'outputs/results/matd3/baseline_exp009/best_model/best_stage_0.pth'}
    summary={}
    for name,path in sources.items():
        result=rollout(cfg,path,66001,32)
        (args.output/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        summary[name]={k:v for k,v in result.items() if k!='rows'}
        print(name,json.dumps(summary[name]),flush=True)
    result=rollout(cfg,sources['final'],66001,8,full=True)
    (args.output/'final_full_episode.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary['final_full_episode']={k:v for k,v in result.items() if k!='rows'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
