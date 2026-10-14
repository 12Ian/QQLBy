"""Inspect collision approaches for a frozen feedforward baseline policy."""
import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv


def load_network(state, prefix, sizes, actor=False):
    layers = []
    for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(sizes) - 2:
            layers.append(torch.nn.LeakyReLU())
        elif actor:
            layers.append(torch.nn.Tanh())
    network = torch.nn.Sequential(*layers)
    network.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
    network.eval()
    return network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--env-seed', type=int, default=21001)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    assert config['representation'] == 'Basic_Identical' and config['activation'] == 'leaky_relu'
    assert config['use_parameter_sharing'] and not config.get('use_graph_module') and not config.get('use_obstacle_gat')
    assert all(config[name] for name in ['terminate_on_collision', 'terminate_on_boundary_collision',
                                         'terminate_on_building_collision']), 'This diagnostic stops at the first collision'
    config.update(env_seed=args.env_seed, curriculum_start_level=args.level, render=False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['policy']
    assert all(torch.isfinite(value).all().item() for value in state.values())
    env = UAVPursuitCoverage3DEnv(Namespace(**config))
    count = env.num_agents
    key = env.agents[0]
    ids = torch.eye(count)
    obs_size = env.observation_space[key].shape[0]
    actor = load_network(state, f'actor.{key}.model.', [obs_size + count] + config['actor_hidden_size'] + [3], True)
    prefixes = [f'critic_A.{key}.model.', f'critic_B.{key}.model.'] if config['agent'] == 'MATD3' else [f'critic.{key}.model.']
    critics = [load_network(state, prefix, [(obs_size + 3) * count + count] + config['critic_hidden_size'] + [1]) for prefix in prefixes]
    rows, actions_all = [], []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            discounted_return = np.zeros(count)
            discount = 1.0
            predicted_q = None
            for step in range(env.max_episode_steps):
                before_pos = env.uav_positions.copy()
                before_vel = env.uav_velocities.copy()
                before_acc = env.uav_accelerations.copy()
                before_target = env.target_position.copy()
                observation = torch.as_tensor(np.stack([obs[a] for a in env.agents]))
                with torch.no_grad():
                    actions = actor(torch.cat([observation, ids], dim=1))
                    if predicted_q is None:
                        joint = torch.cat([observation.flatten(), actions.flatten()]).repeat(count, 1)
                        values = [critic(torch.cat([joint, ids], dim=1)).squeeze(1) for critic in critics]
                        predicted_q = torch.stack(values).min(dim=0).values.numpy()
                actions_all.append(actions.numpy())
                obs, reward, terminated, truncated, info = env.step({a: actions[i].numpy() for i, a in enumerate(env.agents)})
                discounted_return += discount * np.array([reward[a] for a in env.agents])
                discount *= config['gamma']
                if all(terminated.values()) or truncated:
                    break
            approaches = []
            for agent in info['collision_agents']:
                index = env.agents.index(agent)
                pos, vel = before_pos[index], before_vel[index]
                boundary = np.array([pos[0] - env.uav_radius, env.map_size - env.uav_radius - pos[0],
                                     pos[1] - env.uav_radius, env.map_size - env.uav_radius - pos[1],
                                     pos[2] - env.z_min, env.z_max - pos[2]])
                inward_speed = np.array([-vel[0], vel[0], -vel[1], vel[1], -vel[2], vel[2]])
                side = int(boundary.argmin())
                speed = float(max(0.0, inward_speed[side]))
                building_approaches = []
                nearest = np.argsort(env._building_distances_3d(pos))[:env.nearest_building_k].tolist()
                for source in info['collision_sources'][agent]:
                    if not source.startswith('building:'):
                        continue
                    building_index = int(source.split(':', 1)[1])
                    building = env.buildings[building_index]
                    closest = np.array([np.clip(pos[0], building[0], building[1]),
                                        np.clip(pos[1], building[2], building[3]),
                                        np.clip(pos[2], env.z_min, building[4])])
                    offset = closest - pos
                    distance = float(np.linalg.norm(offset))
                    direction = offset / max(distance, 1e-6)
                    closing = float(max(0.0, direction @ vel))
                    braking = closing * getattr(env, 'baseline_braking_reaction_time', 1.5) + closing ** 2 / (2 * env.max_accel)
                    building_approaches.append(dict(building_index=building_index, bounds=building.tolist(),
                                                   surface_distance_before=distance, clearance_before=distance-env.uav_radius,
                                                   closing_speed_before=closing, braking_distance=braking,
                                                   normalized_command_toward_building=float(direction @ actions[index].numpy()),
                                                   actual_acceleration_toward_building=float(direction @ before_acc[index]),
                                                   visible_in_nearest_building_observation=building_index in nearest,
                                                   nearest_observation_rank=nearest.index(building_index) + 1 if building_index in nearest else None))
                approaches.append(dict(agent=agent, sources=info['collision_sources'][agent],
                                       position_before=pos.tolist(), velocity_before=vel.tolist(),
                                       speed_before=float(np.linalg.norm(vel)), boundary_side=side,
                                       boundary_clearance_before=float(boundary[side]),
                                       boundary_approach_speed=speed,
                                       ideal_braking_distance=speed * env.acceleration_lag + speed ** 2 / (2 * env.max_accel),
                                       target_distance_before=float(np.linalg.norm(pos-before_target)),
                                       normalized_action=actions[index].numpy().tolist(),
                                       acceleration_before=before_acc[index].tolist(), buildings=building_approaches))
            rows.append(dict(episode=episode, steps=step + 1, success=info['is_success'],
                             collision=info['is_collision'], truncated=bool(truncated),
                             predicted_q=float(predicted_q.mean()), discounted_return=float(discounted_return.mean()),
                             bias=None if truncated else float((predicted_q - discounted_return).mean()),
                             approaches=approaches))
    finally:
        env.close()
    terminated_rows = [row for row in rows if row['bias'] is not None]
    actions = np.concatenate(actions_all)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  env_seed=args.env_seed, level=args.level, episodes=args.episodes,
                  collision_termination=all(config[name] for name in ['terminate_on_collision', 'terminate_on_boundary_collision', 'terminate_on_building_collision']),
                  safe_captures=sum(row['success'] and not row['collision'] for row in rows),
                  collisions=sum(row['collision'] for row in rows), timeouts=sum(row['truncated'] for row in rows),
                  mean_bias_on_terminated_episodes=float(np.mean([row['bias'] for row in terminated_rows])) if terminated_rows else None,
                  action_saturation_fraction=float(np.mean(np.abs(actions) > .95)), data=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'data'}), flush=True)


if __name__ == '__main__':
    main()
