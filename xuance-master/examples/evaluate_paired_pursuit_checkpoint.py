"""Evaluate a frozen native policy on explicit episode seeds and record each case."""
import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from tools.render_optimized_success_videos import native_policy
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--env-seed',type=int,required=True)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--parallels',type=int,default=8)
    parser.add_argument('--level',type=int,choices=range(5),required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and min(args.episodes,args.parallels)>0
    cfg=yaml.safe_load(args.config.read_text())
    assert cfg['reward_mode']=='baseline'
    cfg.update(env_seed=args.env_seed,curriculum_start_level=args.level,render=False,
               terminate_on_collision=False,terminate_on_building_collision=False,terminate_on_boundary_collision=False)
    torch.set_num_threads(1)
    pool=[UAVPursuitCoverage3DEnv(Namespace(**cfg)) for _ in range(min(args.episodes,args.parallels))]
    assert all(env.max_episode_steps==cfg['episode_length'] for env in pool)
    policy,key,_=native_policy(cfg,pool[0],args.checkpoint,args.device)
    agent_keys=pool[0].agents
    count=pool[0].num_agents
    slots={}
    cases={}
    next_case=0
    started=time.time()
    def assign(slot,case):
        env=pool[slot]
        env.seed(args.env_seed+case)
        observation,_=env.reset()
        initial=dict(building_geometry_sha256=hashlib.sha256(env.buildings.tobytes()).hexdigest(),
                     positions=env.uav_positions.tolist(),velocities=env.uav_velocities.tolist(),
                     accelerations=env.uav_accelerations.tolist(),target_position=env.target_position.tolist(),
                     target_velocity=env.target_velocity.tolist(),target_acceleration=env.target_acceleration.tolist(),
                     observations={a:observation[a].tolist() for a in agent_keys})
        slots[slot]=dict(case=case,obs=observation,initial=initial,
                         initial_sha256=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),
                         steps=0,returns=np.zeros(count),collision=False,building=False,boundary=False,uav=False)
    try:
        for slot in range(len(pool)):
            assign(slot,next_case)
            next_case+=1
        while slots:
            active=sorted(slots)
            observations=torch.tensor(np.stack([slots[slot]['obs'][a] for slot in active for a in agent_keys]),
                                      dtype=torch.float32,device=args.device)
            ids=torch.eye(count,device=args.device).repeat(len(active),1)
            with torch.inference_mode():
                _,values=policy({key:observations},agent_ids=ids)
                actions=values[key].reshape(len(active),count,3).cpu().numpy()
            assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1.000001
            for row,slot in enumerate(active):
                state=slots[slot]
                env=pool[slot]
                obs,rewards,term,trunc,info=env.step({a:actions[row,i] for i,a in enumerate(agent_keys)})
                state['obs']=obs
                state['steps']+=1
                state['returns']+=np.array([rewards[a] for a in agent_keys])
                for field,flag in [('collision','is_collision'),('building','is_building_collision'),
                                   ('boundary','is_boundary_collision'),('uav','is_uav_collision')]:
                    state[field]|=bool(info[flag])
                if not (all(term.values()) or trunc):
                    continue
                case=state['case']
                cases[case]=dict(case=case,env_seed=args.env_seed+case,initial=state['initial'],
                    initial_sha256=state['initial_sha256'],steps=state['steps'],success=bool(info['is_success']),
                    safe_success=bool(info['is_success'] and not state['collision']),collision=state['collision'],
                    building_collision=state['building'],boundary_collision=state['boundary'],uav_collision=state['uav'],
                    capture_type=info['capture_type'],
                    capture_time=float(info['capture_time']) if info['capture_time'] is not None else None,
                    mean_return=float(state['returns'].mean()),elapsed_seconds=float(env.episode_elapsed_seconds))
                if next_case<args.episodes:
                    assign(slot,next_case)
                    next_case+=1
                else:
                    del slots[slot]
        records=[cases[i] for i in range(args.episodes)]
        metrics=dict(episodes=len(records),success_rate=float(np.mean([r['success'] for r in records])),
            safe_success_rate=float(np.mean([r['safe_success'] for r in records])),
            collision_rate=float(np.mean([r['collision'] for r in records])),
            building_collision_rate=float(np.mean([r['building_collision'] for r in records])),
            boundary_collision_rate=float(np.mean([r['boundary_collision'] for r in records])),
            uav_collision_rate=float(np.mean([r['uav_collision'] for r in records])),
            mean_episode_steps=float(np.mean([r['steps'] for r in records])))
        result=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            config=str(args.config.resolve()),level=args.level,env_seed=args.env_seed,seed_protocol='base_seed + episode_index',
            exploration_noise=False,collision_termination=False,max_episode_steps=cfg['episode_length'],
            active_environments=len(pool),batched_native_policy=True,metrics=metrics,episodes=records,
            mean_return=float(np.mean([r['mean_return'] for r in records])),elapsed_seconds=time.time()-started,
            actor_activation=cfg['activation_action'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as file:
            json.dump(result,file,indent=2,ensure_ascii=False)
        print(json.dumps(dict(output=str(args.output),metrics=metrics)),flush=True)
    finally:
        for env in pool:
            env.close()


if __name__=='__main__':
    main()
