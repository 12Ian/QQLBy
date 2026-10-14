"""Compare native RL with explicit controllers without changing the simulator."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/pursuit_feasibility_v1'


def bounded_norm(value,maximum):
    return value*min(1.,maximum/max(float(np.linalg.norm(value)),1e-9))


def controller(env,clearance):
    commands=[]
    for index,pos in enumerate(env.uav_positions):
        desired=bounded_norm(env.target_velocity+.35*(env.target_position-pos),env.uav_max_speed)
        if clearance:
            # A velocity limit from d=v*T+v^2/(2*a), T=2 s, margin=20 m.
            def limit(distance):
                return env.max_accel*(np.sqrt(4.+2.*max(distance-20.,0.)/env.max_accel)-2.)
            lower=pos-np.array([env.uav_radius,env.uav_radius,env.z_min])
            upper=np.array([env.map_size-env.uav_radius,env.map_size-env.uav_radius,env.z_max])-pos
            desired=np.clip(desired,-np.array([limit(d) for d in lower]),np.array([limit(d) for d in upper]))
            nearest=np.argsort(env._building_distances_3d(pos))[:env.nearest_building_k]
            for building in env.buildings[nearest]:
                closest=np.array([np.clip(pos[0],building[0],building[1]),np.clip(pos[1],building[2],building[3]),
                                  np.clip(pos[2],env.z_min,building[4])])
                offset=closest-pos
                distance=float(np.linalg.norm(offset))
                if distance<1e-6:
                    continue
                toward=offset/distance
                component=float(toward@desired)
                maximum=limit(distance-env.uav_radius)
                if component>maximum:
                    desired=desired-toward*(component-maximum)
        command=bounded_norm((desired-env.uav_velocities[index])/1.,env.max_accel)/env.max_accel
        commands.append(command)
    return np.asarray(commands,dtype=np.float32)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cfg=json.loads((ROOT/'outputs/results/matd3/baseline_exp009/config.json').read_text())
    cfg.update(curriculum_start_level=0,render=False,terminate_on_building_collision=True,terminate_on_boundary_collision=True)
    model=ROOT/'outputs/reviews/exp009_final_v1/final009.pth'
    state=torch.load(model,map_location='cpu',weights_only=False)['policy']
    env=UAVPursuitCoverage3DEnv(Namespace(**cfg))
    key=env.agents[0]
    ids=torch.eye(env.num_agents)
    actor=load_network(state,f'actor.{key}.model.',[env.observation_space[key].shape[0]+env.num_agents]+cfg['actor_hidden_size']+[3],True)
    result=dict(controllers_are_not_RL_results=True,level=0,episodes_per_mode=32,seeds=list(range(68101,68133)),
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),modes={})
    try:
        for mode in ('rl','pd','pd_clearance'):
            records=[]
            for episode,seed in enumerate(result['seeds']):
                env.seed(seed)
                obs,_=env.reset()
                initial_target=env.target_position.copy()
                initial_positions=env.uav_positions.copy()
                trajectory=[]
                actions_all=[]
                discounted=np.zeros(env.num_agents)
                discount=1.
                for step in range(env.max_episode_steps):
                    if mode=='rl':
                        with torch.no_grad():
                            x=torch.tensor(np.stack([obs[a] for a in env.agents]))
                            actions=actor(torch.cat((x,ids),dim=1)).numpy()
                    else:
                        actions=controller(env,mode=='pd_clearance')
                    actions_all.append(actions)
                    assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1.000001
                    obs,reward,terminated,truncated,info=env.step({a:actions[i] for i,a in enumerate(env.agents)})
                    discounted+=discount*np.array([reward[a] for a in env.agents])
                    discount*=cfg['gamma']
                    if episode==0:
                        trajectory.append(dict(step=step+1,positions=env.uav_positions.tolist(),evader=env.target_position.tolist(),actions=actions.tolist()))
                    if all(terminated.values()) or truncated:
                        break
                item=dict(seed=seed,steps=step+1,success=bool(info['is_success']),collision=bool(info['is_collision']),
                          initial_target=initial_target.tolist(),initial_positions=initial_positions.tolist(),
                          capture_type=info['capture_type'],collision_sources=info['collision_sources'],
                          discounted_return=float(discounted.mean()),sub_rewards=env.episode_sub_rewards,
                          action_saturation=float(np.mean(np.abs(np.concatenate(actions_all))>.95)),trajectory=trajectory)
                records.append(item)
            mode_result=dict(captures=sum(row['success'] for row in records),
                             safe_captures=sum(row['success'] and not row['collision'] for row in records),
                             collisions=sum(row['collision'] for row in records),
                             mean_steps=float(np.mean([row['steps'] for row in records])),
                             mean_discounted_return=float(np.mean([row['discounted_return'] for row in records])),records=records)
            result['modes'][mode]=mode_result
            (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(mode,json.dumps({k:v for k,v in mode_result.items() if k!='records'}),flush=True)
        for episode in range(32):
            reference=result['modes']['rl']['records'][episode]
            assert all(result['modes'][mode]['records'][episode]['initial_target']==reference['initial_target'] and
                       result['modes'][mode]['records'][episode]['initial_positions']==reference['initial_positions']
                       for mode in ('pd','pd_clearance'))
        result['paired_initial_states_verified']=True
        result['simulator_modified']=False
        result['controller_uses_nearest_buildings_only']=True
        (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('FEASIBILITY_COMPLETE',flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
