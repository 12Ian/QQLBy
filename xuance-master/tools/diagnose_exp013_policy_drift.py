"""Compare frozen best/recent native policies on identical diagnostic cases."""
from argparse import Namespace
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/exp013_policy_drift_v1'


def save(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not OUT.exists()
    cfg=json.loads((ROOT/'outputs/results/matd3/baseline_exp013/config.json').read_text())
    assert cfg['activation_action']=='tanh'
    folders=list((ROOT/'outputs/models/matd3/baseline_exp013').glob('seed_*'))
    assert len(folders)==1
    sources={'best':ROOT/'outputs/results/matd3/baseline_exp013/best_model/best_stage_0.pth',
             'recent':folders[0]/'latest_train_model.pth'}
    snapshots={}
    for label,path in sources.items():
        before=path.stat()
        data=path.read_bytes()
        after=path.stat()
        assert before.st_mtime_ns==after.st_mtime_ns and len(data)==before.st_size
        snapshots[label]=(data,dict(source=str(path),sha256=digest(data),mtime_ns=after.st_mtime_ns))
    time.sleep(.1)
    assert all(digest(path.read_bytes())==snapshots[label][1]['sha256'] for label,path in sources.items()),'Checkpoint changed during freeze'
    OUT.mkdir(parents=True,exist_ok=False)
    process=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    save(OUT/'process.json',process)
    try:
        for label,(data,_) in snapshots.items():(OUT/f'{label}.pth').write_bytes(data)
        status=json.loads((ROOT/'outputs/results/matd3/baseline_exp013/training_status.json').read_text())
        evals=list(csv.DictReader((ROOT/'outputs/results/matd3/baseline_exp013/curriculum_eval.csv').open()))
        save(OUT/'manifest.json',dict(frozen_models={k:v[1] for k,v in snapshots.items()},
             diagnostic_seed_protocol='69701 + case_index',episodes_per_model=16,training_status_at_freeze=status,
             curriculum_evaluations_at_freeze=evals,config=cfg,training_unchanged=True))
        torch.set_num_threads(1)
        summary={}
        for label in sources:
            configuration=dict(cfg,curriculum_start_level=0,render=False,
                               terminate_on_building_collision=True,terminate_on_boundary_collision=True)
            env=UAVPursuitCoverage3DEnv(Namespace(**configuration))
            try:
                n=env.num_agents
                d=env.observation_space[env.agents[0]].shape[0]
                ids=torch.eye(n)
                key=env.agents[0]
                state=torch.load(OUT/f'{label}.pth',map_location='cpu',weights_only=False)['policy']
                actor=load_network(state,f'actor.{key}.model.',[d+n]+cfg['actor_hidden_size']+[3],True)
                critics=[load_network(state,f'critic_{letter}.{key}.model.',[(d+3)*n+n]+cfg['critic_hidden_size']+[1]) for letter in ('A','B')]
                records=[]
                saturation=0
                components=0
                for case in range(16):
                    env.seed(69701+case)
                    obs,_=env.reset()
                    initial=dict(pursuers=env.uav_positions.tolist(),evader=env.target_position.tolist(),
                                 pursuer_velocities=env.uav_velocities.tolist(),evader_velocity=env.target_velocity.tolist(),
                                 observations={a:obs[a].tolist() for a in env.agents},
                                 buildings_sha256=digest(env.buildings.tobytes()))
                    returns=np.zeros(n)
                    discount=1.
                    trace=[]
                    for step in range(env.max_episode_steps):
                        x=torch.tensor(np.stack([obs[a] for a in env.agents]))
                        with torch.inference_mode():
                            actions=actor(torch.cat((x,ids),dim=1))
                            if step==0:
                                joint=torch.cat((x.flatten(),actions.flatten())).repeat(n,1)
                                qs=[net(torch.cat((joint,ids),dim=1)).flatten().numpy().copy() for net in critics]
                        action=actions.numpy()
                        assert np.isfinite(action).all() and np.max(np.abs(action))<=1.000001
                        saturation+=int(np.sum(np.abs(action)>.95))
                        components+=action.size
                        before=dict(step=step,positions=env.uav_positions.tolist(),velocities=env.uav_velocities.tolist(),
                                    evader=env.target_position.tolist(),actions=action.tolist())
                        obs,reward,term,trunc,info=env.step({a:action[i] for i,a in enumerate(env.agents)})
                        returns+=discount*np.array([reward[a] for a in env.agents])
                        discount*=cfg['gamma']
                        if case==0:trace.append(dict(**before,collision=bool(info['is_collision'])))
                        if all(term.values()) or trunc:break
                    records.append(dict(case=case,env_seed=69701+case,initial=initial,initial_sha256=digest(json.dumps(initial,sort_keys=True).encode()),
                        steps=step+1,success=bool(info['is_success']),safe_success=bool(info['is_success'] and not info['is_collision']),
                        collision=bool(info['is_collision']),collision_sources=info['collision_sources'],
                        q_min=np.minimum(*qs).tolist(),discounted_return=returns.tolist(),
                        q_bias=(np.minimum(*qs)-returns).tolist(),terminal_not_timeout=not bool(trunc),trace=trace))
                terminal=[r for r in records if r['terminal_not_timeout']]
                result=dict(label=label,model_sha256=snapshots[label][1]['sha256'],episodes=16,
                    collision_termination=True,seed_protocol='69701 + case_index',records=records,
                    captures=sum(r['success'] for r in records),safe_captures=sum(r['safe_success'] for r in records),
                    collisions=sum(r['collision'] for r in records),mean_steps=float(np.mean([r['steps'] for r in records])),
                    saturation_fraction=saturation/components,mean_initial_Q_min=float(np.mean([r['q_min'] for r in records])),
                    mean_discounted_return=float(np.mean([r['discounted_return'] for r in records])),
                    terminal_episodes=len(terminal),mean_terminal_Q_bias=float(np.mean([r['q_bias'] for r in terminal])) if terminal else None)
                save(OUT/f'{label}.json',result)
                summary[label]={k:v for k,v in result.items() if k!='records'}
                print('DRIFT',label,json.dumps(summary[label]),flush=True)
            finally:env.close()
        a=json.loads((OUT/'best.json').read_text())['records']
        b=json.loads((OUT/'recent.json').read_text())['records']
        assert all(x['initial']==y['initial'] and x['initial_sha256']==y['initial_sha256'] for x,y in zip(a,b))
        summary['paired_initial_states_exact']=True
        save(OUT/'summary.json',summary)
        process.update(status='completed',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
        save(OUT/'process.json',process)
    except BaseException as error:
        process.update(status='failed',error=repr(error),exit_code=1)
        save(OUT/'process.json',process)
        raise


if __name__=='__main__':
    main()
