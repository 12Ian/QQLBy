"""Check whether target Actor lag/smoothing explains the positive predicted Q."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/exp009_target_actor_diagnostic_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cfg=json.loads((ROOT/'outputs/results/matd3/baseline_exp009/config.json').read_text())
    cfg.update(curriculum_start_level=0,render=False,terminate_on_boundary_collision=True,terminate_on_building_collision=True)
    env=UAVPursuitCoverage3DEnv(Namespace(**cfg))
    path=ROOT/'outputs/reviews/exp009_final_v1/final009.pth'
    state=torch.load(path,map_location='cpu',weights_only=False)['policy']
    key=env.agents[0]
    n=env.num_agents
    d=env.observation_space[key].shape[0]
    actors={name:load_network(state,f'{name}.{key}.model.',[d+n]+cfg['actor_hidden_size']+[3],True)
            for name in ('actor','target_actor')}
    critics=[load_network(state,f'critic_{letter}.{key}.model.',[(d+3)*n+n]+cfg['critic_hidden_size']+[1]) for letter in ('A','B')]
    ids=torch.eye(n)
    result=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),level=0,seeds=list(range(68101,68133)),modes={})
    try:
        for mode in ('current','target','target_smoothed'):
            rows=[]
            for seed in result['seeds']:
                env.seed(seed)
                obs,_=env.reset()
                initial=env.target_position.tolist()
                rng=np.random.RandomState(seed+100000)
                discount=1.
                total=np.zeros(n)
                model=actors['actor' if mode=='current' else 'target_actor']
                for step in range(env.max_episode_steps):
                    x=torch.tensor(np.stack([obs[a] for a in env.agents]))
                    with torch.no_grad():
                        actions=model(torch.cat((x,ids),dim=1)).numpy()
                        if mode=='target_smoothed':
                            noise=np.clip(rng.normal(0,cfg['target_policy_noise'],actions.shape),-cfg['target_noise_clip'],cfg['target_noise_clip'])
                            actions=np.clip(actions+noise,-1,1).astype(np.float32)
                        if step==0:
                            joint=torch.cat((x.flatten(),torch.tensor(actions).flatten())).repeat(n,1)
                            q=torch.stack([critic(torch.cat((joint,ids),dim=1)).flatten() for critic in critics]).min(dim=0).values.numpy()
                    obs,rewards,term,trunc,info=env.step({a:actions[i] for i,a in enumerate(env.agents)})
                    total+=discount*np.array([rewards[a] for a in env.agents])
                    discount*=cfg['gamma']
                    if all(term.values()) or trunc:
                        break
                rows.append(dict(seed=seed,initial_target=initial,steps=step+1,success=bool(info['is_success']),
                                 collision=bool(info['is_collision']),truncated=bool(trunc),initial_q_min=float(q.mean()),
                                 discounted_return=float(total.mean()),bias=float((q-total).mean())))
            result['modes'][mode]=dict(rows=rows,captures=sum(x['success'] for x in rows),
                collisions=sum(x['collision'] for x in rows),mean_steps=float(np.mean([x['steps'] for x in rows])),
                mean_q=float(np.mean([x['initial_q_min'] for x in rows])),mean_return=float(np.mean([x['discounted_return'] for x in rows])))
            (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(mode,json.dumps({k:v for k,v in result['modes'][mode].items() if k!='rows'}),flush=True)
        result['paired_initial_states_verified']=all(all(result['modes'][mode]['rows'][i]['initial_target']==result['modes']['current']['rows'][i]['initial_target']
                                                        for mode in ('target','target_smoothed')) for i in range(32))
        assert result['paired_initial_states_verified']
        result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('TARGET_ACTOR_DIAG_COMPLETE',flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
