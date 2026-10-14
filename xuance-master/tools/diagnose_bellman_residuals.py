"""Locate actual-policy value errors; do not train or mutate any checkpoint."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/baseline_bellman_profile_v1'


def measure(number,path):
    name=f'baseline_exp{number}' if number=='009' else f'baseline_exp{number}_pilot'
    cfg=json.loads((ROOT/f'outputs/results/matd3/{name}/config.json').read_text())
    assert cfg['representation']=='Basic_Identical' and cfg['activation_action']=='tanh'
    assert cfg['use_parameter_sharing'] and not cfg.get('use_graph_module') and not cfg.get('use_obstacle_gat')
    cfg.update(curriculum_start_level=0,render=False,terminate_on_boundary_collision=True,terminate_on_building_collision=True)
    env=UAVPursuitCoverage3DEnv(Namespace(**cfg))
    count=env.num_agents
    key=env.agents[0]
    size=env.observation_space[key].shape[0]
    ids=torch.eye(count)
    saved=torch.load(path,map_location='cpu',weights_only=False)['policy']
    actors={prefix:load_network(saved,f'{prefix}.{key}.model.',[size+count]+cfg['actor_hidden_size']+[3],True)
            for prefix in ('actor','target_actor')}
    networks={prefix:load_network(saved,f'{prefix}.{key}.model.',[(size+3)*count+count]+cfg['critic_hidden_size']+[1])
              for prefix in ('critic_A','critic_B','target_critic_A','target_critic_B')}
    def q_values(x,actions,prefix):
        joint=torch.cat((x.flatten(),actions.flatten())).repeat(count,1)
        return networks[prefix](torch.cat((joint,ids),dim=1)).flatten().numpy().astype(np.float64)
    records=[]
    try:
        for case in range(4):
            seed=68101+case
            env.seed(seed)
            obs,_=env.reset()
            q_current=[]
            rewards=[]
            training_residuals=[]
            actions_all=[]
            smoothing=np.random.RandomState(seed+200000)
            for step in range(env.max_episode_steps):
                x=torch.tensor(np.stack([obs[a] for a in env.agents]))
                with torch.inference_mode():
                    act=actors['actor'](torch.cat((x,ids),dim=1))
                    qa=q_values(x,act,'critic_A')
                    qb=q_values(x,act,'critic_B')
                actions_all.append(act.numpy().tolist())
                q_current.append(np.minimum(qa,qb))
                obs,rew,term,trunc,info=env.step({a:act[i].numpy() for i,a in enumerate(env.agents)})
                reward=np.array([rew[a] for a in env.agents],dtype=np.float64)
                rewards.append(reward)
                next_target=np.zeros(count)
                if not all(term.values()):
                    nxt=torch.tensor(np.stack([obs[a] for a in env.agents]))
                    estimates=[]
                    with torch.inference_mode():
                        base=actors['target_actor'](torch.cat((nxt,ids),dim=1)).numpy()
                        for _ in range(8):
                            noise=np.clip(smoothing.normal(0,cfg['target_policy_noise'],base.shape),
                                          -cfg['target_noise_clip'],cfg['target_noise_clip'])
                            action=torch.tensor(np.clip(base+noise,-1,1),dtype=torch.float32)
                            estimates.append(np.minimum(q_values(nxt,action,'target_critic_A'),q_values(nxt,action,'target_critic_B')))
                    next_target=np.mean(estimates,axis=0)
                target=reward+cfg['gamma']*next_target
                training_residuals.append(np.stack((qa-target,qb-target)))
                if all(term.values()) or trunc:
                    break
            assert all(term.values()) and not trunc,'Profile requires a true terminal return'
            q=np.array(q_current)
            r=np.array(rewards)
            weights=cfg['gamma']**np.arange(len(r))
            following=np.concatenate((q[1:],np.zeros((1,count))),axis=0)
            residual=q-r-cfg['gamma']*following
            weighted=residual*weights[:,None]
            actual=np.sum(weights[:,None]*r,axis=0)
            bias=q[0]-actual
            np.testing.assert_allclose(weighted.sum(axis=0),bias,atol=1e-6,rtol=1e-9)
            learner=np.array(training_residuals)
            row=dict(seed=seed,steps=len(r),q_initial=float(q[0].mean()),discounted_return=float(actual.mean()),
                q_bias=float(bias.mean()),weighted_residual_sum=float(weighted.sum(axis=0).mean()),
                positive_current_residual_fraction=float(np.mean(residual>0)),
                current_residual_rms=float(np.sqrt(np.mean(residual**2))),
                learner_target_residual_rms=float(np.sqrt(np.mean(learner**2))),
                terminal_current_residual=float(residual[-1].mean()),
                terminal_weighted_bias=float(weighted[-1].mean()),
                terminal_bias_share=float(weighted[-1].mean()/bias.mean()),
                terminal_q=float(q[-1].mean()),terminal_reward=float(r[-1].mean()),
                telescope_identity_verified=True,success=bool(info['is_success']),collision=bool(info['is_collision']),
                mean_q_by_step=q.mean(axis=1).tolist(),mean_reward_by_step=r.mean(axis=1).tolist(),
                mean_current_residual_by_step=residual.mean(axis=1).tolist(),
                mean_weighted_residual_by_step=weighted.mean(axis=1).tolist(),
                learner_residual_rms_by_step=np.sqrt(np.mean(learner**2,axis=(1,2))).tolist())
            records.append(row)
    finally:
        env.close()
    report=dict(model=number,checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        level=0,episodes=4,diagnostic_seeds=list(range(68101,68105)),
        learner_target_noise_samples_per_state=8,not_training_replay_loss=True,records=records,
        mean_bias=float(np.mean([x['q_bias'] for x in records])),
        mean_terminal_bias_share=float(np.mean([x['terminal_bias_share'] for x in records])),
        mean_learner_target_residual_rms=float(np.mean([x['learner_target_residual_rms'] for x in records])))
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    paths={'009':ROOT/'outputs/reviews/exp009_final_v1/final009.pth',
           '010':ROOT/'outputs/experiments/exp010_012_pilots_v1/010/final.pth',
           '011':ROOT/'outputs/experiments/exp010_012_pilots_v1/011/final.pth'}
    summary={}
    for number,path in paths.items():
        result=measure(number,path)
        (OUT/f'{number}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        summary[number]={k:v for k,v in result.items() if k!='records'}
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(number,json.dumps(summary[number]),flush=True)
    print('BELLMAN_PROFILE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
