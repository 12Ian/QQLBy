"""Check critic invariance for action encodings with equal physical commands."""
from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from xuance.environment.multi_agent_env.uav_kinematics_3d import limit_norm

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/reviews/exp013_policy_drift_v1'
OUT=ROOT/'outputs/reviews/exp013_action_equivalence_v1'


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True,exist_ok=False)
    meta=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    cfg=meta['config']
    assert cfg['activation_action']=='tanh' and cfg['activation']=='leaky_relu'
    torch.set_num_threads(1)
    summary={}
    for label in ('best','recent'):
        path=SOURCE/f'{label}.pth'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['frozen_models'][label]['sha256']
        original=json.loads((SOURCE/f'{label}.json').read_text(encoding='utf-8'))['records'][0]
        config=dict(cfg,curriculum_start_level=0,render=False,
                    terminate_on_building_collision=True,terminate_on_boundary_collision=True)
        env=UAVPursuitCoverage3DEnv(Namespace(**config))
        records=[]
        try:
            env.seed(original['env_seed'])
            obs,_=env.reset()
            n=env.num_agents
            d=env.observation_space[env.agents[0]].shape[0]
            ids=torch.eye(n)
            key=env.agents[0]
            state=torch.load(path,map_location='cpu',weights_only=False)['policy']
            actor=load_network(state,f'actor.{key}.model.',[d+n]+cfg['actor_hidden_size']+[3],True)
            critics=[load_network(state,f'critic_{letter}.{key}.model.',[(d+3)*n+n]+cfg['critic_hidden_size']+[1]) for letter in ('A','B')]
            for row in original['trace']:
                assert np.array_equal(env.uav_positions,np.asarray(row['positions'],dtype=np.float32))
                x=torch.tensor(np.stack([obs[a] for a in env.agents]))
                with torch.no_grad():raw=actor(torch.cat((x,ids),dim=1))
                assert np.allclose(raw.numpy(),row['actions'],rtol=0,atol=1e-7)
                norms=raw.norm(dim=-1,keepdim=True)
                projected=raw/norms.clamp(min=1.)
                raw_joint=raw.flatten().clone().requires_grad_(True)
                joint=torch.cat((x.flatten(),raw_joint)).repeat(n,1)
                qa=critics[0](torch.cat((joint,ids),dim=1)).flatten()
                gradient=torch.autograd.grad(qa.mean(),raw_joint)[0].reshape(n,3)
                with torch.no_grad():
                    qp_input=torch.cat((x.flatten(),projected.flatten())).repeat(n,1)
                    projected_q=[net(torch.cat((qp_input,ids),dim=1)).flatten() for net in critics]
                    raw_q=[qa.detach(),critics[1](torch.cat((joint.detach(),ids),dim=1)).flatten()]
                commands_raw=np.asarray([limit_norm(a.numpy()*env.max_accel,env.max_accel) for a in raw])
                commands_projected=np.asarray([limit_norm(a.numpy()*env.max_accel,env.max_accel) for a in projected])
                command_error=float(np.max(np.abs(commands_raw-commands_projected)))
                assert command_error<1e-6
                cloned=copy.deepcopy(env)
                try:
                    canonical_obs,canonical_rewards,canonical_term,canonical_trunc,canonical_info=cloned.step(
                        {a:projected[i].numpy() for i,a in enumerate(env.agents)})
                    obs,rewards,term,trunc,info=env.step({a:raw[i].numpy() for i,a in enumerate(env.agents)})
                    state_error=max(float(np.max(np.abs(getattr(env,field)-getattr(cloned,field))))
                        for field in ('uav_positions','uav_velocities','uav_accelerations',
                                      'target_position','target_velocity','target_acceleration'))
                    reward_error=max(abs(rewards[a]-canonical_rewards[a]) for a in env.agents)
                    assert state_error<2e-4 and reward_error<1e-4
                    assert term==canonical_term and trunc==canonical_trunc
                    assert info['capture_type']==canonical_info['capture_type'] and info['collision_sources']==canonical_info['collision_sources']
                finally:cloned.close()
                directions=raw/norms.clamp(min=1e-9)
                radial=(gradient*directions).sum(-1)
                records.append(dict(step=row['step'],norms=norms.flatten().tolist(),
                    raw_actions=raw.tolist(),projected_actions=projected.tolist(),
                    q_A_raw=raw_q[0].tolist(),q_A_projected=projected_q[0].tolist(),
                    q_min_raw=torch.minimum(*raw_q).tolist(),q_min_projected=torch.minimum(*projected_q).tolist(),
                    actor_objective_radial_derivative=radial.tolist(),command_error=command_error,
                    native_next_state_error=state_error,native_reward_error=reward_error))
            norms=np.asarray([r['norms'] for r in records])
            radial=np.asarray([r['actor_objective_radial_derivative'] for r in records])
            outside=norms>1.0001
            delta_a=np.asarray([r['q_A_raw'] for r in records])-np.asarray([r['q_A_projected'] for r in records])
            delta_min=np.asarray([r['q_min_raw'] for r in records])-np.asarray([r['q_min_projected'] for r in records])
            result=dict(label=label,seed=original['env_seed'],episodes=1,states=len(records),
                model_sha256=meta['frozen_models'][label]['sha256'],native_trace_replayed=True,
                physical_equivalence_native_steps_verified=True,unit_ball_exceed_fraction=float(np.mean(outside)),
                mean_abs_Q_A_encoding_difference=float(np.mean(np.abs(delta_a))),
                mean_abs_Q_min_encoding_difference=float(np.mean(np.abs(delta_min))),
                max_abs_Q_A_encoding_difference=float(np.max(np.abs(delta_a))),
                mean_Q_A_raw_minus_projected=float(delta_a.mean()),
                mean_absolute_radial_derivative_outside=float(np.mean(np.abs(radial[outside]))),
                radial_derivative_above_1e3_fraction=float(np.mean(np.abs(radial[outside])>1e-3)),
                max_native_state_error=max(r['native_next_state_error'] for r in records),
                max_native_reward_error=max(r['native_reward_error'] for r in records),records=records)
            (OUT/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            summary[label]={k:v for k,v in result.items() if k!='records'}
            print('ACTION_EQUIVALENCE',label,json.dumps(summary[label]),flush=True)
        finally:env.close()
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/'xuance/environment/multi_agent_env/uav_kinematics_3d.py',ROOT/'xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py']}
    (OUT/'summary.json').write_text(json.dumps(dict(models=summary,source_hashes=source_hashes,
        training_unchanged=True,diagnostic_not_a_performance_evaluation=True),indent=2),encoding='utf-8')
    print('ACTION_EQUIVALENCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
