"""Measure actual logits and candidate bounded-output gradients, without training."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_baseline_collisions import load_network
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/actor_output_saturation_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    checkpoints={'009_final':ROOT/'outputs/reviews/exp009_final_v1/final009.pth',
                 '010_final':ROOT/'outputs/experiments/exp010_012_pilots_v1/010/final.pth'}
    result=dict(diagnostic_only=True,candidate_not_trained=True,seed=66001,level=0,models={})
    for label,path in checkpoints.items():
        number=label[:3]
        name=f'baseline_exp{number}' if number=='009' else f'baseline_exp{number}_pilot'
        cfg=json.loads((ROOT/f'outputs/results/matd3/{name}/config.json').read_text())
        cfg.update(env_seed=66001,curriculum_start_level=0,render=False)
        env=UAVPursuitCoverage3DEnv(Namespace(**cfg))
        try:
            obs,_=env.reset()
            key=env.agents[0]
            count=env.num_agents
            ids=torch.eye(count)
            size=env.observation_space[key].shape[0]
            state=torch.load(path,map_location='cpu',weights_only=False)['policy']
            actor=load_network(state,f'actor.{key}.model.',[size+count]+cfg['actor_hidden_size']+[3],True)
            x=torch.cat((torch.tensor(np.stack([obs[a] for a in env.agents])),ids),dim=1)
            with torch.no_grad():
                logits=actor[:-1](x)
                original=actor(x)
                torch.testing.assert_close(original,torch.tanh(logits),rtol=0,atol=0)
                alternative=torch.nn.functional.softsign(logits)
                derivative_tanh=1-original.square()
                derivative_softsign=1/(1+logits.abs()).square()
            result['models'][label]=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                logits=logits.tolist(),tanh_actions=original.tolist(),softsign_actions=alternative.tolist(),
                tanh_derivative=derivative_tanh.tolist(),softsign_derivative=derivative_softsign.tolist(),
                zero_tanh_derivatives=int((derivative_tanh==0).sum()),
                all_softsign_derivatives_positive=bool((derivative_softsign>0).all()),
                mean_tanh_derivative=float(derivative_tanh.mean()),mean_softsign_derivative=float(derivative_softsign.mean()))
        finally:
            env.close()
    logits=torch.tensor([6.,10.,15.,20.],requires_grad=True)
    tanh=torch.tanh(logits)
    dt=torch.autograd.grad(tanh.sum(),logits,retain_graph=True)[0]
    soft=torch.nn.functional.softsign(logits)
    ds=torch.autograd.grad(soft.sum(),logits)[0]
    assert bool((soft.abs()<1).all()) and bool((ds>0).all()) and float(dt[-1])==0.
    result['float32_probe']=dict(logits=logits.detach().tolist(),tanh_derivative=dt.tolist(),softsign_derivative=ds.tolist())
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
