"""CUDA integration audit with deliberately oversized exploration noise."""
from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import torch
from xuance import get_runner

ROOT=Path(__file__).resolve().parents[1]

def main():
    name='exp010_bound_smoke_v1'
    out=ROOT/'outputs/reviews'/name
    assert not out.exists()
    torch.set_num_threads(1)
    args=Namespace(env_id='baseline_'+name, seed=1, env_seed=1, device='cuda:0',parallels=2,
        running_steps=1024,eval_interval=1024,test_episode=2,baseline_eval_parallels=2,
        episode_length=32,buffer_size=4096,batch_size=32,start_training=32,
        start_noise=3.,end_noise=3.,noise_decay_steps=1024,baseline_final_eval_interval=1024)
    runner=get_runner('matd3','uav_pursuit_coverage_3d','baseline_'+name,
        str(ROOT/'experiments/baseline_exp010_pilot.yaml'),args)
    initial={k:v.clone() for k,v in runner.agent.policy.state_dict().items()}
    try:
        runner.run(mode='benchmark')
        memory=runner.agent.memory
        maxima={key:float(np.max(np.abs(value))) for key,value in memory.data['actions'].items()}
        assert memory.size>0 and max(maxima.values())<=1.
        state=runner.agent.policy.state_dict()
        changed={prefix:any(not torch.equal(initial[k],v) for k,v in state.items() if k.startswith(prefix+'.'))
                 for prefix in ('actor','critic_A','critic_B')}
        assert all(changed.values()) and all(torch.isfinite(v).all() for v in state.values())
        report=dict(status='passed',transfers=1024,exploration_sigma=3.,bounded_replay_maxima=maxima,
                    replay_size=int(memory.size),networks_updated=changed,note='Integration only; not training performance')
        out.mkdir(parents=True,exist_ok=False)
        (out/'audit.json').write_text(json.dumps(report,indent=2))
        print('BOUND_AUDIT',json.dumps(report),flush=True)
    finally:
        runner.agent.train_envs.close()
        runner.agent.writer.close()

if __name__=='__main__':
    main()
