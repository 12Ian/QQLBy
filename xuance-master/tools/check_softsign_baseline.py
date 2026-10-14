"""Verify the candidate Actor output and real CUDA training before a long run."""
from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import torch
from xuance import get_runner
from xuance.torch.utils import ActivationFunctions

ROOT=Path(__file__).resolve().parents[1]


def main():
    ActivationFunctions['softsign']=torch.nn.Softsign
    torch.set_num_threads(1)
    name='exp013_softsign_gpu_smoke_v1'
    out=ROOT/'outputs/reviews'/name
    assert not out.exists()
    cfg=Namespace(env_id='baseline_'+name,seed=1,env_seed=1,device='cuda:0',parallels=2,
        running_steps=1024,eval_interval=1024,test_episode=2,baseline_eval_parallels=2,
        episode_length=32,buffer_size=4096,batch_size=32,start_training=32,baseline_final_eval_interval=1024)
    runner=get_runner('matd3','uav_pursuit_coverage_3d','baseline_'+name,
                      str(ROOT/'experiments/baseline_exp013_softsign.yaml'),cfg)
    before={k:v.clone() for k,v in runner.agent.policy.state_dict().items()}
    try:
        key=runner.agent.model_keys[0]
        assert isinstance(runner.agent.policy.actor[key].model[-1],torch.nn.Softsign)
        assert isinstance(runner.agent.policy.target_actor[key].model[-1],torch.nn.Softsign)
        assert not runner.agent.policy.use_graph_module and not runner.agent.policy.use_obstacle_gat
        logits=torch.tensor([10.,15.,20.],device='cuda:0',requires_grad=True)
        actions=runner.agent.policy.actor[key].model[-1](logits)
        gradients=torch.autograd.grad(actions.sum(),logits)[0]
        assert bool((actions.abs()<1).all()) and bool((gradients>0).all())
        runner.run(mode='benchmark')
        after=runner.agent.policy.state_dict()
        changed={prefix:any(not torch.equal(before[k],v) for k,v in after.items() if k.startswith(prefix+'.'))
                 for prefix in ('actor','critic_A','critic_B')}
        assert all(changed.values()) and all(torch.isfinite(v).all() for v in after.values())
        maxima={k:float(np.max(np.abs(v))) for k,v in runner.agent.memory.data['actions'].items()}
        assert max(maxima.values())<=1.
        report=dict(status='passed',transfers=1024,actor_activation='softsign',target_actor_activation='softsign',
                    both_critics_updated=changed,actor_input=83,critic_input=332,
                    bounded_replay=maxima,gradients_at_logits_10_15_20=gradients.tolist(),
                    no_teacher_or_initial_checkpoint=True,note='Implementation verification, not policy performance')
        out.mkdir(parents=True,exist_ok=False)
        (out/'audit.json').write_text(json.dumps(report,indent=2))
        print('SOFTSIGN_GPU_AUDIT',json.dumps(report),flush=True)
    finally:
        runner.agent.train_envs.close()
        runner.agent.writer.close()


if __name__=='__main__':
    main()
