"""Short learning validation, not a capture-rate experiment."""
from argparse import Namespace,ArgumentParser
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from xuance import get_runner
from xuance.engine.run_marl import initialize_baseline_policy
from uav_lstm_prediction.core import sha256,write_json

def state_hash(model):
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def run_smoke(run,steps=1024):
    run=Path(run)
    torch.set_num_threads(2)
    records=[]
    for mode in ["none","cv","lstm"]:
        options=Namespace(device="cuda:0",parallels=2,vectorize="SubprocVecMultiAgentEnv",
             buffer_size=2048,batch_size=64,start_training=64,running_steps=steps,
             training_frequency=1,n_epochs=1,seed=17,env_seed=610290001,
             curriculum_start_level=4,curriculum_auto_advance_by_steps=False,
             log_dir=str(run/"smoke_logs"/mode),model_dir=str(run/"smoke_models"/mode),
             uav_prediction_device="cuda:0")
        runner=get_runner("matd3","uav_pursuit_coverage_3d","integration_learning_"+mode,
                         config_path=str(run/"configs"/f"matd3_prediction_{mode}_600step.yaml"),
                         parser_args=options)
        agent=runner.agent;env=agent.train_envs
        try:
            initialization=initialize_baseline_policy(agent,runner.config)
            before={k:v.detach().cpu().clone() for k,v in agent.policy.state_dict().items()}
            predictor=env.predictor if mode!="none" else None
            pred_before=state_hash(predictor.model) if mode=="lstm" else None
            if predictor and predictor.model:
                pred_ids={id(p) for p in predictor.model.parameters()}
                optimizer_ids={id(p) for opts in agent.learner.optimizer.values()
                               for opt in opts.values() for group in opt.param_groups for p in group["params"]}
                assert not pred_ids & optimizer_ids
            started=time.perf_counter()
            metrics=agent.train(steps//2)
            elapsed=time.perf_counter()-started
            actor_changed=[];critic_changed=[]
            for key,value in agent.policy.state_dict().items():
                assert torch.isfinite(value).all()
                if not torch.equal(before[key],value.detach().cpu()):
                    if key.startswith("actor."):actor_changed.append(key)
                    if key.startswith(("critic_A.","critic_B.")):critic_changed.append(key)
            assert actor_changed and critic_changed
            assert agent.current_step==steps,(agent.current_step,steps)
            sample=agent.memory.sample()
            width=79 if mode=="none" else 88
            for key in agent.agent_keys:
                assert sample["obs"][key].shape[-1]==width
                assert sample["obs_next"][key].shape[-1]==width
                assert np.isfinite(sample["obs"][key]).all()
                assert np.isfinite(sample["obs_next"][key]).all()
            if mode=="lstm":
                assert predictor.model.training is False
                assert all(not p.requires_grad and p.grad is None for p in predictor.model.parameters())
                assert pred_before==state_hash(predictor.model)
                assert sha256(runner.config.uav_prediction_checkpoint)==runner.config.uav_prediction_sha256
            agent.save_model("integration_smoke_final.pth")
            record=dict(mode=mode,passed=True,environment_steps=steps,observation_dim=width,
                 learner_updates=agent.learner.iterations,actor_tensors_changed=len(actor_changed),
                 critic_tensors_changed=len(critic_changed),elapsed_seconds=elapsed,
                 replay_size_vector_steps=int(agent.memory.size),
                 prediction_seconds=getattr(env,"prediction_seconds",0.),
                 prediction_batches=getattr(env,"prediction_batches",0),
                 predictor_frozen=mode=="lstm",predictor_parameter_hash=pred_before,
                 initialization=initialization,
                 final_training_metrics={k:float(v) for k,v in metrics.items() if np.isscalar(v)})
            records.append(record);write_json(run/"learning_smoke_results.json",records)
            print(json.dumps(record),flush=True)
        finally:env.close()
    write_json(run/"learning_smoke_summary.json",dict(passed=True,arms=records,
               note="Short optimizer/replay validation only; no capture-rate improvement claim."))
    print("LEARNING_SMOKE_COMPLETE",flush=True)

if __name__=="__main__":
    parser=ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=1024)
    args=parser.parse_args()
    if args.steps<128 or args.steps%2:raise ValueError("Steps must be even and >=128")
    run_smoke(args.run,args.steps)
