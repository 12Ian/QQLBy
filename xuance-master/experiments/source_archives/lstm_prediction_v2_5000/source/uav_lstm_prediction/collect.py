"""Collect new paired scenes using frozen policies, without learning updates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import Namespace
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
from .core import sha256, write_json, feature_stream, decision_grid

_WORK = {}

def initialize(protocol_path):
    global _WORK
    protocol_path=Path(protocol_path)
    p=json.loads(protocol_path.read_text())
    run=protocol_path.parent
    os.environ["NUMBA_CACHE_DIR"]=str(run/"numba_cache")
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    sys.path.insert(0,p["runtime_source"])
    import torch
    torch.set_num_threads(1)
    from xuance import get_runner
    from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
    import yaml
    _WORK=dict(protocol=p,run=run,torch=torch,get_runner=get_runner,Env=UAVPursuitCoverage3DEnv,
               configs={k:yaml.safe_load(Path(v["config"]).read_text()) for k,v in p["methods"].items()},
               runners={},envs={})

def controller(method):
    w=_WORK
    if method not in w["runners"]:
        spec=w["protocol"]["methods"][method]
        name=f"prediction_collect_{method}_{os.getpid()}"
        options=Namespace(env_id=name,device="cpu",parallels=1,vectorize="DummyVecMultiAgentEnv",
                          buffer_size=128,batch_size=32,render=False,curriculum_start_level=0,
                          baseline_init_actor=None,baseline_init_checkpoint=None,
                          log_dir=str(w["run"]/"controller_logs"),
                          model_dir=str(w["run"]/"controller_models"))
        runner=w["get_runner"]("matd3","uav_pursuit_coverage_3d",name,
                               config_path=spec["config"],parser_args=options)
        saved=w["torch"].load(spec["checkpoint"],map_location="cpu",weights_only=True)["policy"]
        runner.agent.policy.load_state_dict(saved,strict=True)
        runner.agent.policy.eval()
        w["runners"][method]=runner
    return w["runners"][method].agent.policy

def run_episode(job):
    w=_WORK
    method,level,seed=job["method"],job["level"],job["seed"]
    destination=w["run"]/job["path"]
    sidecar=destination.with_suffix(".json")
    if destination.exists() and sidecar.exists():
        result=json.loads(sidecar.read_text())
        if sha256(destination)!=result["trajectory_sha256"]:
            raise ValueError("Existing trace checksum mismatch")
        return result
    cfg=dict(w["configs"][method],curriculum_start_level=level,render=False)
    if job.get("smoke"):
        cfg["episode_length"]=32
    assert cfg["episode_length"]==(32 if job.get("smoke") else 600)
    env=w["Env"](Namespace(**cfg))
    env.seed(seed)
    obs,_=env.reset()
    policy=controller(method)
    torch=w["torch"]
    keys=env.agents
    ids=torch.eye(4)
    rows={k:[] for k in ["times","target","pursuers","target_velocity","target_acceleration",
                         "pursuer_velocity","pursuer_acceleration","observations","target_contacts"]}
    actions_history=[]
    diagnostics=[]
    def record():
        rows["times"].append(float(env.episode_elapsed_seconds))
        rows["target"].append(env.target_position.copy())
        rows["pursuers"].append(env.uav_positions.copy())
        rows["target_velocity"].append(env.target_velocity.copy())
        rows["target_acceleration"].append(env.target_acceleration.copy())
        rows["pursuer_velocity"].append(env.uav_velocities.copy())
        rows["pursuer_acceleration"].append(env.uav_accelerations.copy())
        rows["observations"].append(np.stack([obs[k] for k in keys]).copy())
        rows["target_contacts"].append(int(env.episode_target_building_contacts))
    record()
    import hashlib
    initial=hashlib.sha256(b"".join(np.asarray(rows[k][0]).tobytes() for k in
                          ("target","pursuers","target_velocity","pursuer_velocity","observations"))+
                          env.buildings.tobytes()).hexdigest()
    started=time.time()
    for step in range(cfg["episode_length"]):
        inputs=torch.from_numpy(np.stack([obs[k] for k in keys]))
        with torch.inference_mode():
            action=policy({keys[0]:inputs},agent_ids=ids)[1][keys[0]].cpu().numpy()
        if not np.isfinite(action).all() or abs(action).max()>1.000001:
            raise ValueError("Invalid frozen policy action")
        actions_history.append(action.copy())
        obs,reward,terminated,truncated,info=env.step({k:action[i] for i,k in enumerate(keys)})
        record()
        diagnostics.append([info["infos"]["safety_checks_episode"],
                            info["infos"]["safety_interventions_episode"]])
        if all(terminated.values()) or truncated:
            break
    else:
        raise AssertionError("Episode did not terminate at configured budget")
    trace={k:np.asarray(v) for k,v in rows.items()}
    trace.update(buildings=env.buildings.copy(),actions=np.asarray(actions_history),
                 safety_counts=np.asarray(diagnostics),
                 min_speed=np.array(env.target_min_speed,dtype=np.float32),
                 max_speed=np.array(env.target_max_speed,dtype=np.float32),
                 capture_radius=np.array(env.catch_radius,dtype=np.float32),
                 level=np.array(level,dtype=np.int64),seed=np.array(seed,dtype=np.int64))
    grid=decision_grid(trace)
    if len(grid["times"])>10:
        actual=feature_stream(grid)[0][:,:,41:].reshape(-1,4,5,6)
        from .core import local_buildings_from_map
        expected=local_buildings_from_map(grid["pursuers"],grid["target"],grid["buildings"])
        if not np.allclose(actual,expected,atol=2e-6,rtol=0):
            raise AssertionError("Offline visible-building reconstruction differs from original observation")
    if job.get("smoke"):
        # Re-run identical actions without recording to ensure the recorder is read-only.
        env.seed(seed)
        obs2,_=env.reset()
        for j,action in enumerate(actions_history):
            obs2,_,term2,trunc2,_=env.step({k:action[i] for i,k in enumerate(keys)})
            assert np.array_equal(env.target_position,trace["target"][j+1])
            assert np.array_equal(env.uav_positions,trace["pursuers"][j+1])
            assert np.array_equal(np.stack([obs2[k] for k in keys]),trace["observations"][j+1])
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**trace)
    temporary.replace(destination)
    result=dict(**job,initial_sha256=initial,trajectory_sha256=sha256(destination),
                steps=len(actions_history),seconds=float(trace["times"][-1]),
                target_contacts=int(trace["target_contacts"][-1]),
                success=bool(info["is_success"]),timeout=bool(truncated),
                safety_checks=int(diagnostics[-1][0]),safety_interventions=int(diagnostics[-1][1]),
                uncertified_substeps=int(info["infos"]["safety_uncertified_substeps_episode"]),
                elapsed_seconds=time.time()-started,observation_dim=int(trace["observations"].shape[-1]),
                min_speed=float(env.target_min_speed),max_speed=float(env.target_max_speed))
    write_json(sidecar,result)
    env.close()
    return result

def collect(protocol_path,smoke=False,workers=8):
    protocol_path=Path(protocol_path).resolve()
    p=json.loads(protocol_path.read_text())
    if smoke:
        jobs=[]
        for level in [0,4]:
            for method in p["methods"]:
                jobs.append(dict(level=level,method=method,seed=310199900+level,
                                 split="smoke",smoke=True,
                                 path=f"smoke/level_{level+1}/{method}.npz"))
    else:
        jobs=p["episodes"]
    results=[]
    started=time.time()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"),
                             initializer=initialize,initargs=(str(protocol_path),)) as pool:
        futures=[pool.submit(run_episode,job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%10==0 or len(results)==len(jobs):
                status=dict(stage="smoke" if smoke else "collection",completed=len(results),
                            total=len(jobs),elapsed_seconds=time.time()-started)
                write_json(protocol_path.parent/"status.json",status)
                print(json.dumps(status),flush=True)
    results.sort(key=lambda x:(x["level"],x["seed"],x["method"]))
    paired={}
    for r in results:
        paired.setdefault((r["level"],r["seed"]),[]).append(r)
    assert all(len(rows)==2 and rows[0]["initial_sha256"]==rows[1]["initial_sha256"]
               for rows in paired.values())
    output="smoke_collection.json" if smoke else "collection.json"
    write_json(protocol_path.parent/output,dict(episodes=results,paired_scenes=len(paired),
                paired_initial_states_identical=True,elapsed_seconds=time.time()-started))
    print("COLLECTION_COMPLETE",len(results),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    collect(args.protocol,args.smoke,args.workers)

if __name__=="__main__":
    main()
