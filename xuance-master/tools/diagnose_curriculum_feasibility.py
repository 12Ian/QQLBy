"""Profile unchanged five-level tasks with explicit diagnostic controllers."""
import argparse
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import yaml
from tools.diagnose_pursuit_feasibility import controller
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=16)
    parser.add_argument('--seed',type=int,default=69601)
    args=parser.parse_args()
    assert args.episodes>0 and not args.output.exists()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    files=[args.config,Path(__file__),root/'tools/diagnose_pursuit_feasibility.py',
           root/'xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py']
    hashes={str(p.resolve()):sha(p) for p in files}
    process=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    save(args.output/'process.json',process)
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    result=dict(diagnostic_controllers_not_RL=True,simulator_modified=False,
                collision_termination=True,episodes_per_controller_per_level=args.episodes,
                episode_seeds=list(range(args.seed,args.seed+args.episodes)),source_hashes=hashes,levels=[])
    started=time.time()
    try:
        for level in range(5):
            cfg=dict(config,curriculum_start_level=level,render=False,
                     terminate_on_building_collision=True,terminate_on_boundary_collision=True)
            env=UAVPursuitCoverage3DEnv(Namespace(**cfg))
            assert env.curriculum_level==level
            stage=dict(level=level+1,course=env.get_curriculum_info(),controllers={})
            try:
                for mode in ('pd','pd_clearance'):
                    records=[]
                    for episode,seed in enumerate(result['episode_seeds']):
                        env.seed(seed)
                        obs,_=env.reset()
                        initial=dict(buildings_sha256=hashlib.sha256(env.buildings.tobytes()).hexdigest(),
                                     pursuers=env.uav_positions.tolist(),evader=env.target_position.tolist(),
                                     pursuer_velocities=env.uav_velocities.tolist(),evader_velocity=env.target_velocity.tolist(),
                                     observations={a:obs[a].tolist() for a in env.agents})
                        initial_sha=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
                        rewards=np.zeros(env.num_agents)
                        discount=1.
                        peak_speed=float(env.target_speed)
                        sensed_steps=0
                        trace=[]
                        for step in range(env.max_episode_steps):
                            actions=controller(env,mode=='pd_clearance')
                            assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1.000001
                            obs,reward,terminated,truncated,info=env.step({a:actions[i] for i,a in enumerate(env.agents)})
                            rewards+=discount*np.array([reward[a] for a in env.agents])
                            discount*=cfg['gamma']
                            peak_speed=max(peak_speed,float(env.target_speed))
                            sensed_steps+=int(np.min(np.linalg.norm(env.uav_positions-env.target_position,axis=1))<env.evader_sense)
                            if episode==0:
                                trace.append(dict(step=step+1,positions=env.uav_positions.tolist(),
                                                  evader=env.target_position.tolist(),target_velocity=env.target_velocity.tolist()))
                            if all(terminated.values()) or truncated:
                                break
                        records.append(dict(seed=seed,initial=initial,initial_sha256=initial_sha,
                            steps=step+1,elapsed_seconds=float(env.episode_elapsed_seconds),
                            success=bool(info['is_success']),safe_success=bool(info['is_success'] and not info['is_collision']),
                            collision=bool(info['is_collision']),collision_sources=info['collision_sources'],
                            capture_type=info['capture_type'],mean_discounted_return=float(rewards.mean()),
                            target_peak_speed=peak_speed,target_sensed_fraction=sensed_steps/(step+1),
                            final_distances=np.linalg.norm(env.uav_positions-env.target_position,axis=1).tolist(),trace=trace))
                    stage['controllers'][mode]=dict(records=records,
                        captures=sum(r['success'] for r in records),safe_captures=sum(r['safe_success'] for r in records),
                        collisions=sum(r['collision'] for r in records),
                        mean_episode_steps=float(np.mean([r['steps'] for r in records])),
                        mean_target_peak_speed=float(np.mean([r['target_peak_speed'] for r in records])),
                        mean_target_sensed_fraction=float(np.mean([r['target_sensed_fraction'] for r in records])))
                    print('LEVEL',level+1,mode,json.dumps({k:v for k,v in stage['controllers'][mode].items() if k!='records'}),flush=True)
                a=stage['controllers']['pd']['records']
                b=stage['controllers']['pd_clearance']['records']
                assert all(x['initial']==y['initial'] and x['initial_sha256']==y['initial_sha256'] for x,y in zip(a,b))
                stage['paired_initial_states_exact']=True
                result['levels'].append(stage)
                save(args.output/f'level{level+1}.json',stage)
                save(args.output/'results.json',result)
            finally:
                env.close()
        assert all(sha(Path(p))==h for p,h in hashes.items())
        result['elapsed_seconds']=time.time()-started
        save(args.output/'results.json',result)
        process.update(status='completed',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
        save(args.output/'process.json',process)
        print('CURRICULUM_FEASIBILITY_COMPLETE',flush=True)
    except BaseException as error:
        process.update(status='failed',error=repr(error),exit_code=1)
        save(args.output/'process.json',process)
        raise


if __name__=='__main__':
    main()
