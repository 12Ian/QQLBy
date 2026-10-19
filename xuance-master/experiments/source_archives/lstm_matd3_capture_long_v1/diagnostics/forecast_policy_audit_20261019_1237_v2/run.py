"""Read-only forecast audit on new scenes, with no learner updates."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os,json,time,hashlib
import numpy as np
import torch
from uav_lstm_prediction import fixed_eval as fe
from uav_lstm_prediction.core import TAUS,write_json,sha256
RUN=Path('/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/lstm_matd3_capture_long_v1_20261019')
OUT=RUN/'forecast_policy_audit_20261019_1237_v2'
TRACE_DIR=None
def init(config,checkpoint,name):
    global TRACE_DIR
    os.nice(5)
    TRACE_DIR=OUT/name;TRACE_DIR.mkdir(parents=True,exist_ok=True)
    fe.initialize(config,checkpoint,4,str(TRACE_DIR/'workers'))
def episode(case):
    raw=fe._RAW;env=fe._ENV;keys=fe._KEYS
    raw.seed(641004000+case)
    obs,_=env.reset()
    targets=[env.history[0].frames[-1]['target'].copy()]
    initial=hashlib.sha256(np.concatenate((raw.buildings.flatten(),raw.uav_positions.flatten(),raw.target_position.flatten())).tobytes()).hexdigest()
    ends=[];pred=[];cv=[];ca=[]
    for step in range(600):
        h=env.history[0]
        if h.ready:
            assert np.allclose(h.frames[-1]['target'],targets[-1])
            v=h.velocity();frames=list(h.frames)
            prev=(frames[-2]['target']-frames[-3]['target'])/.5
            acc=(v-prev)/.5
            ends.append(step)
            pred.append(env.last_forecasts[0].copy())
            base=v[None,:]*TAUS[:,None]
            cv.append(np.repeat(base[None,:,:],4,axis=0))
            ca.append(np.repeat((base+.5*acc[None,:]*TAUS[:,None]**2)[None,:,:],4,axis=0))
        x=torch.from_numpy(np.stack([obs[0][k] for k in keys]))
        with torch.inference_mode():actions=fe._POLICY({keys[0]:x},agent_ids=fe._IDS)[1][keys[0]].numpy()
        assert np.isfinite(actions).all() and np.abs(actions).max()<=1.000001
        obs,rew,term,trunc,infos=env.step([{k:actions[i] for i,k in enumerate(keys)}])
        context=infos[0]['prediction_context']
        done=all(term[0].values()) or bool(trunc[0])
        uniform=np.isclose(context['times'],(step+1)*.5)
        assert uniform or done,'Nonterminal decision timestamp changed'
        if uniform:targets.append(np.array(context['target'],copy=True))
        if done:break
    target=np.asarray(targets);ends=np.asarray(ends)
    keep=ends+12<len(target);ends=ends[keep]
    assert len(ends)>0
    truth=target[ends[:,None]+np.arange(1,13)[None,:]]-target[ends,None,:]
    errors={}
    metrics={}
    for method,values in [('lstm',pred),('cv',cv),('ca',ca)]:
        errors[method]=np.linalg.norm(np.asarray(values)[keep]-truth[:,None,:,:],axis=-1).astype(np.float32)
        e=errors[method]
        metrics[method]=dict(ade_m=float(e.mean()),horizons_m=e.mean(axis=(0,1)).tolist(),p90_m=float(np.quantile(e,.9)))
    np.savez_compressed(TRACE_DIR/f'case_{case:02d}.npz',target=target,ends=ends,lstm=np.asarray(pred)[keep],cv=np.asarray(cv)[keep],ca=np.asarray(ca)[keep],**{'error_'+k:v for k,v in errors.items()})
    return dict(case=case,seed=641004000+case,initial_sha256=initial,steps=step+1,terminal_time_seconds=float(context['times']),terminal_on_grid=bool(uniform),success=bool(infos[0]['is_success']),full_6s_windows=len(ends),metrics=metrics)
def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    config=str(RUN/'configs/lstm_seed_3.yaml')
    paths={name:json.loads((RUN/'jobs'/job/'model_paths.json').read_text())[which] for name,job,which in [('initial','lstm_seed_1','initial'),('final_seed_3','lstm_seed_3','final')]}
    result=dict(scene_count=20,levels=[5],physical_dt=.5,full_future_only=True,trajectory_source={},results={})
    initial=None
    for name,checkpoint in paths.items():
        write_json(OUT/'status.json',dict(stage='collecting_and_scoring',policy=name))
        result['trajectory_source'][name]=dict(checkpoint=checkpoint,sha256=sha256(checkpoint))
        with ProcessPoolExecutor(max_workers=8,mp_context=get_context('spawn'),initializer=init,initargs=(config,checkpoint,name)) as pool:
            rows=list(pool.map(episode,range(20)))
        fingerprints=[(r['case'],r['initial_sha256']) for r in rows]
        if initial is None:initial=fingerprints
        assert initial==fingerprints
        aggregate={}
        for method in ['lstm','cv','ca']:
            errors=np.concatenate([np.load(OUT/name/f"case_{r['case']:02d}.npz")['error_'+method] for r in rows])
            episode_ade=np.array([r['metrics'][method]['ade_m'] for r in rows])
            aggregate[method]=dict(ade_m=float(errors.mean()),episode_macro_ade_m=float(episode_ade.mean()),horizons_m=errors.mean(axis=(0,1)).tolist(),p90_m=float(np.quantile(errors,.9)),full_windows=len(errors))
        delta=np.array([r['metrics']['lstm']['ade_m']-r['metrics']['cv']['ade_m'] for r in rows])
        bootstrap=np.random.default_rng(123).choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1)
        result['results'][name]=dict(aggregate=aggregate,episodes=rows,lstm_minus_cv_macro_ade_m=float(delta.mean()),scene_bootstrap_95ci_m=np.quantile(bootstrap,[.025,.975]).tolist())
        print('RESULT',name,aggregate,flush=True)
        write_json(OUT/'partial.json',result)
    result.update(elapsed_seconds=time.time()-started,paired_initial_states_verified=True,note='Read-only diagnostic on 20 new level-five scenes under two fixed policies. Predictions precede labels; no future after episode termination; four views and overlapping windows are correlated.')
    write_json(OUT/'summary.json',result);write_json(OUT/'status.json',dict(stage='complete',elapsed_seconds=time.time()-started))
if __name__=='__main__':main()
