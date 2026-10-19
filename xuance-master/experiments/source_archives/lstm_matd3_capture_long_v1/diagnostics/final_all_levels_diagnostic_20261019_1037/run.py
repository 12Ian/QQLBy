from pathlib import Path
import json,os,time
from uav_lstm_prediction.fixed_eval import evaluate_fixed
from uav_lstm_prediction.core import write_json
RUN=Path('/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/lstm_matd3_capture_long_v1_20261019')
OUT=RUN/'final_all_levels_diagnostic_20261019_1037'
def main():
    os.nice(5)
    started=time.time();rows=[]
    for seed in [1,2]:
        d=RUN/'jobs'/f'lstm_seed_{seed}'
        models=json.loads((d/'model_paths.json').read_text())
        config=RUN/'configs'/f'lstm_seed_{seed}.yaml'
        for level in range(5):
            write_json(OUT/'status.json',dict(stage='evaluating',seed=seed,level=level+1,elapsed_seconds=time.time()-started))
            path=OUT/f'final_seed_{seed}_level_{level+1}.json'
            if level==4:
                result=json.loads((d/'test/final_level_5.json').read_text())
                write_json(path,result)
            else:
                cases=[dict(case=i,seed=631000000+level*1000+i,level=level) for i in range(200)]
                result=evaluate_fixed(config,models['final'],cases,path,workers=8)
            reference=json.loads((d/'test'/f'best_level_{level+1}.json').read_text())
            fingerprint=lambda x:[(r['case'],r['seed'],r['initial_sha256']) for r in x['records']]
            assert fingerprint(result)==fingerprint(reference),'Different test scenes'
            rows.append(dict(seed=seed,level=level+1,final=result['metrics'],initial=reference['metrics']))
            print('DONE',seed,level+1,result['metrics'],flush=True)
    write_json(OUT/'summary.json',dict(rows=rows,paired_scenes_verified=True,selection_used=False,elapsed_seconds=time.time()-started))
    write_json(OUT/'status.json',dict(stage='complete',elapsed_seconds=time.time()-started))
if __name__=='__main__':main()
