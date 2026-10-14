"""Freeze a completed best-stage save and render one actual safe pilot success."""
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from tools import render_optimized_success_videos
from tools import render_baseline_success_videos as native

ROOT=Path(__file__).resolve().parents[1]
PROCESS=ROOT/'outputs/experiments/exp010_012_pilots_v1/stage1_video_process.json'
OUTPUT=ROOT/'outputs/reviews/exp012_pilot_stage1_video_v1'
RESULT=ROOT/'outputs/results/matd3/baseline_exp012_pilot'


def save(value):
    temp=PROCESS.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(PROCESS)


def main():
    import torch
    torch.set_num_threads(1)
    assert not PROCESS.exists() and not OUTPUT.exists()
    state=dict(status='waiting_for_stable_best_stage',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
               experiment='012_pilot',level=0,video_seed=69501,illustrative_only=True)
    save(state)
    try:
        while True:
            batch=json.loads((PROCESS.parent/'process.json').read_text())
            run=next(x for x in batch['runs'] if x['number']=='012')
            metrics=[row for row in csv.DictReader((RESULT/'curriculum_eval.csv').open()) if row['phase']=='current' and int(row['level'])==0]
            assert any(float(row['safe_success_rate'])>0 for row in metrics),'No formal safe capture'
            if run['status']!='running':
                assert run.get('exit_code')==0
                break
            progress=json.loads((RESULT/'training_status.json').read_text())
            if progress['level']!=0 or int(metrics[-1]['step'])+1024<=progress['step']<int(metrics[-1]['step'])+45000:
                break
            if batch['status']=='failed':
                raise RuntimeError('Pilot failed during video preparation')
            time.sleep(10)
        OUTPUT.mkdir(parents=True,exist_ok=False)
        level_dir=OUTPUT/'level_1'
        level_dir.mkdir()
        config,provenance=native.freeze_stage(ROOT,'012_pilot',0,level_dir)
        native.save_json(level_dir/'provenance.json',provenance)
        state.update(status='selecting',checkpoint_sha256=provenance['checkpoint_sha256'])
        save(state)
        data,report=native.find_success(config,level_dir/'model.pth',0,69501,30,level_dir)
        report.update(provenance)
        report['pilot_video_not_exp013']=True
        native.save_json(level_dir/'manifest.json',report)
        state['status']='rendering'
        save(state)
        native.render(data,report,level_dir)
        native.save_json(level_dir/'manifest.json',report)
        native.save_json(OUTPUT/'index.json',[report])
        state.update(status='completed',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),
                     capture_time=report['capture_time'],video_sha256=report['video_sha256'],
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     native_environment_replay_identical=True)
        save(state)
        print('ACTUAL_PILOT_VIDEO_COMPLETE',json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed',exit_code=1,error=repr(error))
        save(state)
        raise


if __name__=='__main__':
    main()
