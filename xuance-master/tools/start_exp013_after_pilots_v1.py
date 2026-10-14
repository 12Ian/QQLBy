"""Finish the controlled pilot decision before one selected full-budget RL run."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'outputs/experiments/exp013'
PILOTS=ROOT/'outputs/experiments/exp010_012_pilots_v1'
RESULT=ROOT/'outputs/results/matd3/baseline_exp013'


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not RESULT.exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='feat-coverage-reaward'
    assert json.loads((ROOT/'outputs/reviews/exp013_softsign_gpu_smoke_v1/audit.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'outputs/reviews/exp013_softsign_gpu_smoke_v1/adapter_audit.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'outputs/reviews/exp013_softsign_gpu_smoke_v1/paired_evaluator_audit.json').read_text())['status']=='passed'
    EXP.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting_pilots',supervisor_pid=os.getpid(),created_at=now(),evaluations=[],videos=[])
    save(EXP/'process.json',state)
    environment=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',TQDM_MININTERVAL='30')
    try:
        while True:
            batch=json.loads((PILOTS/'process.json').read_text())
            if batch['status']=='completed':
                break
            if batch['status']=='failed':
                raise RuntimeError('Pilot batch failed; inspect it rather than restarting')
            handle=Path('/proc')/str(batch['supervisor_pid'])/'cmdline'
            if not handle.exists() or b'run_pursuit_improvement_pilots.py' not in handle.read_bytes():
                raise RuntimeError('Pilot supervisor missing before a terminal state')
            time.sleep(20)
        choices=[]
        for row in batch['runs']:
            assert row['status']=='completed' and row['exit_code']==0
            assert row['initialization_matched_exp009']
            for evaluation in row['evaluations']:
                metrics=evaluation['metrics']
                assert metrics['episodes']==100
                if metrics['safe_success_rate']>=.8 and metrics['collision_rate']<=.1:
                    choices.append(dict(pilot=row['number'],checkpoint_label=evaluation['label'],metrics=metrics))
        if choices:
            selection=max(choices,key=lambda x:(x['metrics']['safe_success_rate'],-x['metrics']['collision_rate'],x['metrics']['success_rate']))
            source_config=ROOT/f'experiments/baseline_exp{selection["pilot"]}_pilot.yaml'
            selection.update(method='validated_pilot_configuration',initialization='random, not selected checkpoint')
        else:
            source_config=ROOT/'experiments/baseline_exp013_softsign.yaml'
            selection=dict(method='softsign_only_relative_to_010',reason='No pilot met validation safe80/collision10',
                           initialization='random; no teacher, no policy transfer')
        parent=json.loads((PILOTS/'metadata.json').read_text())
        assert all(sha(ROOT/name)==value for name,value in parent['source_hashes'].items()),'Pilot sources changed'
        cfg=yaml.safe_load(source_config.read_text())
        cfg.update(env_id='baseline_exp013',running_steps=3000000,baseline_init_checkpoint=None,baseline_init_actor=None)
        configuration=ROOT/'experiments/baseline_exp013.yaml'
        assert not configuration.exists()
        configuration.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        names=set(parent['source_hashes'])
        names.update(['experiments/baseline_exp013.yaml','experiments/baseline_exp013_softsign.yaml',
                      'examples/run_softsign_baseline.py','examples/evaluate_softsign_baseline_checkpoint.py',
                      'examples/evaluate_paired_pursuit_checkpoint.py',
                      'tools/check_softsign_baseline.py','tools/start_exp013_after_pilots.py',
                      'tools/report_exp013.py','tools/render_optimized_success_videos.py'])
        hashes={name:sha(ROOT/name) for name in sorted(names)}
        with zipfile.ZipFile(EXP/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(ROOT/name,name)
        metadata=dict(selection=selection,source_configuration=str(source_config),source_hashes=hashes,
            branch='feat-coverage-reaward',training_from_scratch=True,budget=3000000,parallels=16,seed=1,
            actor_activation=cfg['activation_action'],no_teacher_or_safety_controller=True,
            final_review=dict(seed=67201,episode_seeds='67201 + case_index',episodes_per_level=100,
                              batched_environments=8,levels=[1,2,3,4,5],paired_frozen_reference='009 final'),
            video_seed=69301,pilot_selection=batch)
        save(EXP/'metadata.json',metadata)
        command=[sys.executable,'-u','examples/run_softsign_baseline.py','--name','exp013','--config',str(configuration),
                 '--steps','3000000','--parallels','16','--seed','1','--device','cuda:0',
                 '--start-level','0','--eval-interval','50000','--test-episodes','100']
        with (EXP/'train.log').open('x') as log:
            worker=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',child_pid=worker.pid,started_at=now(),command=command,selection=selection)
            save(EXP/'process.json',state)
            code=worker.wait()
        state.update(training_exit_code=code,finished_training_at=now())
        save(EXP/'process.json',state)
        if code!=0:
            raise RuntimeError(f'Training exited {code}; no restart')
        summary=json.loads((RESULT/'baseline_summary.json').read_text())
        assert summary['environment_steps']==3000000
        mismatches=[name for name,value in hashes.items() if sha(ROOT/name)!=value]
        assert not mismatches,f'Training source mismatch: {mismatches}'
        review=EXP/'final_review'
        review.mkdir(exist_ok=False)
        folders=list((ROOT/'outputs/models/matd3/baseline_exp013').glob('seed_*'))
        assert len(folders)==1
        initial=folders[0]/'initial_random_model.pth'
        old=torch.load(ROOT/'outputs/experiments/exp009/initial_random_model.pth',map_location='cpu',weights_only=False)['policy']
        current=torch.load(initial,map_location='cpu',weights_only=False)['policy']
        metadata['initial_weight_tensors_match_009']=all(torch.equal(current[k],v) for k,v in old.items())
        save(EXP/'metadata.json',metadata)
        shutil.copy2(initial,EXP/'initial_random_model.pth')
        model=review/'final.pth'
        shutil.copy2(folders[0]/'final_train_model.pth',model)
        save(review/'manifest.json',dict(checkpoint_sha256=sha(model),configuration_sha256=sha(configuration),
             source_hash_mismatches=mismatches,actor_activation=cfg['activation_action'],seed=67201,
             episodes_per_level=100,exploration=False,collision_termination=False,max_steps=cfg['episode_length']))
        reference_config=ROOT/'outputs/reviews/exp009_final_v1/config.yaml'
        reference_model=ROOT/'outputs/reviews/exp009_final_v1/final009.pth'
        reference_manifest=json.loads((reference_model.parent/'manifest.json').read_text())
        assert sha(reference_model)==reference_manifest['model_sha256']
        reference_cfg=yaml.safe_load(reference_config.read_text())
        task_keys=['num_agents','z_min','z_max','physics_dt','decision_dt','episode_length','acceleration_lag',
                   'uav_max_speed','max_accel','target_accel','uav_jerk_max','target_jerk_max','evader_sense',
                   'target_velocity_response','target_spawn_box','pursuer_spawn_positions','spawn_min_target_distance',
                   'curriculum_schedule','spawn_safety_enabled','spawn_safety_margin','direct_intercept_enabled',
                   'capture_hold_required','uav_collision_enabled']
        assert all(cfg.get(k)==reference_cfg.get(k) for k in task_keys),'Paired task configurations differ'
        save(review/'reference_manifest.json',dict(checkpoint_sha256=sha(reference_model),
             configuration_sha256=sha(reference_config),task_keys_verified_equal=task_keys,
             seed_protocol='67201 + episode_index; exact initial states checked for every case'))
        state.update(status='evaluating',final_checkpoint_sha256=sha(model))
        save(EXP/'process.json',state)
        pairing=[]
        for level in range(5):
            for label,checkpoint,config in [('final',model,configuration),('baseline009',reference_model,reference_config)]:
                output=review/f'{label}_level{level+1}.json'
                command=[sys.executable,'-u','examples/evaluate_paired_pursuit_checkpoint.py','--config',str(config),
                         '--checkpoint',str(checkpoint),'--env-seed','67201','--episodes','100','--parallels','8',
                         '--level',str(level),'--device','cuda:0','--output',str(output)]
                with (review/f'{label}_level{level+1}.log').open('x') as log:
                    evaluation=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    state['evaluation_pid']=evaluation.pid
                    save(EXP/'process.json',state)
                    code=evaluation.wait()
                assert code==0,f'Final evaluation {label}, level {level+1} failed: {code}'
                state['evaluations'].append(dict(model=label,level=level+1,metrics=json.loads(output.read_text())['metrics']))
                save(EXP/'process.json',state)
            a=json.loads((review/f'final_level{level+1}.json').read_text())['episodes']
            b=json.loads((review/f'baseline009_level{level+1}.json').read_text())['episodes']
            assert len(a)==len(b)==100 and all(x['case']==y['case'] and x['env_seed']==y['env_seed'] and
                       x['initial']==y['initial'] and x['initial_sha256']==y['initial_sha256'] for x,y in zip(a,b)), 'Initial-state pairing failed'
            pairing.append(dict(level=level+1,cases=100,initial_states_exact=True))
            save(review/'pairing_audit.json',dict(levels=pairing,seed_protocol='67201 + episode_index'))
            subprocess.run([sys.executable,'tools/report_exp013.py'],cwd=ROOT,env=environment,check=True)
        state['status']='rendering'
        save(EXP/'process.json',state)
        for text,metrics in summary['best_by_stage'].items():
            level=int(text)
            if metrics['safe_success_rate']<=0:
                state['videos'].append(dict(level=level+1,status='no_safe_example_in_formal_evaluation'))
                continue
            destination=ROOT/f'outputs/reviews/exp013_stage{level+1}_video_v1'
            command=[sys.executable,'-u','tools/render_optimized_success_videos.py','--root',str(ROOT),
                     '--output',str(destination),'--stages',f'013:{level}','--env-seed','69301','--max-attempts','30']
            with (review/f'video_level{level+1}.log').open('x') as log:
                renderer=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                state['render_pid']=renderer.pid
                save(EXP/'process.json',state)
                code=renderer.wait()
            state['videos'].append(dict(level=level+1,status='completed' if code==0 else 'selection_or_render_failed',exit_code=code))
            save(EXP/'process.json',state)
        state.update(status='completed',finished_at=now())
        save(EXP/'process.json',state)
        subprocess.run([sys.executable,'tools/report_exp013.py'],cwd=ROOT,env=environment,check=True)
        files=set(p for directory in (EXP,RESULT) for p in directory.rglob('*') if p.is_file())
        files.add(ROOT/'experiments/baseline_exp013.yaml')
        files.update(p for p in (ROOT/'docs').glob('baseline_exp013*') if p.suffix!='.zip')
        for directory in (ROOT/'outputs/reviews').glob('exp013_stage*_video_v1'):
            files.update(p for p in directory.rglob('*') if p.is_file())
        package=ROOT/'docs/baseline_exp013_delivery_v1.zip'
        checksums={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
        with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as archive:
            for name in checksums:
                archive.write(ROOT/name,name)
            archive.writestr('delivery_manifest.json',json.dumps(dict(files=checksums),ensure_ascii=False))
        save(EXP/'delivery.json',dict(archive=str(package),bytes=package.stat().st_size,sha256=sha(package)))
    except BaseException as error:
        state.update(status='failed',error=repr(error),failed_at=now())
        save(EXP/'process.json',state)
        subprocess.run([sys.executable,'tools/report_exp013.py'],cwd=ROOT,env=environment)
        raise


if __name__=='__main__':
    main()
