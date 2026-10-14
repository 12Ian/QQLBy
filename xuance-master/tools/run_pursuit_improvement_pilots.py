"""Sequential, matched-initialization pilots; preserve every real exit/result."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/exp010_012_pilots_v1'
NUMBERS=('010','011','012')


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='feat-coverage-reaward'
    audit=json.loads((ROOT/'outputs/reviews/exp010_bound_smoke_v1/audit.json').read_text())
    assert audit['status']=='passed'
    assert '67 passed' in Path('/tmp/exp010_tests.log').read_text()
    assert all(not (ROOT/f'outputs/results/matd3/baseline_exp{n}_pilot').exists() for n in NUMBERS)
    OUT.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'outputs/experiments/exp009/metadata.json').read_text())
    names=set(old['source_hashes'])
    names.update(f'experiments/baseline_exp{n}_pilot.yaml' for n in NUMBERS)
    names.update(['xuance/torch/agents/multi_agent_rl/matd3_agents.py','tests/test_matd3_action_bounds.py',
                  'tools/run_pursuit_improvement_pilots.py','tools/check_action_replay_bounds.py'])
    hashes={name:sha(ROOT/name) for name in sorted(names)}
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(ROOT/name,name)
    save(OUT/'metadata.json',dict(source_hashes=hashes,budget_per_pilot=750000,parallels=16,seed=1,
        initialization='random policy initialization matched tensor-for-tensor to exp009',
        validation_seed=67101,blind_seed_reserved=67201,
        variants={'010':'bound action contract only','011':'010 plus exploration .3 to .1 over 3M',
                  '012':'010 plus pursuit weight .1 to 1.0; formation unchanged'},
        gates='same safety promotion gate; pilots cannot establish full task success',tests='67 passed',gpu_audit=audit))
    shutil.copy2('/tmp/exp010_tests.log',OUT/'tests.log')
    environment=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',TQDM_MININTERVAL='30')
    state=dict(status='running',supervisor_pid=os.getpid(),started_at=now(),runs=[])
    save(OUT/'process.json',state)
    try:
        for n in NUMBERS:
            assert all(sha(ROOT/name)==value for name,value in hashes.items()),'Source changed during pilots'
            run_dir=OUT/n
            run_dir.mkdir(exist_ok=False)
            config=ROOT/f'experiments/baseline_exp{n}_pilot.yaml'
            name=f'exp{n}_pilot'
            command=[sys.executable,'-u','examples/run_baseline.py','--name',name,'--config',str(config),
                     '--steps','750000','--parallels','16','--seed','1','--device','cuda:0',
                     '--start-level','0','--eval-interval','50000','--test-episodes','100']
            row=dict(number=n,status='running',started_at=now(),command=command,evaluations=[])
            state['runs'].append(row)
            with (run_dir/'train.log').open('x') as log:
                worker=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=worker.pid
                save(OUT/'process.json',state)
                code=worker.wait()
            row.update(status='trained' if code==0 else 'failed',exit_code=code,finished_training_at=now())
            save(OUT/'process.json',state)
            if code!=0:
                raise RuntimeError(f'{name} exited {code}; no restart')
            result=ROOT/f'outputs/results/matd3/baseline_{name}'
            summary=json.loads((result/'baseline_summary.json').read_text())
            assert summary['environment_steps']==750000
            row['training_summary']=summary
            models=list((ROOT/f'outputs/models/matd3/baseline_{name}').glob('seed_*'))
            assert len(models)==1
            reference=torch.load(ROOT/'outputs/experiments/exp009/initial_random_model.pth',map_location='cpu',weights_only=False)['policy']
            initial=torch.load(models[0]/'initial_random_model.pth',map_location='cpu',weights_only=False)['policy']
            row['initialization_matched_exp009']=all(torch.equal(initial[key],value) for key,value in reference.items())
            assert row['initialization_matched_exp009']
            checkpoints={'final':models[0]/'final_train_model.pth','best':result/'best_model/best_stage_0.pth'}
            for label,source in checkpoints.items():
                frozen=run_dir/f'{label}.pth'
                shutil.copy2(source,frozen)
                output=run_dir/f'{label}_level1_seed67101.json'
                evaluate=[sys.executable,'-u','examples/evaluate_baseline_checkpoint.py','--config',str(config),
                          '--checkpoint',str(frozen),'--env-seed','67101','--episodes','100','--parallels','8',
                          '--level','0','--device','cuda:0','--output',str(output)]
                with (run_dir/f'{label}_evaluation.log').open('x') as log:
                    evaluation=subprocess.Popen(evaluate,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    row['evaluation_pid']=evaluation.pid
                    save(OUT/'process.json',state)
                    exit_code=evaluation.wait()
                assert exit_code==0,f'Evaluation failed {exit_code}'
                metrics=json.loads(output.read_text())['metrics']
                row['evaluations'].append(dict(label=label,sha256=sha(frozen),output=str(output),metrics=metrics))
                save(OUT/'process.json',state)
            row.update(status='completed',finished_at=now())
            save(OUT/'process.json',state)
        state.update(status='completed',finished_at=now())
        save(OUT/'process.json',state)
    except BaseException as error:
        state.update(status='failed',finished_at=now(),error=repr(error))
        save(OUT/'process.json',state)
        raise


if __name__=='__main__':
    main()
