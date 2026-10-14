"""One-time Q/real-return diagnosis of each completed improvement pilot."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import time
import torch
from tools.analyze_exp009_policy import rollout

ROOT=Path(__file__).resolve().parents[1]
PILOTS=ROOT/'outputs/experiments/exp010_012_pilots_v1'
OUT=ROOT/'outputs/reviews/exp010_012_q_diagnostic_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    summary={}
    for number in ('010','011','012'):
        while True:
            state=json.loads((PILOTS/'process.json').read_text())
            row=next((row for row in state['runs'] if row['number']==number),{})
            if row.get('status')=='completed':
                break
            if state['status']=='failed':
                raise RuntimeError('Pilot batch failed; never restart it here')
            supervisor=Path('/proc')/str(state['supervisor_pid'])/'cmdline'
            if not supervisor.exists():
                raise RuntimeError('Pilot supervisor missing before completion; inspect original state')
            time.sleep(20)
        directory=OUT/number
        directory.mkdir(exist_ok=False)
        cfg=json.loads((ROOT/f'outputs/results/matd3/baseline_exp{number}_pilot/config.json').read_text())
        summary[number]={}
        for label in ('final','best'):
            checkpoint=PILOTS/number/f'{label}.pth'
            result=rollout(cfg,checkpoint,66001,32)
            (directory/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            summary[number][label]={k:v for k,v in result.items() if k!='rows'}
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print(number,label,json.dumps(summary[number][label]),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_seed=66001,episodes_per_model=32,collision_termination=True,not_holdout_performance=True),indent=2))
    print('PILOT_Q_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
