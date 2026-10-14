"""Correct captions from the exact frozen, replay-verified successful trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from matplotlib.figure import Figure
from tools import render_optimized_success_videos
from tools import render_baseline_success_videos as native


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation',type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    report=json.loads((args.source/'manifest.json').read_text())
    assert not report['course']['direct_intercept_enabled'] and report['capture_type']=='sustained'
    assert report['unmodified_environment_replay_identical'] and sum(report['contact_events'].values())==0
    assert sha(args.source/'trajectory.npz')==report['trajectory_sha256']
    assert sha(args.source/'model.pth')==report['checkpoint_sha256']
    validation=None
    if args.validation:
        validation=json.loads(args.validation.read_text())
        assert validation['checkpoint_sha256']==report['checkpoint_sha256']
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['model.pth','trajectory.npz','training_config.json','rollout_config.json','provenance.json','attempts.json']:
        if (args.source/name).exists():
            shutil.copy2(args.source/name,args.output/name)
    data=dict(np.load(args.source/'trajectory.npz'))
    report.update(source_video_sha256=report['video_sha256'],caption_revision='hold-only criterion; distinguish curriculum evaluation from validation',
                  caption_only_revision=True,validation_metrics=validation['metrics'] if validation else None)
    old=Figure.text
    def text(self,x,y,s='',*positional,**kwargs):
        if '成功判据：' in s:
            m=report['checkpoint_eval']
            s=f"课程评估100回合：安全{float(m['safe_success_rate']):.0%}、碰撞{float(m['collision_rate']):.0%}"
            if validation:
                m=validation['metrics']
                s+=f" | 独立验证100回合：安全{m['safe_success_rate']:.0%}、碰撞{m['collision_rate']:.0%}"
            s+=' | 判据：同机在半径内连续2秒'
        return old(self,x,y,s,*positional,**kwargs)
    Figure.text=text
    try:
        native.render(data,report,args.output)
    finally:
        Figure.text=old
    native.save_json(args.output/'manifest.json',report)
    print('CORRECT_CAPTION_COMPLETE',report['video_sha256'],flush=True)


if __name__=='__main__':
    main()
