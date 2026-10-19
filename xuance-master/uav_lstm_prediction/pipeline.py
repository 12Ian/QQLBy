"""Server-only experiment coordinator; ends before any MATD3 integration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .core import sha256,write_json

def main(run):
    run=Path(run).resolve()
    started=time.time()
    stages=[("tests",["-m","uav_lstm_prediction.tests","--run",str(run)]),
            ("training",["-m","uav_lstm_prediction.train","--run",str(run)]),
            ("rendering",["-m","uav_lstm_prediction.render","--run",str(run)])]
    if (run/"training_summary.json").exists() and (run/"test_metrics.json").exists():
        summary=json.loads((run/"training_summary.json").read_text())
        if sha256(run/"model/best.pt")!=summary["checkpoint_sha256"]:
            raise ValueError("Resume checkpoint hash mismatch")
        stages=[entry for entry in stages if entry[0]=="rendering"]
    try:
        for stage,command in stages:
            write_json(run/"pipeline_status.json",dict(stage=stage,pid=os.getpid()))
            with (run/(stage+".log")).open("w") as log:
                process=subprocess.run([sys.executable,"-u",*command],cwd=run,
                                       stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"{stage} failed; see {stage}.log")
        p=json.loads((run/"protocol.json").read_text())
        protected={**p["protected_sha256"],**json.loads((run/"branch_protected_sha256.json").read_text())}
        mismatches=[path for path,value in protected.items() if sha256(path)!=value]
        for method,spec in p["methods"].items():
            if sha256(spec["checkpoint"])!=spec["checkpoint_sha256"]:
                mismatches.append(spec["checkpoint"])
        assert not mismatches,mismatches
        write_json(run/"verification.json",dict(protected_source_and_checkpoints_unchanged=True,
                   source_files=len(protected),matd3_integrated=False,videos_delivered_for_user_review=True))
        write_json(run/"pipeline_status.json",dict(stage="complete_waiting_for_video_review",
                   elapsed_seconds=time.time()-started,pid=os.getpid()))
        print("PIPELINE_COMPLETE_WAITING_FOR_VIDEO_REVIEW",flush=True)
    except Exception as error:
        write_json(run/"pipeline_status.json",dict(stage="failed",error=str(error),pid=os.getpid()))
        raise

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    main(parser.parse_args().run)
