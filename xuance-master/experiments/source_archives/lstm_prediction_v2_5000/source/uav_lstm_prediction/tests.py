"""Invariant checks for causal data construction, grouping and learning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .core import (unit_checks, make_windows, load_trace, supervised_loss,
                   ContextLSTM, decode, sha256, write_json)

def run_checks(run):
    torch.set_num_threads(1)
    unit_checks()
    p=json.loads((run/"protocol.json").read_text())
    groups={}
    for e in p["episodes"]:
        group=(e["level"],e["seed"])
        groups.setdefault(group,set()).add(e["split"])
    assert len(groups)==p.get("total_scenes",250) and all(len(s)==1 for s in groups.values())
    for level in range(5):
        assert [sum(1 for g,s in groups.items() if g[0]==level and split in s)
                for split in ["train","validation","test"]]==p.get("split_counts_per_level",[35,7,8])
    result=json.loads((run/"collection.json").read_text())
    assert len(result["episodes"])==len(p["episodes"])
    assert result["paired_initial_states_identical"]
    initial_groups={}
    for item in result["episodes"]:
        key=item["initial_sha256"]
        initial_groups.setdefault(key,set()).add((item["level"],item["seed"]))
    assert all(len(v)==1 for v in initial_groups.values()),"Different scene seeds produced duplicate initial conditions"
    parent=json.loads((Path(p["parent_run"])/"protocol.json").read_text())
    assert all(groups[(e["level"],e["seed"])]=={e["split"]} for e in parent["episodes"])
    e=next(e for e in p["episodes"] if e["level"]==4)
    trace=load_trace(run/e["path"])
    windows=make_windows(trace)
    modified={k:v.copy() for k,v in trace.items()}
    modified["target"][11:]+=100.
    modified["pursuers"][11:]-=30.
    modified["observations"][11:]=0.
    changed=make_windows(modified)
    assert np.array_equal(windows["x"][:4],changed["x"][:4])
    assert not np.array_equal(windows["y"][:4],changed["y"][:4])
    truth=torch.from_numpy(windows["y"][-8:])
    mask=torch.from_numpy(windows["mask"][-8:])
    pred=torch.zeros_like(truth,requires_grad=True)
    first=supervised_loss(pred,truth,mask)
    contaminated=truth.clone()
    contaminated[~mask]=1e7
    second=supervised_loss(pred,contaminated,mask)
    assert torch.equal(first,second), "Missing labels affected supervised loss"
    torch.manual_seed(7)
    net=ContextLSTM()
    net.train()
    x=torch.randn(16,9,71)
    vel=torch.zeros(16,3)
    y=torch.arange(1,13)[None,:,None].float()**2 * torch.tensor([.05,-.02,.01])[None,None,:]
    y=y.expand(16,-1,-1).contiguous()
    m=torch.ones(16,12,dtype=torch.bool)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    with torch.no_grad():
        initial=float(supervised_loss(decode(net(x),vel),y,m))
    for _ in range(120):
        loss=supervised_loss(decode(net(x),vel),y,m)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    net.eval()
    with torch.no_grad():
        final=float(supervised_loss(decode(net(x),vel),y,m))
    assert final<initial*.5, (initial,final)
    protected={**p["protected_sha256"],**json.loads((run/"branch_protected_sha256.json").read_text())}
    assert all(sha256(path)==value for path,value in protected.items())
    outcome=dict(passed=True,checks=["synthetic motion and units","future causal isolation",
                 "missing-label masks","scene-group splits","paired-policy episodes",
                 "read-only policy collection smoke","small-model supervised learning",
                 "protected source hashes"],overfit_initial_loss=initial,overfit_final_loss=final)
    write_json(run/"checks.json",outcome)
    print(json.dumps(outcome),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    run_checks(args.run)
