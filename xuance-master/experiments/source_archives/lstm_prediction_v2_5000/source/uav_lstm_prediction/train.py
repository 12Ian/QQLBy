"""Train and independently evaluate the frozen-history target predictor."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .core import (ContextLSTM, FEATURE_VERSION, FEATURE_DIM, TAUS, make_windows,
                   load_trace, decode, supervised_loss, sha256, write_json)

def assemble(run, protocol, split):
    keys=("x","y","mask","v0","a0","level","trace_id","near","turning","sensed_switch","future_contact")
    cache=run/"window_cache"/split
    cache.mkdir(parents=True,exist_ok=True)
    handles={k:(cache/(k+".bin")).open("wb") for k in keys}
    specs={}
    sources=[]
    total=0
    try:
        for job in protocol["episodes"]:
            if job["split"]!=split:
                continue
            path=run/job["path"]
            side=json.loads(path.with_suffix(".json").read_text())
            if sha256(path)!=side["trajectory_sha256"]:
                raise ValueError("Data checksum mismatch")
            windows=make_windows(load_trace(path))
            if windows is None:
                continue
            count=len(windows["x"])
            trace_id=len(sources)
            sources.append(dict(**job,samples=count,trajectory_sha256=side["trajectory_sha256"]))
            for key in keys:
                value=(np.full(count,job["level"],np.int64) if key=="level" else
                       np.full(count,trace_id,np.int64) if key=="trace_id" else windows[key])
                specs[key]=(value.shape[1:],value.dtype)
                value.tofile(handles[key])
            total+=count
    finally:
        for handle in handles.values():
            handle.close()
    if not sources:
        raise ValueError("No usable windows for "+split)
    result={key:np.memmap(cache/(key+".bin"),mode="r+",dtype=dtype,shape=(total,*shape))
            for key,(shape,dtype) in specs.items()}
    result["sources"]=sources
    print(json.dumps(dict(split=split,windows=total,episodes=len(sources),
                          x_gib=result["x"].nbytes/1024**3,data_residency="cpu_memmap")),flush=True)
    return result

def tensors(data,device):
    return {key:torch.from_numpy(data[key]).to(device) for key in ("x","y","mask","v0")}

@torch.inference_mode()
def validation_score(model,data,batch=2048):
    device=next(model.parameters()).device
    total=0.
    count=0
    for start in range(0,len(data["x"]),batch):
        sl=slice(start,start+batch)
        pred=decode(model(data["x"][sl].to(device)),data["v0"][sl].to(device))
        error=torch.linalg.vector_norm(pred-data["y"][sl].to(device),dim=-1)[:,1:6]
        mask=data["mask"][sl,1:6].to(device)
        total+=float(error[mask].sum())
        count+=int(mask.sum())
    return total/max(count,1)

@torch.inference_mode()
def infer(model,data,device,batch=2048):
    values=[]
    for start in range(0,len(data["x"]),batch):
        x=torch.from_numpy(data["x"][start:start+batch]).to(device)
        velocity=torch.from_numpy(data["v0"][start:start+batch]).to(device)
        values.append(decode(model(x),velocity).cpu().numpy())
    return np.concatenate(values)

def error_summary(errors,mask):
    valid=errors[mask]
    full=mask.all(axis=1)
    return dict(ade_m=float(valid.mean()),p50_m=float(np.percentile(valid,50)),
                p90_m=float(np.percentile(valid,90)),p95_m=float(np.percentile(valid,95)),
                valid_points=int(mask.sum()),windows=int(len(mask)),
                full_6s_windows=int(full.sum()),
                full_6s_ade_m=float(errors[full].mean()) if full.any() else None,
                horizons=[dict(seconds=float(tau),samples=int(mask[:,k].sum()),
                               mean_error_m=float(errors[mask[:,k],k].mean()),
                               p90_m=float(np.percentile(errors[mask[:,k],k],90)))
                          for k,tau in enumerate(TAUS) if mask[:,k].any()])

def evaluate(model,data,device):
    predicted=infer(model,data,device)
    tau=TAUS[None,:,None]
    baselines=dict(lstm=predicted,cv=data["v0"][:,None,:]*tau,
                   ca=data["v0"][:,None,:]*tau+.5*data["a0"][:,None,:]*tau*tau)
    report=dict(samples=len(predicted),source_episodes=len(data["sources"]),methods={},
                input_mode=FEATURE_VERSION,
                note="Overlapping windows and four views are correlated; no independent-window confidence claims.")
    for name,value in baselines.items():
        errors=np.linalg.norm(value-data["y"],axis=-1)
        groups={"all":np.ones(len(errors),bool)}
        groups.update({f"level_{i+1}":data["level"]==i for i in range(5)})
        groups.update({key:data[key] for key in ("near","turning","sensed_switch","future_contact")})
        metrics={key:error_summary(errors[selected],data["mask"][selected])
                 for key,selected in groups.items() if selected.any()}
        episode_means=[]
        for i in range(len(data["sources"])):
            selected=data["trace_id"]==i
            episode_means.append(float(errors[selected][data["mask"][selected]].mean()))
        metrics["episode_macro_ade_m"]=float(np.mean(episode_means))
        report["methods"][name]=metrics
    return report

def train(run,epochs=100,batch=256):
    run=Path(run)
    p=json.loads((run/"protocol.json").read_text())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    seed=20261019
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the verified server GPU")
    data=assemble(run,p,"train")
    validation=assemble(run,p,"validation")
    write_json(run/"dataset_summary.json",dict(
        train=dict(windows=len(data["x"]),sources=data["sources"]),
        validation=dict(windows=len(validation["x"]),sources=validation["sources"]),
        feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,
        future_seconds=TAUS.tolist()))
    training_bytes=sum(data[k].nbytes for k in ("x","y","mask","v0"))
    free_bytes,_=torch.cuda.mem_get_info(device)
    storage_device=device if training_bytes<free_bytes-2*1024**3 else torch.device("cpu")
    gpu=tensors(data,storage_device)
    print(json.dumps(dict(training_storage=str(storage_device),training_bytes=training_bytes,free_gpu_bytes=free_bytes)),flush=True)
    val_gpu=tensors(validation,torch.device("cpu"))
    model=ContextLSTM().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    config=dict(feature_version=FEATURE_VERSION,feature_dim=71,history_frames=9,
                raw_position_frames=11,dt=.5,future_steps=12,hidden_size=128,
                layers=2,dropout=.1,batch_size=batch,learning_rate=.001,
                max_epochs=epochs,patience=10,seed=seed,huber_delta=.1,
                velocity_loss_weight=.2,parameters=sum(x.numel() for x in model.parameters()),
                protocol_sha256=sha256(run/"protocol.json"),data_residency="cpu_memmap_with_optional_gpu_training_cache",training_storage=str(storage_device),shuffle_device=str(storage_device))
    write_json(run/"training_config.json",config)
    best=float("inf")
    bad=0
    curve=[]
    started=time.time()
    (run/"model").mkdir(exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train()
        order=torch.randperm(len(gpu["x"]),device=storage_device)
        total_loss=0.
        batches=0
        epoch_started=time.time()
        for indices in order.split(batch):
            part={key:value[indices].to(device) for key,value in gpu.items()}
            output=decode(model(part["x"]),part["v0"])
            loss=supervised_loss(output,part["y"],part["mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total_loss+=float(loss.detach())
            batches+=1
        model.eval()
        score=validation_score(model,val_gpu)
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        improved=score<best-1e-5
        if improved:
            best=score
            bad=0
            torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,
                            validation_1to3s_error_m=score),
                       run/"model/best.pt")
        else:
            bad+=1
        row=dict(epoch=epoch,train_loss=total_loss/batches,
                 validation_1to3s_error_m=score,best_validation_error_m=best,
                 epoch_seconds=time.time()-epoch_started)
        curve.append(row)
        write_json(run/"learning_curve.json",curve)
        write_json(run/"status.json",dict(stage="training",**row))
        print(json.dumps(row),flush=True)
        if bad>=10:
            break
    del gpu,val_gpu,data,validation
    torch.cuda.empty_cache()
    best_state=torch.load(run/"model/best.pt",map_location=device,weights_only=True)
    model.load_state_dict(best_state["model"])
    model.eval()
    summaries={}
    for split in ("validation","test"):
        samples=assemble(run,p,split)
        summaries[split]=evaluate(model,samples,device)
        write_json(run/(split+"_metrics.json"),summaries[split])
        write_json(run/(split+"_sources.json"),samples["sources"])
        if split=="test":
            rows=[]
            for method,values in summaries[split]["methods"].items():
                for group,result in values.items():
                    if not isinstance(result,dict):
                        continue
                    for horizon in result["horizons"]:
                        rows.append(dict(method=method,group=group,**horizon))
            with (run/"test_horizon_errors.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        del samples
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),dpi=150)
    axes[0].plot([r["epoch"] for r in curve],[r["train_loss"] for r in curve])
    axes[0].set(xlabel="Epoch",ylabel="Training supervised loss")
    axes[1].plot([r["epoch"] for r in curve],[r["validation_1to3s_error_m"] for r in curve])
    axes[1].set(xlabel="Epoch",ylabel="Validation 1-3 s position error (m)")
    fig.tight_layout()
    fig.savefig(run/"learning_curve.png")
    plt.close(fig)
    summary=dict(stage="trained_and_evaluated",epochs=len(curve),
                 best_epoch=best_state["epoch"],best_validation_error_m=best,
                 elapsed_seconds=time.time()-started,checkpoint_sha256=sha256(run/"model/best.pt"),
                 test_ade_m={k:v["all"]["ade_m"] for k,v in summaries["test"]["methods"].items()})
    write_json(run/"training_summary.json",summary)
    write_json(run/"status.json",summary)
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=100)
    args=parser.parse_args()
    train(args.run,args.epochs)
