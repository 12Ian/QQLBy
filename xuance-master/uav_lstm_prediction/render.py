"""Render fixed held-out examples: history, LSTM, CV and scoring-only future."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
from .core import (ContextLSTM,TAUS,DT,load_trace,decision_grid,make_windows,
                   decode,write_json,sha256)
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
COLORS=dict(history="#1f4e79",lstm="#b5229d",cv="#777777",truth="#18864b",target="#d92344")

def building_faces(b):
    x0,x1,y0,y1,z1=map(float,b)
    z0=0.
    return [[(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
            [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
            [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]]

def render_case(payload):
    run=Path(payload["run"])
    job=payload["job"]
    torch.set_num_threads(1)
    model=ContextLSTM()
    saved=torch.load(run/"model/best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    raw=load_trace(run/job["path"])
    trace=decision_grid(raw)
    samples=make_windows(raw)
    selected=np.arange(0,len(samples["x"]),4)
    with torch.inference_mode():
        relative=decode(model(torch.from_numpy(samples["x"][selected])),
                        torch.from_numpy(samples["v0"][selected])).numpy()
    predictions={int(end):relative[i] for i,end in enumerate(samples["ends"][selected])}
    estimates={int(end):samples["v0"][selected[i]] for i,end in enumerate(samples["ends"][selected])}
    folder=run/"videos"/f"level_{job['level']+1}_seed_{job['seed']}"
    folder.mkdir(parents=True,exist_ok=True)
    original=folder/"prediction_1x.mp4"
    fast=folder/"prediction_10x.mp4"
    buildings=trace["buildings"]
    target=trace["target"]
    times=trace["times"]
    fig=plt.figure(figsize=(19.2,10.8),dpi=100,facecolor="#f5f7fa")
    gs=fig.add_gridspec(2,3,width_ratios=[1.3,1.05,.9],hspace=.28,wspace=.22,
                       left=.035,right=.98,bottom=.10,top=.85)
    ax3=fig.add_subplot(gs[:,0],projection="3d")
    ax2=fig.add_subplot(gs[0,1])
    axerr=fig.add_subplot(gs[1,1])
    axmap=fig.add_subplot(gs[0,2])
    axtext=fig.add_subplot(gs[1,2])
    axtext.axis("off")
    faces=[face for building in buildings for face in building_faces(building)]
    ax3.add_collection3d(Poly3DCollection(faces,facecolors="#a5b6c8",edgecolors="#7890a8",
                                         linewidths=.25,alpha=.13))
    ax3.set(xlim=(0,1000),ylim=(0,1000),zlim=(50,350),xlabel="X (m)",ylabel="Y (m)",
            zlabel="Z (m)",title="3-D scene / observed execution")
    ax3.set_box_aspect((1,1,.55))
    ax3.view_init(elev=24,azim=-62)
    for ax in [ax2,axmap]:
        for b in buildings:
            ax.add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],
                                   facecolor="#d5dde6",edgecolor="#9dafc0",linewidth=.4,alpha=.7))
        ax.set_aspect("equal",adjustable="box")
        ax.grid(alpha=.2)
    ax2.set(title="Local forecast top view / adaptive zoom",xlabel="X (m)",ylabel="Y (m)")
    axmap.set(xlim=(0,1000),ylim=(0,1000),title="Full map / past target path")
    axerr.set(xlim=(.5,6),ylim=(0,80),xlabel="Forecast horizon (s)",
              ylabel="3-D error (m)",title="Current forecast vs observed future")
    axerr.grid(alpha=.2)
    handles={}
    for name in ["history","lstm","cv","truth"]:
        color=COLORS[name]
        style="--" if name in ("lstm","cv") else "-"
        handles[name]=dict(line3=ax3.plot([],[],[],style,color=color,lw=2.4)[0],
                           line2=ax2.plot([],[],style,color=color,lw=2.3,zorder={"truth":2,"cv":3,"history":3,"lstm":4}[name])[0])
    current3=ax3.plot([],[],[],"o",color=COLORS["target"],ms=7)[0]
    current2=ax2.plot([],[],"o",color=COLORS["target"],ms=7)[0]
    pursuer3=ax3.plot([],[],[],"o",color="#137f96",alpha=.65,ms=4)[0]
    pursuer2=ax2.plot([],[],"o",color="#137f96",alpha=.65,ms=4)[0]
    pastmap=axmap.plot([],[],color=COLORS["history"],lw=1.7)[0]
    currentmap=axmap.plot([],[],"o",color=COLORS["target"],ms=4)[0]
    errl=axerr.plot([],[],color=COLORS["lstm"],marker="o",ms=3,label="LSTM")[0]
    errc=axerr.plot([],[],color=COLORS["cv"],ls="--",marker="o",ms=3,label="CV")[0]
    axerr.legend(loc="upper left")
    subtitle=fig.text(.5,.918,"",ha="center",fontsize=14,color="#33475b")
    fig.text(.5,.965,"LSTM TARGET TRAJECTORY FORECAST | HELD-OUT SCENE",
             ha="center",fontsize=21,weight="bold")
    legends=[Line2D([0],[0],color=COLORS[n],lw=2.5,
                     ls="--" if n in ("lstm","cv") else "-") for n in
                     ("history","lstm","cv","truth")]
    fig.legend(legends,["Past 4 s (observed)","LSTM next 6 s","Constant velocity",
                       "Actual next 6 s (scoring only)"],
               loc="upper center",bbox_to_anchor=(.5,.9),ncol=4,frameon=False,fontsize=12)
    text=axtext.text(0,1,"",va="top",fontfamily="monospace",fontsize=12,color="#203449",
                     linespacing=1.6,transform=axtext.transAxes)
    fig.text(.5,.018,"Future truth is used only for drawing and scoring. Forecast uses past observations only. "
             "Predictor not connected to MATD3.",ha="center",fontsize=11,color="#52677e")
    cmd=[FFMPEG,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-vcodec","rawvideo",
         "-pix_fmt","rgb24","-s","1920x1080","-r","2","-i","-","-an","-c:v","libx264",
         "-preset","veryfast","-crf","24","-pix_fmt","yuv420p","-vf","fps=24",
         "-threads","2","-movflags","+faststart",str(original)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    frames=[]
    for raw_index,current_time in enumerate(raw["times"]):
        nearest=int(round(float(current_time)/DT))
        regular=nearest<len(times) and abs(float(times[nearest])-float(current_time))<1e-6
        if not regular or raw_index==len(raw["times"])-1:
            index=len(times)-1
            origin=raw["target"][raw_index]
            history=raw["target"][max(0,raw_index-8):raw_index+1]
            future=np.empty((0,3),np.float32)
            lstm=np.empty((0,3),np.float32)
            cv=np.empty((0,3),np.float32)
            status="Episode ended / no post-terminal forecast"
        else:
            index=nearest
            origin=target[index]
            history=target[max(0,index-8):index+1]
            future=target[index+1:min(index+13,len(target))]
            vel=(target[index]-target[index-1])/DT if index else np.zeros(3)
            cv=origin+TAUS[:,None]*vel
            lstm=origin+predictions[index] if index in predictions else np.empty((0,3),np.float32)
            status="LSTM / complete 5 s raw history" if index in predictions else "Warm-up or final frame / CV only"
        paths=dict(history=history,lstm=np.vstack((origin,lstm)) if len(lstm) else lstm,
                   cv=np.vstack((origin,cv)) if len(cv) else cv,
                   truth=np.vstack((origin,future)) if len(future) else future)
        for name,path in paths.items():
            line3,line2=handles[name]["line3"],handles[name]["line2"]
            line3.set_data_3d(path[:,0],path[:,1],path[:,2])
            line2.set_data(path[:,0],path[:,1])
        current3.set_data_3d([origin[0]],[origin[1]],[origin[2]])
        current2.set_data([origin[0]],[origin[1]])
        pp=raw["pursuers"][raw_index]
        pursuer3.set_data_3d(pp[:,0],pp[:,1],pp[:,2])
        pursuer2.set_data(pp[:,0],pp[:,1])
        pastmap.set_data(raw["target"][:raw_index+1,0],raw["target"][:raw_index+1,1])
        currentmap.set_data([origin[0]],[origin[1]])
        extent=max(float(np.linalg.norm(cv[:,:2]-origin[:2],axis=1).max()) if len(cv) else 0.,
                   float(np.linalg.norm(lstm[:,:2]-origin[:2],axis=1).max()) if len(lstm) else 0.)
        zoom=max(20.,min(120.,extent*1.3))
        ax2.set_xlim(float(origin[0])-zoom,float(origin[0])+zoom)
        ax2.set_ylim(float(origin[1])-zoom,float(origin[1])+zoom)
        count=len(future)
        le=np.linalg.norm(lstm[:count]-future,axis=1) if len(lstm) and count else np.array([])
        ce=np.linalg.norm(cv[:count]-future,axis=1) if len(cv) and count else np.array([])
        errl.set_data(TAUS[:len(le)],le)
        errc.set_data(TAUS[:len(ce)],ce)
        maximum=max(float(le.max()) if len(le) else 0,float(ce.max()) if len(ce) else 0,1)
        axerr.set_ylim(0,max(1,maximum*1.15))
        distance=float(np.linalg.norm(pp-origin,axis=1).min())
        speed=float(np.linalg.norm(raw["target_velocity"][raw_index]))
        def show_error(k):
            return f"{le[k-1]:5.2f} m" if len(le)>=k else "  N/A"
        text.set_text(f"Level {job['level']+1} | seed {job['seed']}\n"
                      f"Time          {current_time:6.1f} s\n"
                      f"Target speed  {speed:6.2f} m/s\n"
                      f"Nearest UAV   {distance:6.1f} m\n\n"
                      f"LSTM error @ 1 s: {show_error(2)}\n"
                      f"LSTM error @ 3 s: {show_error(6)}\n"
                      f"LSTM error @ 6 s: {show_error(12)}\n\n"
                      f"Prediction view: P1\n"
                      f"Checkpoint epoch: {saved['epoch']}\n"
                      f"Forecast refresh: 0.5 s\n"
                      f"Future available: {count*.5:.1f} s")
        subtitle.set_text(f"Level {job['level']+1} | target speed setting "
                          f"{float(raw['min_speed']):g}-{float(raw['max_speed']):g} m/s | {status}")
        fig.canvas.draw()
        rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
        process.stdin.write(rgb.tobytes())
        if raw_index==len(raw["times"])//2:
            fig.savefig(folder/"preview.png",dpi=100)
        if raw_index%100==0:
            write_json(folder/"progress.json",dict(frames=raw_index,total=len(raw["times"])))
        if payload.get("max_frames") and raw_index+1>=payload["max_frames"]:
            break
    process.stdin.close()
    errors=process.stderr.read().decode()
    if process.wait()!=0:
        raise RuntimeError(errors)
    plt.close(fig)
    subprocess.run([FFMPEG,"-v","error","-y","-i",str(original),
                    "-vf","setpts=PTS/10,fps=24","-an","-c:v","libx264","-preset","veryfast",
                    "-crf","24","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",
                    str(fast)],check=True)
    for path in [original,fast]:
        subprocess.run([FFMPEG,"-v","error","-threads","2","-i",str(path),
                        "-f","null","-"],check=True,stdout=subprocess.DEVNULL)
    manifest=dict(**job,original=str(original.relative_to(run)),fast=str(fast.relative_to(run)),
                  trajectory_sha256=sha256(run/job["path"]),
                  checkpoint_sha256=sha256(run/"model/best.pt"),
                  video_sha256=sha256(original),fast_sha256=sha256(fast),
                  resolution=[1920,1080],fps=24,source_refresh_seconds=.5,
                  prediction_view=0,selection="first two sorted test seeds per level, frozen before training",
                  render_version="qa_v2_adaptive_zoom",decoded_verified=True,source_seconds=float(raw["times"][-1]))
    write_json(folder/"manifest.json",manifest)
    return manifest

def render_all(run,workers=2):
    run=Path(run).resolve()
    p=json.loads((run/"protocol.json").read_text())
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
        tasks=[pool.submit(render_case,dict(run=str(run),job=job)) for job in p["video_cases"]]
        for future in as_completed(tasks):
            results.append(future.result())
            write_json(run/"status.json",dict(stage="rendering",completed=len(results),total=10))
            print("RENDER_COMPLETE",len(results),flush=True)
    results.sort(key=lambda x:(x["level"],x["seed"]))
    playlist=run/"videos/playlist.txt"
    playlist.write_text("".join("file '"+str((run/r["fast"]).resolve())+"'\n" for r in results))
    overview=run/"videos/all_levels_prediction_10x.mp4"
    subprocess.run([FFMPEG,"-v","error","-y","-f","concat","-safe","0","-i",str(playlist),
                    "-c","copy","-movflags","+faststart",str(overview)],check=True)
    subprocess.run([FFMPEG,"-v","error","-i",str(overview),"-f","null","-"],
                   check=True,stdout=subprocess.DEVNULL)
    from PIL import Image,ImageOps,ImageDraw
    sheet=Image.new("RGB",(960*2,540*5),"white")
    for k,r in enumerate(results):
        path=run/r["original"]
        preview=path.parent/"preview.png"
        image=Image.open(preview).convert("RGB").resize((960,540))
        sheet.paste(image,((k%2)*960,(k//2)*540))
    sheet.save(run/"videos/fixed_test_previews.jpg",quality=90)
    write_json(run/"videos/video_manifest.json",dict(cases=results,overview=str(overview.relative_to(run)),
                overview_sha256=sha256(overview),all_decoded_verified=True))
    print("ALL_VIDEOS_COMPLETE",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    render_all(args.run,args.workers)
