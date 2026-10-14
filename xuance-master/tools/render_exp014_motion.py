"""Replay the frozen 014 policy and render honest motion examples for five levels."""
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from tools.render_baseline_success_videos import RecordedEnv, native_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reviews/exp014_motion_v1'
MODEL = ROOT / 'outputs/reviews/exp014_final_v1/final.pth'
CONFIG = ROOT / 'experiments/baseline_exp014.yaml'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def rollout(config, level, seed, model):
    options = dict(config, env_seed=seed, curriculum_start_level=level, render=False,
                   terminate_on_collision=False, terminate_on_building_collision=False,
                   terminate_on_boundary_collision=False)
    env = RecordedEnv(Namespace(**options))
    replay = UAVPursuitCoverage3DEnv(Namespace(**options))
    try:
        obs, _ = env.reset()
        original, _ = replay.reset()
        for agent in env.agents:
            np.testing.assert_array_equal(obs[agent], original[agent])
        agents = list(env.agents)
        collisions = {'building': 0, 'boundary': 0, 'uav': 0}
        first_contact = None
        actions = []
        for step in range(env.max_episode_steps):
            tensor = torch.as_tensor(np.stack([obs[a] for a in agents]), dtype=torch.float32)
            with torch.inference_mode():
                _, values = model({agents[0]: tensor}, agent_ids=torch.eye(len(agents)))
            action = values[agents[0]].numpy()
            assert np.isfinite(action).all() and float(np.max(np.abs(action))) <= 1.000001
            actions.append(action.copy())
            command = {a: action[i].copy() for i, a in enumerate(agents)}
            obs, reward, terminated, truncated, info = env.step(command)
            obs2, reward2, term2, trunc2, info2 = replay.step(command)
            np.testing.assert_array_equal(env.uav_positions, replay.uav_positions)
            np.testing.assert_array_equal(env.target_position, replay.target_position)
            for a in agents:
                np.testing.assert_array_equal(obs[a], obs2[a])
            assert reward == reward2 and terminated == term2 and truncated == trunc2
            for field in ('is_success', 'is_collision', 'is_building_collision',
                          'is_boundary_collision', 'capture_time', 'capture_agent'):
                assert info[field] == info2[field]
            for name in collisions:
                collisions[name] += int(info['is_' + name + '_collision'])
            if first_contact is None and info['is_collision']:
                first_contact = float(env.physics_trace[-1]['t'])
            np.testing.assert_allclose(env.physics_trace[-1]['pursuers'], env.uav_positions, atol=1e-5)
            np.testing.assert_allclose(env.physics_trace[-1]['evader'], env.target_position, atol=1e-5)
            if all(terminated.values()) or truncated:
                break
        success = bool(info['is_success'])
        contact = sum(collisions.values()) > 0
        trace = env.physics_trace
        data = dict(times=np.asarray([p['t'] for p in trace]),
                    pursuers=np.stack([p['pursuers'] for p in trace]),
                    evader=np.stack([p['evader'] for p in trace]),
                    hold=np.stack([p['hold'] for p in trace]),
                    buildings=env.buildings.copy(), actions=np.stack(actions))
        report = dict(level=level+1, seed=seed, steps=step+1, seconds=float(data['times'][-1]),
            success=success, safe_success=success and not contact, collision=contact,
            collision_decision_steps=collisions, first_contact_seconds=first_contact,
            timed_out=bool(truncated), capture_agent=info['capture_agent'],
            capture_type=info['capture_type'], capture_time=info['capture_time'],
            capture_radius=float(env.catch_radius), capture_hold_seconds=float(env.capture_hold_required),
            target_speed_range=[float(env.target_min_speed),float(env.target_max_speed)],
            building_count=int(len(env.buildings)), physics_samples=len(trace),
            exact_unmodified_simulator_replay=True, exploration_noise=False,
            collision_termination=False, full_episode=True)
        return data, report
    finally:
        env.close()
        replay.close()


def render(data, report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle, Circle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from tools.render_baseline_success_videos import building_faces

    t, p, e = data['times'], data['pursuers'], data['evader']
    colors = ['#2679d8', '#16a06b', '#914bd2', '#e59c1b', '#e64b5d']
    all_points = np.concatenate([p.reshape(-1, 3), e])
    lo = np.maximum(all_points.min(axis=0)-[60,60,30],[0,0,50])
    hi = np.minimum(all_points.max(axis=0)+[60,60,30],[1000,1000,350])
    fig = plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f7f9fc')
    grid = fig.add_gridspec(2,2,width_ratios=[1.7,1],height_ratios=[1.35,1],
                           left=.03,right=.985,bottom=.08,top=.82,wspace=.12,hspace=.34)
    ax3 = fig.add_subplot(grid[:,0],projection='3d')
    axxy = fig.add_subplot(grid[0,1])
    axd = fig.add_subplot(grid[1,1])
    ax3.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(50,350),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax3.set_box_aspect((max(1,hi[0]-lo[0]),max(1,hi[1]-lo[1]),300))
    ax3.view_init(elev=28,azim=-55)
    ax3.set_title('3D flight paths',fontsize=11)
    for b in data['buildings']:
        x0,x1,y0,y1,top = b
        axxy.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#b5c2d1',
                                 edgecolor='#738397',linewidth=.4,alpha=.8))
        if x1>=lo[0] and x0<=hi[0] and y1>=lo[1] and y0<=hi[1] and top>50:
            ax3.add_collection3d(Poly3DCollection(building_faces(b,50),facecolors='#9eafc0',
                                                   edgecolors='#738397',linewidths=.3,alpha=.25))
    axxy.add_patch(Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],fill=False,
                             edgecolor='#c85958',linestyle='--',linewidth=1))
    axxy.set(xlim=(0,1000),ylim=(0,1000),aspect='equal',title='City map (top view)')
    axxy.grid(alpha=.15)
    radius=report['capture_radius']
    distances=np.linalg.norm(p-e[:,None,:],axis=-1)
    axd.set(xlim=(0,max(1,t[-1])),ylim=(0,max(100,distances.max()*1.05)),
            xlabel='Simulated time (s)',ylabel='3D distance to evader (m)',title='Pursuer distances')
    axd.axhline(radius,color='#657488',linestyle='--',linewidth=1,label=f'Hold radius {radius:g} m')
    if report['first_contact_seconds'] is not None:
        axd.axvline(report['first_contact_seconds'],color='#c62828',linestyle=':',lw=1,label='First collision')
    axd.grid(alpha=.2)
    axd.legend(fontsize=8)
    trace3=[]; dots3=[]; trace2=[]; dots2=[]; distance_lines=[]
    for i,color in enumerate(colors):
        name=f'P{i+1}' if i<4 else 'Evader'
        trace3.append(ax3.plot([],[],[],color=color,lw=1.7)[0])
        dots3.append(ax3.plot([],[],[],'o',color=color,markersize=7)[0])
        trace2.append(axxy.plot([],[],color=color,lw=1.2)[0])
        dots2.append(axxy.plot([],[],'o',color=color,markersize=5,label=name)[0])
        if i<4: distance_lines.append(axd.plot([],[],color=color,lw=1.2)[0])
    axxy.legend(loc='upper center',bbox_to_anchor=(.5,1.36),ncol=5,fontsize=8,frameon=False)
    circle=Circle(e[0,:2],radius,fill=False,edgecolor='#e64b5d',lw=.8,alpha=.5)
    axxy.add_patch(circle)
    cursor=axd.axvline(0,color='#9daab6',lw=1)
    first=report['first_contact_seconds']
    fig.text(.035,.963,f"Experiment 014 final policy | Level {report['level']} | {report['building_count']} buildings | evader {report['target_speed_range'][0]:g}-{report['target_speed_range'][1]:g} m/s",fontsize=15,weight='bold')
    fig.text(.035,.919,f"Seed {report['seed']} | no exploration | same model across all levels | 0.1 s physics / 0.5 s decisions | 8x playback",fontsize=10,color='#46566b')
    outcome=('SAFE CAPTURE' if report['safe_success'] else 'CAPTURE AFTER COLLISION' if report['success'] else 'TIMEOUT / NO CAPTURE')
    fig.text(.035,.881,f"Actual outcome: {outcome} | full-episode collision: {report['collision']} | success criterion: one pursuer holds within radius for 2 s",fontsize=10,color='#46566b')
    status=fig.text(.035,.839,'',fontsize=11,weight='bold')
    footer=fig.text(.035,.024,'',fontsize=9,color='#46566b')
    def draw(index):
        for i in range(5):
            points=p[:index+1,i] if i<4 else e[:index+1]
            q=points[-1]
            trace3[i].set_data(points[:,0],points[:,1]); trace3[i].set_3d_properties(points[:,2])
            dots3[i].set_data([q[0]],[q[1]]); dots3[i].set_3d_properties([q[2]])
            trace2[i].set_data(points[:,0],points[:,1]); dots2[i].set_data([q[0]],[q[1]])
            if i<4: distance_lines[i].set_data(t[:index+1],distances[:index+1,i])
        circle.center=e[index,:2]
        cursor.set_xdata([t[index],t[index]])
        contacted=first is not None and t[index]>=first-1e-6
        label=f"t={t[index]:.1f} s | "
        if index==len(t)-1:
            label+=outcome
        elif contacted:
            label+='Collision has occurred'
        else:
            label+='No collision so far'
        status.set_text(label)
        status.set_color('#c62828' if contacted else '#11774d')
        footer.set_text('Distance: '+'   '.join(f'P{i+1} {d:.1f} m' for i,d in enumerate(distances[index])))
    for name,index in [('start',0),('middle',len(t)//2),('end',len(t)-1)]:
        draw(index)
        fig.savefig(output/f'{name}.png',dpi=100,facecolor=fig.get_facecolor())
    indices=list(range(0,len(t),8))
    if indices[-1]!=len(t)-1: indices.append(len(t)-1)
    writer=FFMpegWriter(fps=10,codec='libx264',bitrate=1500,
                        extra_args=['-pix_fmt','yuv420p','-preset','veryfast','-threads','1','-movflags','+faststart'])
    with writer.saving(fig,str(output/'motion.mp4'),dpi=100):
        for _ in range(10): draw(0); writer.grab_frame()
        for ordinal,index in enumerate(indices):
            draw(index); writer.grab_frame()
            if ordinal%100==0: print('FRAME',report['level'],ordinal,len(indices),flush=True)
        for _ in range(20): draw(len(t)-1); writer.grab_frame()
    plt.close(fig)
    report.update(video_sha256=digest(output/'motion.mp4'),video_frames=len(indices)+30,
                  video_fps=10,playback_speed=8,video_width=1280,video_height=720)


def main():
    assert not OUT.exists(), 'Preserve the existing render directory'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='feat-coverage-reaward'
    manifest=json.loads((MODEL.parent/'manifest.json').read_text())
    assert digest(MODEL)==manifest['model_sha256'] and digest(CONFIG)==manifest['config_sha256']
    final_state=json.loads((MODEL.parent/'process.json').read_text())
    assert final_state['status']=='completed' and final_state['completed_levels']==[1,2,3,4,5]
    config=yaml.safe_load(CONFIG.read_text())
    config['buildings_dir']=str(ROOT/'xuance/environment/multi_agent_env/buildings')
    torch.set_num_threads(1)
    # Policy shape and scene are common to all five levels.
    initial=RecordedEnv(Namespace(**dict(config,env_seed=72001,curriculum_start_level=0)))
    try:
        policy,_,_=native_policy(config,initial,MODEL)
    finally:
        initial.close()
    OUT.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        model_sha256=manifest['model_sha256'],config_sha256=manifest['config_sha256'],levels=[])
    save(OUT/'process.json',state)
    try:
        for level in range(5):
            folder=OUT/f'level_{level+1}'
            folder.mkdir()
            selected=None
            attempts=[]
            for offset in range(30 if level<3 else 20):
                seed=72001+offset
                data,report=rollout(config,level,seed,policy)
                attempts.append({k:report[k] for k in ('seed','success','safe_success','collision','steps')})
                print('ROLLOUT',level+1,json.dumps(attempts[-1]),flush=True)
                if (level<3 and report['safe_success']) or (level>=3 and not report['success']):
                    selected=(data,report)
                    break
            if selected is None:
                raise RuntimeError(f'No matching episode for level {level+1} in {len(attempts)} attempts')
            data,report=selected
            report.update(experiment='014',model_sha256=manifest['model_sha256'],
                          config_sha256=manifest['config_sha256'],selection_rule='first safe capture' if level<3 else 'first no-capture episode',
                          attempts=attempts,formal_100_episode_metrics=json.loads((MODEL.parent/f'final_level{level+1}.json').read_text())['metrics'],
                          illustrative_only=True)
            np.savez_compressed(folder/'trajectory.npz',**data)
            report['trajectory_sha256']=digest(folder/'trajectory.npz')
            save(folder/'manifest.json',report)
            render(data,report,folder)
            save(folder/'manifest.json',report)
            state['levels'].append(dict(level=level+1,seed=report['seed'],safe_success=report['safe_success'],
                                        collision=report['collision'],video_sha256=report['video_sha256']))
            save(OUT/'process.json',state)
        state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
        save(OUT/'process.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=datetime.now(timezone.utc).isoformat())
        save(OUT/'process.json',state)
        raise


if __name__=='__main__':
    main()
