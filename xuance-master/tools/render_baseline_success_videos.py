"""Render genuine, deterministic baseline successes with unchanged simulator dynamics.

The recording subclass observes the existing capture calculation once per physics
substep. A second, unmodified environment replays every action to check that the
recording hook has not changed the rollout. Videos are illustrative successes,
not estimates of success rate. No training process or checkpoint is modified.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from xuance.torch.policies.deterministic_marl import MATD3_Policy
from xuance.torch.representations import Basic_Identical


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class RecordedEnv(UAVPursuitCoverage3DEnv):
    def reset(self):
        result = super().reset()
        self.physics_trace = [dict(t=0.0, pursuers=self.uav_positions.copy(),
                                   evader=self.target_position.copy(),
                                   hold=self.capture_hold_times.copy())]
        return result

    def _capture_event(self, old_relative, new_relative, dt):
        event = super()._capture_event(old_relative, new_relative, dt)
        previous = self.physics_trace[-1]
        fraction = event[2] if event is not None else 1.0
        # The base environment uses exactly this interpolation at a capture.
        positions = (self.uav_positions.copy() if event is None else
                     previous["pursuers"] + (self.uav_positions - previous["pursuers"]) * fraction)
        target = (self.target_position.copy() if event is None else
                  previous["evader"] + (self.target_position - previous["evader"]) * fraction)
        self.physics_trace.append(dict(t=previous["t"] + dt * fraction,
                                       pursuers=positions.copy(), evader=target.copy(),
                                       hold=self.capture_hold_times.copy()))
        return event


def native_policy(config, env, checkpoint):
    assert config["agent"] == "MATD3" and config["representation"] == "Basic_Identical"
    assert config["activation"] == "leaky_relu" and config["activation_action"] == "tanh"
    assert config["use_parameter_sharing"] and not config.get("use_rnn", False)
    assert not config.get("use_graph_module", False) and not config.get("use_obstacle_gat", False)
    assert not config.get("normalize"), "This renderer requires the current unnormalized MLP baseline"
    key = env.agents[0]
    obs_size = env.observation_space[key].shape[0]
    actor_rep = torch.nn.ModuleDict({key: Basic_Identical((obs_size,), device="cpu")})
    critic_rep = torch.nn.ModuleDict({key: Basic_Identical(((obs_size + 3) * env.num_agents,), device="cpu")})
    policy = MATD3_Policy(action_space=env.action_space, n_agents=env.num_agents,
                         actor_representation=actor_rep, critic_representation=critic_rep,
                         actor_hidden_size=config["actor_hidden_size"], critic_hidden_size=config["critic_hidden_size"],
                         activation=torch.nn.LeakyReLU, activation_action=torch.nn.Tanh, device="cpu",
                         use_parameter_sharing=True, model_keys=[key], use_rnn=False, rnn=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["policy"]
    assert all(torch.isfinite(value).all().item() for value in state.values())
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, key, torch.eye(env.num_agents)


def freeze_stage(root, experiment, level, output):
    result = root / f"outputs/results/matd3/baseline_exp{experiment}"
    source = result / "best_model" / f"best_stage_{level}.pth"
    # CSV is written before the checkpoint. Stable file metadata, a successful
    # load and a second read protect against a live checkpoint write.
    for attempt in range(4):
        before = source.stat()
        data = source.read_bytes()
        after = source.stat()
        rows = list(csv.DictReader((result / "curriculum_eval.csv").open()))
        selected = [row for row in rows if row["phase"] == "current" and int(row["level"]) == level]
        best = max(selected, key=lambda row: (float(row["safe_success_rate"]),
                                              -float(row["collision_rate"]), float(row["mean_return"])))
        try:
            torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            stable = (before.st_mtime_ns == after.st_mtime_ns and before.st_size == len(data)
                      and source.read_bytes() == data)
        except (EOFError, RuntimeError, OSError):
            stable = False
        if stable:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Could not read a stable stage checkpoint")
    destination = output / "model.pth"
    destination.write_bytes(data)
    config = json.loads((result / "config.json").read_text())
    save_json(output / "training_config.json", config)
    return config, dict(experiment=experiment, source_checkpoint=str(source.resolve()),
                        frozen_checkpoint=str(destination.resolve()), checkpoint_sha256=sha256(destination),
                        source_mtime_utc=datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat(),
                        checkpoint_eval=best, frozen_utc=datetime.now(timezone.utc).isoformat())


def find_success(config, checkpoint, level, seed_start, max_attempts, output):
    attempts = []
    for attempt in range(max_attempts):
        seed = seed_start + attempt
        runtime = dict(config, env_seed=seed, curriculum_start_level=level, render=False,
                       terminate_on_collision=False, terminate_on_building_collision=False,
                       terminate_on_boundary_collision=False)
        env = RecordedEnv(Namespace(**runtime))
        replay = UAVPursuitCoverage3DEnv(Namespace(**runtime))
        try:
            obs, _ = env.reset()
            reference_obs, _ = replay.reset()
            for key in env.agents:
                np.testing.assert_array_equal(obs[key], reference_obs[key])
            policy, key, ids = native_policy(runtime, env, checkpoint)
            any_collision = False
            actions_log = []
            for step in range(env.max_episode_steps):
                observations = torch.as_tensor(np.stack([obs[a] for a in env.agents]), dtype=torch.float32)
                with torch.inference_mode():
                    _, action_dict = policy({key: observations}, agent_ids=ids)
                actions = action_dict[key].numpy()
                assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.000001
                actions_log.append(actions.copy())
                command = {a: actions[index].copy() for index, a in enumerate(env.agents)}
                obs, rewards, terminated, truncated, info = env.step(command)
                other_obs, other_reward, other_term, other_trunc, other_info = replay.step(command)
                np.testing.assert_array_equal(env.uav_positions, replay.uav_positions)
                np.testing.assert_array_equal(env.target_position, replay.target_position)
                for agent in env.agents:
                    np.testing.assert_array_equal(obs[agent], other_obs[agent])
                assert rewards == other_reward and terminated == other_term and truncated == other_trunc
                for field in ["is_collision", "is_success", "capture_type", "capture_agent", "capture_time"]:
                    assert info[field] == other_info[field]
                np.testing.assert_allclose(env.physics_trace[-1]["pursuers"], env.uav_positions, atol=1e-5)
                np.testing.assert_allclose(env.physics_trace[-1]["evader"], env.target_position, atol=1e-5)
                any_collision |= bool(info["is_collision"])
                if all(terminated.values()) or truncated:
                    break
            safe_success = bool(info["is_success"] and not any_collision)
            item = dict(attempt=attempt + 1, env_seed=seed, success=bool(info["is_success"]),
                        collision=any_collision, safe_success=safe_success, steps=step + 1,
                        capture_time=info["capture_time"], capture_type=info["capture_type"])
            attempts.append(item)
            save_json(output / "attempts.json", attempts)
            print("VIDEO_ATTEMPT", level + 1, json.dumps(item), flush=True)
            if not safe_success:
                continue
            trace = env.physics_trace
            data = dict(times=np.array([x["t"] for x in trace]),
                        pursuers=np.stack([x["pursuers"] for x in trace]),
                        evader=np.stack([x["evader"] for x in trace]),
                        hold=np.stack([x["hold"] for x in trace]),
                        buildings=env.buildings.copy(), actions=np.stack(actions_log))
            assert abs(data["times"][-1] - info["capture_time"]) < 1e-6
            assert sum(env.episode_contact_events.values()) == 0
            np.savez_compressed(output / "trajectory.npz", **data)
            course = env.get_curriculum_info()
            capture_agent = env.agents.index(info["capture_agent"])
            report = dict(level=level, display_level=level + 1, agents=env.agents,
                          env_seed=seed, selected_attempt=attempt + 1, attempts=attempts,
                          illustrative_success_only=True, exploration_noise=False,
                          collision_termination=False, max_episode_steps=env.max_episode_steps,
                          capture_type=info["capture_type"], capture_agent=info["capture_agent"],
                          capture_agent_index=capture_agent, capture_time=info["capture_time"],
                          success=True, episode_collision=False, native_policy=True,
                          unmodified_environment_replay_identical=True, physics_samples=len(trace),
                          contact_events=env.episode_contact_events, course=course,
                          building_count=len(env.buildings), trajectory_sha256=sha256(output / "trajectory.npz"))
            save_json(output / "rollout_config.json", runtime)
            return data, report
        finally:
            env.close()
            replay.close()
    raise RuntimeError(f"No collision-free success in {max_attempts} attempts at level {level}")


def building_faces(b, z_min):
    x0, x1, y0, y1, top = b
    return [[(x0, y0, z_min), (x1, y0, z_min), (x1, y0, top), (x0, y0, top)],
            [(x1, y0, z_min), (x1, y1, z_min), (x1, y1, top), (x1, y0, top)],
            [(x1, y1, z_min), (x0, y1, z_min), (x0, y1, top), (x1, y1, top)],
            [(x0, y1, z_min), (x0, y0, z_min), (x0, y0, top), (x0, y1, top)],
            [(x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)]]


def render(data, report, output, playback_speed=2.0):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle, Circle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    font_file = subprocess.check_output(["fc-match", "Noto Serif CJK SC", "--format=%{file}"], text=True).strip()
    font_manager.fontManager.addfont(font_file)
    font_name = font_manager.FontProperties(fname=font_file).get_name()
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    t, p, e = data["times"], data["pursuers"], data["evader"]
    c = report["course"]
    radius = float(c["capture_radius"])
    distances = np.linalg.norm(p - e[:, None, :], axis=-1)
    colors = ["#2679d8", "#16a06b", "#914bd2", "#e59c1b"]
    all_points = np.concatenate([p.reshape(-1, 3), e])
    low = np.maximum(all_points.min(axis=0) - [65, 65, 35], [0, 0, 50])
    high = np.minimum(all_points.max(axis=0) + [65, 65, 35], [1000, 1000, 350])
    # Keep enough visible altitude to make rooftop clearance interpretable.
    low[2], high[2] = 50, 350
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f7f9fc")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[1.35, 1],
                           left=.025, right=.985, bottom=.07, top=.80, wspace=.10, hspace=.34)
    ax3 = fig.add_subplot(grid[:, 0], projection="3d", facecolor="#f7f9fc")
    axxy = fig.add_subplot(grid[0, 1], facecolor="white")
    axd = fig.add_subplot(grid[1, 1], facecolor="white")
    ax3.set(xlim=(low[0], high[0]),
            ylim=(low[1], high[1]), zlim=(50, 350), xlabel="X / m", ylabel="Y / m", zlabel="高度 / m")
    ax3.set_box_aspect((high[0] - low[0], high[1] - low[1], 300))
    ax3.view_init(elev=28, azim=-55)
    ax3.set_title("三维局部视角 · 线条为已飞过的真实轨迹", fontsize=11, pad=1)
    for b in data["buildings"]:
        x0, x1, y0, y1, top = b
        shade = "#c6cfda" if top < 150 else "#9eaebf"
        axxy.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=shade,
                                edgecolor="#7b8998", linewidth=.4, alpha=.8))
        if top > 50 and x1 >= low[0] and x0 <= high[0] and y1 >= low[1] and y0 <= high[1]:
            ax3.add_collection3d(Poly3DCollection(building_faces(b, 50), facecolors=shade,
                                                 edgecolors="#7b8998", linewidths=.5, alpha=.33))
    axxy.add_patch(Rectangle(low[:2], high[0] - low[0], high[1] - low[1],
                            fill=False, edgecolor="#c85958", linestyle="--", linewidth=1))
    axxy.set(xlim=(0, 1000), ylim=(0, 1000), aspect="equal", title="整张地图 · 虚线框对应左侧视角")
    axxy.tick_params(labelsize=8)
    axxy.grid(alpha=.18)
    axd.set(xlim=(0, max(1, t[-1])), ylim=(0, max(radius * 1.5, distances.max() * 1.05)),
            xlabel="仿真时间 / s", ylabel="距逃逸机 / m", title="真实三维距离（不是俯视投影距离）")
    axd.axhline(radius, color="#657488", linestyle="--", linewidth=1, label=f"保持半径 {radius:g} m")
    axd.axhline(c["direct_intercept_radius"], color="#ed6774", linestyle=":", linewidth=1)
    axd.grid(alpha=.18)
    distance_cursor = axd.axvline(0, color="#97a4b2", linewidth=1)
    lines3, dots3, linesxy, dotsxy, linesd = [], [], [], [], []
    for i, color in enumerate(colors + ["#e64b5d"]):
        label = f"追捕机 P{i + 1}" if i < len(colors) else "逃逸机 E"
        lines3.append(ax3.plot([], [], [], color=color, linewidth=1.8, alpha=.8)[0])
        dots3.append(ax3.plot([], [], [], "o", color=color, markersize=7 if i < 4 else 8,
                             markeredgecolor="white", markeredgewidth=.8)[0])
        linesxy.append(axxy.plot([], [], color=color, linewidth=1.3, alpha=.8)[0])
        dotsxy.append(axxy.plot([], [], "o", color=color, markersize=4.5, label=label)[0])
        if i < 4:
            linesd.append(axd.plot([], [], color=color, linewidth=1.4)[0])
    axxy.legend(loc="upper center", bbox_to_anchor=(.5, 1.38), ncol=3, fontsize=8,
                frameon=False, columnspacing=1.0, handletextpad=.35)
    angle = np.linspace(0, 2 * np.pi, 48)
    sphere_lines = [ax3.plot([], [], [], color="#e64b5d", linewidth=.7, alpha=.5)[0] for _ in range(3)]
    capture_circle = Circle(e[0, :2], radius=radius, fill=False, edgecolor="#e64b5d", alpha=.55, linewidth=.8)
    axxy.add_patch(capture_circle)
    model_eval = report["checkpoint_eval"]
    fig.text(.035, .953, f"第 {report['display_level']} 级：{report['building_count']} 栋建筑 | "
             f"逃逸机 {c['target_min_speed']:g}–{c['target_max_speed']:g} m/s | 保持半径 {radius:g} m", fontsize=16, weight="bold")
    fig.text(.035, .913, f"模型：实验 {report['experiment']}，{int(model_eval['step']):,} 步 | "
             f"种子 {report['env_seed']} | 本视频为筛选的无碰撞成功样例 | 回放速度 {playback_speed:g}×", fontsize=10, color="#46566b")
    fig.text(.035, .875, f"该模型整批 100 回合评估：安全捕获 {float(model_eval['safe_success_rate']):.0%}，"
             f"碰撞 {float(model_eval['collision_rate']):.0%} | 成功判据：单机 ≤5 m，或同机在保持半径内连续 2 s", fontsize=10, color="#46566b")
    status = fig.text(.035, .831, "", fontsize=11, color="#164b80", weight="bold")
    footer = fig.text(.035, .018, "", fontsize=9, color="#46566b")
    def draw(index):
        for i in range(5):
            points = p[:index + 1, i] if i < 4 else e[:index + 1]
            current = points[-1]
            lines3[i].set_data(points[:, 0], points[:, 1])
            lines3[i].set_3d_properties(points[:, 2])
            dots3[i].set_data([current[0]], [current[1]])
            dots3[i].set_3d_properties([current[2]])
            linesxy[i].set_data(points[:, 0], points[:, 1])
            dotsxy[i].set_data([current[0]], [current[1]])
            if i < 4:
                linesd[i].set_data(t[:index + 1], distances[:index + 1, i])
        center = e[index]
        for plane, line in enumerate(sphere_lines):
            xyz = np.repeat(center[:, None], len(angle), axis=1)
            axes = [(0, 1), (0, 2), (1, 2)][plane]
            xyz[axes[0]] += radius * np.cos(angle)
            xyz[axes[1]] += radius * np.sin(angle)
            line.set_data(xyz[0], xyz[1])
            line.set_3d_properties(xyz[2])
        capture_circle.center = center[:2]
        distance_cursor.set_xdata([t[index], t[index]])
        if index == len(t) - 1:
            method = "5 m 直接拦截" if report["capture_type"] == "direct" else "保持半径内连续 2 s"
            status.set_text(f"t = {t[index]:.2f} s | 捕获成功：P{report['capture_agent_index'] + 1}，{method} | 全回合零碰撞")
            status.set_color("#11774d")
        else:
            holds = " / ".join(f"P{i + 1} {value:.1f}s" for i, value in enumerate(data["hold"][index]))
            status.set_text(f"t = {t[index]:.1f} s | 连续保持：{holds} | 到当前帧零碰撞")
            status.set_color("#164b80")
        footer.set_text("距离：" + "   ".join(f"P{i + 1} {value:.1f} m" for i, value in enumerate(distances[index]))
                        + "     建筑底面显示为飞行下界 50 m；低于该高度的楼不构成飞行层障碍。")
    # The simulator produces exact 0.1 s samples. Select them for 2x playback;
    # no invented interpolated path and no change to the 1 s control interval.
    fps = 10
    indices = list(range(0, len(t), max(1, round(playback_speed / (fps * float(c["physics_dt"]))))))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)
    poster_indices = [0, len(t) // 2, len(t) - 1]
    for name, index in zip(["start", "middle", "capture"], poster_indices):
        draw(index)
        fig.savefig(output / f"{name}.png", dpi=100, facecolor=fig.get_facecolor())
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1500,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-threads", "1", "-movflags", "+faststart"])
    with writer.saving(fig, str(output / "success.mp4"), dpi=100):
        draw(0)
        for _ in range(fps):
            writer.grab_frame()
        for ordinal, index in enumerate(indices):
            draw(index)
            writer.grab_frame()
            if ordinal % 100 == 0:
                print("VIDEO_RENDER", report["display_level"], ordinal, len(indices), flush=True)
        for _ in range(2 * fps):
            writer.grab_frame()
    plt.close(fig)
    report.update(playback_speed=playback_speed, output_fps=fps, video_frames=len(indices) + 3 * fps,
                  video_width=1280, video_height=720, video_sha256=sha256(output / "success.mp4"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", default="005:0,005:1,006:2,006:3",
                        help="experiment:zero-based-level pairs; each needs an actual saved stage model")
    parser.add_argument("--env-seed", type=int, default=23001)
    parser.add_argument("--max-attempts", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; preserve each video snapshot")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    jobs = []
    for spec in args.stages.split(","):
        experiment, level_text = spec.split(":")
        level = int(level_text)
        assert level in range(5) and experiment.isdigit()
        output = args.output / f"level_{level + 1}"
        output.mkdir()
        config, provenance = freeze_stage(args.root, experiment, level, output)
        save_json(output / "provenance.json", provenance)
        jobs.append((level, output, config, provenance))
    completed = []
    for level, output, config, provenance in jobs:
        data, report = find_success(config, output / "model.pth", level, args.env_seed, args.max_attempts, output)
        report.update(provenance)
        save_json(output / "manifest.json", report)
        render(data, report, output)
        save_json(output / "manifest.json", report)
        completed.append(report)
        save_json(args.output / "index.json", completed)
        print("VIDEO_COMPLETE", level + 1, str(output), report["capture_time"], flush=True)
    print("ALL_VIDEOS_COMPLETE", len(completed), flush=True)


if __name__ == "__main__":
    main()
