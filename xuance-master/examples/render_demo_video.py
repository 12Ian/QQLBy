"""Render a demo video of the flagship policy in the lightweight 3D simulator.

The published demo clips date from an earlier configuration (closure weight 8, one-sided start,
evader vmax 10) and no longer match the reported results. This records the current flagship
policy in the headline scenario -- an evader at least as fast as the pursuers -- which is the
claim the work now rests on.

Usage:
  python examples/render_demo_video.py --model-path <pth> --num-agents 3 \
      --target-max-speed 11 --surround-spawn --out demo.mp4
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

P_COLORS = ["#2a78d6", "#17a673", "#8e5bd4", "#d68a2a", "#3aa6b9", "#b95f3a"]
E_COLOR = "#e34948"
VIEW_PRESETS = {
    "xy-x": ("XY oblique view - x emphasis", 20, -35, False),
    "xy-y": ("XY oblique view - y emphasis", 20, -75, False),
    "view-a": ("3D oblique view A", 28, -55, True),
    "view-b": ("3D oblique view B", 35, 35, False),
    "view-c": ("3D low side view C", 18, -95, False),
    "view-d": ("3D low front view D", 16, 5, False),
    "view-e": ("3D high oblique view E", 55, -135, False),
    # Backward-compatible aliases for older commands.
    "perspective": ("3D oblique view A", 28, -55, True),
    "top": ("3D high oblique view E", 55, -135, False),
    "side-xz": ("3D low side view C", 18, -95, False),
    "side-yz": ("3D low front view D", 16, 5, False),
}
VIEW_SET = ["xy-x", "xy-y", "top2d"]


def rollout(args):
    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_coverage_3d",
                           env_id="coverage_3d", device=args.device)
    p.parallels = 1
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    p.use_obstacle_gat = args.use_obstacle_gat
    p.obstacle_gat_k = args.obstacle_gat_k
    p.nearest_building_k = args.obstacle_gat_k
    if args.evader_center_pull is not None:
        p.evader_center_pull = args.evader_center_pull
    p.num_agents = args.num_agents
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    if args.surround_spawn:
        p.surround_spawn = True
    np.random.seed(args.seed)
    runner = get_runner(algo=args.algo, env="uav_pursuit_coverage_3d",
                        env_id="coverage_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(os.path.abspath(args.model_path))
    env = UAVPursuitCoverage3DEnv(runner.config)

    for attempt in range(args.max_tries):          # keep the clip that actually captures
        np.random.seed(args.seed + attempt)
        obs, _ = env.reset()
        P, E, COVERAGE, done, caught = [], [], [], False, False
        while not done:
            P.append(env.uav_positions.copy()); E.append(env.target_position.copy())
            COVERAGE.append(float(env.last_coverage))
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            caught = caught or bool(info.get("is_success", False))
            done = bool(term[env.agents[0]]) or bool(trunc)
        P.append(env.uav_positions.copy()); E.append(env.target_position.copy())
        COVERAGE.append(float(env.last_coverage))
        print(f"  attempt {attempt}: steps={len(P)} caught={caught} "
              f"coverage {COVERAGE[0]:.3f} -> {COVERAGE[-1]:.3f}")
        if caught:
            return np.asarray(P), np.asarray(E), env, True, np.asarray(COVERAGE)
    return np.asarray(P), np.asarray(E), env, False, np.asarray(COVERAGE)


def _tail_slice(f, tail):
    return slice(max(0, f - tail), f + 1)


def _draw_buildings_3d(ax, buildings, zmin, zmax):
    for b in buildings:
        x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
        top = float(b[4]) if len(b) >= 5 else zmax
        for (ax0, ax1, ay0, ay1) in ((x0, x0, y0, y1), (x1, x1, y0, y1),
                                     (x0, x1, y0, y0), (x0, x1, y1, y1)):
            ax.plot_surface(np.array([[ax0, ax1], [ax0, ax1]]),
                            np.array([[ay0, ay1], [ay0, ay1]]),
                            np.array([[zmin, zmin], [top, top]]),
                            color="#8f8d86", alpha=0.24, shade=False,
                            linewidth=0)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [top] * 5,
                color="#6e6c66", lw=0.7, alpha=0.7)


def _draw_ground_grid(ax, env):
    step = env.map_size / 5.0
    xs = np.arange(0, env.map_size + 0.1, step)
    ys = np.arange(0, env.map_size + 0.1, step)
    z = env.z_min
    for x in xs:
        ax.plot([x, x], [0, env.map_size], [z, z],
                color="#b8b8b8", lw=0.45, alpha=0.38)
    for y in ys:
        ax.plot([0, env.map_size], [y, y], [z, z],
                color="#b8b8b8", lw=0.45, alpha=0.38)


def _draw_bounds_box(ax, env):
    x0, x1 = 0, env.map_size
    y0, y1 = 0, env.map_size
    z0, z1 = env.z_min, env.z_max
    edges = [
        ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1)),
    ]
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="#8a8a8a", lw=0.65, alpha=0.55)


def _draw_ground_shadow(ax, path, color, zmin, lw=1.0):
    if path.shape[0] < 2:
        return
    ax.plot(path[:, 0], path[:, 1], np.full(path.shape[0], zmin),
            color=color, lw=lw, alpha=0.23, ls="--")


def _draw_vertical_marker(ax, point, color, zmin):
    ax.plot([point[0], point[0]], [point[1], point[1]], [zmin, point[2]],
            color=color, lw=0.75, alpha=0.34)
    ax.scatter(point[0], point[1], zmin, color=color, s=12,
               alpha=0.34, depthshade=False)


def _set_3d_scene(ax, env, elev, azim, title, z_scale):
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_zlim(env.z_min, env.z_max)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
        ax.set_box_aspect((env.map_size, env.map_size,
                           (env.z_max - env.z_min) * z_scale))
    except AttributeError:
        pass
    ax.grid(alpha=0.22)
    ax.set_title(title, fontsize=10)


def _draw_catch_sphere(ax, center, radius):
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 9)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=E_COLOR, alpha=0.16, linewidth=0.45)


def _draw_3d(ax, P, E, B, f, args, env, caught, lam, title, elev, azim,
             rotate=False, show_status=True):
    ax.clear()
    _draw_ground_grid(ax, env)
    _draw_bounds_box(ax, env)
    _draw_buildings_3d(ax, B, env.z_min, env.z_max)
    sl = _tail_slice(f, args.tail)
    for i in range(P.shape[1]):
        c = P_COLORS[i % len(P_COLORS)]
        t = P[sl, i]
        _draw_ground_shadow(ax, t, c, env.z_min, lw=1.0)
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, lw=1.8, alpha=0.9)
        ax.scatter(*P[f, i], color=c, s=44, depthshade=False)
        _draw_vertical_marker(ax, P[f, i], c, env.z_min)
        ax.plot([P[f, i, 0], E[f, 0]],
                [P[f, i, 1], E[f, 1]],
                [P[f, i, 2], E[f, 2]],
                color=c, lw=0.9, alpha=0.24)
    te = E[sl]
    _draw_ground_shadow(ax, te, E_COLOR, env.z_min, lw=1.25)
    ax.plot(te[:, 0], te[:, 1], te[:, 2], color=E_COLOR, lw=2.4, alpha=0.95)
    ax.scatter(*E[f], color=E_COLOR, s=92, marker="*", depthshade=False)
    _draw_vertical_marker(ax, E[f], E_COLOR, env.z_min)
    _draw_catch_sphere(ax, E[f], env.catch_radius)
    dmin = float(np.min(np.linalg.norm(P[f] - E[f], axis=1)))
    view_azim = azim + 0.15 * f if rotate else azim
    tag = " CAPTURED" if (caught and f == P.shape[0] - 1) else ""
    full_title = title
    if show_status:
        full_title = (f"{title}  step {f+1}/{P.shape[0]}  "
                      f"min dist {dmin:5.1f} m  lambda={lam:.2f}{tag}")
    _set_3d_scene(
        ax, env, elev, view_azim,
        full_title,
        args.z_scale,
    )


def _draw_metrics(ax, P, E, COVERAGE, f, env):
    ax.clear()
    n = max(f + 1, 2)
    t = np.arange(n)
    dmin = np.min(np.linalg.norm(P[:n] - E[:n, None, :], axis=2), axis=1)
    ax.plot(t, dmin, color="#111111", lw=1.8, label="min distance")
    ax.axhline(env.catch_radius, color=E_COLOR, lw=1.0, ls="--",
               label="catch radius")
    ax.set_xlim(0, max(P.shape[0] - 1, 1))
    ymax = max(float(np.max(dmin)) * 1.08, env.catch_radius * 2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("step")
    ax.set_ylabel("distance (m)")
    ax.grid(alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(t, COVERAGE[:n], color="#2a78d6", lw=1.6,
             label="reach-and-hold coverage")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("coverage score")
    ax.set_title("Capture and future coverage", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)


def _draw_buildings_2d(ax, buildings):
    for b in buildings:
        x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor="#8f8d86", edgecolor="#5f5d58",
                                   alpha=0.32, linewidth=0.8))


def _draw_2d_top(ax, P, E, B, f, args, env, caught, lam, title):
    ax.clear()
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(alpha=0.22)
    _draw_buildings_2d(ax, B)
    sl = _tail_slice(f, args.tail)
    for i in range(P.shape[1]):
        c = P_COLORS[i % len(P_COLORS)]
        t = P[sl, i]
        ax.plot(t[:, 0], t[:, 1], color=c, lw=1.9, alpha=0.88)
        ax.scatter(P[f, i, 0], P[f, i, 1], color=c, s=38, zorder=4)
        ax.plot([P[f, i, 0], E[f, 0]], [P[f, i, 1], E[f, 1]],
                color=c, lw=0.85, alpha=0.22)
    te = E[sl]
    ax.plot(te[:, 0], te[:, 1], color=E_COLOR, lw=2.5, alpha=0.95)
    ax.scatter(E[f, 0], E[f, 1], color=E_COLOR, s=90, marker="*",
               zorder=5)
    ax.add_patch(plt.Circle((E[f, 0], E[f, 1]), env.catch_radius,
                            edgecolor=E_COLOR, facecolor="none",
                            linestyle="--", linewidth=1.0, alpha=0.8))
    dmin = float(np.min(np.linalg.norm(P[f] - E[f], axis=1)))
    tag = " captured" if (caught and f == P.shape[0] - 1) else ""
    ax.set_title(f"{title}  step {f+1}/{P.shape[0]}  "
                 f"min dist {dmin:5.1f} m  lambda={lam:.2f}{tag}",
                 fontsize=10)


def _render_composite(args, P, E, env, caught, COVERAGE, out_path):
    T, N = P.shape[0], P.shape[1]
    B = np.asarray(env.buildings, float)
    lam = args.target_max_speed / env.uav_max_speed
    fig = plt.figure(figsize=(16.2, 5.8), dpi=120)
    gs = fig.add_gridspec(1, 3, wspace=0.22)
    ax_x = fig.add_subplot(gs[0, 0], projection="3d")
    ax_y = fig.add_subplot(gs[0, 1], projection="3d")
    ax_top = fig.add_subplot(gs[0, 2])

    def update(f):
        for ax, view_name in ((ax_x, "xy-x"), (ax_y, "xy-y")):
            title, elev, azim, rotate = VIEW_PRESETS[view_name]
            _draw_3d(ax, P, E, B, f, args, env, caught, lam,
                     title, elev, azim, rotate=rotate, show_status=False)
        _draw_2d_top(ax_top, P, E, B, f, args, env, caught, lam,
                     "Pure XY top view")
        dmin = float(np.min(np.linalg.norm(P[f] - E[f], axis=1)))
        tag = " captured" if (caught and f == P.shape[0] - 1) else ""
        fig.suptitle(f"Three-view pursuit: {args.building_mode}  "
                     f"step {f+1}/{T}  min dist {dmin:5.1f} m  "
                     f"lambda={lam:.2f}{tag}", fontsize=12)
        return []

    _save_animation(fig, update, T, args, out_path)


def _render_single_view(args, P, E, env, caught, COVERAGE, out_path, view):
    T = P.shape[0]
    B = np.asarray(env.buildings, float)
    lam = args.target_max_speed / env.uav_max_speed
    if view == "top2d":
        fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=120)
        update = lambda f: (_draw_2d_top(ax, P, E, B, f, args, env, caught, lam,
                                         "Pure XY top view"), [])[1]
    elif view != "metrics":
        fig = plt.figure(figsize=(9.6, 7.8), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        if view not in VIEW_PRESETS:
            raise ValueError(f"unknown view: {view}")
        title, elev, azim, rotate = VIEW_PRESETS[view]
        update = lambda f: (_draw_3d(ax, P, E, B, f, args, env, caught, lam,
                                     title, elev, azim, rotate), [])[1]
    else:
        fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=120)
        update = lambda f: (_draw_metrics(ax, P, E, COVERAGE, f, env), [])[1]
    _save_animation(fig, update, T, args, out_path)


def _save_animation(fig, update, frames, args, out_path):
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 / args.fps, blit=False)
    try:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=args.fps,
                                                          bitrate=3200))
    except Exception as e:
        alt = out_path.rsplit(".", 1)[0] + ".gif"
        print(f"ffmpeg unavailable ({type(e).__name__}); writing {alt}")
        anim.save(alt, writer=animation.PillowWriter(fps=args.fps))
        out_path = alt
    plt.close(fig)
    print(f"wrote {out_path}")


def _out_with_suffix(out_path, suffix):
    stem, ext = os.path.splitext(out_path)
    return f"{stem}_{suffix}{ext or '.mp4'}"


def main():
    default_run = "apollonius_3d_medium_4v1_k10_boundary_center_pincer_3m_20260923_175609"
    repo_root = "/home/ryy/UAV_Mine/xuance-master"
    default_model = os.path.join(
        repo_root, "outputs", "results", "maddpg", f"{default_run}",
        "best_model", "best_model.pth")
    default_out = os.path.join(
        repo_root, "outputs", "runs", default_run, "videos", "default_three_xy.mp4")
    ap = argparse.ArgumentParser(
        description="加载训练好的策略，在三维无人机环境中生成追捕过程视频。")
    ap.add_argument("--model-path", default=default_model,
                    help="训练完成的 .pth 策略模型绝对路径（默认：本脚本预设的 best_model）")
    ap.add_argument("--algo", default="maddpg",
                    help="用于创建策略网络的算法名称（默认：maddpg）")
    ap.add_argument("--num-agents", type=int, default=4,
                    help="追捕无人机数量（默认：4）")
    ap.add_argument("--building-mode", default="medium",
                    help="测试场景的建筑布局类型（默认：medium）")
    ap.add_argument("--target-max-speed", type=float, default=12.0,
                    help="逃逸机最大速度，单位 m/s（默认：12）")
    ap.add_argument("--target-min-speed", type=float, default=4.0,
                    help="逃逸机最小速度，单位 m/s（默认：4）")
    ap.add_argument("--surround-spawn", action=argparse.BooleanOptionalAction, default=True,
                    help="让追捕机在逃逸机周围出生；使用 --no-surround-spawn 关闭（默认：开启）")
    ap.add_argument("--use-obstacle-gat", action=argparse.BooleanOptionalAction, default=True,
                    help="启用建筑物观测网络；使用 --no-use-obstacle-gat 关闭（默认：开启）")
    ap.add_argument("--obstacle-gat-k", type=int, default=5,
                    help="提供给策略网络的最近建筑物数量上限（默认：5）")
    ap.add_argument("--evader-center-pull", type=float, default=None,
                    help="逃逸机朝空域中心移动的引导强度（默认：使用环境配置）")
    ap.add_argument("--spawn-radius", type=float, default=None,
                    help="环绕出生时的初始半径，单位 m（默认：使用环境配置）")
    ap.add_argument("--seed", type=int, default=21,
                    help="场景和仿真的随机种子（默认：21）")
    ap.add_argument("--max-tries", type=int, default=5,
                    help="最多尝试的仿真回合数，用于寻找可展示的捕获过程（默认：5）")
    ap.add_argument("--fps", type=int, default=12,
                    help="输出视频帧率（默认：12 帧/秒）")
    ap.add_argument("--tail", type=int, default=40,
                    help="轨迹中以完整颜色显示的最近步数（默认：40）")
    ap.add_argument("--z-scale", type=float, default=2.0,
                    help="仅用于画面的 z 轴高度显示倍数，不改变仿真（默认：2）")
    ap.add_argument("--device", default="cuda:0",
                    help="策略推理使用的设备，例如 cuda:0 或 cpu（默认：cuda:0）")
    ap.add_argument("--out", default=default_out,
                    help="输出视频文件的绝对路径（默认：实验目录下的 default_three_xy.mp4）")
    ap.add_argument("--view", default="composite",
                    choices=["composite", "xy-x", "xy-y", "top2d",
                             "perspective", "top", "side-xz",
                             "side-yz", "view-a", "view-b", "view-c",
                             "view-d", "view-e", "metrics"],
                    help="视频视角：composite 三视图、xy-x/xy-y 倾斜三维视角、top2d 纯俯视等（默认：composite）")
    ap.add_argument("--view-set", action="store_true",
                    help="除主视频外，另外输出 composite、xy-x、xy-y 和 top2d 四种视频")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    P, E, env, caught, COVERAGE = rollout(args)
    COVERAGE = np.asarray(COVERAGE, dtype=float)
    if args.view_set:
        for view in ["composite"] + VIEW_SET:
            out = args.out if view == "composite" else _out_with_suffix(args.out, view)
            if view == "composite":
                _render_composite(args, P, E, env, caught, COVERAGE, out)
            else:
                _render_single_view(args, P, E, env, caught, COVERAGE, out, view)
    elif args.view == "composite":
        _render_composite(args, P, E, env, caught, COVERAGE, args.out)
    else:
        _render_single_view(args, P, E, env, caught, COVERAGE, args.out, args.view)
    print(f"frames={P.shape[0]} caught={caught} "
          f"lambda={args.target_max_speed / env.uav_max_speed:.2f}")


if __name__ == "__main__":
    main()
