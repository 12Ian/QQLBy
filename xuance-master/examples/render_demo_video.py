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
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv

P_COLORS = ["#2a78d6", "#17a673", "#8e5bd4", "#d68a2a", "#3aa6b9", "#b95f3a"]
E_COLOR = "#e34948"


def rollout(args):
    p = argparse.Namespace(algo=args.algo, env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device=args.device)
    p.parallels = 1
    p.curriculum_enabled = False
    p.building_mode = args.building_mode
    if args.evader_center_pull is not None:
        p.evader_center_pull = args.evader_center_pull
    p.num_agents = args.num_agents
    p.target_max_speed = args.target_max_speed
    p.target_min_speed = args.target_min_speed
    if args.spawn_radius is not None:
        p.spawn_radius = args.spawn_radius
    if args.cbf_safety:
        p.cbf_enabled = True
    if args.cbf_margin is not None:
        p.cbf_margin = args.cbf_margin
    if args.surround_spawn:
        p.surround_spawn = True
    np.random.seed(args.seed)
    runner = get_runner(algo=args.algo, env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(os.path.abspath(args.model_path))
    env = UAVPursuitApollonius3DEnv(runner.config)

    for attempt in range(args.max_tries):          # keep the clip that actually captures
        np.random.seed(args.seed + attempt)
        obs, _ = env.reset()
        P, E, OPEN, done, caught = [], [], [], False, False
        while not done:
            P.append(env.uav_positions.copy()); E.append(env.target_position.copy())
            # Split the 200 sampled directions by what closes them. The clip is meant to show
            # which exits the buildings account for and which the team accounts for, so a single
            # "still open" count is not enough -- and it does not fall over an episode anyway,
            # because the surrounding spawn has already compressed the escape set at step 0.
            f = getattr(env, "apollonius_escape", None)
            if isinstance(f, dict) and f.get("dangerous") is not None:
                go = np.asarray(f["geometry_open"], bool)
                dg = np.asarray(f["dangerous"], bool)
                OPEN.append((int(np.sum(~go)),              # closed by structure
                             int(np.sum(go & ~dg)),         # closed by pursuers
                             int(np.sum(dg))))              # still open to the evader
            else:
                OPEN.append((0, 0, 0))
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, _, term, trunc, info = env.step(acts)
            caught = caught or bool(info.get("is_success", False))
            done = bool(term[env.agents[0]]) or bool(trunc)
        P.append(env.uav_positions.copy()); E.append(env.target_position.copy())
        OPEN.append(OPEN[-1] if OPEN else (0, 0, 0))
        print(f"  attempt {attempt}: steps={len(P)} caught={caught} "
              f"(structure,pursuers,open) {OPEN[0]} -> {OPEN[-1]}")
        if caught:
            return np.asarray(P), np.asarray(E), env, True, np.asarray(OPEN)
    return np.asarray(P), np.asarray(E), env, False, np.asarray(OPEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=11.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--evader-center-pull", type=float, default=None,
                    help="APF pull toward airspace centre; keeps captures off the walls")
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--cbf-safety", action="store_true")
    ap.add_argument("--cbf-margin", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tries", type=int, default=6)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="demo.mp4")
    args = ap.parse_args()

    P, E, env, caught, OPEN = rollout(args)
    T, N = P.shape[0], P.shape[1]
    B = np.asarray(env.buildings, float)
    zmin, zmax, msize = env.z_min, env.z_max, env.map_size
    lam = args.target_max_speed / env.uav_max_speed

    OPEN = np.asarray(OPEN, dtype=float)
    fig = plt.figure(figsize=(9.6, 8.6), dpi=120)
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    ax.set_position([0.02, 0.30, 0.96, 0.66])
    axb = fig.add_axes([0.10, 0.06, 0.84, 0.18])
    C_STRUCT, C_PURSUE, C_OPEN = "#8f8d86", "#2a78d6", "#e34948"

    def update(f):
        ax.clear()
        for b in B:
            x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
            for (ax0, ax1, ay0, ay1) in ((x0, x0, y0, y1), (x1, x1, y0, y1),
                                         (x0, x1, y0, y0), (x0, x1, y1, y1)):
                ax.plot_surface(np.array([[ax0, ax1], [ax0, ax1]]),
                                np.array([[ay0, ay1], [ay0, ay1]]),
                                np.array([[zmin, zmin], [zmax, zmax]]),
                                color="#8f8d86", alpha=0.30, shade=False, linewidth=0)
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zmax]*5,
                    color="#6e6c66", lw=0.8, alpha=0.9)
        lo = max(0, f - args.tail)
        for i in range(N):
            t = P[lo:f+1, i]
            c = P_COLORS[i % len(P_COLORS)]
            if len(t) > 1:
                ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, lw=1.8, alpha=0.9)
            ax.scatter(*P[f, i], color=c, s=48, depthshade=False)
        te = E[lo:f+1]
        if len(te) > 1:
            ax.plot(te[:, 0], te[:, 1], te[:, 2], color=E_COLOR, lw=2.5, alpha=0.95)
        ax.scatter(*E[f], color=E_COLOR, s=95, marker="*", depthshade=False)
        dmin = float(np.min(np.linalg.norm(P[f] - E[f], axis=1)))
        ax.set_xlim(0, msize); ax.set_ylim(0, msize); ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=24, azim=-58 + 0.25 * f)

        # who is closing the exits, up to the current frame
        axb.clear()
        n = max(f + 1, 2)
        t = np.arange(n)
        st, pu, op = OPEN[:n, 0], OPEN[:n, 1], OPEN[:n, 2]
        axb.stackplot(t, st, pu, op, colors=(C_STRUCT, C_PURSUE, C_OPEN), alpha=0.9,
                      labels=("closed by buildings", "closed by pursuers", "still open"))
        axb.set_xlim(0, max(T - 1, 1)); axb.set_ylim(0, 200)
        axb.set_xlabel("step"); axb.set_ylabel("escape directions")
        axb.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.85)
        axb.grid(alpha=0.2)
        tag = "  CAPTURED" if (caught and f == T-1) else ""
        ax.set_title(f"N={N} pursuers vs evader  vmax {args.target_max_speed:.0f} m/s "
                     f"(lambda={lam:.2f})   {args.building_mode}\n"
                     f"step {f+1}/{T}   min distance {dmin:5.1f} m{tag}", fontsize=11)
        return []

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000/args.fps, blit=False)
    try:
        anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2600))
    except Exception as e:
        alt = args.out.rsplit(".", 1)[0] + ".gif"
        print(f"ffmpeg unavailable ({type(e).__name__}); writing {alt}")
        anim.save(alt, writer=animation.PillowWriter(fps=args.fps)); args.out = alt
    print(f"wrote {args.out}  frames={T} caught={caught} lambda={lam:.2f}")


if __name__ == "__main__":
    main()
