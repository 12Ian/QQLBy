"""Roll out a trained policy until a successful capture, render each step with the env's
3D view, and save an MP4 + GIF of the pursuit. Mirrors evaluate_3d's model/env setup."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv

COLORS = ["#FF8C00", "#1CA64C", "#00A6D6", "#8E44AD", "#E84393", "#2E86DE",
          "#8c564b", "#e377c2"]


def frame(env, step, azim, captured):
    """Polished 3D frame: rotating camera, large markers, encircling net, status."""
    fig = plt.figure(figsize=(9, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, env.map_size); ax.set_ylim(0, env.map_size); ax.set_zlim(env.z_min, env.z_max)
    ax.view_init(elev=26, azim=azim)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    for b in env.buildings:
        xmin, xmax, ymin, ymax = b
        ax.bar3d(xmin, ymin, env.z_min, xmax - xmin, ymax - ymin, env.z_max - env.z_min,
                 color="#9aa3ad", alpha=0.16, shade=True)
    tp = env.target_position
    if len(env.target_trail) > 1:
        tt = np.array(env.target_trail)
        ax.plot(tt[:, 0], tt[:, 1], tt[:, 2], color="#D6202A", lw=1.7, alpha=0.75)
    ax.scatter(tp[0], tp[1], tp[2], color="#D6202A", s=170, marker="*",
               edgecolors="black", linewidths=0.6, zorder=8, label="Evader")
    P = env.uav_positions
    for i, a in enumerate(env.agents):
        c = COLORS[i % len(COLORS)]
        up = P[i]
        tr = np.array(list(env.uav_trails[a]))
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=1.8, alpha=0.8)
        ax.scatter(up[0], up[1], up[2], color=c, s=75, marker="o",
                   edgecolors="black", linewidths=0.6, zorder=8)
    # encircling net: connect pursuers in azimuth order around the evader
    if env.num_agents >= 3:
        ang = np.arctan2(P[:, 1] - tp[1], P[:, 0] - tp[0])
        ring = P[np.argsort(ang)]
        ring = np.vstack([ring, ring[0]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#2F6DB0", lw=1.3, alpha=0.5)
    if captured:
        ax.set_title(f"3D Cooperative Pursuit   ·   step {step}   ·   CAPTURED",
                     color="#1CA64C", fontsize=13, fontweight="bold")
    else:
        ax.set_title(f"3D Cooperative Pursuit   ·   step {step}", fontsize=13)
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--target-max-speed", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-episodes", type=int, default=10)
    ap.add_argument("--out", default="p3_results/pursuit")
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()

    np.random.seed(a.seed)
    p = argparse.Namespace(algo="maddpg", env="uav_pursuit_apollonius_3d",
                           env_id="apollonius_3d", device="cuda:0")
    p.num_agents = a.num_agents
    p.seed = a.seed
    p.curriculum_enabled = False
    p.building_mode = a.building_mode
    p.target_max_speed = a.target_max_speed
    runner = get_runner(algo="maddpg", env="uav_pursuit_apollonius_3d",
                        env_id="apollonius_3d", parser_args=p)
    agent = runner.agent
    agent.load_model(a.model_path)
    env = UAVPursuitApollonius3DEnv(runner.config)

    frames, chosen = None, None
    for ep in range(a.max_episodes):
        obs, info = env.reset()
        step, azim0 = 0, -60.0
        fs = [frame(env, 0, azim0, False)]
        done, caught = False, False
        while not done:
            acts = agent.action([obs], test_mode=True)["actions"][0]
            obs, rew, term, trunc, info = env.step(acts)
            step += 1
            caught = bool(info.get("is_success", False))
            done = bool(term[env.agents[0]]) or bool(trunc)
            fs.append(frame(env, step, azim0 + 1.6 * step, caught))
        print(f"ep {ep}: caught={caught} frames={len(fs)}", flush=True)
        if caught:
            frames, chosen = fs, ep
            break
        if frames is None:
            frames = fs  # fallback to last episode if none succeed
    print(f"chosen episode={chosen} (None=fallback) len={len(frames)}", flush=True)

    frames = frames + [frames[-1]] * a.fps           # hold last frame ~1s
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    import imageio.v2 as imageio

    try:
        imageio.mimsave(a.out + ".mp4", frames, fps=a.fps, codec="libx264", quality=8,
                        macro_block_size=None)
        print("saved", a.out + ".mp4", flush=True)
    except Exception as e:
        print("mp4 failed:", e, flush=True)

    try:
        from PIL import Image
        small = [np.array(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)))
                 for f in frames]
    except Exception:
        small = frames
    imageio.mimsave(a.out + ".gif", small, fps=a.fps, loop=0)
    print("saved", a.out + ".gif", "frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
