"""Render a successful CRL-G trajectory with its LOS-normal coverage geometry.

The source is a saved crlg_evaluate.py --save-trajectories JSON file. This
does not rerun the policy or substitute closest approach for terminal success.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Ellipse, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


COLORS = ("#dc4b43", "#2878b9", "#2d9b64", "#db8a20", "#8659b5", "#a66c48")
TARGET_COLOR = "#20252b"


def interpolate(times, values, render_times):
    values = np.asarray(values, dtype=float)
    flat = values.reshape(len(times), -1)
    result = np.column_stack([np.interp(render_times, times, flat[:, i])
                              for i in range(flat.shape[1])])
    return result.reshape((len(render_times),) + values.shape[1:])


def render(record, output, fps=10, frame_dt=0.25, hold_seconds=2.0):
    trajectory = record["trajectory"]
    if any("projection_geometry" not in step for step in trajectory):
        raise ValueError("evaluation JSON needs --save-projection-geometry data")
    geometry = [step["projection_geometry"] for step in trajectory]
    times = np.array([step["time_s"] for step in trajectory], dtype=float)
    pursuers = np.array([step["pursuer_positions_m"] for step in trajectory], dtype=float)
    target = np.array([step["target_position_m"] for step in trajectory], dtype=float)
    coverage = np.array([step["coverage_probability"] for step in trajectory], dtype=float)
    encounter = np.array([step["estimated_encounter_time_s"] for step in trajectory],
                         dtype=float)
    render_times = np.r_[np.arange(times[0], times[-1], frame_dt), times[-1]]
    p = interpolate(times, pursuers, render_times)
    q = interpolate(times, target, render_times)
    c = np.interp(render_times, times, coverage)
    te = np.interp(render_times, times, encounter)
    distances = np.linalg.norm(p - q[:, None, :], axis=2)
    min_distances = np.min(distances, axis=1)
    png_index = next(i for i, step in enumerate(trajectory)
                     if step["guidance_phase"] == "PNG")
    png_start = times[png_index - 1] if png_index else times[0]
    n_agents = p.shape[1]
    winners = set(np.flatnonzero(distances[-1] <= 5.))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(16, 10), dpi=110, facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.96, top=0.83,
                            bottom=0.09, wspace=0.24, hspace=0.43,
                            height_ratios=(1., 1.))
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    axp = fig.add_subplot(grid[1, 0])
    axd = fig.add_subplot(grid[1, 1])
    axc = axd.twinx()

    points = np.concatenate((p.reshape(-1, 3), q), axis=0)
    low = points.min(axis=0) - np.array([22., 22., 16.])
    high = points.max(axis=0) + np.array([22., 22., 16.])
    ax3.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
            zlim=(low[2], high[2]), xlabel="x (m)", ylabel="y (m)",
            zlabel="z (m)", title="Three-dimensional flight paths")
    ax3.view_init(elev=25, azim=-55)
    ax3.set_proj_type("ortho")
    ax3.set_box_aspect((high[0]-low[0], high[1]-low[1],
                        2.2*(high[2]-low[2])))
    ax3.grid(alpha=0.2)
    ax3.text2D(0.01, 0.02, "Dashed segments: terminal PNG | vertical scale enlarged for display",
               transform=ax3.transAxes, fontsize=8, color="#5b6470")
    plane = Poly3DCollection([np.zeros((4, 3))], facecolor="#397fb9",
                             edgecolor="#397fb9", alpha=0.09, linewidth=0.8)
    ax3.add_collection3d(plane)

    ax2.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
            xlabel="x (m)", ylabel="y (m)",
            title="XY trajectory projection")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.22)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    circle = Circle(q[0, :2], 5., fill=False, ec="#d63232", lw=1.4, ls="--",
                    label="5 m terminal threshold")
    ax2.add_patch(circle)

    axp.set(xlabel="Projected axis $e_1$ (m)", ylabel="Projected axis $e_2$ (m)",
            title="LOS-normal plane: target probability and reachability")
    axp.set_aspect("equal", adjustable="box")
    axp.grid(alpha=0.18)
    axp.axhline(0., lw=0.7, color="#929daa", zorder=0)
    axp.axvline(0., lw=0.7, color="#929daa", zorder=0)
    axp.plot(0., 0., "*", color="#d18420", markersize=12, zorder=8)
    projection_note = axp.text(0.015, 0.03, "", transform=axp.transAxes,
                               fontsize=8, color="#435364", va="bottom")
    projection_artists = []

    axd.set(xlim=(times[0], times[-1]), ylim=(0.5, max(500., min_distances[0]*1.15)),
            xlabel="Time (s)", ylabel="Nearest 3D distance (m)",
            title="Terminal timing and target coverage")
    axd.set_yscale("log")
    axd.grid(alpha=0.22, which="both")
    axd.axhline(5., color="#d63232", lw=1.1, ls="--", label="5 m threshold")
    axd.axvspan(png_start, times[-1], color="#eabf72", alpha=0.18,
                label="PNG stage")
    axc.set(ylim=(0., 1.02), ylabel="Coverage probability")
    axc.spines["right"].set_visible(True)

    mid3, png3, dots3, mid2, png2, dots2 = [], [], [], [], [], []
    for i in range(n_agents):
        color = COLORS[i]
        width = 2.4 if i in winners else 1.7
        m3, = ax3.plot([], [], [], color=color, lw=width, label=f"Interceptor {i+1}")
        g3, = ax3.plot([], [], [], color=color, lw=width, ls="--")
        d3 = ax3.scatter(*p[0, i], color=color, s=58 if i in winners else 38,
                         depthshade=False)
        m2, = ax2.plot([], [], color=color, lw=width, label=f"Interceptor {i+1}")
        g2, = ax2.plot([], [], color=color, lw=width, ls="--")
        d2 = ax2.scatter(p[0, i, 0], p[0, i, 1], color=color,
                         s=50 if i in winners else 33, zorder=4)
        ax2.scatter(p[0, i, 0], p[0, i, 1], s=20, facecolors="none",
                    edgecolors=color, linewidths=0.9)
        mid3.append(m3); png3.append(g3); dots3.append(d3)
        mid2.append(m2); png2.append(g2); dots2.append(d2)

    target3, = ax3.plot([], [], [], color=TARGET_COLOR, lw=2.4, label="Target")
    target_dot3 = ax3.scatter(*q[0], color=TARGET_COLOR, marker="*", s=125,
                               depthshade=False)
    target2, = ax2.plot([], [], color=TARGET_COLOR, lw=2.4, label="Target")
    target_dot2 = ax2.scatter(q[0, 0], q[0, 1], color=TARGET_COLOR, marker="*",
                               s=120, zorder=5)
    link3, = ax3.plot([], [], [], color="#d63232", lw=1.1, alpha=0.7)
    link2, = ax2.plot([], [], color="#d63232", lw=1.1, alpha=0.7)
    dist_line, = axd.plot([], [], color="#1c4d78", lw=2.2, label="Nearest distance")
    cov_line, = axc.plot([], [], color="#318a69", lw=1.8, label="Coverage")
    time_line = axd.axvline(times[0], color="#424b54", lw=1., alpha=0.7)
    dist_dot, = axd.plot([], [], "o", color="#1c4d78", ms=6)
    ax2.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)
    axd.legend([dist_line, cov_line, circle],
               ["Nearest distance", "Coverage", "5 m threshold"],
               loc="upper right", fontsize=8)

    fig.text(0.045, 0.957, "CRL-G  |  Cooperative guidance", fontsize=20,
             weight="bold", color="#192330")
    fig.text(0.045, 0.924,
             f"Evaluation episode {record['episode']}  •  {record['active_agents']} interceptors"
             f"  •  {record['target_mode']} target  •  MADDPG midcourse + PNG terminal",
             fontsize=11, color="#52606d")
    status = fig.text(0.045, 0.882, "", fontsize=12, color="#283746")
    fig.text(0.045, 0.036,
             f"Result: terminal miss {record['terminal_min_miss_m']:.2f} m  |  "
             f"{len(winners)}/{n_agents} interceptors within 5 m  |  "
             f"locked encounter time {record['encounter_time_s']:.2f} s",
             fontsize=11, weight="bold", color="#256846")

    def update(frame):
        k = min(frame, len(render_times) - 1)
        source_index = min(np.searchsorted(times, render_times[k], side="right") - 1,
                           len(times) - 1)
        g = geometry[source_index]
        for artist in projection_artists:
            artist.remove()
        projection_artists.clear()
        mean = np.asarray(g["target_mean_2d_m"])
        cov = np.asarray(g["target_covariance_2d_m2"])
        eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2.)
        major = np.argmax(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, major], eigvecs[0, major]))
        contour_scale = np.sqrt(-2. * np.log(0.05))
        centers = np.asarray(g["reach_centers_2d_m"]) - mean
        radii = np.asarray(g["reach_radii_m"])
        extent = max(12., np.max(np.abs(centers) + radii[:, None]) + 4.,
                     contour_scale * np.sqrt(max(eigvals[major], 0.)) + 4.)
        axp.set(xlim=(-extent, extent), ylim=(-extent, extent))
        target_region = Ellipse((0., 0.),
                                2. * contour_scale * np.sqrt(max(eigvals[major], 0.)),
                                2. * contour_scale * np.sqrt(max(eigvals[1-major], 0.)),
                                angle=angle, facecolor="#e7a54c", edgecolor="#a96818",
                                lw=1.7, alpha=0.38, zorder=4)
        axp.add_patch(target_region)
        projection_artists.append(target_region)
        for i in range(n_agents):
            center = centers[i]
            radius = radii[i]
            polygon = g["reach_polygons_2d_m"][i]
            if polygon is not None:
                reachable = Polygon(np.asarray(polygon) - mean, closed=True,
                                    facecolor="#2878b9", edgecolor="#17609b",
                                    lw=1., alpha=0.22, zorder=2)
                axp.add_patch(reachable)
                projection_artists.append(reachable)
                reference = Circle(center, radius, fill=False, edgecolor="#2878b9",
                                   lw=0.8, ls="--", alpha=0.55, zorder=3)
                axp.add_patch(reference)
                projection_artists.append(reference)
            else:
                reachable = Circle(center, radius, facecolor="#2878b9",
                                   edgecolor="#17609b", lw=1., alpha=0.22, zorder=2)
                axp.add_patch(reachable)
                projection_artists.append(reachable)
            marker, = axp.plot(center[0], center[1], "o", ms=4.5,
                               color=COLORS[i], mec="white", mew=0.5, zorder=6)
            projection_artists.append(marker)
        projection_note.set_text("Fixed initial LOS basis  |  amber: target 95% Gaussian region\n"
                                 "blue: reachable sets  |  dashed: radius in PNG stage")
        axes = np.asarray(g["plane_axes_world"])
        origin = np.asarray(g["plane_origin_world_m"])
        plane_half_width = min(35., extent / 2.)
        plane.set_verts([[origin + a * plane_half_width * axes[0]
                         + b * plane_half_width * axes[1]
                         for a, b in ((-1,-1), (1,-1), (1,1), (-1,1))]])
        history = np.arange(k + 1)
        mid = history[render_times[:k+1] <= png_start + 1e-8]
        terminal = history[render_times[:k+1] >= png_start - 1e-8]
        for i in range(n_agents):
            for line3, line2, indices in ((mid3[i], mid2[i], mid),
                                          (png3[i], png2[i], terminal)):
                line3.set_data(p[indices, i, 0], p[indices, i, 1])
                line3.set_3d_properties(p[indices, i, 2])
                line2.set_data(p[indices, i, 0], p[indices, i, 1])
            dots3[i]._offsets3d = ([p[k, i, 0]], [p[k, i, 1]], [p[k, i, 2]])
            dots2[i].set_offsets(p[k, i, :2][None, :])
        target3.set_data(q[:k+1, 0], q[:k+1, 1])
        target3.set_3d_properties(q[:k+1, 2])
        target2.set_data(q[:k+1, 0], q[:k+1, 1])
        target_dot3._offsets3d = ([q[k, 0]], [q[k, 1]], [q[k, 2]])
        target_dot2.set_offsets(q[k, :2][None, :])
        circle.center = q[k, :2]
        nearest = int(np.argmin(distances[k]))
        link3.set_data([p[k, nearest, 0], q[k, 0]],
                       [p[k, nearest, 1], q[k, 1]])
        link3.set_3d_properties([p[k, nearest, 2], q[k, 2]])
        link2.set_data([p[k, nearest, 0], q[k, 0]],
                       [p[k, nearest, 1], q[k, 1]])
        dist_line.set_data(render_times[:k+1], min_distances[:k+1])
        cov_line.set_data(render_times[:k+1], c[:k+1])
        time_line.set_xdata([render_times[k], render_times[k]])
        dist_dot.set_data([render_times[k]], [min_distances[k]])
        phase = "PNG" if render_times[k] > png_start else "CRL-G"
        tag = "  |  SUCCESS" if k == len(render_times) - 1 else ""
        status.set_text(f"t = {render_times[k]:5.2f} s  |  phase: {phase}  |  "
                        f"nearest distance: {min_distances[k]:5.2f} m  |  "
                        f"coverage: {c[k]:.2f}  |  estimated t_e: {te[k]:.2f} s{tag}")
        return []

    output.parent.mkdir(parents=True, exist_ok=True)
    frames = len(render_times) + round(hold_seconds * fps)
    movie = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps,
                                    blit=False)
    movie.save(output, writer=animation.FFMpegWriter(
        fps=fps, bitrate=4200, extra_args=["-pix_fmt", "yuv420p"]))
    update(frames - 1)
    poster = output.with_suffix(".png")
    fig.savefig(poster, dpi=140)
    plt.close(fig)
    details = {
        "source_episode": record["episode"], "target_mode": record["target_mode"],
        "active_agents": n_agents, "terminal_min_miss_m": record["terminal_min_miss_m"],
        "agents_within_5m": len(winners), "trajectory_steps": len(trajectory),
        "frames": frames, "fps": fps, "png_start_s": png_start,
        "video": str(output), "poster": str(poster),
    }
    output.with_suffix(".json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    record = next(item for item in payload["episodes"]
                  if item["episode"] == args.episode)
    if not record["valid"] or record["terminal_min_miss_m"] > 5.:
        raise ValueError("episode must be a valid terminal success at the 5 m threshold")
    if not record.get("trajectory"):
        raise ValueError("evaluation JSON needs --save-trajectories data")
    print(json.dumps(render(record, Path(args.output), fps=args.fps), indent=2))


if __name__ == "__main__":
    main()
