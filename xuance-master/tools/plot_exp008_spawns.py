"""Plot actual reset positions on the five fixed maps for the far-side experiment."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Rectangle
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv


def main():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "experiments/baseline_exp008.yaml").read_text())
    config["buildings_dir"] = str(root / "xuance/environment/multi_agent_env/buildings")
    font = subprocess.check_output(["fc-match", "Noto Serif CJK SC", "--format=%{file}"], text=True).strip()
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(), "axes.unicode_minus": False})
    env = UAVPursuitCoverage3DEnv(Namespace(**config))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=140)
    records = []
    colors = ["#2679d8", "#16a06b", "#914bd2", "#e59c1b"]
    try:
        for level, axis in enumerate(axes.flat[:5]):
            env.set_curriculum_level(level)
            env.seed(1)
            env.reset()
            distances = np.linalg.norm(env.uav_positions - env.target_position, axis=1)
            records.append(dict(level=level + 1, seed=1, pursuers=env.uav_positions.tolist(),
                                target=env.target_position.tolist(), distances=distances.tolist(),
                                safe_spawn_offsets=env.spawn_offsets.tolist()))
            for xmin, xmax, ymin, ymax, top in env.buildings:
                color = "#d5dee7" if top < 60 else ("#a5b6c7" if top < 150 else "#627d97")
                axis.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         facecolor=color, edgecolor="#738699", linewidth=.3))
            axis.add_patch(Rectangle((650, 350), 100, 300, facecolor="#f7dedf", alpha=.5,
                                     edgecolor="#d44758", linestyle="--", linewidth=1.2))
            for index, (position, color) in enumerate(zip(env.uav_positions, colors)):
                axis.scatter(*position[:2], color=color, s=50, edgecolor="white", zorder=5)
                axis.text(position[0] + 15, position[1] + 8, f"P{index+1}", color=color, fontsize=9)
                axis.plot([position[0], env.target_position[0]], [position[1], env.target_position[1]],
                          color=color, alpha=.4, linestyle=":", linewidth=1)
            axis.scatter(*env.target_position[:2], color="#df3b50", marker="*", s=150, edgecolor="white", zorder=6)
            axis.text(env.target_position[0] + 15, env.target_position[1] + 10, "E", color="#bf263d")
            axis.set(xlim=(0, 1000), ylim=(0, 1000), aspect="equal", xlabel="X / m", ylabel="Y / m")
            axis.grid(alpha=.12)
            axis.set_title(f"第{level+1}级 · {len(env.buildings)}栋楼\n实际三维初距 {distances.min():.0f}–{distances.max():.0f} m", fontsize=12)
        axes.flat[5].axis("off")
        axes.flat[5].text(.02, .9, "P1–P4：西侧集结\nE：东侧自由空间随机出生\n\n虚线矩形：逃逸机出生区域\n点线：初始位置之间的直线距离\n不是规划路线\n\n建筑固定；每次目标位置随机。\n高度：追捕机180–220 m，\n逃逸机150–250 m。\n出生点检查建筑与边界。", va="top", fontsize=12, linespacing=1.65)
        fig.suptitle("实验008：城市一侧出发，延长接近阶段（真实环境重置，种子1）", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(root / "docs/baseline_exp008_spawns_2026-10-13.png")
        (root / "docs/baseline_exp008_spawns_2026-10-13.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records, indent=2))
    finally:
        plt.close(fig)
        env.close()


if __name__ == "__main__":
    main()
