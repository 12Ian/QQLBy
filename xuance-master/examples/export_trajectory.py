"""Dump one successful episode's trajectory to JSON so AirSim can replay it.

The closed-loop bridge flies SimpleFlight quadrotors under live policy control, which is the
honest way to run in AirSim but currently captures 0/6: the policy reasons about the 26 prisms in
buildings_complex.json while the bridge was flying the stock Blocks map, the quadrotor's response
lag is absent from the training kinematics, and dt=1s is coarse for a real airframe. Untangling
those is a research task.

Recording a presentable video is not. The trajectory here comes from the training environment --
where the same model captures in 89.3% of episodes -- and AirSim renders it by setting vehicle
poses directly. The pixels are a real Unreal render of the real scene geometry; the motion is a
replay of a verified capture rather than live closed-loop flight. Label it that way: it is a
visualisation, not an AirSim result.

Usage:
    python examples/export_trajectory.py --model-path <model.pth> --building-mode complex \
        --num-agents 3 --target-max-speed 9 --target-min-speed 8 --evader-center-pull 1.0 \
        --surround-spawn --spawn-radius 200 --cbf-safety --cbf-margin 2 \
        --require-capture --out traj_complex.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_demo_video import rollout          # noqa: E402  (same rollout the clip renderer uses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--building-mode", default="complex")
    ap.add_argument("--target-max-speed", type=float, default=9.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--surround-spawn", action="store_true")
    ap.add_argument("--evader-center-pull", type=float, default=None)
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--cbf-safety", action="store_true")
    ap.add_argument("--cbf-margin", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tries", type=int, default=12)
    ap.add_argument("--require-capture", action="store_true",
                    help="exit non-zero unless the recorded episode ends in a capture")
    ap.add_argument("--out", default="trajectory.json")
    # rollout() reads these two but the clip renderer owns them; harmless defaults here
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    P, E, env, caught, OPEN = rollout(args)
    P = np.asarray(P, float)                                  # (T, N, 3) ENU metres
    E = np.asarray(E, float)                                  # (T, 3)

    if args.require_capture and not caught:
        print("NO_CAPTURE: episode did not end in a capture; raise --max-tries or change --seed")
        return 1

    out = {
        "caught": bool(caught),
        "steps": int(P.shape[0]),
        "num_agents": int(P.shape[1]),
        "dt": float(getattr(env, "dt", 1.0)),
        "frame": "ENU, metres, z measured up from ground",
        "map_size": float(env.map_size),
        "z_min": float(env.z_min),
        "z_max": float(env.z_max),
        "catch_radius": float(getattr(env, "catch_radius", 15.0)),
        "building_mode": args.building_mode,
        "buildings": np.asarray(env.buildings, float).tolist(),
        "pursuers": P.tolist(),
        "evader": E.tolist(),
        # per-step direction census (buildings / pursuers / still open), same split the
        # matplotlib clip plots underneath the 3D view
        "open_census": (np.asarray(OPEN, float).tolist() if OPEN is not None else None),
        "model_path": args.model_path,
        "provenance": "training environment rollout, replayed in AirSim by pose-setting "
                      "(NOT closed-loop AirSim flight)",
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print("wrote %s: %d steps, %d pursuers, caught=%s"
          % (args.out, out["steps"], out["num_agents"], out["caught"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
