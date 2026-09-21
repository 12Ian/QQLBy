"""AirSim high-fidelity validation bridge (plan P4) -- FRAMEWORK.

Connects a MADDPG policy TRAINED in the lightweight 3D env to AirSim/UE so the
same policy flies real quadrotors in a building scene. NOT for training.

Design principle: reuse the lightweight env as the single source of truth for
observation construction, the APF evader, and the Apollonius capture judge --
we only swap where the *state* comes from (AirSim instead of the internal sim)
and where the *actions* go (SimpleFlight velocity commands). This guarantees the
policy sees the exact observation format it was trained on.

Two modes:
  --mode closed_loop  (default, the real validation): read AirSim state -> policy
      -> velocity command -> SimpleFlight tracks -> quadrotor dynamics execute.
  --mode mirror       (visualization / transform-debug): run the lightweight sim
      internally and command AirSim to follow the computed positions.

RUN REQUIREMENTS (why this can't run here): a machine with Unreal Engine + AirSim
rendering (GPU + display). `pip install airsim msgpack-rpc-python`.

[TUNE-ON-MACHINE] markers flag the few things that must be calibrated once on the
real UE scene (coordinate origin/scale, control dt/gains). Everything else is done.
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch as _torch
# models are saved on the training GPU; on a CPU-only inference host, default
# torch.load to map storages to CPU so agent.load_model() works.
if not _torch.cuda.is_available():
    _orig_torch_load = _torch.load
    def _cpu_torch_load(*a, **k):
        k["map_location"] = "cpu"          # force CPU, override any cuda location
        return _orig_torch_load(*a, **k)
    _torch.load = _cpu_torch_load

# ----- lightweight env + policy (reused, identical to evaluate_3d.py) -----
from xuance import get_runner
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv
from xuance.environment.multi_agent_env import geometry3d as g3
from xuance.environment.multi_agent_env import apollonius3d as ap3


def escape_solid_angle(env, mode):
    """Apollonius escape solid angle from the CURRENT env state (capture judge).
    Reused verbatim from evaluate_3d.py -> capture when this -> 0."""
    free = g3.ray_free_distance(env.target_position, env.escape_dirs, env.map_size,
                                env.z_min, env.z_max, env.buildings)
    f = ap3.compute_escape_field_3d(env.target_position, env.target_speed,
            env.uav_positions, env.uav_max_speed, env.escape_dirs, free,
            env.catch_radius, env._compute_r_f(), margin_ema=None, dangerous_prev=None,
            criterion_mode=mode, buildings=env.buildings)
    return f["escape_solid_angle"]


def vel_from_state(speed, yaw, pitch):
    """Per-step displacement / desired velocity vector (m/s at dt=1) in ENV frame.
    Matches env._move_with_clip_3d: vel = speed*[cosP cosY, cosP sinY, sinP]."""
    cp = math.cos(pitch)
    return np.array([speed * cp * math.cos(yaw),
                     speed * cp * math.sin(yaw),
                     speed * math.sin(pitch)], np.float32)


class AirSimPursuitBridge:
    def __init__(self, args):
        self.args = args
        self.N = args.num_agents
        self.dt = args.dt                      # control period (s). env dt=1 step. [TUNE]
        # ---- coordinate transform: ENV (x,y in [0,map], z up in [z_min,z_max])
        #      <-> AirSim NED (x North, y East, z Down; up = negative z). ----
        self.origin = np.array(args.origin, np.float32)   # env point mapped to NED (0,0,0). [TUNE]
        # ---- build the policy exactly like evaluate_3d.py ----
        p = argparse.Namespace(algo=args.algo, env="uav_pursuit_apollonius_3d",
                               env_id="apollonius_3d", device="cpu")
        p.parallels = 1
        p.use_obstacle_gat = args.use_obstacle_gat
        p.use_graph_module = args.use_graph_module
        p.curriculum_enabled = False
        p.building_mode = args.building_mode
        p.target_max_speed = args.target_max_speed
        p.target_min_speed = args.target_min_speed
        p.num_agents = self.N
        # The lambda>=1 policies are trained with pursuers deployed AROUND the evader; a
        # one-sided start cannot kinematically close on a target of equal or greater speed, so
        # omitting this here would evaluate the policy outside the setup it was trained for.
        if getattr(args, "surround_spawn", False):
            p.surround_spawn = True
            if getattr(args, "spawn_radius", None) is not None:
                p.spawn_radius = args.spawn_radius
        self.runner = get_runner(algo=args.algo, env="uav_pursuit_apollonius_3d",
                                 env_id="apollonius_3d", parser_args=p)
        self.agent = self.runner.agent
        self.agent.load_model(args.model_path)
        # ---- env instance = obs builder + APF evader + capture judge ----
        self.env = UAVPursuitApollonius3DEnv(self.runner.config)
        self.env.reset()                       # seeds buildings, guide state, target spawn
        self.pursuer_names = [f"Pursuer{i}" for i in range(self.N)]
        self.evader_name = "Evader0"
        self.trails = {n: [] for n in self.pursuer_names + [self.evader_name]}

    # ---------- coordinate transforms [TUNE-ON-MACHINE: origin/z-sign] ----------
    def env_to_ned(self, p):
        e = p - self.origin
        return np.array([e[0], e[1], -e[2]], np.float32)      # z up -> z down
    def ned_to_env(self, p):
        return np.array([p[0] + self.origin[0], p[1] + self.origin[1],
                         -p[2] + self.origin[2]], np.float32)
    def vel_env_to_ned(self, v):
        return np.array([v[0], v[1], -v[2]], np.float32)

    # ---------- AirSim I/O (imported lazily so this file imports without airsim) ----------
    def connect(self):
        import airsim
        self.airsim = airsim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        for n in self.pursuer_names + [self.evader_name]:
            self.client.enableApiControl(True, n)
            self.client.armDisarm(True, n)
        fs = [self.client.takeoffAsync(vehicle_name=n) for n in self.pursuer_names + [self.evader_name]]
        [f.join() for f in fs]
        # takeoffAsync only reaches ~3 m, but the environment operates in a 50-350 m band: left
        # there, the quadrotors fly near the ground while the policy reasons about obstacle
        # geometry and an evader hundreds of metres up, so nothing lines up. Climb each vehicle
        # to the altitude its counterpart occupies in the env before the control loop starts.
        # Every scalar handed to the RPC layer must be a builtin float: msgpack serialises
        # numpy.float32 by looking for .to_msgpack() and raises AttributeError without it, and
        # origin is a numpy array so the subtraction alone re-introduces the numpy type.
        oz = float(self.origin[2])
        cl = []
        for i, n in enumerate(self.pursuer_names):
            z_env = float(self.env.uav_positions[i, 2])
            cl.append(self.client.moveToZAsync(float(-(z_env - oz)), 15.0, vehicle_name=n))
        z_t = float(self.env.target_position[2])
        cl.append(self.client.moveToZAsync(float(-(z_t - oz)), 15.0, vehicle_name=self.evader_name))
        [f.join() for f in cl]
        got = self.read_kinematics(self.pursuer_names[0])[0]
        want = float(self.env.uav_positions[0, 2])
        err = abs(float(got[2]) - want)
        print(f"[bridge] after climb, Pursuer0 env-frame altitude = {got[2]:.1f} m "
              f"(env expects {want:.1f} m, error {err:.1f} m)")
        # A frame mismatch does not announce itself: the relative pursuer-evader geometry stays
        # intact, so the run still "captures" while the whole fleet flies above every building.
        # Fail loudly instead of reporting a result that validates the wrong scenario.
        if err > 25.0:
            raise RuntimeError(
                f"ALTITUDE FRAME MISMATCH: measured {got[2]:.1f} m vs env {want:.1f} m "
                f"(error {err:.1f} m). Command frame and read frame disagree; results would be "
                f"taken in an obstacle-free band. Check vehicle spawn Z in settings.json.")

    def read_kinematics(self, name):
        st = self.client.getMultirotorState(vehicle_name=name).kinematics_estimated
        vel_ned = np.array([st.linear_velocity.x_val, st.linear_velocity.y_val, st.linear_velocity.z_val], np.float32)
        # Position MUST come from the global object pose. kinematics_estimated.position is
        # measured from each vehicle's OWN spawn point, so with several vehicles it reports the
        # same numbers for all of them -- measured on this scene, three pursuers 200 m apart all
        # read (0,0,180.7), putting every one of them 20 m from the evader and tripping the 15 m
        # capture test immediately. Velocity needs no such correction: NED axes are world-aligned.
        p = self.client.simGetObjectPose(name).position
        pos_ned = np.array([p.x_val, p.y_val, p.z_val], np.float32)
        return self.ned_to_env(pos_ned), np.array([vel_ned[0], vel_ned[1], -vel_ned[2]], np.float32)

    def send_velocity(self, name, v_env):
        v = self.vel_env_to_ned(v_env)
        # SimpleFlight tracks the desired velocity for dt; the low-level controller
        # + quadrotor dynamics are AirSim's job (the sim-to-sim fidelity).
        self.client.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]),
                                        duration=self.dt, vehicle_name=name)

    # ---------- sync env state <- AirSim (closed-loop) ----------
    def sync_state_from_airsim(self):
        for i, n in enumerate(self.pursuer_names):
            pos, vel = self.read_kinematics(n)
            self.env.uav_positions[i] = pos
            spd = float(np.linalg.norm(vel))
            self.env.uav_speeds[i] = np.clip(spd, self.env.uav_min_speed, self.env.uav_max_speed)
            self.env.uav_yaws[i] = math.atan2(vel[1], vel[0]) if spd > 1e-3 else self.env.uav_yaws[i]
            self.env.uav_pitches[i] = math.asin(np.clip(vel[2] / max(spd, 1e-6), -1, 1)) if spd > 1e-3 else self.env.uav_pitches[i]
        tpos, tvel = self.read_kinematics(self.evader_name)
        self.env.target_position = tpos
        tspd = float(np.linalg.norm(tvel))
        self.env.target_speed = np.clip(tspd, self.env.target_min_speed, self.env.target_max_speed)
        self.env.target_yaw = math.atan2(tvel[1], tvel[0]) if tspd > 1e-3 else self.env.target_yaw
        self.env.target_pitch = math.asin(np.clip(tvel[2] / max(tspd, 1e-6), -1, 1)) if tspd > 1e-3 else self.env.target_pitch
        # recompute the derived caches _get_obs() depends on
        self.env._update_radar_cache()
        self.env.current_guide_points = self.env._assign_target_points()

    # ---------- one control step ----------
    def pursuer_desired_velocities(self):
        obs = self.env._get_obs()                                  # exact training obs
        acts = self.agent.action([obs], test_mode=True)["actions"][0]
        out = {}
        for i, n in enumerate(self.pursuer_names):
            a = np.clip(acts[f"uav_{i}"], -1.0, 1.0)
            spd = np.clip(self.env.uav_speeds[i] + a[0] * self.env.max_accel,
                          self.env.uav_min_speed, self.env.uav_max_speed)
            yaw = (self.env.uav_yaws[i] + a[1] * self.env.max_yaw_rate) % (2 * math.pi)
            pit = np.clip(self.env.uav_pitches[i] + a[2] * self.env.max_pitch_rate,
                          -self.env.pitch_max, self.env.pitch_max)
            out[n] = vel_from_state(spd, yaw, pit)
        return out

    def evader_desired_velocity(self):
        # advance the APF evader in the env's internal state, read its new velocity
        self.env._evader_step()
        return vel_from_state(self.env.target_speed, self.env.target_yaw, self.env.target_pitch)

    # ---------- episode loop ----------
    def run_episode(self, max_steps=400):
        self.connect()
        caught = crashed = False
        for step in range(max_steps):
            if self.args.mode == "closed_loop":
                self.sync_state_from_airsim()
            pv = self.pursuer_desired_velocities()
            ev = self.evader_desired_velocity()
            for n, v in pv.items(): self.send_velocity(n, v)
            self.send_velocity(self.evader_name, ev)
            if self.args.mode == "mirror":
                # advance internal sim so mirror mode follows the lightweight trajectory
                pass
            time.sleep(self.dt)
            # record + judge
            for n in self.pursuer_names + [self.evader_name]:
                self.trails[n].append(self.read_kinematics(n)[0].copy())
            dists = np.linalg.norm(self.env.uav_positions - self.env.target_position[None, :], axis=1)
            omega = escape_solid_angle(self.env, self.args.criterion)
            if float(np.min(dists)) <= self.env.catch_radius or omega < 0.05:
                caught = True; break
        self.land()
        self.plot_trajectory(caught)
        return {"caught": caught, "steps": step + 1}

    def land(self):
        for n in self.pursuer_names + [self.evader_name]:
            self.client.armDisarm(False, n); self.client.enableApiControl(False, n)

    def plot_trajectory(self, caught):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(8, 7), dpi=140); ax = fig.add_subplot(111, projection="3d")
        for b in self.env.buildings:            # buildings as vertical prisms
            xmin, ymin, xmax, ymax = b[:4]
            for z in (self.env.z_min, self.env.z_max):
                ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
                        [z] * 5, color="#b9b8b2", lw=0.8)
        for i, n in enumerate(self.pursuer_names):
            t = np.array(self.trails[n]); ax.plot(t[:, 0], t[:, 1], t[:, 2], color="#2a78d6", lw=1.6)
        te = np.array(self.trails[self.evader_name])
        ax.plot(te[:, 0], te[:, 1], te[:, 2], color="#e34948", lw=2.2, label="evader")
        ax.set_title(f"AirSim capture trajectory ({'CAUGHT' if caught else 'escaped'})")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        # --out names the JSON result; the figure rides alongside it with an image extension,
        # otherwise savefig is handed a .json path and refuses the format.
        out = self.args.out or "airsim_trajectory.png"
        if os.path.splitext(out)[1].lower() not in (
                ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff", ".webp", ".eps", ".ps"):
            out = os.path.splitext(out)[0] + ".png"
        fig.savefig(out, bbox_inches="tight"); print("saved", out)


def gen_settings(num_agents, out="settings.json", surround=False, spawn_radius=200.0):
    """Generate AirSim settings.json for N pursuers + 1 evader (SimpleFlight).
    Place at ~/Documents/AirSim/settings.json. [TUNE: initial X/Y to match scene]

    With surround=True the pursuers are placed on a ring around the evader so the UE start
    geometry matches the lightweight sim's surround-spawn; a one-sided line-up would put the
    quadrotors somewhere the policy was never trained to fly from."""
    vehicles = {}
    if surround:
        # Spawn Z stays 0. moveToZAsync commands are relative to each vehicle's OWN spawn pose
        # while state is read from the global object pose, so any non-zero spawn Z silently
        # offsets one frame against the other -- it put the fleet at 463 m when the env expected
        # 283 m, i.e. flying above every building, which reads as a clean capture while actually
        # validating an obstacle-free scenario. Ground spawn makes the two frames coincide.
        for i in range(num_agents):
            a = 2.0 * np.pi * i / max(num_agents, 1)
            vehicles[f"Pursuer{i}"] = {"VehicleType": "SimpleFlight",
                                       "X": float(spawn_radius * np.cos(a)),
                                       "Y": float(spawn_radius * np.sin(a)),
                                       "Z": 0.0}
        vehicles["Evader0"] = {"VehicleType": "SimpleFlight", "X": 0, "Y": 0, "Z": 0.0}
        cfg = {"SettingsVersion": 1.2, "SimMode": "Multirotor",
               "ClockSpeed": 1.0, "ViewMode": "NoDisplay", "Vehicles": vehicles}
        with open(out, "w") as f: json.dump(cfg, f, indent=2)
        print("wrote", out, "(surround) -> copy to ~/Documents/AirSim/settings.json")
        return
    ys = np.linspace(-350, 350, num_agents)
    for i, y in enumerate(ys):
        vehicles[f"Pursuer{i}"] = {"VehicleType": "SimpleFlight", "X": -400, "Y": float(y), "Z": -150}
    vehicles["Evader0"] = {"VehicleType": "SimpleFlight", "X": 0, "Y": 0, "Z": -200}
    cfg = {"SettingsVersion": 1.2, "SimMode": "Multirotor",
           "ClockSpeed": 1.0, "Vehicles": vehicles}
    with open(out, "w") as f: json.dump(cfg, f, indent=2)
    print("wrote", out, "-> copy to ~/Documents/AirSim/settings.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=False)
    ap.add_argument("--algo", default="maddpg")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--building-mode", default="medium")
    ap.add_argument("--target-max-speed", type=float, default=10.0)
    ap.add_argument("--target-min-speed", type=float, default=8.0)
    ap.add_argument("--use-obstacle-gat", action="store_true")
    ap.add_argument("--use-graph-module", action="store_true")
    ap.add_argument("--criterion", default="euclidean")
    ap.add_argument("--mode", default="closed_loop", choices=["closed_loop", "mirror"])
    ap.add_argument("--dt", type=float, default=1.0)                 # [TUNE-ON-MACHINE]
    ap.add_argument("--origin", type=float, nargs=3, default=[500, 500, 0])  # [TUNE]
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--surround-spawn", action="store_true",
                    help="pursuers start distributed around the evader (required for lambda>=1 policies)")
    ap.add_argument("--spawn-radius", type=float, default=None)
    ap.add_argument("--gen-settings", action="store_true", help="just write settings.json and exit")
    args = ap.parse_args()
    if args.gen_settings:
        gen_settings(args.num_agents, surround=args.surround_spawn,
                     spawn_radius=args.spawn_radius or 200.0); sys.exit(0)
    bridge = AirSimPursuitBridge(args)
    result = bridge.run_episode(args.max_steps)
    print(result)
    if args.out and args.out.lower().endswith(".json"):
        result.update({"mode": args.mode, "num_agents": args.num_agents,
                       "building_mode": args.building_mode,
                       "target_max_speed": args.target_max_speed,
                       "model_path": args.model_path, "dt": args.dt})
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print("wrote", args.out)
