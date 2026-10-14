"""Episode outcomes and numeric update summaries for numbered baseline runs."""
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path


class RewardLossTensorBoard:
    """Average episode rewards in small batches; expose only rewards and losses."""
    reward_parts = {"r_purs", "r_obstacle", "r_boundary", "r_collision", "r_terminal", "r_failure"}

    def __init__(self, writer):
        self.writer = writer
        self.reward_sums = defaultdict(float)
        self.reward_count = 0
        self.level = None
        self.last_step = 0

    def log(self, info, step):
        if "Train-Returns/team_mean" in info:
            level = info["Train-Curriculum/Level"]
            if self.level is not None and self.level != level:
                self.flush_rewards()
            self.level = level
            self.last_step = step
            for name, value in info.items():
                if (name.startswith("Train-Returns/") or
                        (name.startswith("Train-Rewards/") and name.rsplit("/", 1)[-1] in self.reward_parts)):
                    self.reward_sums[name] += float(value)
            self.reward_count += 1
            if self.reward_count == 20:
                self.flush_rewards()
        for metric in ("actor", "critic"):
            value = info.get(f"Train-Learning/loss_{metric}_mean")
            if value is not None:
                self.writer.add_scalar(f"Loss/loss_{metric}", float(value), step)
        prefix = "Test-Results/Episode-Rewards-Level-"
        for name, value in info.items():
            if name.startswith(prefix):
                level = int(name[len(prefix):]) + 1
                self.writer.add_scalar(f"Eval-Rewards/Level-{level}", float(value), step)

    def flush_rewards(self):
        if self.reward_count:
            for name, total in self.reward_sums.items():
                self.writer.add_scalar(name, total / self.reward_count, self.last_step)
            self.reward_sums.clear()
            self.reward_count = 0


class BaselineTrainingLog:
    def __init__(self, directory, log_infos):
        self.directory = Path(directory)
        self.log_infos = log_infos
        self.episode_count = self.success_count = self.update_count = 0
        self.actor_update_count = 0
        self.recent = defaultdict(lambda: deque(maxlen=100))
        self.reward_windows = defaultdict(lambda: deque(maxlen=100))
        self.pending_updates = []
        self.episode_stream = (self.directory / "training_episodes.csv").open("w", newline="", encoding="utf-8")
        self.update_stream = (self.directory / "training_updates.csv").open("w", newline="", encoding="utf-8")
        episode_fields = ["step", "env", "level", "building_count", "episode_steps", "success",
                          "safe_success", "collision", "capture_type", "mean_return", "min_target_distance",
                          "building_contact_seconds", "boundary_contact_seconds", "uav_contact_seconds",
                           "r_distance", "r_obstacle", "r_boundary", "r_teammate", "r_collision", "r_terminal",
                           "r_progress", "r_failure", "r_purs", "r_form", "episode_seconds",
                           "pursuit_raw_mean", "formation_raw_mean", "initial_target_distance_mean", "final_target_distance_mean",
                           "target_speed_mean", "target_speed_max", "target_sensed_fraction",
                           "safety_checks", "safety_interventions", "safety_uncertified_substeps",
                           "safety_mean_correction"]
        self.episodes = csv.DictWriter(self.episode_stream, fieldnames=episode_fields)
        self.updates = csv.DictWriter(self.update_stream, fieldnames=[
            "step", "updates", "window_updates", "actor_updates", "window_actor_updates",
            "loss_actor_mean", "loss_critic_mean",
            "predictQ_mean", "loss_critic_max", "predictQ_min", "predictQ_max", "noise_scale",
            "predict_value_mean"])
        self.episodes.writeheader()
        self.updates.writeheader()

    def record_episode(self, step, env_id, info):
        diagnostics = info["infos"]
        level = int(diagnostics["curriculum_level"])
        success = bool(info["is_success"])
        collision = (any(diagnostics[f"{kind}_contact_events_episode"] > 0
                         for kind in ("building", "boundary", "uav")) or info["episode_uav_conflicts"] > 0)
        sub_rewards = info["episode_sub_rewards"]
        row = dict(step=step, env=env_id, level=level, building_count=diagnostics["building_count"],
                   episode_steps=info["episode_step"], success=int(success), safe_success=int(success and not collision),
                   collision=int(collision), capture_type=info["capture_type"] or (
                       "collision" if info.get("is_collision") else "timeout"),
                   mean_return=sum(info["episode_score"].values()) / len(info["episode_score"]),
                   min_target_distance=diagnostics["min_target_distance_episode"])
        for kind in ("building", "boundary", "uav"):
            row[f"{kind}_contact_seconds"] = diagnostics[f"{kind}_contact_seconds_episode"]
        for key in next(iter(sub_rewards.values())):
            row[key] = sum(rewards[key] for rewards in sub_rewards.values()) / len(sub_rewards)
        initial_distances = diagnostics.get("initial_target_distances", [])
        final_distances = diagnostics.get("target_distances_current", [])
        row.update(initial_target_distance_mean=(sum(initial_distances) / len(initial_distances)
                                                 if initial_distances else None),
                   final_target_distance_mean=(sum(final_distances) / len(final_distances)
                                               if final_distances else None),
                   target_speed_mean=diagnostics.get("target_speed_mean_episode"),
                   target_speed_max=diagnostics.get("target_speed_max_episode"),
                   target_sensed_fraction=diagnostics.get("target_sensed_fraction_episode"),
                   episode_seconds=diagnostics.get("episode_elapsed_seconds"),
                   pursuit_raw_mean=diagnostics.get("pursuit_raw_mean_episode"),
                   formation_raw_mean=diagnostics.get("formation_raw_mean_episode"),
                   safety_checks=diagnostics.get("safety_checks_episode", 0),
                   safety_interventions=diagnostics.get("safety_interventions_episode", 0),
                   safety_uncertified_substeps=diagnostics.get("safety_uncertified_substeps_episode", 0),
                   safety_mean_correction=diagnostics.get("safety_mean_correction_episode", 0.))
        self.episodes.writerow(row)
        self.episode_stream.flush()
        self.episode_count += 1
        self.success_count += int(success)
        self.recent[level].append((int(success), int(success and not collision), int(collision),
                                  int(diagnostics["building_contact_events_episode"] > 0),
                                  int(diagnostics["boundary_contact_events_episode"] > 0),
                                  int(row["capture_type"] == "timeout")))
        recent = self.recent[level]
        rates = [sum(item[i] for item in recent) / len(recent) for i in range(3)]
        self.log_infos({f"Baseline-Train/Success-Rate-Level-{level}": rates[0],
                        f"Baseline-Train/Safe-Success-Rate-Level-{level}": rates[1],
                        f"Baseline-Train/Collision-Rate-Level-{level}": rates[2]}, step)
        self._log_episode_details(step, info, row)
        if success and self.success_count == 1:
            print(f"[first training success] step={step} level={level} type={info['capture_type']}", flush=True)
        if self.episode_count % 100 == 0:
            print(f"[training episodes] step={step} episodes={self.episode_count} successes={self.success_count} "
                  f"level={level} recent_success={rates[0]:.3f} recent_safe={rates[1]:.3f} "
                  f"recent_collision={rates[2]:.3f}", flush=True)
        status = dict(step=step, episodes=self.episode_count, successes=self.success_count,
                      level=level, recent_success_rate=rates[0], recent_safe_success_rate=rates[1],
                      recent_collision_rate=rates[2], updates=self.update_count,
                      actor_updates=self.actor_update_count)
        temporary = self.directory / "training_status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "training_status.json")

    def _log_episode_details(self, step, info, row):
        """Write flat scalars to the main run; rewards are scaled episode sums."""
        diagnostics = info["infos"]
        level = row["level"]
        episode_steps = max(1, int(row["episode_steps"]))
        sub_rewards = info["episode_sub_rewards"]
        values = {"Train-Episode/Steps": row["episode_steps"],
                  "Train-Episode/Seconds": row["episode_seconds"],
                  "Train-Pursuit/Raw-Mean": row["pursuit_raw_mean"],
                  "Train-Formation/Raw-Mean-Radians": row["formation_raw_mean"],
                  "Train-Episode/Completed-Episodes": self.episode_count,
                  "Train-Episode/Success": row["success"],
                  "Train-Episode/Safe-Success": row["safe_success"],
                  "Train-Episode/Collision": row["collision"],
                  "Train-Episode/Timeout": int(row["capture_type"] == "timeout"),
                  "Train-Curriculum/Level": level + 1,
                  "Train-Returns/team_mean": row["mean_return"],
                  "Train-Distance/Initial-Mean": row["initial_target_distance_mean"],
                  "Train-Distance/Final-Mean": row["final_target_distance_mean"],
                  "Train-Distance/Closest-Reached": row["min_target_distance"],
                  "Train-Evader/Speed-Mean": row["target_speed_mean"],
                  "Train-Evader/Speed-Max": row["target_speed_max"],
                  "Train-Evader/Speed-Final": diagnostics.get("target_speed_current"),
                  "Train-Evader/Sensed-Decision-Fraction": row["target_sensed_fraction"],
                  "Train-Evader/Wall-Contacts": diagnostics.get("target_wall_contacts_episode")}
        reward_names = next(iter(sub_rewards.values())).keys()
        pursuit = diagnostics.get("pursuit_diagnostics", {})
        for name in ("closing_mean", "closing_min", "r_purs_raw", "r_form_raw"):
            values[f"Train-Pursuit/Final/{name}"] = pursuit.get(name)
        for index, agent in enumerate(sub_rewards):
            for name in ("closing_speeds", "angles_radians", "nearest_teammates"):
                measurements = pursuit.get(name, [])
                if len(measurements) > index:
                    values[f"Train-Pursuit/{agent}/{name}"] = measurements[index]
        team_rewards = {name: row[name] for name in reward_names}
        self.reward_windows[level].append(dict(team_rewards, total=row["mean_return"]))
        for index, (agent, components) in enumerate(sub_rewards.items()):
            values[f"Train-Returns/{agent}"] = info["episode_score"][agent]
            for name, value in components.items():
                values[f"Train-Rewards/{agent}/{name}"] = value
                values[f"Train-Rewards-Per-Step/{agent}/{name}"] = value / episode_steps
                if row["episode_seconds"]:
                    values[f"Train-Rewards-Per-Second/{agent}/{name}"] = value / row["episode_seconds"]
            for name, diagnostic in (("Initial", "initial_target_distances"),
                                     ("Final", "target_distances_current")):
                measurements = diagnostics.get(diagnostic, [])
                if len(measurements) > index:
                    values[f"Train-Distance/{agent}/{name}"] = measurements[index]
            for name, diagnostic in (("Final-Speed", "uav_speeds_current"),
                                     ("Final-Acceleration", "uav_accelerations_current")):
                measurements = diagnostics.get(diagnostic, [])
                if len(measurements) > index:
                    values[f"Train-UAV/{agent}/{name}"] = measurements[index]
        for name, value in team_rewards.items():
            values[f"Train-Rewards/team/{name}"] = value
            values[f"Train-Rewards-Per-Step/team/{name}"] = value / episode_steps
        window = self.reward_windows[level]
        for name in list(reward_names) + ["total"]:
            values[f"Train-Rewards-Rolling100/team/{name}"] = sum(r[name] for r in window) / len(window)
        for index, name in enumerate(("Success", "Safe-Success", "Collision", "Building-Collision",
                                      "Boundary-Collision", "Timeout")):
            recent = self.recent[level]
            values[f"Train-Outcomes-Rolling100/{name}"] = sum(r[index] for r in recent) / len(recent)
        for kind in ("building", "boundary", "uav"):
            values[f"Train-Safety/{kind}-Contact-Seconds"] = row[f"{kind}_contact_seconds"]
        for name in ("building_count", "capture_radius", "target_min_speed", "target_max_speed",
                     "target_accel", "uav_accel_max", "evader_sense"):
            values[f"Train-Curriculum/{name}"] = diagnostics.get(name)
        self.log_infos({name: float(value) for name, value in values.items()
                        if value is not None and math.isfinite(float(value))}, step)

    def record_update(self, step, info):
        def metric(*names):
            return next((float(v) for k, v in info.items()
                         if any(k.endswith('/' + name) for name in names)), None)

        actor = metric("loss_actor", "actor_loss")
        critic = metric("loss_critic", "critic_loss")
        if critic is None:
            raise ValueError("Baseline update has no critic loss")
        values = [actor, critic, metric("predictQ"), metric("predict_value")]
        if not all(value is None or math.isfinite(value) for value in values):
            raise FloatingPointError(f"Nonfinite baseline update at step {step}: {values}")
        self.update_count += 1
        self.actor_update_count += int(actor is not None)
        self.pending_updates.append(values)
        self.last_update_step, self.last_noise = step, info.get("noise_scale")
        if len(self.pending_updates) >= 1000:
            self.flush_updates()

    def flush_updates(self):
        if not self.pending_updates:
            return
        values = self.pending_updates
        actor_values = [v[0] for v in values if v[0] is not None]
        q_values = [v[2] for v in values if v[2] is not None]
        value_predictions = [v[3] for v in values if v[3] is not None]
        row = dict(step=self.last_update_step, updates=self.update_count, window_updates=len(values),
                   actor_updates=self.actor_update_count, window_actor_updates=len(actor_values),
                   loss_actor_mean=sum(actor_values) / len(actor_values) if actor_values else None,
                   loss_critic_mean=sum(v[1] for v in values) / len(values),
                   predictQ_mean=sum(q_values) / len(q_values) if q_values else None,
                   loss_critic_max=max(v[1] for v in values),
                   predictQ_min=min(q_values) if q_values else None,
                   predictQ_max=max(q_values) if q_values else None, noise_scale=self.last_noise,
                   predict_value_mean=(sum(value_predictions) / len(value_predictions)
                                       if value_predictions else None))
        self.updates.writerow(row)
        self.update_stream.flush()
        self.pending_updates.clear()
        self.log_infos({f"Train-Learning/{name}": float(value) for name, value in row.items()
                        if name != "step" and value is not None and math.isfinite(float(value))},
                       self.last_update_step)
        actor_text = f"{row['loss_actor_mean']:.3f}" if actor_values else "skipped"
        prediction = (f"Q={row['predictQ_mean']:.3f}" if q_values else
                      f"value={row['predict_value_mean']:.3f}" if value_predictions else "prediction=n/a")
        noise = f"{row['noise_scale']:.3f}" if row['noise_scale'] is not None else "n/a"
        print(f"[training updates] step={row['step']} updates={self.update_count} "
              f"actor_updates={self.actor_update_count} actor={actor_text} critic={row['loss_critic_mean']:.3f} "
              f"{prediction} noise={noise}", flush=True)

    def close(self):
        self.flush_updates()
        self.episode_stream.close()
        self.update_stream.close()
