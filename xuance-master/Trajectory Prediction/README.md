# Trajectory Prediction: runnable first stage

This folder implements the first reproducible stage of Yu et al. (2026): a
paper-scale 3-D engagement generator, track-level data splits, nine-class
BiLSTM intention recognition, and constant-velocity/constant-acceleration
prediction baselines. The full module plan and the paper's unresolved details
are in [REPRODUCTION_FRAMEWORK.md](REPRODUCTION_FRAMEWORK.md).

Run from this directory on the server with the existing `rl_env` environment:

```bash
export PYTHONPATH=src
PY=/home/ryy/miniconda3/envs/rl_env/bin/python
$PY -m trajectory_prediction.cli generate --config configs/paper_intention_scene.yaml --output outputs/intention_tracks --tracks 100 --seed 1
$PY -m trajectory_prediction.cli train-intention --data outputs/intention_tracks --epochs 70 --seed 1
$PY -m trajectory_prediction.cli generate --config configs/paper_scene.yaml --output outputs/prediction_tracks --tracks 100 --seed 100001
$PY -m trajectory_prediction.cli eval-baselines --data outputs/prediction_tracks --seed 1
```

These 100-track commands are a functional example, not the paper's roughly
50,000-window scale. With the current explicit window strides and 24 s tracks,
each track yields six intention windows and eleven prediction windows. Use
roughly 8,334 intention tracks and 4,546 prediction tracks for that scale;
choose fresh output directories and record the exact counts from the run.
For a small smoke run, use at least three tracks and one epoch. Generated
datasets and checkpoints are stored under `outputs/` and ignored by Git.

## Reproduction decisions in this stage

- Paper facts: one target and one defender; constant-speed velocity-normal
  point-mass dynamics; target/defender overload limits; initial geometry and
  speed ranges; terminal anti-LOS evasion trigger at two thirds of initial range.
- Explicit assumptions: defender turns toward current LOS; primitive segments
  last 2–5 s; terminal evasion is blended with the ongoing primitive when the
  defender is in the target FOV. Nine-class labels are derived from the applied
  acceleration in the target's local left/up basis.
- The intention library requests 50% initially visible engagements; the main
  trajectory configuration starts inside the target FOV. These are separate
  generated datasets, matching the paper's distinct experiments.
- Samples are split by complete engagement before sliding windows are built.
  Intention recognition uses 70/15/15 train/validation/test and fits
  standardization on training windows only. CV/CA evaluation uses a separate
  80/20 train/evaluation split of prediction tracks.
- Stored acceleration at time `t` describes the completed interval ending at
  `t`; the prediction baseline does not read an upcoming command.

The current cov environment is a separate low-speed, multi-pursuer scenario.
Its adapter and the paper's graph/cropper/teacher-loss predictor are subsequent
modules described in the framework document; they are not claimed as present in
this first executable stage.
