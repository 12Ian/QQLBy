# Trajectory Prediction: runnable reproduction framework

This folder implements a paper-scale 3-D engagement generator, track-level
data splits, nine-class BiLSTM intention recognition, CV/CA baselines, and an
initial intention-conditioned trajectory predictor. The predictor uses a
two-layer target/defender interaction graph, nine physical maneuver candidates,
feasibility masking with Top-K selection, and a GRU displacement decoder. The
training objective now adds six candidate-attention heads partitioned into
energy, interception-risk, and task-timeliness groups. Teacher scores supervise
these heads and the soft cropper during training only. The full module plan and
the paper's unresolved details are in
[REPRODUCTION_FRAMEWORK.md](REPRODUCTION_FRAMEWORK.md).

Run from this directory on the server with the existing `rl_env` environment:

```bash
export PYTHONPATH=src
PY=/home/ryy/miniconda3/envs/rl_env/bin/python
$PY -m trajectory_prediction.cli generate --config configs/paper_intention_scene.yaml --output outputs/intention_tracks --tracks 100 --seed 1
$PY -m trajectory_prediction.cli train-intention --data outputs/intention_tracks --epochs 70 --seed 1
$PY -m trajectory_prediction.cli generate --config configs/paper_scene.yaml --output outputs/prediction_tracks --tracks 100 --seed 100001
$PY -m trajectory_prediction.cli eval-baselines --data outputs/prediction_tracks --seed 1
$PY -m trajectory_prediction.cli train-prediction --data outputs/prediction_tracks --intention-checkpoint outputs/intention/intention_best.pt --output outputs/prediction --epochs 200 --seed 1
$PY -m trajectory_prediction.cli generate --config configs/paper_scene.yaml --output outputs/independent_test_tracks --tracks 100 --seed 200001
$PY -m trajectory_prediction.cli eval-prediction --data outputs/independent_test_tracks --intention-checkpoint outputs/intention/intention_best.pt --prediction-checkpoint outputs/prediction/prediction_best.pt --output outputs/prediction_test
$PY -m trajectory_prediction.cli rollout --track outputs/independent_test_tracks/track_000000.npz --intention-checkpoint outputs/intention/intention_best.pt --prediction-checkpoint outputs/prediction/prediction_best.pt --output outputs/rolling_test --update-seconds 0.5
$PY -m trajectory_prediction.cli probe-cov-map --output outputs/cov_map_probe --live-resets 20
```

For the geometric-loss ablation, run `train-prediction` again with
`--disable-teacher` and a different `--output` directory. The resolved setting
is saved with the checkpoint.

These 100-track commands are a functional example, not the paper's roughly
50,000-window scale. With the current explicit window strides and 24 s tracks,
each track yields six intention windows, eleven CV/CA windows, and nine model
windows (the model also requires a 10 s intention prefix). Use roughly 8,334
intention tracks and 5,556 model prediction tracks for that scale;
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
- The predictor receives frozen BiLSTM intention probabilities computed from
  the preceding 10 s. It uses only observed states at inference. A separate
  generated directory is required for the final `eval-prediction` command;
  that command reports model, CV, and CA on exactly the same windows.
- `rollout` refreshes the intention posterior and 6 s forecast every 0.5 s
  from an independent full track. It saves each forecast, the matching truth,
  candidate weights, and per-update ADE/FDE; truth is used only for scoring.
- Candidate generation uses nine 6g constant-speed maneuvers, a ground-plane
  feasibility mask, an FOV/threat-gated escape-rate bias, and Top-K 8. Straight
  flight is retained when feasible; if every candidate is blocked, it serves
  as a flagged numerical fallback. These are declared implementation choices
  where the paper omits exact parameters.
  Their values and the `0.4 ADE + 0.6 FDE` weights are stored in
  `configs/prediction.yaml`; the model is trained with AdamW.
- Teacher energy scores use overload, jerk, and braking terms; risk scores use
  defender separation, range rate, and a reachable-radius proxy; timeliness
  scores use a mission corridor. The missing mission corridor is explicitly
  represented by straight flight from the last observed target state, while
  defender future motion uses observed constant velocity. Risk-head attention
  is trained to identify threats; cropper desirability subtracts risk. The
  teacher targets never use future truth, and are not computed at inference.
- Earlier `prediction_best.pt` files from before the factor-head addition have
  incompatible model weights; rerun `train-prediction` for this version.

The current cov environment is a separate low-speed, multi-pursuer scenario.
`cov_map.py` reads its configured building prisms and can snapshot a live
`UAVPursuitCoverage3DEnv` after reset; passing the map to the predictor masks
candidates that leave its 1000 m square, cross the 50–350 m altitude limits,
or enter a building with the target radius included. The `probe-cov-map`
command uses the actual cov `open` map and target acceleration, writes a JSON
report and top/side-view PNG, then samples real environment resets. In a
controlled approach to the first building, 3/9 candidates remain valid; 20
seeded resets of the sparse `open` map yielded 9/9 valid candidates at their
initial states. The probe also feeds 80 actual cov states to the map-conditioned
predictor; to match its 0.1 s history it overrides the cov decision interval to
0.1 s and uses zero acceleration actions. This is a **map-geometry test with
untrained predictor weights**,
not a trajectory accuracy comparison. Applying the map to paper-scale tracks
raises an error because those positions lie outside cov's coordinate bounds.

Cov trajectory collection and scale-matched model training, the paper's
mission corridor specification, and exact edgewise teacher supervision remain
subsequent modules. The available
graph/candidate model is an explicit implementation of the paper's method
structure, not a claim to reproduce the unpublished candidate and mask rules.

The concrete cov data and evaluation plan is in
[NEXT_STEPS_COV.md](NEXT_STEPS_COV.md).
