# Frozen LSTM observations for standard MATD3

This integration uses the validated epoch-4 predictor from the 5,000-episode supervised run.
It does not train the predictor jointly with MATD3.

## Configuration

- `uav_prediction_mode`: `none` (default), `cv`, or `lstm`.
- `uav_prediction_checkpoint` and `uav_prediction_sha256`: required for LSTM.
- `uav_prediction_device`: inference device in the parent vector-environment process.
- `uav_prediction_migrate_checkpoint`: explicitly migrate a baseline policy's input columns.
- `uav_prediction_diagnostic_dir`: save causal-history snapshots if inference fails.

Use the three `experiments/matd3_prediction_{none,cv,lstm}_600step.yaml` templates.
Each template retains the same source policy, task, reward and safety settings.
The predictor adds target displacement at 1/2/3 seconds, divided by 100, before the building block.
Observations are 88-dimensional (79 without prediction); no validity flag is added.
Histories contain eleven raw frames at 0.5-second intervals. Before five seconds, use CV.
The original four-dimensional MATD3 curriculum context is retained; the LSTM has no level,
building count or capture radius input.

The current checkpoint supports four agents, the recorded 79-dimensional observation layout,
five local buildings, a 1,000-metre map, altitude 50--350 metres and decision dt=0.5.
The old analytic forecast observation switch must remain disabled.
The predictor is an observation feature, not a new reward or a replacement safety filter.

## Runtime and provenance

Implementation and validation were performed on the server branch `feat-coverage-reaward`.
The repository and the earlier experiment runtime already differ in evader avoidance and
other prior experiment changes. This integration does not silently merge those historical changes.

For reproducible runs, use:
`/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/lstm_matd3_integration_v1_20261019/code`

That directory copies the previously validated 600-step runtime and applies the same integration.
It preserves the dynamics used to collect the predictor dataset. Earlier experiment snapshots
and source policy/predictor checkpoints remain unchanged.

## Validation commands

Run on the server using the existing experiment Python environment:

```bash
RUN=/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/lstm_matd3_integration_v1_20261019
PY=/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/exp022_perf_v1/venv/bin/python
cd "$RUN/code"
UAV_LSTM_TEST_RUN="$RUN" PYTHONPATH="$RUN/code" "$PY" -m pytest tests/test_uav_lstm_integration.py -q
PYTHONPATH="$RUN/code" "$PY" tools/validate_lstm_matd3_integration.py --run "$RUN" --steps 1024
```

The learning check runs all three modes for 1,024 environment transitions each at level five.
It checks real actor/critic updates, observation/replay dimensions and predictor immutability.
It is not a capture-rate comparison and does not establish improved pursuit success.

The replay buffer starts empty and stores augmented obs/obs_next. Predictions are computed
during collection, never from future labels or recomputed from shuffled replay samples.
Terminal observations use old-episode history; reset observations use a newly cleared history.
All six online/target actor and twin-critic first layers are migrated by explicit column mapping,
with new columns initialized to zero. Unexplained key/shape changes fail.

A sustained three-arm, multi-seed experiment remains necessary before judging capture-rate gains.
