# Standalone UAV LSTM target predictor

Server implementation for the accepted 71-feature prediction-video experiment.
No MATD3 actor, critic, observation, reward or replay changes are included.

Experiment: outputs/experiments/lstm_prediction_v1_20261019
See REPORT.md and protocol.json there for data provenance, fixed splits and selected video cases.

Run using the verified experiment Python and PYTHONPATH pointing to the frozen source directory:
python -m uav_lstm_prediction.tests --run EXPERIMENT
python -m uav_lstm_prediction.train --run EXPERIMENT
python -m uav_lstm_prediction.render --run EXPERIMENT

The collector loads read-only policy checkpoints and the existing frozen 600-step simulator.
Data uses causal position differences, never upcoming controls or future states as input.
