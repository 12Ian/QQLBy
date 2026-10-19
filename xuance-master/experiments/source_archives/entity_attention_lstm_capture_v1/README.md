# Entity attention + frozen LSTM, seed 1

Episode limit: 600 decisions / 300 seconds. Training: six million environment steps,
16 parallel environments, fixed curriculum level five. Validation/test scenes,
reward, safety filtering and LSTM checkpoint match lstm_matd3_capture_long_v1_20261019.
The entity policy starts randomly; the historical MLP policy warm-started from exp020.
Consequently this is an outcome comparison with different initialization.

Restore source in two steps, from the xuance-master repository folder:

1. python tools/restore_experiment_source.py --archive experiments/source_archives/lstm_matd3_capture_long_v1 --destination /new/checkout
2. python experiments/source_archives/entity_attention_lstm_capture_v1/apply_overlay.py --destination /new/checkout

The destination is the repository checkout containing xuance-master, not the code folder.
Only source is restored; configuration paths and checkpoint dependencies still refer
to the original server. Set new artifact paths explicitly before any new run.
