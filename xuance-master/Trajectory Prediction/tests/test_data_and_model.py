import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import (Track, Windows, fit_normalizer, generate_tracks,
                                       load_tracks, split_tracks, validate_split)
from trajectory_prediction.model import Forecaster, TemporalGraphAttention, constant_velocity
from trajectory_prediction.objective import displacement_errors, loss_components


def small_config(**values):
    defaults = dict(observation_steps=6, prediction_steps=4, stride=3, lstm_hidden=8,
                    projection_dim=8, graph_dim=12, heads=3, graph_layers=1,
                    batch_size=8, dropout=0.0, epochs=2, device="cpu", cpu_threads=1)
    defaults.update(values)
    return Config(**defaults).validate()


def test_euclidean_ade_fde_and_constant_velocity():
    truth = torch.zeros(2, 3, 3)
    prediction = torch.tensor([[[3., 4., 0.]] * 3, [[0., 0., 2.]] * 3])
    ade, fde = displacement_errors(prediction, truth)
    torch.testing.assert_close(ade, torch.tensor([5., 2.]))
    torch.testing.assert_close(fde, torch.tensor([5., 2.]))
    observed = torch.zeros(1, 6, 6)
    observed[:, -1, :3] = torch.tensor([10., 2., -3.])
    observed[:, -1, 3:] = torch.tensor([2., -1., 0.])
    forecast = constant_velocity(observed, 4, 0.5)
    torch.testing.assert_close(forecast[0, -1], torch.tensor([14., 0., -3.]))


def test_track_split_no_leakage_train_only_normalization(tmp_path):
    cfg = small_config()
    path = tmp_path / "tracks.npz"
    generate_tracks(path, count=12, steps=24)
    tracks = load_tracks(path, cfg.dt, 10)
    splits = split_tracks(tracks, 0.7, 0.15, 42)
    validate_split(tracks, splits)
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    dataset = Windows(tracks, splits["train"], cfg)
    statistics = fit_normalizer(dataset)
    for track in tracks:
        if track.track_id not in splits["train"]:
            track.states *= 1000
    altered = fit_normalizer(Windows(tracks, splits["train"], cfg))
    torch.testing.assert_close(statistics["mean"], altered["mean"])
    torch.testing.assert_close(statistics["std"], altered["std"])


def test_observation_label_never_uses_future():
    cfg = small_config()
    labels = np.asarray([1] * 6 + [8] * 4)
    track = Track("a", np.arange(10) * 0.5, np.zeros((10, 6), dtype=np.float32), labels)
    assert Windows([track], ["a"], cfg)[0]["label"].item() == 1


def test_invalid_dt_and_noninteger_labels_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    values = dict(states=np.zeros((10, 6)), track_id=np.asarray(["a"] * 10),
                  time=np.arange(10) * 0.5, labels=np.zeros(10))
    values["time"][4] = values["time"][3]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="timestamps"):
        load_tracks(path, 0.5, 10)
    values["time"] = np.arange(10) * 0.5
    values["labels"][0] = 1.5
    np.savez(path, **values)
    with pytest.raises(ValueError, match="integers"):
        load_tracks(path, 0.5, 10)


@pytest.mark.parametrize("radius", [0, 2])
def test_attention_masks_normalize_and_receive_gradients(radius):
    torch.manual_seed(3)
    layer = TemporalGraphAttention(12, 3, radius, 0)
    nodes = torch.randn(2, 6, 12, requires_grad=True)
    # Tiny gates must remain stable rather than yield 0/0.
    with torch.no_grad():
        layer.node_gate.bias.fill_(-100)
        layer.edge_gate.bias.fill_(-100)
    output, attention = layer(nodes)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3, 6))
    index = torch.arange(6)
    masked = (index[:, None] - index[None, :]).abs() > radius
    assert (attention[:, :, masked] == 0).all()
    output.square().sum().backward()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
    assert layer.edge_gate.weight.grad is not None


def test_forecast_translation_equivariance_and_shapes():
    torch.manual_seed(1)
    cfg = small_config()
    model = Forecaster(cfg).eval()
    with torch.no_grad():
        model.decoder[-1].weight.normal_(std=0.03)
    observed = torch.randn(2, 6, 6)
    shift = torch.tensor([20., -10., 30.])
    shifted = observed.clone()
    shifted[:, :, :3] += shift
    original, translated = model(observed), model(shifted)
    assert original["trajectory"].shape == (2, 4, 3)
    torch.testing.assert_close(translated["trajectory"], original["trajectory"] + shift, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(original["probabilities"].sum(-1), torch.ones(2))


def test_gradients_and_single_batch_learning(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    cfg = small_config()
    data = tmp_path / "tracks.npz"
    generate_tracks(data, count=12, steps=24)
    tracks = load_tracks(data, cfg.dt, 10)
    dataset = Windows(tracks, [t.track_id for t in tracks], cfg)
    examples = [dataset[i] for i in range(8)]
    batch = {k: torch.stack([sample[k] for sample in examples]) for k in ("observed", "future", "label")}
    model = Forecaster(cfg, fit_normalizer(dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = float(loss_components(model(batch["observed"]), batch, cfg)["loss"].detach())
    gate_gradient = 0.0
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_components(model(batch["observed"]), batch, cfg)["loss"]
        loss.backward()
        if step > 0:
            gate_gradient += float(model.graph[0].edge_gate.weight.grad.abs().sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = float(loss_components(model(batch["observed"]), batch, cfg)["loss"].detach())
    assert final < initial * 0.7, (initial, final)
    assert gate_gradient > 0


def test_frozen_intention_encoder_stays_deterministic():
    cfg = small_config(stage="forecast", freeze_intention=True, dropout=0.5)
    model = Forecaster(cfg).train()
    assert not model.intention.training
    assert not any(p.requires_grad for p in model.intention.parameters())
