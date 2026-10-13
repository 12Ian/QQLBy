import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import generate_tracks, rotate_xy
from trajectory_prediction.engine import train, load_checkpoint, choose_device
from trajectory_prediction.metrics import EvaluationDetails
from trajectory_prediction.model import Forecaster, TemporalGraphAttention, bound_increments, build_forecaster
from trajectory_prediction.objective import attention_terms
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.workflows import classifier_config, pipeline, compare
from trajectory_prediction.xuance_bridge import framework, runner


def settings(**updates):
    values = dict(observation_steps=6, prediction_steps=4, stride=5,
                  lstm_hidden=8, projection_dim=8, graph_dim=12, heads=6, graph_head_dim=4,
                  graph_layers=1, graph_top_k=3, temporal_radius=2, temporal_layers=1,
                  decoder_type="autoregressive_gru", static_state_branch=True, relative_frame="yaw",
                  attention_supervision=True, loss_balance="uncertainty", optimizer="adamw",
                  schedule="warmup_cosine", warmup_epochs=1, schedule_epochs=4,
                  dropout=0.1, batch_size=8, epochs=1, device="cpu", cpu_threads=1,
                  max_speed_mps=10, max_acceleration_mps2=1, baseline_width=12)
    values.update(updates)
    return Config(**values).validate()


def test_native_registry_runner_learner_and_cleanup(tmp_path):
    framework()
    from xuance.engine import RunnerBase
    from xuance.torch.learners.learner import Learner
    from xuance.extensions.trajectory_prediction.learner import XuanceTrajectoryLearner
    assert issubclass(XuanceTrajectoryLearner, Learner)
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24)
    native = runner(data, tmp_path / "native", settings())
    assert isinstance(native, RunnerBase)
    result = native.run(mode="benchmark")
    assert native.agent.closed
    assert result["test"]["samples"] > 0
    manifest = json.loads((tmp_path / "native/manifest.json").read_text(encoding="utf-8"))
    assert manifest["integration"]["framework"] == "xuance"
    assert manifest["integration"]["learner"] == "CivilianTrajectoryLearner"
    bad = runner(tmp_path / "absent.npz", tmp_path / "bad", settings())
    with pytest.raises(FileNotFoundError):
        bad.run(mode="train")
    assert bad.agent.closed


def test_independent_head_width_topk_and_feasible_self_edges():
    torch.set_num_threads(1)
    layer = TemporalGraphAttention(16, 6, 4, 0, head_dim=4, top_k=2, temperature=0.1)
    nodes = torch.randn(2, 6, 16, requires_grad=True)
    output, weights = layer(nodes, feasible=torch.zeros(2, 6, dtype=torch.bool))
    torch.testing.assert_close(weights, torch.eye(6)[None, None].expand(2, 6, -1, -1))
    output.square().sum().backward()
    assert torch.isfinite(nodes.grad).all()
    _, weights = layer(nodes)
    assert (weights.gt(0).sum(-1) <= 2).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 6, 6))


def test_civilian_teachers_are_detached_and_train_all_groups():
    cfg = settings(dropout=0)
    observed = torch.randn(2, 6, 6, requires_grad=True)
    logits = torch.randn(2, 6, 6, 6, requires_grad=True)
    groups, separation = attention_terms([logits.softmax(-1)], observed, cfg)
    (sum(groups) + separation).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    torch.testing.assert_close(observed.grad, torch.zeros_like(observed))


def test_yaw_rotation_equivariance_and_physical_limits():
    cfg = settings(dropout=0)
    model = Forecaster(cfg).eval()
    with torch.no_grad():
        model.decoder[-1].weight.normal_(std=0.02)
    observed = torch.randn(2, 6, 6)
    observed[..., 3:] *= 0.5
    angle = torch.tensor([0.8, -0.4])
    rotated = torch.cat((rotate_xy(observed[..., :3], angle, inverse=True),
                         rotate_xy(observed[..., 3:], angle, inverse=True)), -1)
    original, changed = model(observed), model(rotated)
    torch.testing.assert_close(changed["trajectory"], rotate_xy(original["trajectory"], angle, inverse=True), atol=2e-5, rtol=2e-5)
    velocity = torch.tensor([[2., 0., 0.]])
    increments = bound_increments(torch.randn(1, 4, 3) * 100, velocity, cfg)
    speeds = increments / cfg.dt
    accelerations = torch.cat((velocity[:, None], speeds), 1).diff(dim=1) / cfg.dt
    assert speeds.norm(dim=-1).max() <= cfg.max_speed_mps + 1e-5
    assert accelerations.norm(dim=-1).max() <= cfg.max_acceleration_mps2 + 1e-5
    with pytest.raises(ValueError, match="speed exceeds"):
        bound_increments(increments, velocity * 10, cfg)


def test_adamw_cosine_uncertainty_resume_is_exact(tmp_path):
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24, seed=19)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    train(data, continuous, settings(epochs=3))
    train(data, resumed, settings(epochs=1))
    train(data, resumed, settings(epochs=3), resume=resumed / "last.pt")
    left, right = (load_checkpoint(path / "last.pt") for path in (continuous, resumed))
    for collection in ("model", "criterion"):
        for key in left[collection]:
            torch.testing.assert_close(left[collection][key], right[collection][key], rtol=0, atol=0)
    assert left["validation"] == right["validation"]
    assert left["learner_iterations"] == right["learner_iterations"]
    assert left["criterion"]["log_variance"].abs().sum() > 0


@pytest.mark.parametrize("model", ["lstm", "gru", "tcn", "transformer"])
def test_baselines_have_real_trainable_forecast_gradients(model):
    cfg = settings(model_type=model, stage="forecast", attention_supervision=False, dropout=0)
    policy = build_forecaster(cfg)
    learner = TrajectoryLearner(cfg, policy, torch.device("cpu"))
    observed = torch.zeros(2, 6, 6)
    observed[..., 3] = 2
    future = torch.ones(2, 4, 3)
    before = policy.output.weight.detach().clone()
    learner.update(dict(observed=observed, future=future, label=torch.full((2,), -1)))
    assert not torch.equal(before, policy.output.weight)
    assert torch.isfinite(policy(observed)["trajectory"]).all()


def test_two_stage_native_pipeline_and_controlled_comparison(tmp_path):
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24)
    cfg = settings(dropout=0)
    result = pipeline(data, tmp_path / "pipeline", cfg, intention_epochs=1)
    assert result["forecast_test"]["stage"] == "forecast"
    assert (tmp_path / "pipeline/forecast/test_metrics_horizon.csv").exists()
    before = load_checkpoint(tmp_path / "pipeline/intention/best.pt")
    after = load_checkpoint(tmp_path / "pipeline/forecast/best.pt")
    for key in before["model"]:
        if key.startswith("intention."):
            torch.testing.assert_close(before["model"][key], after["model"][key], rtol=0, atol=0)
    same = pipeline(data, tmp_path / "pipeline", cfg, intention_epochs=1, resume=True)
    assert same["forecast_training"]["completed_epoch"] == 1
    compared = compare(data, tmp_path / "comparison", cfg,
                       ["graph", "lstm", "gru", "tcn", "transformer", "graph_no_intention", "graph_no_attention_supervision", "graph_no_both"], [7], intention_epochs=1)
    assert len(compared["runs"]) == 8
    assert compared["partitions"] == before["splits"]


def test_intention_stage_drops_engagement_teacher_supervision():
    """IV-A is the BiLSTM stage (Eqs. 12-14); the engagement teacher scores
    supervise the GAT heads of the trajectory stage (Definition 1, Eqs. 36-53).
    Letting them leak into the intention stage used to trip the config check
    ("military_teachers requires attention_supervision") and abort `pipeline`."""
    cfg = settings(military_teachers=True, teacher_edge_subsets=True,
                   context_entities=1, engagement_edges=True, max_speed_mps=0,
                   max_acceleration_mps2=0, interaction_time_radius=2).validate()
    assert cfg.military_teachers
    classifier = classifier_config(cfg, epochs=1)
    classifier.validate()                      # must not raise
    assert classifier.attention_supervision is False
    assert classifier.military_teachers is False
    assert classifier.teacher_edge_subsets is False
    # The forecast stage keeps the paper's teacher supervision untouched.
    forecast = replace(cfg, stage="forecast", freeze_intention=True).validate()
    assert forecast.military_teachers and forecast.teacher_edge_subsets


def test_evaluation_track_cluster_ci_and_exact_known_metrics(tmp_path):
    cfg = settings(dropout=0)
    details = EvaluationDetails(cfg)
    prediction = torch.tensor([[[3., 4., 0.]] * 4, [[0., 0., 2.]] * 4])
    batch = dict(observed=torch.zeros(2, 6, 6), future=torch.zeros(2, 4, 3),
                 track_id=["a", "b"], last_time=torch.tensor([1., 2.]))
    details.add(dict(trajectory=prediction), batch)
    matrix = np.zeros((9, 9), dtype=int)
    matrix[0, 0], matrix[1, 0], matrix[1, 1] = 1, 1, 2
    report = dict(confusion_matrix=matrix.tolist(), ade_m=3.5, fde_m=3.5, cv_ade_m=0., cv_fde_m=0.)
    details.finish(report, tmp_path / "metrics.json")
    assert report["track_mean_ade_m"] == 3.5
    assert report["track_cluster_bootstrap_95ci"]["ade_m"] == [2., 5.]
    assert report["error_by_horizon"][-1]["displacement_m"] == 3.5
    assert report["intention_balanced_accuracy"] == pytest.approx((1 + 2 / 3) / 2)
    assert len(report["artifacts"]) == 5


def test_early_stopping_and_strict_cuda(tmp_path):
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24)
    result = train(data, tmp_path / "early", settings(epochs=4, early_stopping_patience=1, early_stopping_min_delta_m=1e6))
    assert result["completed_epoch"] == 2 and result["early_stopped"]
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            choose_device("cuda:0")
