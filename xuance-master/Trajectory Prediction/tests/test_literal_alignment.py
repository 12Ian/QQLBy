"""Literal-alignment guard tests (audit E1 + B4, round 4).

The paper text pins two knobs that the engineering defaults deliberately
deviate from:

* E1 - the paper specifies no initialisation, so PyTorch defaults are the
  literal reading; our default zero-initialises the decoder output layer so
  training starts at the constant-velocity baseline.  ``decoder_zero_init``
  exposes that choice and must default to True (status quo).
* B4 - the paper's KL distillation (Eq. 36-37) carries no temperature; our
  default sharpens the teachers by ``teacher_temperature=0.1``.  The literal
  reading is 1.0, set through the JSON config only.

These tests pin the switch behaviour, the default, and the literal config
file so a future refactor cannot silently change either round.
"""

import json
from pathlib import Path

import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.model import build_forecaster

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LITERAL_CONFIG = CONFIG_DIR / "engagement_forecast_literal.json"


def _config(**overrides):
    base = dict(observation_steps=20, prediction_steps=10, dt=.1, stride=2,
                static_state_branch=True, decoder_type="autoregressive_gru",
                context_entities=1, engagement_edges=True,
                intention_task="semantic9", attention_supervision=True,
                military_teachers=True, teacher_regularizers="paper")
    base.update(overrides)
    return Config(**base).validate()


def _decoder_weight_norm(model):
    return model.decoder[-1].weight.norm().item()


def test_decoder_zero_init_defaults_on():
    assert Config().decoder_zero_init is True


def test_zero_init_off_uses_pytorch_default_weights():
    torch.manual_seed(0)
    default = build_forecaster(_config(decoder_zero_init=True))
    torch.manual_seed(0)
    literal = build_forecaster(_config(decoder_zero_init=False))
    assert _decoder_weight_norm(default) == pytest.approx(0.0, abs=1e-9)
    assert _decoder_weight_norm(literal) > 0.0
    # Everything except the output-layer init must be bit-identical: same
    # seed, same architecture, only the init branch differs.
    for (name_a, param_a), (name_b, param_b) in zip(default.named_parameters(), literal.named_parameters()):
        assert name_a == name_b
        if name_a.endswith("decoder.0.weight") or name_a.endswith("decoder.0.bias"):
            continue
        assert torch.equal(param_a, param_b), name_a


def test_literal_config_carries_the_two_paper_values():
    payload = json.loads(LITERAL_CONFIG.read_text(encoding="utf-8"))
    baseline = json.loads((CONFIG_DIR / "engagement_forecast.json").read_text(encoding="utf-8"))
    assert payload["teacher_temperature"] == 1.0
    assert payload["decoder_zero_init"] is False
    # decoder_head stays at the "linear" default (IV-B) without being spelled
    # out, matching the engineering baseline file key-for-key.
    assert Config(**payload).validate().decoder_head == "linear"
    # Everything else must stay identical to the engineering baseline.
    differences = {key for key in set(payload) | set(baseline)
                   if payload.get(key) != baseline.get(key)}
    assert differences == {"teacher_temperature", "decoder_zero_init"}, differences
    # And it must validate as a whole.
    Config(**payload).validate()


def test_teacher_temperature_literal_is_identity():
    cfg = _config(teacher_temperature=1.0)
    assert cfg.teacher_temperature == 1.0
    cfg = _config(teacher_temperature=0.1)
    assert cfg.teacher_temperature == 0.1
