import json
from argparse import Namespace
from pathlib import Path

import pytest

import examples.run_n4_hard_stage as stage


def args_for(tmp_path, **overrides):
    values = dict(name="unit", seed=1, building_mode="open", randomize_density=None,
                  steps=8, load_from=str(tmp_path / "warm.pth"), parallels=4,
                  eval_every=8, eval_mode="open", eval_seed=17, eval_k=2,
                  out_dir=str(tmp_path / "out"), device="cpu")
    values.update(overrides)
    Path(values["load_from"]).write_bytes(b"warm")
    return Namespace(**values)


def test_checkpoint_key_orders_success_collision_then_return():
    assert stage.checkpoint_key(dict(success_rate=.8, collision_rate=.1, mean_episode_return=1)) > stage.checkpoint_key(dict(success_rate=.7, collision_rate=0, mean_episode_return=999))
    assert stage.checkpoint_key(dict(success_rate=.8, collision_rate=.1, mean_episode_return=1)) > stage.checkpoint_key(dict(success_rate=.8, collision_rate=.2, mean_episode_return=999))
    assert stage.checkpoint_key(dict(success_rate=.8, collision_rate=.1, mean_episode_return=2)) > stage.checkpoint_key(dict(success_rate=.8, collision_rate=.1, mean_episode_return=1))


@pytest.mark.parametrize("summary", [
    {}, {"success_rate": .1, "collision_rate": .1}, {"success_rate": .1, "mean_episode_return": 1}, {"collision_rate": .1, "mean_episode_return": 1}, {"success_rate": float("nan"), "collision_rate": .1, "mean_episode_return": 1}, {"success_rate": .1, "collision_rate": float("inf"), "mean_episode_return": 1}, {"success_rate": .1, "collision_rate": .1, "mean_episode_return": float("nan")},
    {"success_rate": 1.1, "collision_rate": .1, "mean_episode_return": 1},
    {"success_rate": .1, "collision_rate": -0.1, "mean_episode_return": 1},
])
def test_invalid_evaluator_summary_is_rejected(summary):
    with pytest.raises((KeyError, FloatingPointError, ValueError)):
        stage.validate_summary(summary)


def test_recursive_nonfinite_training_metric_is_rejected():
    with pytest.raises(FloatingPointError, match="critic"):
        stage.assert_finite_metrics({"nested": [1, {"critic": float("inf")} ]})


def test_output_claim_rejects_nonempty_and_concurrent(tmp_path):
    out = tmp_path / "out"; out.mkdir(); (out / "old").write_text("x")
    with pytest.raises(FileExistsError, match="nonempty"):
        stage.claim_output(out)
    empty = tmp_path / "empty"; claim = stage.claim_output(empty)
    with pytest.raises(FileExistsError, match="already active"):
        stage.claim_output(empty)
    claim.unlink()


def test_parser_requires_exactly_one_density_and_rejects_soft(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stage", "--name", "x", "--seed", "1", "--steps", "8", "--load-from", "x", "--eval-mode", "open", "--out-dir", "o"])
    with pytest.raises(SystemExit): stage.parse_args()
    monkeypatch.setattr("sys.argv", ["stage", "--name", "x", "--seed", "1", "--building-mode", "open", "--randomize-density", "open", "--steps", "8", "--load-from", "x", "--eval-mode", "open", "--out-dir", "o"])
    with pytest.raises(SystemExit): stage.parse_args()
    monkeypatch.setattr("sys.argv", ["stage", "--name", "x", "--seed", "1", "--building-mode", "open", "--steps", "8", "--load-from", "x", "--eval-mode", "open", "--out-dir", "o", "--soft"])
    with pytest.raises(SystemExit) as exc: stage.parse_args()
    assert exc.value.code == 2
    assert "unrecognized arguments: --soft" in capsys.readouterr().err


def test_claim_is_cleaned_if_config_write_fails(tmp_path, monkeypatch):
    original = Path.write_text
    def fail_config(self, *args, **kwargs):
        if self.name == "training_config.json": raise OSError("config write")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail_config)
    with pytest.raises(OSError, match="config write"):
        stage.main(args_for(tmp_path))
    assert not (tmp_path / "out" / ".n4_hard_stage.claim").exists()


def test_step_delta_mismatch_fails_and_cleans_up(tmp_path, monkeypatch):
    closed=[]
    agent=Namespace(current_step=0, policy=Namespace(load_state_dict=lambda *a, **k: None),
                    train=lambda **k: {}, save_model=lambda *a, **k: None,
                    finish=lambda: closed.append("finish"))
    runner=Namespace(agent=agent, envs=Namespace(close=lambda: closed.append("close")))
    monkeypatch.setattr(stage, "get_runner", lambda **_: runner)
    monkeypatch.setattr(stage.torch, "load", lambda *a, **k: {"policy": {}})
    with pytest.raises(RuntimeError, match="step delta"):
        stage.main(args_for(tmp_path))
    assert closed == ["close", "finish"]


def test_missing_snapshot_fails_explicitly(tmp_path, monkeypatch):
    agent=Namespace(current_step=0, policy=Namespace(load_state_dict=lambda *a, **k: None), train=lambda **k: (setattr(agent, "current_step", 8) or {}), save_model=lambda *a, **k: None, finish=lambda: None)
    monkeypatch.setattr(stage, "get_runner", lambda **k: Namespace(agent=agent, envs=Namespace(close=lambda: None)))
    monkeypatch.setattr(stage.torch, "load", lambda *a, **k: {"policy": {}})
    with pytest.raises(FileNotFoundError, match="snapshot"):
        stage.main(args_for(tmp_path))


def test_missing_trace_fails_explicitly(tmp_path, monkeypatch):
    agent=Namespace(current_step=0, policy=Namespace(load_state_dict=lambda *a, **k: None), train=lambda **k: (setattr(agent, "current_step", 8) or {}), save_model=lambda n, model_path=None: Path(model_path,n).write_bytes(b"x"), finish=lambda: None)
    monkeypatch.setattr(stage, "get_runner", lambda **k: Namespace(agent=agent, envs=Namespace(close=lambda: None)))
    monkeypatch.setattr(stage.torch, "load", lambda *a, **k: {"policy": {}})
    monkeypatch.setattr(stage.subprocess, "run", lambda c, **k: Path(c[c.index("--out")+1]).write_text('{"success_rate": .8, "collision_rate": .1, "mean_episode_return": 1}'))
    with pytest.raises(FileNotFoundError, match="trace"):
        stage.main(args_for(tmp_path))


def test_fake_orchestration_wires_hard_config_command_step_and_selection(tmp_path, monkeypatch):
    class Policy:
        def load_state_dict(self, state, strict=False): assert state == {"weight": 1} and not strict
    class Agent:
        def __init__(self): self.policy=Policy(); self.current_step=0; self.finished=False
        def train(self, train_steps): self.current_step += train_steps * 4; return {"loss": {"value": 1.0}}
        def save_model(self, name, model_path=None): Path(model_path, name).write_bytes(str(self.current_step).encode())
        def finish(self): self.finished=True
    class Envs:
        def __init__(self): self.closed=False
        def close(self): self.closed=True
    runner=Namespace(agent=Agent(), envs=Envs())
    monkeypatch.setattr(stage, "get_runner", lambda **_: runner)
    monkeypatch.setattr(stage.torch, "load", lambda *_, **__: {"policy":{"weight":1}})
    commands=[]
    def fake_run(command, cwd=None, check=None):
        assert check is True and cwd
        commands.append(command); Path(command[command.index("--out")+1]).write_text(json.dumps({"success_rate":.8,"collision_rate":.1,"mean_episode_return":float(len(commands))})); Path(command[command.index("--trace-out")+1]).write_text("{}\n")
    monkeypatch.setattr(stage.subprocess, "run", fake_run)
    assert stage.main(args_for(tmp_path, steps=16, eval_every=8)) == 0
    out=tmp_path/"out"; config=json.loads((out/"training_config.json").read_text()); rows=[json.loads(x) for x in (out/"checkpoint_metrics.jsonl").read_text().splitlines()]
    assert config["terminate_on_collision"] is True and config["spawn_safety_enabled"] is True and config["target_max_speed"] == 6.0 and config["closure_weight"] == 8.0 and config["w_pos"] == .6 and config["w_gap"] == .8 and config["separation_weight"] == 4.0 and config["separation_distance"] == 20.0 and config["load_from"] == str((tmp_path/"warm.pth").resolve()) and (config["eval_every"],config["eval_mode"],config["eval_seed"],config["eval_k"],config["parallels"],config["building_mode"]) == (8,"open",17,2,4,"open") and "randomize_density" not in config
    assert [r["step_before"] for r in rows] == [0,8] and [r["step_after"] for r in rows] == [8,16] and [row["step_delta"] for row in rows] == [8,8] and (out/"selected_model.pth").read_bytes()==b"16"
    command=commands[0]
    for flag, value in [("--collision-mode","hard"),("--spawn-safety","on"),("--seed","17"),("--k","2"),("--building-mode","open"),("--num-agents","4"),("--target-max-speed","6"),("--closure-weight","8"),("--w-pos","0.6"),("--w-gap","0.8"),("--separation-weight","4"),("--separation-distance","20")]: assert command[command.index(flag)+1] == value
    assert command[command.index("--model-path")+1].endswith("snapshot_8.pth") and len(rows) == 2 and (out/"snapshot_8.pth").is_file() and (out/"snapshot_16.pth").is_file() and (out/"selected_summary.json").is_file()
    assert json.loads((out/"selected_summary.json").read_text())["environment_step"] == 16 and (out/"eval_8.jsonl").is_file() and (out/"eval_16.jsonl").is_file()
    assert len(commands) == 2
    for command, step in zip(commands, (8,16)):
        assert command[command.index("--model-path")+1] == str(out/f"snapshot_{step}.pth") and command[command.index("--out")+1] == str(out/f"eval_{step}.json") and command[command.index("--trace-out")+1] == str(out/f"eval_{step}.jsonl")
    selected=json.loads((out/"selected_summary.json").read_text()); assert selected["environment_step"] == 16 and selected["summary"]["mean_episode_return"] == 2 and selected["snapshot"] == str(out/"snapshot_16.pth")
    assert runner.envs.closed and runner.agent.finished


@pytest.mark.parametrize("failure", ["torch_load", "load", "train", "eval"])
def test_cleanup_runs_independently_on_runtime_failures(tmp_path, monkeypatch, failure):
    calls=[]
    class Agent:
        policy=Namespace(load_state_dict=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("load")) if failure=="load" else None)
        current_step=0
        def train(self, train_steps):
            if failure=="train": raise RuntimeError("train")
            self.current_step=8; return {}
        def save_model(self, n, model_path=None): Path(model_path,n).write_bytes(b"x")
        def finish(self): calls.append("finish"); raise RuntimeError("finish")
    runner=Namespace(agent=Agent(), envs=Namespace(close=lambda: calls.append("close")))
    monkeypatch.setattr(stage,"get_runner",lambda **_: runner); monkeypatch.setattr(stage.torch,"load",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("torch_load")) if failure=="torch_load" else {"policy":{}})
    if failure=="eval": monkeypatch.setattr(stage.subprocess,"run",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("eval")))
    with pytest.raises(RuntimeError, match=failure): stage.main(args_for(tmp_path))
    assert calls == ["close", "finish"]
    assert not (tmp_path / "out" / ".n4_hard_stage.claim").exists()


def test_cleanup_only_close_failure_removes_claim(tmp_path, monkeypatch):
    calls=[]
    agent=Namespace(current_step=0, policy=Namespace(load_state_dict=lambda *a, **k: None),
        train=lambda **k: (setattr(agent, "current_step", 8) or {}),
        save_model=lambda n, model_path=None: Path(model_path,n).write_bytes(b"x"),
        finish=lambda: calls.append("finish"))
    runner=Namespace(agent=agent, envs=Namespace(close=lambda: (_ for _ in ()).throw(RuntimeError("close"))))
    monkeypatch.setattr(stage, "get_runner", lambda **k: runner)
    monkeypatch.setattr(stage.torch, "load", lambda *a, **k: {"policy": {}})
    def evaluate(command, **kwargs):
        Path(command[command.index("--out")+1]).write_text('{"success_rate": 0.8, "collision_rate": 0.1, "mean_episode_return": 1}')
        Path(command[command.index("--trace-out")+1]).write_text('{}\n')
    monkeypatch.setattr(stage.subprocess, "run", evaluate)
    with pytest.raises(RuntimeError, match="close"):
        stage.main(args_for(tmp_path))
    assert calls == ["finish"]
    assert not (tmp_path / "out" / ".n4_hard_stage.claim").exists()
