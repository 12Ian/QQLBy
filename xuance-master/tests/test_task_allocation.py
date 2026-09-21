"""Standalone tests for the multi-target pursuer->target allocation module.
Imports the module file directly (no xuance package __init__) so it runs clean
without pulling in torch/tensorboard. Run: python tests/test_task_allocation.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "xuance", "environment", "multi_agent_env"))
import task_allocation as ta  # noqa: E402


def test_balanced_even_split():
    # 5 pursuers clustered near T0, 1 near T1; balanced assignment must still be 3+3
    # (each target gets ~N/M pursuers for coverage), NOT a naive 5+1 nearest split.
    P = np.array([[100, 100, 200], [110, 100, 200], [100, 110, 200],
                  [120, 100, 200], [100, 120, 200], [900, 900, 200]], float)
    T = np.array([[100, 100, 200], [900, 900, 200]], float)
    assign = ta.assign_pursuers_to_targets(P, T, np.full(6, -1),
                                           np.array([True, True]))
    counts = np.bincount(assign, minlength=2)
    assert counts.tolist() == [3, 3], counts
    assert set(assign.tolist()) == {0, 1}


def test_all_to_only_alive_target():
    rng = np.random.RandomState(0)
    P = rng.uniform(0, 1000, (6, 3))
    T = np.array([[100, 100, 200], [900, 900, 200]], float)
    assign = ta.assign_pursuers_to_targets(P, T, np.full(6, -1),
                                           np.array([True, False]))  # T1 dead
    assert set(assign.tolist()) == {0}, assign


def test_single_target_all_assigned():
    P = np.random.RandomState(1).uniform(0, 1000, (6, 3))
    T = np.array([[500, 500, 200]], float)
    assign = ta.assign_pursuers_to_targets(P, T, np.full(6, -1), np.array([True]))
    assert set(assign.tolist()) == {0} and len(assign) == 6


def test_stickiness_keeps_prev_when_no_clear_gain():
    # Symmetric geometry: two targets equidistant; a valid prev assignment should be
    # kept rather than thrashed to an equivalent-cost alternative.
    P = np.array([[500, 400, 200], [500, 410, 200], [500, 420, 200],
                  [500, 600, 200], [500, 590, 200], [500, 580, 200]], float)
    T = np.array([[500, 300, 200], [500, 700, 200]], float)
    prev = np.array([0, 0, 0, 1, 1, 1])
    assign = ta.assign_pursuers_to_targets(P, T, prev, np.array([True, True]),
                                           sticky=1e6)  # huge stickiness -> keep prev
    assert assign.tolist() == prev.tolist(), assign


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL PASS")
