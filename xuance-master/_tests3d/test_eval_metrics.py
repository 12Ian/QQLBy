import math
from xuance.environment.multi_agent_env.eval_metrics import wilson_ci, summarize_eval

EPS = 0.1 * 4 * math.pi


def test_wilson_ci_bounds_and_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.5


def _rec(caught, crashed, steps, oe, ov):
    return {"caught": caught, "crashed": crashed, "steps": steps,
            "min_omega_euc": oe, "min_omega_vis": ov}


def test_summarize_rates_and_capture_steps():
    recs = [
        _rec(True, False, 100, 0.0, 0.0),
        _rec(True, False, 200, 0.0, 0.0),
        _rec(False, True, 50, 5.0, 5.0),
        _rec(False, False, 400, 0.0, 9.0),   # timeout; euc says encircled (FALSE), vis open (correct)
    ]
    s = summarize_eval(recs, eps=EPS)
    assert s["n_episodes"] == 4
    assert abs(s["success_rate"] - 0.5) < 1e-9
    assert abs(s["collision_rate"] - 0.25) < 1e-9
    assert abs(s["timeout_rate"] - 0.25) < 1e-9
    assert abs(s["mean_capture_steps"] - 150.0) < 1e-9
    assert s["judged_encircled_euc"] == 3
    assert abs(s["false_capture_rate_euclidean"] - (1 / 3)) < 1e-9
    assert s["judged_encircled_vis"] == 2
    assert s["false_capture_rate_visibility"] == 0.0


def test_summarize_empty_capture_and_judged():
    recs = [_rec(False, True, 30, 9.0, 9.0)]
    s = summarize_eval(recs, eps=EPS)
    assert s["mean_capture_steps"] is None
    assert s["false_capture_rate_euclidean"] is None
    assert s["false_capture_rate_visibility"] is None
