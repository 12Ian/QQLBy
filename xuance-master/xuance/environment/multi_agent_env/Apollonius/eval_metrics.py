"""Pure evaluation metrics for the 3D pursuit experiments (no torch/env deps)."""
import math
import statistics


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n. n==0 -> (0,0)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_eval(records, eps=0.1 * 4 * math.pi):
    """records: list of dicts with caught(bool), crashed(bool), steps(int),
    min_omega_euc(float), min_omega_vis(float). Returns the metric dict."""
    n = len(records)
    if n == 0:
        return {"n_episodes": 0}
    n_c = sum(1 for r in records if r["caught"])
    n_cr = sum(1 for r in records if r["crashed"])
    n_to = sum(1 for r in records if (not r["caught"] and not r["crashed"]))
    steps = [r["steps"] for r in records if r["caught"]]
    mean_steps = (sum(steps) / len(steps)) if steps else None
    std_steps = statistics.pstdev(steps) if len(steps) > 1 else (0.0 if steps else None)

    def false_capture(key):
        judged = [r for r in records if r[key] < eps]
        m = len(judged)
        if m == 0:
            return None, (None, None), 0
        fc = sum(1 for r in judged if not r["caught"])
        return fc / m, wilson_ci(fc, m), m

    fce, fce_ci, je = false_capture("min_omega_euc")
    fcv, fcv_ci, jv = false_capture("min_omega_vis")
    return {
        "n_episodes": n,
        "success_rate": n_c / n, "success_ci": wilson_ci(n_c, n),
        "collision_rate": n_cr / n, "collision_ci": wilson_ci(n_cr, n),
        "timeout_rate": n_to / n,
        "mean_capture_steps": mean_steps, "std_capture_steps": std_steps,
        "false_capture_rate_euclidean": fce, "fc_euc_ci": fce_ci, "judged_encircled_euc": je,
        "false_capture_rate_visibility": fcv, "fc_vis_ci": fcv_ci, "judged_encircled_vis": jv,
    }
