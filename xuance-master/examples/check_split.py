"""Standalone check: the finish/closure gate split must keep the full reward identical
(additive partition). full - both == (full - nofin) + (full - noclo), per agent."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_3d import UAVPursuitApollonius3DEnv


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def step_rew(disable):
    np.random.seed(11)
    e = UAVPursuitApollonius3DEnv(Cfg(num_agents=6, building_mode="medium",
                                      curriculum_enabled=False, closure_weight=8.0,
                                      reward_disable=disable))
    e.reset()
    acts = {a: np.zeros(3, np.float32) for a in e.agents}
    _, rew, _, _, _ = e.step(acts)
    return np.array([rew[a] for a in e.agents], dtype=np.float64)


full = step_rew([])
nofin = step_rew(["r_finish"])
noclo = step_rew(["r_closure"])
noboth = step_rew(["r_finish", "r_closure"])
lhs = full - noboth
rhs = (full - nofin) + (full - noclo)
print("full              :", np.round(full, 4))
print("full-noboth       :", np.round(lhs, 4))
print("(f-nofin)+(f-noclo):", np.round(rhs, 4))
print("ADDITIVE_IDENTITY_OK:", bool(np.allclose(lhs, rhs, atol=1e-6)))
print("finish_part_nonzero :", bool(np.any(np.abs(full - nofin) > 1e-9)))
