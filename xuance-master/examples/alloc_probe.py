"""Does Apollonius allocation actually differ from the distance criterion in the real env?
Run both side by side on an identical trajectory before committing GPU hours to the arm."""
import argparse, numpy as np
from xuance.environment.multi_agent_env.uav_pursuit_apollonius_multitarget_3d import (
    UAVPursuitApolloniusMultiTarget3DEnv as E)

def cfg(apo):
    c = argparse.Namespace()
    c.num_agents, c.num_targets = 6, 2
    c.building_mode, c.curriculum_enabled = "medium", False
    c.target_max_speed, c.target_min_speed = 10.0, 8.0
    c.apollonius_alloc = apo
    return c

tot_diff = tot_steps = 0
for ep in range(5):
    np.random.seed(100 + ep); ea = E(cfg(True));  np.random.seed(100 + ep); ea.reset()
    np.random.seed(100 + ep); ed = E(cfg(False)); np.random.seed(100 + ep); ed.reset()
    rng = np.random.default_rng(ep)
    for t in range(150):
        acts = {a: rng.uniform(-1, 1, 3).astype(np.float32) for a in ea.agents}
        ea.step(acts); ed.step(acts)
        tot_steps += 1
        if not np.array_equal(ea.assign, ed.assign):
            tot_diff += 1
            if tot_diff <= 3:
                print(f"  ep{ep} step{t:3d}  dist={ed.assign}  apollonius={ea.assign}")
print(f"\n{tot_steps} steps, assignments differ on {tot_diff} ({tot_diff/tot_steps*100:.1f}%)")
print("RESULT:", "DIFFERENT - arm is effective" if tot_diff else "IDENTICAL - arm is a no-op")
