"""Evaluate the exact implemented reward on explicit instantaneous geometries."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import numpy as np
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/formation_objective_counterexamples_v1'


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True,exist_ok=False)
    target=np.array([700.,500.,200.])
    layouts={
        'square_around_target':target+np.array([[50,0,0],[0,50,0],[-50,0,0],[0,-50,0]]),
        'tetrahedron_around_target':target+40/np.sqrt(3)*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]),
        'one_sided_near':target+np.array([[-45,0,0],[-42,0,0],[-33,0,0],[-22,0,0]]),
        'one_sided_far':target+np.array([[-300,0,0],[-297,0,0],[-288,0,0],[-277,0,0]])}
    records=[]
    for name,positions in layouts.items():
        directions=(target-positions)/np.linalg.norm(target-positions,axis=1)[:,None]
        state=Namespace(num_agents=4,uav_positions=positions,target_position=target,
                        uav_velocities=4*directions,target_velocity=np.zeros(3))
        pursuit,formation,diagnostics=UAVPursuitCoverage3DEnv._pursuit_reward_terms(state)
        assert np.isclose(pursuit,8.)
        records.append(dict(name=name,positions=positions.tolist(),target=target.tolist(),
            pursuit_raw=pursuit,formation_raw_radians=formation,formation_mean_degrees=float(np.degrees(formation)),
            target_distances=np.linalg.norm(target-positions,axis=1).tolist(),
            diagnostics=diagnostics,decision_formation_reward=.1*.1*.5*formation))
    assert np.isclose(records[0]['formation_mean_degrees'],45,atol=.001)
    assert np.isclose(records[2]['formation_mean_degrees'],135,atol=.001)
    source=ROOT/'xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py'
    result=dict(static_formula_examples_not_rollouts=True,training_unchanged=True,
        implemented_formula_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        formula='mean arccos(nearest teammate direction dot target direction)',records=records)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('name','pursuit_raw','formation_mean_degrees','decision_formation_reward')} for r in records]),flush=True)


if __name__=='__main__':
    main()
