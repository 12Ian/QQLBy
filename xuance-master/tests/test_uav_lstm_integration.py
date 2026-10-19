"""Server integration tests against the frozen predictor and recorded trajectories."""
from argparse import Namespace
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

RUN=Path(os.environ.get("UAV_LSTM_TEST_RUN",Path(__file__).resolve().parents[1]/"outputs/experiments/lstm_matd3_integration_v1_20261019"))
if not (RUN/"configs").exists():
    pytest.skip("Set UAV_LSTM_TEST_RUN to the server integration artifacts",allow_module_level=True)
from uav_lstm_prediction.core import make_windows,load_trace,decision_grid,decode,sha256
from uav_lstm_prediction.online import PredictionHistory,FrozenTrajectoryPredictor,augment_observations
from uav_lstm_prediction.migration import migrate_prediction_state
from xuance.environment import make_envs
from xuance.environment.multi_agent_env.uav_pursuit_coverage_3d import UAVPursuitCoverage3DEnv
from xuance.environment.vector_envs.prediction_vec_maenv import PredictionVecMultiAgentEnv

torch.set_num_threads(2)
AGENTS=["uav_0","uav_1","uav_2","uav_3"]

def config(mode="lstm",**overrides):
    cfg=yaml.safe_load((RUN/"configs"/f"matd3_prediction_{mode}_600step.yaml").read_text())
    cfg.update(device="cpu",uav_prediction_device="cpu",vectorize="DummyVecMultiAgentEnv",
               parallels=1,render=False)
    cfg.update(overrides)
    return Namespace(**cfg)

def predictor(mode="lstm"):
    cfg=config(mode)
    return FrozenTrajectoryPredictor(mode,cfg.uav_prediction_checkpoint,cfg.uav_prediction_sha256,"cpu")

@pytest.mark.parametrize("level",range(5))
def test_online_inputs_and_forecasts_match_recorded_windows(level):
    parent=Path(json.loads((RUN/"provenance.json").read_text())["predictor"]).parents[1]
    protocol=json.loads((parent/"protocol.json").read_text())
    job=next(e for e in protocol["episodes"] if e["level"]==level and e["split"]=="test" and e["method"]=="matd3")
    raw=load_trace(parent/job["path"]);trace=decision_grid(raw)
    offline=make_windows(raw);h=PredictionHistory();net=predictor()
    for i in range(min(25,len(trace["times"])-1)):
        obs={a:trace["observations"][i,j] for j,a in enumerate(AGENTS)}
        context={k:trace[k][i] for k in ("times","target","pursuers")}
        context.update(min_speed=float(trace["min_speed"]),max_speed=float(trace["max_speed"]))
        h.append(obs,context,AGENTS)
        if not h.ready:continue
        x,v=h.inputs();sl=slice((i-10)*4,(i-9)*4)
        np.testing.assert_array_equal(x,offline["x"][sl])
        np.testing.assert_array_equal(v,offline["v0"][sl])
        with torch.inference_mode():
            reference=decode(net.model(torch.from_numpy(offline["x"][sl])),torch.from_numpy(offline["v0"][sl])).numpy()
        np.testing.assert_allclose(net.predict([h])[0],reference,atol=1e-5,rtol=1e-5)

def fake_frame(t,speed=2.,marker=0.):
    obs={a:np.zeros(79,np.float32) for a in AGENTS}
    for a in AGENTS:
        obs[a][0]=marker
        obs[a][39]=speed/12.
    context=dict(times=t,target=np.array([650+speed*t,400,180],np.float32),
                 pursuers=np.array([[100,350,180],[100,450,200],[100,550,220],[100,650,200]],np.float32),
                 min_speed=3.,max_speed=12.)
    return obs,context

def test_warmup_and_partial_time_do_not_invent_lstm_history():
    h=PredictionHistory();net=predictor()
    for i in range(10):
        obs,ctx=fake_frame(i*.5);h.append(obs,ctx,AGENTS)
        np.testing.assert_allclose(net.predict([h])[0][0,:,0],np.arange(1,13),atol=1e-5)
        assert not h.ready
    obs,ctx=fake_frame(4.7);h.append(obs,ctx,AGENTS)
    assert not h.ready
    assert np.isfinite(net.predict([h])[0]).all()
    with pytest.raises(ValueError,match="did not advance"):h.append(obs,ctx,AGENTS)

def test_checkpoint_and_timing_contracts_fail_explicitly():
    cfg=config()
    with pytest.raises(ValueError,match="SHA256"):FrozenTrajectoryPredictor("lstm",cfg.uav_prediction_checkpoint,"bad","cpu")
    with pytest.raises(ValueError,match="validated"):make_envs(config(decision_dt=1.))
    with pytest.raises(ValueError,match="validated"):make_envs(config(policy="UAV_HGAT_MATD3_Policy"))

@pytest.mark.parametrize("level",range(5))
def test_context_collection_does_not_change_physics_or_rewards(level):
    cfg=config("none",curriculum_start_level=level)
    a=UAVPursuitCoverage3DEnv(cfg)
    cfg2=deepcopy(cfg);cfg2.uav_prediction_mode="cv"
    b=UAVPursuitCoverage3DEnv(cfg2)
    try:
        a.seed(610190000+level);b.seed(610190000+level)
        oa,ia=a.reset();ob,ib=b.reset()
        assert "prediction_context" not in ia and "prediction_context" in ib
        for key in a.agents:np.testing.assert_array_equal(oa[key],ob[key])
        for _ in range(16):
            actions={key:np.zeros(3,np.float32) for key in a.agents}
            oa,ra,ta,xa,ia=a.step(actions);ob,rb,tb,xb,ib=b.step(actions)
            assert ra==rb and ta==tb and xa==xb
            np.testing.assert_array_equal(a.target_position,b.target_position)
            np.testing.assert_array_equal(a.uav_positions,b.uav_positions)
            for key in a.agents:np.testing.assert_array_equal(oa[key],ob[key])
            if all(ta.values()) or xa:break
    finally:a.close();b.close()

class ScriptedVec:
    def __init__(self,partial=False):
        from gymnasium.spaces import Box
        self.num_envs=2;self.agents=AGENTS;self.count=[0,0];self.partial=partial
        self.observation_space={a:Box(-1,1,(79,),dtype=np.float32) for a in AGENTS}
        self.action_space={a:Box(-1,1,(3,),dtype=np.float32) for a in AGENTS}
        self.env_info=dict(observation_space=self.observation_space)
    def reset(self):
        self.count=[0,0]
        pairs=[fake_frame(0.,marker=float(i)) for i in range(2)]
        return [x[0] for x in pairs],[dict(prediction_context=x[1]) for x in pairs]
    def step_async(self,actions):pass
    def step_wait(self):
        obs=[];infos=[];term=[];trunc=[]
        for i in range(2):
            self.count[i]+=1;done=i==0 and self.count[i]==12
            t=self.count[i]*.5-(.2 if done and self.partial else 0.)
            o,c=fake_frame(t,marker=float(i));info=dict(prediction_context=c)
            if done:
                ro,rc=fake_frame(0.,speed=7.,marker=9.)
                info.update(reset_obs=ro,reset_prediction_context=rc);self.count[i]=0
            obs.append(o);infos.append(info);term.append({a:False for a in AGENTS});trunc.append(done)
        return obs,[{a:0. for a in AGENTS} for _ in range(2)],term,trunc,infos
    def close(self):pass

@pytest.mark.parametrize("partial",[False,True])
def test_terminal_reset_and_parallel_histories_are_separate(partial):
    wrapped=PredictionVecMultiAgentEnv(ScriptedVec(partial),config("cv"))
    wrapped.reset()
    for i in range(12):
        obs,_,_,trunc,infos=wrapped.step([{},{}])
        if i<11:assert not any(trunc)
    assert trunc==[True,False]
    np.testing.assert_allclose(obs[0][AGENTS[0]][49:58:3],[.02,.04,.06],atol=1e-5)
    np.testing.assert_allclose(infos[0]["reset_obs"][AGENTS[0]][49:58:3],[.07,.14,.21],atol=1e-5)
    assert obs[0][AGENTS[0]][0]==0. and infos[0]["reset_obs"][AGENTS[0]][0]==9.
    assert len(wrapped.history[0].frames)==1 and len(wrapped.history[1].frames)==11
    assert infos[0]["trajectory_prediction"]["history_ready"]== (not partial)
    wrapped.buf_obs[0]=infos[0]["reset_obs"]
    wrapped.step([{},{}])
    assert len(wrapped.history[0].frames)==2
    assert wrapped.history[0].frames[0]["times"]==0.
    wrapped.close()

@pytest.mark.parametrize("vectorize",["DummyVecMultiAgentEnv","SubprocVecMultiAgentEnv"])
def test_real_vector_environments_reset_augmented_observations(vectorize):
    env=make_envs(config(parallels=2,vectorize=vectorize,episode_length=12,curriculum_start_level=4))
    try:
        obs,_=env.reset()
        assert env.observation_space[env.agents[0]].shape==(88,)
        for _ in range(12):
            actions=[{a:np.zeros(3,np.float32) for a in env.agents} for _ in range(2)]
            obs,rewards,term,trunc,infos=env.step(actions)
        assert all(trunc)
        for i in range(2):
            assert obs[i][env.agents[0]].shape==(88,)
            assert infos[i]["reset_obs"][env.agents[0]].shape==(88,)
            assert len(env.history[i].frames)==1
            assert infos[i]["trajectory_prediction"]["history_frames"]==11
        env.step(actions)
        assert all(len(h.frames)==2 for h in env.history)
    finally:env.close()

def test_actual_policy_checkpoint_migration_preserves_all_networks():
    from xuance import get_runner
    from xuance.engine.run_marl import initialize_baseline_policy
    runners=[]
    try:
        for mode in ["none","cv"]:
            options=Namespace(device="cpu",parallels=1,vectorize="DummyVecMultiAgentEnv",
                              buffer_size=128,batch_size=32,uav_prediction_device="cpu",
                              log_dir=str(RUN/"test_logs"),model_dir=str(RUN/"test_models"))
            runner=get_runner("matd3","uav_pursuit_coverage_3d","prediction_migration_"+mode,
                              config_path=str(RUN/"configs"/f"matd3_prediction_{mode}_600step.yaml"),parser_args=options)
            initialize_baseline_policy(runner.agent,runner.config);runners.append(runner)
        old,new=[r.agent.policy for r in runners]
        old.eval();new.eval();key=runners[0].agent.model_keys[0];ids=torch.eye(4)
        torch.manual_seed(17)
        raw=torch.randn(4,79)
        extra=torch.randn(4,9)
        expanded=torch.cat((raw[:,:49],extra,raw[:,49:]),dim=1)
        actions=torch.randn(1,12).clamp(-1,1)
        with torch.inference_mode():
            ao=old({key:raw},agent_ids=ids)[1][key]
            an=new({key:expanded},agent_ids=ids)[1][key]
            torch.testing.assert_close(ao,an,atol=2e-5,rtol=2e-5)
            to=old.Atarget({key:raw},agent_ids=ids)[1][key]
            tn=new.Atarget({key:expanded},agent_ids=ids)[1][key]
            torch.testing.assert_close(to,tn,atol=2e-5,rtol=2e-5)
            qo=old.Qpolicy(raw.reshape(1,-1),actions,agent_ids=ids)
            qn=new.Qpolicy(expanded.reshape(1,-1),actions,agent_ids=ids)
            for a,b in zip(qo,qn):torch.testing.assert_close(a[key],b[key],atol=2e-5,rtol=2e-5)
            qo=old.Qtarget(raw.reshape(1,-1),actions,agent_ids=ids)
            qn=new.Qtarget(expanded.reshape(1,-1),actions,agent_ids=ids)
            torch.testing.assert_close(qo[key],qn[key],atol=2e-5,rtol=2e-5)
        converted,changed=migrate_prediction_state(new.state_dict(),new.state_dict())
        assert not changed
        bad=dict(old.state_dict());name=next(k for k,v in bad.items() if k.startswith("actor.") and v.ndim==2 and v.shape[1]==83)
        bad[name]=bad[name][:,:82]
        with pytest.raises(ValueError,match="Unexplained"):migrate_prediction_state(bad,new.state_dict())
        torch.save(dict(policy=new.state_dict(),source="zero-column migrated baseline"),RUN/"migrated_initial_policy.pth")
    finally:
        for runner in runners:runner.agent.train_envs.close()
