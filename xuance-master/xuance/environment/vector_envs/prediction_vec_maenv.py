"""Parent-process forecasts with independent histories for each vector environment."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from gymnasium.spaces import Box
from .vector_env import VecEnv
from uav_lstm_prediction.online import (BASE_DIM, EXTRA_DIM, INSERT_AT, PredictionHistory,
                                       FrozenTrajectoryPredictor, augment_observations)

class PredictionVecMultiAgentEnv(VecEnv):
    def __init__(self, env, config):
        self.env=env
        self.agents=env.agents
        if len(self.agents)!=4 or any(s.shape!=(BASE_DIM,) for s in env.observation_space.values()):
            raise ValueError("Prediction wrapper requires the four-agent 79-dimensional base observations")
        spaces={}
        for agent,space in env.observation_space.items():
            low=np.concatenate((space.low[:INSERT_AT],np.full(EXTRA_DIM,-np.inf),space.low[INSERT_AT:]))
            high=np.concatenate((space.high[:INSERT_AT],np.full(EXTRA_DIM,np.inf),space.high[INSERT_AT:]))
            spaces[agent]=Box(low.astype(np.float32),high.astype(np.float32),dtype=np.float32)
        super().__init__(env.num_envs,spaces,env.action_space)
        self.env_info=deepcopy(env.env_info)
        self.env_info["observation_space"]=spaces
        self.history=[PredictionHistory() for _ in range(self.num_envs)]
        self.predictor=FrozenTrajectoryPredictor(
            config.uav_prediction_mode,getattr(config,"uav_prediction_checkpoint",None),
            getattr(config,"uav_prediction_sha256",None),getattr(config,"uav_prediction_device","cpu"))
        self.prediction_metadata=deepcopy(self.predictor.metadata)
        self.diagnostic_dir=getattr(config,"uav_prediction_diagnostic_dir",None)
        self.buf_obs=[{} for _ in range(self.num_envs)]
        self.buf_info=[{} for _ in range(self.num_envs)]
        self.last_forecasts=[]
        self.prediction_seconds=0.
        self.prediction_batches=0

    def __getattr__(self, name):
        return getattr(self.env,name)

    def _predict(self, histories):
        started=time.perf_counter()
        try:
            values=self.predictor.predict(histories)
        except Exception:
            if self.diagnostic_dir:
                folder=Path(self.diagnostic_dir)
                folder.mkdir(parents=True,exist_ok=True)
                stamp=time.time_ns()
                for i,history in enumerate(histories):
                    if history.frames:
                        np.savez_compressed(folder/f"prediction_failure_{stamp}_{i}.npz",**history.trace())
            raise
        self.prediction_seconds+=time.perf_counter()-started
        self.prediction_batches+=1
        return values

    def reset(self):
        raw,infos=self.env.reset()
        infos=[dict(info) for info in infos]
        for h,obs,info in zip(self.history,raw,infos):
            h.clear()
            h.append(obs,info["prediction_context"],self.agents)
        forecasts=self._predict(self.history)
        self.buf_obs=[augment_observations(obs,self.agents,p) for obs,p in zip(raw,forecasts)]
        self.buf_info=infos
        self.last_forecasts=forecasts
        return self.buf_obs.copy(),infos

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        raw,rewards,terminated,truncated,infos=self.env.step_wait()
        infos=[dict(info) for info in infos]
        for h,obs,info in zip(self.history,raw,infos):
            h.append(obs,info["prediction_context"],self.agents)
        forecasts=self._predict(self.history)
        next_obs=[augment_observations(obs,self.agents,p) for obs,p in zip(raw,forecasts)]
        # Construct terminal forecasts before clearing any completed history.
        for i,(obs,info,p) in enumerate(zip(next_obs,infos,forecasts)):
            info["trajectory_prediction"]=dict(mode=self.predictor.mode,
                history_frames=len(self.history[i].frames),history_ready=bool(self.history[i].ready),
                time_seconds=self.history[i].frames[-1]["times"],
                displacement_m=p.copy(),checkpoint_sha256=self.prediction_metadata["checkpoint_sha256"])
            if all(terminated[i].values()) or truncated[i]:
                reset_raw=info["reset_obs"]
                h=self.history[i]
                h.clear()
                h.append(reset_raw,info["reset_prediction_context"],self.agents)
                reset_forecast=self._predict([h])[0]
                info["reset_obs"]=augment_observations(reset_raw,self.agents,reset_forecast)
        self.buf_obs=next_obs
        self.buf_info=infos
        self.last_forecasts=forecasts
        return next_obs,rewards,terminated,truncated,infos

    def env_method(self, method_name, *args, **kwargs):
        if hasattr(self.env,"env_method"):
            return self.env.env_method(method_name,*args,**kwargs)
        return [getattr(e.env if hasattr(e,"env") else e,method_name)(*args,**kwargs) for e in self.env.envs]

    def render(self, mode):
        return self.env.render(mode)

    def close_extras(self):
        self.env.close()
