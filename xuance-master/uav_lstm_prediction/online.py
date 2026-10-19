"""Frozen causal predictor shared by vector-environment observation views."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from .core import ContextLSTM, DT, RAW_HISTORY, HISTORY, FEATURE_DIM, FUTURE, FEATURE_VERSION, TAUS, feature_stream, decode, sha256

HORIZON_INDICES = np.array([1, 3, 5])
EXTRA_DIM = 9
BASE_DIM = 79
INSERT_AT = 49

class PredictionHistory:
    def __init__(self):
        self.frames = deque(maxlen=RAW_HISTORY)

    def clear(self):
        self.frames.clear()

    def append(self, observations, context, agents):
        frame = dict(times=float(context["times"]),
                     target=np.array(context["target"], copy=True),
                     pursuers=np.array(context["pursuers"], copy=True),
                     observations=np.stack([observations[a] for a in agents]).copy(),
                     min_speed=float(context["min_speed"]), max_speed=float(context["max_speed"]))
        if frame["target"].shape != (3,) or frame["pursuers"].shape != (4,3) or frame["observations"].shape != (4,BASE_DIM):
            raise ValueError("Prediction context requires the recorded four-agent, 79-dimensional layout")
        if not all(np.isfinite(v).all() for v in frame.values()):
            raise ValueError("Nonfinite prediction context")
        if self.frames and frame["times"] <= self.frames[-1]["times"]:
            raise ValueError("Prediction timestamps did not advance; reset history at episode boundaries")
        if not self.frames and not np.isclose(frame["times"], 0., atol=1e-6):
            raise ValueError("New prediction history must start at the episode reset")
        self.frames.append(frame)

    @property
    def ready(self):
        return len(self.frames) == RAW_HISTORY and np.allclose(
            np.diff([f["times"] for f in self.frames]), DT, atol=1e-6, rtol=0)

    def trace(self):
        return {key:(np.array(self.frames[-1][key]) if key in ("min_speed","max_speed") else
                     np.stack([f[key] for f in self.frames]))
                for key in self.frames[-1]}

    def velocity(self):
        if len(self.frames) == 1:
            return self.frames[-1]["observations"][0,39:42] * self.frames[-1]["max_speed"]
        a,b = list(self.frames)[-2:]
        return (b["target"].astype(np.float32)-a["target"].astype(np.float32))/(b["times"]-a["times"])

    def inputs(self):
        if not self.ready:
            raise ValueError("LSTM requires eleven uniform raw frames")
        trace = self.trace()
        features, velocity = feature_stream(trace)
        x = features[-HISTORY:].transpose(1,0,2).copy()
        x[:,:,:3] -= trace["target"][-1] / 100.
        return x, np.repeat(velocity[-1,None,:], 4, axis=0)

class FrozenTrajectoryPredictor:
    def __init__(self, mode, checkpoint=None, expected_sha256=None, device="cpu"):
        if mode not in ("cv","lstm"):
            raise ValueError("Prediction mode must be cv or lstm")
        self.mode = mode
        self.device = torch.device(device)
        self.model = None
        self.metadata = dict(mode=mode,feature_version=FEATURE_VERSION,
                             horizons_seconds=[1.,2.,3.],normalization_m=100.,
                             checkpoint_sha256=None,device=str(self.device))
        if mode == "lstm":
            if not checkpoint or not expected_sha256:
                raise ValueError("LSTM requires an explicit checkpoint and SHA256")
            checkpoint = Path(checkpoint)
            actual = sha256(checkpoint)
            if actual != expected_sha256:
                raise ValueError("LSTM checkpoint SHA256 mismatch")
            saved = torch.load(checkpoint,map_location=self.device,weights_only=True)
            required = dict(feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,
                            history_frames=HISTORY,raw_position_frames=RAW_HISTORY,dt=DT,
                            future_steps=FUTURE,hidden_size=128,layers=2)
            if any(saved["config"].get(k) != v for k,v in required.items()):
                raise ValueError("LSTM feature/model contract differs from the trained predictor")
            self.model = ContextLSTM().to(self.device)
            self.model.load_state_dict(saved["model"],strict=True)
            self.model.eval()
            self.model.requires_grad_(False)
            if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
                raise ValueError("Nonfinite LSTM weights")
            self.metadata.update(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=actual,
                                 epoch=saved["epoch"])

    @torch.inference_mode()
    def predict(self, histories):
        predictions=[np.repeat((h.velocity()[None,:]*TAUS[:,None])[None,:,:],4,axis=0)
                     for h in histories]
        selected=[i for i,h in enumerate(histories) if self.mode=="lstm" and h.ready]
        if selected:
            blocks=[histories[i].inputs() for i in selected]
            x=torch.from_numpy(np.concatenate([b[0] for b in blocks])).to(self.device)
            v=torch.from_numpy(np.concatenate([b[1] for b in blocks])).to(self.device)
            result=decode(self.model(x),v).cpu().numpy().reshape(len(selected),4,FUTURE,3)
            if not np.isfinite(result).all():
                raise FloatingPointError("Nonfinite LSTM forecast")
            for i,value in zip(selected,result):
                predictions[i]=value
        if not all(np.isfinite(p).all() for p in predictions):
            raise FloatingPointError("Nonfinite forecast")
        return predictions

def augment_observations(observations, agents, displacement):
    result={}
    for i,agent in enumerate(agents):
        raw=observations[agent]
        extra=displacement[i,HORIZON_INDICES].reshape(EXTRA_DIM)/100.
        result[agent]=np.concatenate((raw[:INSERT_AT],extra,raw[INSERT_AT:])).astype(np.float32)
    return result
