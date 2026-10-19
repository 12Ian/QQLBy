"""Causal position-history features and the standalone 3-D LSTM predictor."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn

DT = 0.5
HISTORY = 9
RAW_HISTORY = 11
FUTURE = 12
FEATURE_DIM = 71
TAUS = np.arange(1, FUTURE + 1, dtype=np.float32) * DT
FEATURE_VERSION = "causal_position_v1_71"
LAYOUT = dict(target_motion=[0,9], interaction=[9,33], boundary=[33,39],
              speed_settings=[39,41], local_buildings=[41,71])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)

def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    return trace

def decision_grid(trace):
    times = trace["times"]
    selected = np.flatnonzero(np.isclose(times / DT, np.round(times / DT), atol=1e-6, rtol=0))
    expected = np.arange(len(selected), dtype=np.float64) * DT
    if not np.allclose(times[selected], expected, atol=1e-6, rtol=0):
        raise ValueError("Trace has a missing or nonuniform decision timestamp")
    result = {}
    dynamic = {"times", "target", "pursuers", "target_velocity", "target_acceleration",
               "pursuer_velocity", "pursuer_acceleration", "observations", "target_contacts"}
    for key, value in trace.items():
        result[key] = value[selected] if key in dynamic else value
    return result

def local_buildings_from_map(pursuers, target, buildings):
    """Reconstruct exactly the original nearest-own-five observation geometry."""
    length = len(target)
    result = np.zeros((length, 4, 5, 6), dtype=np.float32)
    if not len(buildings):
        return result
    for i in range(4):
        p = pursuers[:, i]
        dx = np.maximum(np.maximum(buildings[None,:,0]-p[:,None,0],
                                    p[:,None,0]-buildings[None,:,1]), 0)
        dy = np.maximum(np.maximum(buildings[None,:,2]-p[:,None,1],
                                    p[:,None,1]-buildings[None,:,3]), 0)
        dz = np.maximum(np.maximum(50-p[:,None,2], p[:,None,2]-buildings[None,:,4]), 0)
        order = np.argsort(np.sqrt(dx*dx+dy*dy+dz*dz), axis=1)[:,:5]
        chosen = buildings[order]
        count = chosen.shape[1]
        values = np.empty((length,count,6), dtype=np.float32)
        values[:,:,:4] = (chosen[:,:,:4] - target[:,None,[0,0,1,1]]) / 200.
        values[:,:,4] = (chosen[:,:,4] - target[:,None,2]) / 300.
        values[:,:,5] = 1.
        result[:,i,:count] = values
    return result

def feature_stream(trace):
    """Per-frame features; final-window anchoring is performed by make_windows."""
    target = trace["target"].astype(np.float32)
    pursuers = trace["pursuers"].astype(np.float32)
    positions = np.concatenate((target[:,None,:], pursuers), axis=1)
    velocity = np.zeros_like(positions)
    acceleration = np.zeros_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / DT
    acceleration[2:] = np.diff(velocity[1:], axis=0) / DT
    count = len(target)
    features = np.zeros((count,4,FEATURE_DIM), np.float32)
    features[:,:,:3] = target[:,None,:] / 100.
    features[:,:,3:6] = velocity[:,0,None,:] / 12.
    features[:,:,6:9] = acceleration[:,0,None,:] / 4.
    interactions = np.concatenate(((pursuers-target[:,None,:])/200.,
                                    (velocity[:,1:]-velocity[:,0,None,:])/23.), axis=-1)
    features[:,:,9:33] = interactions.reshape(count,1,24)
    x,y,z = target.T
    boundaries = np.stack((x/1000.,(1000.-x)/1000.,y/1000.,(1000.-y)/1000.,
                            (z-50.)/300.,(350.-z)/300.), axis=1)
    features[:,:,33:39] = boundaries[:,None,:]
    features[:,:,39] = float(trace["min_speed"])/12.
    features[:,:,40] = float(trace["max_speed"])/12.
    if "observations" in trace:
        # These are the same five buildings visible in each actor's own observation.
        raw = trace["observations"][:,:,-30:].reshape(count,4,5,6)
        buildings = raw.copy()
        delta = pursuers-target[:,None,:]
        buildings[:,:,:,:4] = (raw[:,:,:,:4]*1000. + delta[:,:,None,[0,0,1,1]]) / 200.
        buildings[:,:,:,4] = (raw[:,:,:,4]*300. + delta[:,:,None,2]) / 300.
        buildings[:,:,:,:5] *= raw[:,:,:,5:6]
    else:
        buildings = local_buildings_from_map(pursuers, target, trace["buildings"])
    features[:,:,41:71] = buildings.reshape(count,4,30)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite prediction features")
    return features, velocity[:,0]

def make_windows(raw_trace, stride=1):
    trace = decision_grid(raw_trace)
    features, velocity = feature_stream(trace)
    target = trace["target"]
    ends = np.arange(RAW_HISTORY-1, len(target)-1, stride, dtype=np.int64)
    if not len(ends):
        return None
    history_indices = ends[:,None] - np.arange(HISTORY-1,-1,-1)[None,:]
    x = features[history_indices].transpose(0,2,1,3).copy()
    x[:,:,:,:3] -= target[ends,None,None,:] / 100.
    future_indices = ends[:,None] + np.arange(1,FUTURE+1)[None,:]
    mask = future_indices < len(target)
    clipped = np.minimum(future_indices, len(target)-1)
    displacement = target[clipped] - target[ends,None,:]
    displacement[~mask] = 0.
    v0 = velocity[ends].copy()
    distances = np.linalg.norm(trace["pursuers"][ends]-target[ends,None,:],axis=2)
    previous = np.linalg.norm(trace["pursuers"][ends-1]-target[ends-1,None,:],axis=2)
    sensed_switch = (distances.min(axis=1)<200) != (previous.min(axis=1)<200)
    vp = velocity[ends-1]
    denom = np.maximum(np.linalg.norm(vp,axis=1)*np.linalg.norm(v0,axis=1),1e-6)
    turning = np.arccos(np.clip((vp*v0).sum(axis=1)/denom,-1,1)) > np.deg2rad(20)
    contacts = trace.get("target_contacts", np.zeros(len(target),np.int64))
    contact_future = contacts[clipped] > contacts[ends,None]
    future_contact = (contact_future & mask).any(axis=1)
    return dict(x=x.reshape(-1,HISTORY,FEATURE_DIM),
                y=np.repeat(displacement,4,axis=0),
                mask=np.repeat(mask,4,axis=0),
                v0=np.repeat(v0,4,axis=0),
                a0=np.repeat((velocity[ends]-velocity[ends-1])/DT,4,axis=0),
                ends=np.repeat(ends,4), views=np.tile(np.arange(4),len(ends)),
                near=np.repeat(distances.min(axis=1)<2*float(trace["capture_radius"]),4),
                turning=np.repeat(turning,4),
                sensed_switch=np.repeat(sensed_switch,4),
                future_contact=np.repeat(future_contact,4))

class ContextLSTM(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim,128,num_layers=2,batch_first=True,dropout=.1)
        self.head = nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Dropout(.1),
                                  nn.Linear(128,FUTURE*3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        states,_ = self.lstm(x)
        return self.head(states[:,-1]).reshape(-1,FUTURE,3)

def decode(residual, velocity):
    tau = torch.as_tensor(TAUS,device=residual.device,dtype=residual.dtype)
    return velocity[:,None,:]*tau[None,:,None] + residual*100.

def supervised_loss(predicted, truth, mask):
    element_mask = mask[:,:,None].to(predicted.dtype)
    position = torch.nn.functional.huber_loss(predicted/100.,truth/100.,
                                               reduction="none",delta=.1)
    position = (position*element_mask).sum() / (element_mask.sum()*3).clamp_min(1)
    zero = torch.zeros_like(predicted[:,:1])
    pv = torch.diff(torch.cat((zero,predicted),dim=1),dim=1)/DT
    tv = torch.diff(torch.cat((zero,truth),dim=1),dim=1)/DT
    valid = mask.clone()
    valid[:,1:] &= mask[:,:-1]
    valid = valid[:,:,None].to(predicted.dtype)
    speed = torch.nn.functional.huber_loss(pv/12.,tv/12.,reduction="none",delta=.1)
    speed = (speed*valid).sum() / (valid.sum()*3).clamp_min(1)
    return position + .2*speed

def unit_checks():
    length=36
    t=np.arange(length,dtype=np.float32)*DT
    target=np.stack((500+2*t,500+0*t,180+0*t),axis=1)
    pursuers=np.tile(np.array([[100,350,180],[100,450,200],
                               [100,550,220],[100,650,200]],np.float32),(length,1,1))
    trace=dict(times=t.astype(np.float64), target=target,pursuers=pursuers,
               buildings=np.zeros((0,5),np.float32),min_speed=np.array(3),
               max_speed=np.array(12),capture_radius=np.array(15))
    samples=make_windows(trace)
    assert samples["x"].shape[1:]==(9,71)
    assert np.allclose(samples["x"][:, -1,:3],0)
    assert np.allclose(samples["x"][0,:,3],2/12.)
    assert np.allclose(samples["x"][0,:,6:9],0)
    assert np.allclose(samples["v0"],[2,0,0])
    assert np.allclose(samples["y"][0,:,0],TAUS*2)
    assert samples["mask"][-1].sum()==1
    changed={k:v.copy() for k,v in trace.items()}
    changed["target"][11:]+=100
    changed_samples=make_windows(changed)
    assert np.array_equal(samples["x"][:4],changed_samples["x"][:4])
    assert not np.array_equal(samples["y"][:4],changed_samples["y"][:4])
    net=ContextLSTM()
    net.eval()
    xt=torch.from_numpy(samples["x"][:8])
    out=decode(net(xt),torch.from_numpy(samples["v0"][:8]))
    assert torch.allclose(out,torch.from_numpy(samples["y"][:8]),atol=1e-5)
    loss=supervised_loss(out,torch.from_numpy(samples["y"][:8]),
                         torch.from_numpy(samples["mask"][:8]))
    loss.backward()
    assert torch.isfinite(loss)
    grid_trace={k:v.copy() for k,v in trace.items()}
    grid_trace["times"][-1]-=.1
    assert len(decision_grid(grid_trace)["times"])==length-1
    print("CORE_CHECKS_PASSED: shapes, units, backward differences, causality, masks, CV init, loss, terminal grid",
          flush=True)
