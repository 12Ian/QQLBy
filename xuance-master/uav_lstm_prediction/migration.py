"""Explicit column mapping from the unaugmented MATD3 checkpoint."""
import torch
from .online import BASE_DIM, EXTRA_DIM, INSERT_AT

def observation_columns():
    return list(range(INSERT_AT))+list(range(INSERT_AT+EXTRA_DIM,BASE_DIM+EXTRA_DIM))

def migrate_prediction_state(saved,current,actor_only=False):
    prefixes=("actor.","target_actor.")
    expected={k for k in current if not actor_only or k.startswith(prefixes)}
    selected={k:v for k,v in saved.items() if not actor_only or k.startswith(prefixes)}
    if set(selected)!=expected:
        raise ValueError("Prediction initialization has incomplete or unexpected policy parameters")
    obs_map=observation_columns()
    actor_map=obs_map+list(range(BASE_DIM+EXTRA_DIM,BASE_DIM+EXTRA_DIM+4))
    critic_map=[i*(BASE_DIM+EXTRA_DIM)+j for i in range(4) for j in obs_map]
    critic_map+=list(range(4*(BASE_DIM+EXTRA_DIM),4*(BASE_DIM+EXTRA_DIM)+12+4))
    result=dict(current)
    migrated=[]
    for key,value in selected.items():
        target=current[key]
        if not torch.is_tensor(value) or not torch.is_tensor(target) or not torch.isfinite(value).all().item():
            raise ValueError("Prediction initialization requires finite policy tensors")
        if value.shape==target.shape:
            result[key]=value
            continue
        actor=key.startswith(prefixes)
        critic=key.startswith(("critic_A.","critic_B.","target_critic_A.","target_critic_B."))
        mapping=actor_map if actor else critic_map
        old_width=83 if actor else 332
        new_width=92 if actor else 368
        if not (actor or critic) or value.ndim!=2 or target.ndim!=2 or value.shape[0]!=target.shape[0] or value.shape[1]!=old_width or target.shape[1]!=new_width:
            raise ValueError("Unexplained prediction initialization shape change: "+key)
        expanded=torch.zeros_like(target)
        expanded[:,mapping]=value.to(device=target.device,dtype=target.dtype)
        result[key]=expanded
        migrated.append(key)
    # A converted checkpoint already has the expanded width, so zero migrations is valid.
    return result,migrated
