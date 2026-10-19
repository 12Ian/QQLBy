"""Entity-attention actor and independent centralized twin Qs for MATD3."""
from copy import deepcopy
import torch
from torch import nn
from .core import ActorNet, CriticNet
from .uav_entity_attention import UAVEntityAttentionEncoder


class UAV_EntityAttention_MATD3_Policy(nn.Module):
    def __init__(self, action_space, model_keys, actor_hidden_size=(512, 512),
                 critic_hidden_size=(512, 512), hidden_dim=128, heads=4,
                 initialize=None, activation=nn.LeakyReLU, activation_action=nn.Tanh,
                 prediction_sha256=None, device="cpu"):
        super().__init__()
        self.device, self.action_space = device, action_space
        self.model_keys, self.agent_keys = list(model_keys), list(action_space)
        self.n_agents, self.use_parameter_sharing, self.use_rnn = 4, True, False
        if len(self.model_keys) != 1 or len(action_space) != 4 or any(
                space.shape != (3,) for space in action_space.values()):
            raise ValueError("Entity MATD3 requires four shared-parameter 3-D agents")
        self.hidden_dim, self.heads = int(hidden_dim), int(heads)
        self._architecture = dict(version=1, policy=type(self).__name__, observation_dim=88,
            self_dim=44, teammate_shape=[3, 6], building_shape=[5, 6],
            model_keys=self.model_keys, agent_keys=self.agent_keys, hidden_dim=self.hidden_dim,
            heads=self.heads, actor_hidden_size=list(actor_hidden_size),
            critic_hidden_size=list(critic_hidden_size), prediction_sha256=prediction_sha256)
        key = self.model_keys[0]
        # Construct A and B separately; copying an online critic would correlate their initialization.
        for name in ("actor_representation", "critic_A_representation", "critic_B_representation"):
            setattr(self, name, nn.ModuleDict({key: UAVEntityAttentionEncoder(hidden_dim, heads, activation)}))
        self.actor = nn.ModuleDict({key: ActorNet(hidden_dim, 3, actor_hidden_size,
            None, initialize, activation, activation_action, device)})
        critic_width = 4 * hidden_dim + 12 + 4
        self.critic_A = nn.ModuleDict({key: CriticNet(critic_width, critic_hidden_size,
            None, initialize, activation, device)})
        self.critic_B = nn.ModuleDict({key: CriticNet(critic_width, critic_hidden_size,
            None, initialize, activation, device)})
        for name in self._online_modules:
            setattr(self, "target_" + name, deepcopy(getattr(self, name)).requires_grad_(False))
        self.to(device)

    _online_modules = ("actor_representation", "actor", "critic_A_representation", "critic_A",
                       "critic_B_representation", "critic_B")

    def get_extra_state(self):
        return deepcopy(self._architecture)

    def set_extra_state(self, state):
        if state != self._architecture:
            raise ValueError("Entity MATD3 checkpoint architecture/predictor mismatch")

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Xuance's general loader can request strict=False; reject partial/old MLP states anyway.
        if state_dict.get("_extra_state") != self._architecture:
            raise ValueError("Entity MATD3 checkpoint architecture/predictor mismatch")
        return super().load_state_dict(state_dict, strict=True, **kwargs)

    @property
    def parameters_actor(self):
        return {k: list(self.actor_representation[k].parameters()) + list(self.actor[k].parameters())
                for k in self.model_keys}

    @property
    def parameters_critic(self):
        return {k: [p for name in self._online_modules[2:] for p in getattr(self, name)[k].parameters()]
                for k in self.model_keys}

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=next(self.parameters()).device)

    def _actions(self, observation, agent_ids, target=False, agent_key=None):
        prefix = "target_" if target else ""
        encoders, heads = getattr(self, prefix + "actor_representation"), getattr(self, prefix + "actor")
        actions = {}
        for key in ([agent_key] if agent_key else self.model_keys):
            state = encoders[key](self._tensor(observation[key]), self._tensor(agent_ids))["state"]
            actions[key] = heads[key](state)
        return {}, actions

    def forward(self, observation, agent_ids, agent_key=None, rnn_hidden=None):
        return self._actions(observation, agent_ids, False, agent_key)

    @torch.no_grad()
    def Atarget(self, next_observation, agent_ids, agent_key=None, rnn_hidden=None):
        return self._actions(next_observation, agent_ids, True, agent_key)

    def _values(self, joint_observation, joint_actions, agent_ids, target=False, agent_key=None):
        obs, actions, ids = map(self._tensor, (joint_observation, joint_actions, agent_ids))
        if obs.ndim != 2 or obs.shape[-1] != 4 * 88:
            raise ValueError("Entity critic requires [B,352] joint observations in agent order")
        batch = len(obs)
        if actions.shape != (batch, 12) or ids.shape != (batch * 4, 4):
            raise ValueError("Entity critic requires [B,12] actions and [B*4,4] query IDs")
        # Local encoder IDs follow the fixed agent order, independently of which Q is queried.
        local_ids = torch.eye(4, device=obs.device).repeat(batch, 1)
        twins = []
        for branch in ("A", "B"):
            prefix = ("target_" if target else "") + "critic_" + branch
            encoder, head = getattr(self, prefix + "_representation"), getattr(self, prefix)
            values = {}
            for key in ([agent_key] if agent_key else self.model_keys):
                state = encoder[key](obs.reshape(batch * 4, 88), local_ids)["state"].reshape(batch, -1)
                joint = torch.cat((state, actions), dim=-1)
                joint = joint.unsqueeze(1).expand(-1, 4, -1).reshape(batch * 4, -1)
                values[key] = head[key](torch.cat((joint, ids), dim=-1))
            twins.append(values)
        return twins

    def Qpolicy(self, joint_observation, joint_actions, agent_ids, agent_key=None, rnn_hidden=None):
        a, b = self._values(joint_observation, joint_actions, agent_ids, False, agent_key)
        return a, b, {k: (a[k] + b[k]) * .5 for k in a}

    @torch.no_grad()
    def Qtarget(self, joint_observation, joint_actions, agent_ids, agent_key=None, rnn_hidden=None):
        a, b = self._values(joint_observation, joint_actions, agent_ids, True, agent_key)
        return {k: torch.minimum(a[k], b[k]) for k in a}

    @torch.no_grad()
    def soft_update(self, tau=.005):
        if not 0. <= tau <= 1.:
            raise ValueError("Soft-update tau must be in [0,1]")
        for name in self._online_modules:
            source = dict(getattr(self, name).named_parameters())
            target = dict(getattr(self, "target_" + name).named_parameters())
            if source.keys() != target.keys():
                raise RuntimeError("Online and target parameter layouts differ")
            for key in source:
                target[key].lerp_(source[key], tau)
