import torch
import numpy as np
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import List, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents.multi_agent_rl.iddpg_agents import IDDPG_Agents


class MATD3_Agents(IDDPG_Agents):
    """The implementation of MATD3 agents.

    Args:
        config: The Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
            num_agents: Optional[int] = None,
            agent_keys: Optional[List[str]] = None,
            state_space: Optional[Space] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[MultiAgentBaseCallback] = None
    ):
        super(MATD3_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

    def action(self, *args, **kwargs):
        result = super().action(*args, **kwargs)
        if getattr(self.config, "matd3_clip_actions", False):
            # Exploration happens after tanh. Bound the command before both
            # environment execution and replay storage see it.
            for commands in result["actions"]:
                for key, command in commands.items():
                    space = self.action_space[key]
                    commands[key] = np.clip(command, space.low, space.high)
        return result

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        if self.config.policy == "UAV_EntityAttention_MATD3_Policy":
            from xuance.torch.policies.uav_attention_matd3 import UAV_EntityAttention_MATD3_Policy
            if not self.use_parameter_sharing or self.use_rnn or self.distributed_training:
                raise ValueError("Entity MATD3 requires shared, feed-forward, single-learner execution")
            if self.config.env_name != "uav_pursuit_coverage_3d" or self.config.representation != "Basic_Identical":
                raise ValueError("Entity MATD3 requires raw UAV pursuit observations")
            if any(s.shape != (88,) for s in self.observation_space.values()):
                raise ValueError("Entity MATD3 requires 88-D prediction observations")
            if (getattr(self.config, "uav_prediction_mode", "none") != "lstm" or
                    not getattr(self.config, "curriculum_context_obs", False) or
                    not getattr(self.config, "use_nearest_building_obs", False) or
                    int(getattr(self.config, "nearest_building_k", 5)) != 5 or
                    getattr(self.config, "use_target_prediction_obs", False)):
                raise ValueError("Entity MATD3 requires the frozen-LSTM, four-context, five-building layout")
            if getattr(self.config, "use_graph_module", False) or getattr(self.config, "use_obstacle_gat", False):
                raise ValueError("Entity attention cannot be combined with legacy graph modules")
            return UAV_EntityAttention_MATD3_Policy(
                action_space=self.action_space, model_keys=self.model_keys,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                hidden_dim=int(getattr(self.config, "entity_attention_dim", 128)),
                heads=int(getattr(self.config, "entity_attention_heads", 4)),
                initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                prediction_sha256=getattr(self.config, "uav_prediction_sha256", None), device=device)

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        critic_in = [sum(self.observation_space[k].shape) + sum(self.action_space[k].shape) for k in self.agent_keys]
        space_critic_in = {k: (sum(critic_in),) for k in self.agent_keys}
        C_representation = self._build_representation(self.config.representation, space_critic_in, self.config)

        # build policies
        if self.config.policy == "MATD3_Policy":
            policy = REGISTRY_Policy["MATD3_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                actor_representation=A_representation, critic_representation=C_representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"MATD3 currently does not support the policy named {self.config.policy}.")

        return policy
