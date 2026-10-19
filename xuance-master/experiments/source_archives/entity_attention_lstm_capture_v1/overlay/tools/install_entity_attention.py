"""Add a dedicated policy construction path to a Xuance source checkout."""
from pathlib import Path


def install(root):
    root = Path(root)
    path = root / "xuance/torch/agents/multi_agent_rl/matd3_agents.py"
    source = path.read_text()
    anchor = "        # build representations\n"
    branch = '''        if self.config.policy == "UAV_EntityAttention_MATD3_Policy":
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

'''
    if branch not in source:
        if source.count(anchor) != 1:
            raise ValueError("Unexpected MATD3 agent construction source")
        path.write_text(source.replace(anchor, branch + anchor))
    path = root / "xuance/torch/policies/__init__.py"
    source = path.read_text()
    registration = '\nfrom .uav_attention_matd3 import UAV_EntityAttention_MATD3_Policy\nREGISTRY_Policy["UAV_EntityAttention_MATD3_Policy"] = UAV_EntityAttention_MATD3_Policy\n'
    if registration not in source:
        path.write_text(source + registration)
    path = root / "xuance/environment/__init__.py"
    source = path.read_text()
    old = 'getattr(config, "policy", None) != "MATD3_Policy" or'
    new = 'getattr(config, "policy", None) not in ("MATD3_Policy", "UAV_EntityAttention_MATD3_Policy") or'
    if new not in source:
        if source.count(old) != 1:
            raise ValueError("Unexpected frozen-predictor environment guard")
        path.write_text(source.replace(old, new))


if __name__ == "__main__":
    import sys
    install(sys.argv[1])
