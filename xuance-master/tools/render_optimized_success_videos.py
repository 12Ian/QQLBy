"""Native MATD3 video adapter for Tanh/Softsign; preserve simulator replay checks."""
import torch
from xuance.torch.policies.deterministic_marl import MATD3_Policy
from xuance.torch.representations import Basic_Identical
from tools import render_baseline_success_videos as native

original_policy=native.native_policy
original_render=native.render


def native_policy(config,env,checkpoint,device='cpu'):
    if config['activation_action']=='tanh' and device=='cpu':
        return original_policy(config,env,checkpoint)
    assert config['activation_action'] in ('tanh','softsign') and config['agent']=='MATD3'
    assert config['representation']=='Basic_Identical' and config['activation']=='leaky_relu'
    assert config['use_parameter_sharing'] and not config.get('use_rnn') and not config.get('normalize')
    assert not config.get('use_graph_module') and not config.get('use_obstacle_gat')
    key=env.agents[0]
    size=env.observation_space[key].shape[0]
    actors=torch.nn.ModuleDict({key:Basic_Identical((size,),device=device)})
    critics=torch.nn.ModuleDict({key:Basic_Identical(((size+3)*env.num_agents,),device=device)})
    policy=MATD3_Policy(action_space=env.action_space,n_agents=env.num_agents,actor_representation=actors,
        critic_representation=critics,actor_hidden_size=config['actor_hidden_size'],
        critic_hidden_size=config['critic_hidden_size'],activation=torch.nn.LeakyReLU,
        activation_action=torch.nn.Softsign if config['activation_action']=='softsign' else torch.nn.Tanh,
        device=device,use_parameter_sharing=True,
        model_keys=[key],use_rnn=False,rnn=None)
    state=torch.load(checkpoint,map_location=device,weights_only=False)['policy']
    assert all(torch.isfinite(v).all().item() for v in state.values())
    policy.load_state_dict(state,strict=True)
    policy.eval()
    return policy,key,torch.eye(env.num_agents,device=device)


def render(data,report,output,playback_speed=2.):
    # Hide the historical 5 m guide when direct capture is disabled.
    from matplotlib.axes import Axes
    original_line=Axes.axhline
    def axhline(self,*args,**kwargs):
        line=original_line(self,*args,**kwargs)
        if kwargs.get('color')=='#ed6774' and not report['course'].get('direct_intercept_enabled',True):
            line.set_visible(False)
        return line
    Axes.axhline=axhline
    try:
        return original_render(data,report,output,playback_speed)
    finally:
        Axes.axhline=original_line


native.native_policy=native_policy
native.render=render


if __name__=='__main__':
    native.main()
