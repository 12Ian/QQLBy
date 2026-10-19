"""Local entity encoding for the fixed 88-D UAV prediction observation."""
import torch
from torch import nn


class UAVEntityAttentionEncoder(nn.Module):
    obs_dim, n_agents, building_k = 88, 4, 5

    def __init__(self, hidden_dim=128, heads=4, activation=nn.LeakyReLU):
        super().__init__()
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads:
            raise ValueError("Attention width must be positive and divisible by heads")
        self.hidden_dim = hidden_dim
        self.output_shapes = {"state": (hidden_dim,)}
        def mlp(width):
            return nn.Sequential(nn.Linear(width, hidden_dim), activation(),
                                 nn.Linear(hidden_dim, hidden_dim), activation())
        self.self_mlp, self.teammate_mlp, self.building_mlp = mlp(44), mlp(6), mlp(6)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), activation(),
                                 nn.Linear(2 * hidden_dim, hidden_dim))

    def split_observation(self, obs, ids):
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim or ids.shape != (len(obs), 4):
            raise ValueError("Entity encoder requires [N,88] observations and [N,4] IDs")
        own = torch.cat((obs[:, :11], obs[:, 29:58], ids), dim=-1)
        teammates = obs[:, 11:29].reshape(-1, 3, 6)
        buildings = obs[:, 58:88].reshape(-1, 5, 6)
        invalid = buildings[..., 5] < .5
        buildings = buildings.masked_fill(invalid.unsqueeze(-1), 0.)
        return own, teammates, buildings, invalid

    def forward(self, obs, ids, return_attention=False):
        own, teammates, buildings, invalid = self.split_observation(obs, ids)
        self_state = self.self_mlp(own)
        tokens = torch.cat((self_state.unsqueeze(1), self.teammate_mlp(teammates),
                            self.building_mlp(buildings)), dim=1)
        mask = torch.cat((torch.zeros((len(obs), 4), dtype=torch.bool, device=obs.device), invalid), dim=1)
        normalized = self.token_norm(tokens)
        attended, weights = self.attention(normalized[:, :1], normalized, normalized,
            key_padding_mask=mask, need_weights=return_attention, average_attn_weights=False)
        state = self_state + attended[:, 0]
        state = state + self.ffn(self.ffn_norm(state))
        if return_attention:
            return state, {"weights": weights, "padding_mask": mask}
        return {"state": state}
