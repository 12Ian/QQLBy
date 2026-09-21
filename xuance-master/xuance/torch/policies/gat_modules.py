"""Obstacle attention module (GAT-O) for the 3D pursuit env."""
import torch
import torch.nn as nn


class ObstacleGAT(nn.Module):
    """Multi-head attention pooling: a pursuer query attends over its k obstacle
    node features, producing a (B, hidden_dim) obstacle-aware embedding.

    Order-invariant over nodes; masked (valid=0) nodes contribute nothing;
    all-invalid -> finite (zero-ish) embedding."""

    def __init__(self, node_dim, query_dim, hidden_dim=32, num_heads=2, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q = nn.Linear(query_dim, hidden_dim * num_heads, device=device)
        self.k = nn.Linear(node_dim, hidden_dim * num_heads, device=device)
        self.v = nn.Linear(node_dim, hidden_dim * num_heads, device=device)
        self.out = nn.Linear(hidden_dim * num_heads, hidden_dim, device=device)

    def forward(self, query, nodes, valid=None, return_attn=False):
        B, K, _ = nodes.shape
        H, nh = self.hidden_dim, self.num_heads
        q = self.q(query).reshape(B, nh, H)
        k = self.k(nodes).reshape(B, K, nh, H)
        v = self.v(nodes).reshape(B, K, nh, H)
        scores = torch.einsum('bhd,bkhd->bhk', q, k) / (H ** 0.5)   # (B, nh, K)
        attn = torch.softmax(scores, dim=-1)
        if valid is not None:
            m = (valid > 0.5).float().unsqueeze(1)                  # (B, 1, K)
            attn = attn * m
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)   # renormalize over valid
        ctx = torch.einsum('bhk,bkhd->bhd', attn, v).reshape(B, nh * H)
        emb = self.out(ctx)
        if return_attn:
            return emb, attn
        return emb
