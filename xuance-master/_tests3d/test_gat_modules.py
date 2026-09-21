import torch
from xuance.torch.policies.gat_modules import ObstacleGAT


def _mod():
    torch.manual_seed(0)
    return ObstacleGAT(node_dim=4, query_dim=8, hidden_dim=16, num_heads=2)


def test_output_shape():
    m = _mod()
    emb = m(torch.randn(5, 8), torch.randn(5, 3, 4))
    assert emb.shape == (5, 16)


def test_attention_sums_to_one_over_valid():
    m = _mod()
    q = torch.randn(2, 8)
    nodes = torch.randn(2, 4, 4)
    valid = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]])
    emb, attn = m(q, nodes, valid, return_attn=True)
    # attn: (B, num_heads, K); valid weights sum to 1, masked nodes ~0
    s = (attn * valid.unsqueeze(1)).sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert torch.allclose(attn[0, :, 2:], torch.zeros_like(attn[0, :, 2:]), atol=1e-6)


def test_all_invalid_is_finite():
    m = _mod()
    valid = torch.zeros(2, 3)
    emb = m(torch.randn(2, 8), torch.randn(2, 3, 4), valid)
    assert torch.isfinite(emb).all()


def test_permutation_invariance():
    m = _mod()
    q = torch.randn(1, 8)
    nodes = torch.randn(1, 3, 4)
    valid = torch.ones(1, 3)
    perm = [2, 0, 1]
    e1 = m(q, nodes, valid)
    e2 = m(q, nodes[:, perm, :], valid[:, perm])
    assert torch.allclose(e1, e2, atol=1e-5)
