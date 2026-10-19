"""Observation semantics, masking and symmetry of the local attention encoder."""
import pytest
import torch
from xuance.torch.policies.uav_entity_attention import UAVEntityAttentionEncoder


def observations(batch=3):
    generator = torch.Generator().manual_seed(17)
    obs = torch.randn(batch * 4, 88, generator=generator) * .1
    obs[:, 47] = 12 / 11  # The legitimate speed ratio exceeds one.
    obs[:, 63::6] = 1.
    return obs


def ids(batch=3):
    return torch.eye(4).repeat(batch, 1)


def test_exact_slices_preserve_all_non_entity_features():
    encoder = UAVEntityAttentionEncoder()
    obs = torch.arange(2 * 88).reshape(2, 88).float()
    identity = torch.eye(4)[:2]
    own, mates, buildings, invalid = encoder.split_observation(obs, identity)
    torch.testing.assert_close(own, torch.cat((obs[:, :11], obs[:, 29:58], identity), -1))
    torch.testing.assert_close(mates.flatten(1), obs[:, 11:29])
    torch.testing.assert_close(buildings.flatten(1), obs[:, 58:])
    assert own.shape == (2, 44) and not invalid.any()


@pytest.mark.parametrize("valid_count", [0, 1, 5])
def test_padding_mask_is_exact_and_gradients_are_finite(valid_count):
    encoder = UAVEntityAttentionEncoder()
    obs = observations()
    buildings = obs[:, 58:].reshape(-1, 5, 6)
    buildings[:, valid_count:, 5] = 0
    original = obs.clone()
    original[:, 58:].reshape(-1, 5, 6)[:, valid_count:, :5] = float("nan")
    state, detail = encoder(original, ids(), True)
    torch.testing.assert_close(state, encoder(obs, ids())["state"])
    assert torch.isfinite(state).all()
    assert detail["weights"].shape == (12, 4, 1, 9)
    assert (detail["weights"][..., 4 + valid_count:] == 0).all()
    torch.testing.assert_close(detail["weights"].sum(-1), torch.ones(12, 4, 1))
    state.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_same_type_entity_permutation_is_invariant():
    encoder = UAVEntityAttentionEncoder()
    obs, other = observations(), observations()
    other[:, 11:29] = obs[:, 11:29].reshape(-1, 3, 6)[:, [2, 0, 1]].flatten(1)
    other[:, 58:] = obs[:, 58:].reshape(-1, 5, 6)[:, [4, 2, 0, 3, 1]].flatten(1)
    torch.testing.assert_close(encoder(obs, ids())["state"], encoder(other, ids())["state"], atol=1e-6, rtol=1e-5)


def test_batches_do_not_communicate_and_layout_is_strict():
    encoder = UAVEntityAttentionEncoder()
    obs = observations()
    changed = obs.clone()
    changed[1:, 49:58] += 100
    torch.testing.assert_close(encoder(obs, ids())["state"][0], encoder(changed, ids())["state"][0])
    with pytest.raises(ValueError, match="88"):
        encoder(obs[:, :79], ids())
