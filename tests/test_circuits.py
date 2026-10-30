"""Interventions: controls change nothing, and path ablation touches only the path it names."""

import pytest
import torch
from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Block, Transformer

from deja_vu_tfm.circuits import (
  composition_scores,
  head_means,
  head_writes,
  identity,
  mean_ablation,
  path_ablation,
)

CFG = TransformerConfig(vocab_size=13, n_layers=2, d_model=16, n_heads=4, max_seq_len=10, mlp='none')


@pytest.fixture
def model() -> Transformer:
  torch.manual_seed(0)
  return Transformer(CFG).eval()


@pytest.fixture
def tokens() -> torch.Tensor:
  return torch.randint(0, 13, (3, 10), generator=torch.Generator().manual_seed(1))


def test_head_writes_add_up_to_the_attention_output(model: Transformer, tokens: torch.Tensor) -> None:
  _, cache = model(tokens, return_cache=True)
  block = model.blocks[0]
  assert isinstance(block, Block)
  writes = head_writes(block.attn, cache['blocks.0.attn.z']).sum(dim=2) + block.attn.W_o.bias
  torch.testing.assert_close(writes, cache['blocks.0.attn_out'])


def test_identity_patch_changes_nothing(model: Transformer, tokens: torch.Tensor) -> None:
  assert torch.equal(model(tokens, patch=identity(2)), model(tokens))


def test_mean_ablation_replaces_only_the_chosen_heads(model: Transformer, tokens: torch.Tensor) -> None:
  _, cache = model(tokens, return_cache=True)
  means = head_means(cache, 2)
  _, patched = model(tokens, return_cache=True, patch=mean_ablation([(1, 2)], means))
  z, clean = patched['blocks.1.attn.z'], cache['blocks.1.attn.z']
  assert torch.equal(z[:, :, 2], means[1][2].expand_as(z[:, :, 2]))
  assert torch.equal(z[:, :, [0, 1, 3]], clean[:, :, [0, 1, 3]])
  assert torch.equal(patched['blocks.0.attn.z'], cache['blocks.0.attn.z'])


@pytest.mark.parametrize('path', ['q', 'k', 'v'])
def test_path_ablation_of_nothing_reproduces_the_clean_model(model: Transformer, tokens: torch.Tensor, path: str) -> None:
  _, cache = model(tokens, return_cache=True)
  means = head_means(cache, 2)[0]
  patch = path_ablation(model, tokens, [], means, path)  # pyright: ignore[reportArgumentType]
  torch.testing.assert_close(model(tokens, patch=patch), model(tokens))


@pytest.mark.parametrize('path', ['q', 'k', 'v'])
def test_path_ablation_leaves_layer_zero_and_the_direct_path_alone(model: Transformer, tokens: torch.Tensor, path: str) -> None:
  _, cache = model(tokens, return_cache=True)
  means = head_means(cache, 2)[0]
  _, patched = model(tokens, return_cache=True, patch=path_ablation(model, tokens, [1], means, path))  # pyright: ignore[reportArgumentType]
  assert torch.equal(patched['blocks.0.attn.z'], cache['blocks.0.attn.z'])
  assert torch.equal(patched['blocks.1.resid_pre'], cache['blocks.1.resid_pre'])
  changed_pattern = not torch.allclose(patched['blocks.1.attn.pattern'], cache['blocks.1.attn.pattern'])
  assert changed_pattern == (path != 'v')
  assert not torch.allclose(patched['blocks.1.attn.z'], cache['blocks.1.attn.z'])


def test_composition_scores_are_ratios_for_every_head_pair(model: Transformer) -> None:
  scores = composition_scores(model)
  assert set(scores) == {'q', 'k', 'v'}
  for value in scores.values():
    assert value.shape == (4, 4)
    assert ((value > 0) & (value <= 1)).all()


def test_composition_needs_two_layers() -> None:
  with pytest.raises(ValueError, match='two-layer'):
    composition_scores(Transformer(TransformerConfig(vocab_size=5, n_layers=1, d_model=8, n_heads=2, mlp='none')))
