"""Head scores: each one reads the key it claims to, checked on hand-built attention patterns."""

import pytest
import torch
from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Transformer

from deja_vu_tfm.heads import (
  chance,
  diagonal_score,
  head_scores,
  induction_heads,
  score_queries,
  scores_from_cache,
)
from deja_vu_tfm.repeats import Repeats

# BOS, filler 7 9, then 3 4 5 twice: period 3, second copy at positions 6, 7, 8.
BATCH = Repeats(torch.tensor([[20, 7, 9, 3, 4, 5, 3, 4, 5]]), torch.tensor([3]))


def one_hot_pattern(keys: list[int]) -> torch.Tensor:
  """A one-head pattern where query q puts all its attention on keys[q]."""
  pattern = torch.zeros(1, 1, len(keys), len(keys))
  for q, k in enumerate(keys):
    assert 0 <= k <= q
    pattern[0, 0, q, k] = 1.0
  return pattern


def scores_of(pattern: torch.Tensor) -> dict[str, float]:
  return {
    name: diagonal_score(pattern, offset, queries).item()
    for name, (offset, queries) in score_queries(BATCH).items()
  }


def test_previous_token_head() -> None:
  pattern = one_hot_pattern([0] + list(range(8)))
  assert scores_of(pattern) == {'previous': 1.0, 'duplicate': 0.0, 'induction': 0.0}


def test_induction_head() -> None:
  # Queries 6, 7, 8 hold 3, 4, 5; the tokens after their earlier copies sit at 4, 5, 6.
  pattern = one_hot_pattern([0, 0, 0, 0, 0, 0, 4, 5, 6])
  assert scores_of(pattern)['induction'] == 1.0
  assert scores_of(pattern)['duplicate'] == 0.0


def test_duplicate_token_head() -> None:
  pattern = one_hot_pattern([0, 0, 0, 0, 0, 0, 3, 4, 5])
  # Query 1's attention to BOS is also attention to its previous token: 1 of 8 queries.
  assert scores_of(pattern) == {'previous': 1 / 8, 'duplicate': 1.0, 'induction': 0.0}


def test_induction_score_averages_over_the_second_copy_only() -> None:
  # Right on query 7 alone of the three second-copy queries; everything else on BOS.
  pattern = one_hot_pattern([0, 0, 0, 0, 0, 0, 0, 5, 0])
  assert scores_of(pattern)['induction'] == pytest.approx(1 / 3)


def test_scores_follow_each_sequences_own_period() -> None:
  # Two sequences with periods 2 and 3, each with a perfect induction pattern of its own.
  batch =Repeats(torch.tensor([[20, 1, 2, 3, 8, 9, 8, 9], [20, 1, 6, 4, 5, 6, 4, 5]]), torch.tensor([2, 3]))
  pattern = torch.zeros(2, 1, 8, 8)
  for b, period in enumerate([2, 3]):
    for q in range(8 - period, 8):
      pattern[b, 0, q, q - period + 1] = 1.0
  offset, queries = score_queries(batch)['induction']
  assert diagonal_score(pattern, offset, queries).item() == 1.0


def test_queries_that_reach_before_the_start_are_rejected() -> None:
  pattern = one_hot_pattern([0] * 9)
  with pytest.raises(ValueError, match='before the start'):
    diagonal_score(pattern, torch.tensor([5]), torch.ones(1, 9, dtype=torch.bool))


def test_chance_is_the_score_of_uniform_attention() -> None:
  uniform = torch.tril(torch.ones(9, 9))
  uniform = (uniform / uniform.sum(dim=1, keepdim=True))[None, None]
  expected = chance(BATCH)
  for name, value in scores_of(uniform).items():
    assert value == pytest.approx(expected[name])


def test_head_scores_cover_every_layer_and_head() -> None:
  torch.manual_seed(0)
  model = Transformer(TransformerConfig(vocab_size=21, n_layers=2, d_model=16, n_heads=4, max_seq_len=9, mlp='none'))
  scores = head_scores(model, BATCH)
  assert set(scores) == {'previous', 'duplicate', 'induction'}
  for value in scores.values():
    assert value.shape == (2, 4)
    assert ((value >= 0) & (value <= 1)).all()
  _, cache = model(BATCH.tokens, return_cache=True)
  assert torch.equal(scores_from_cache(cache, BATCH, 2)['induction'], scores['induction'])


def test_induction_heads_clear_the_threshold_or_fall_back_to_the_best() -> None:
  assert induction_heads(torch.tensor([0.1, 0.6, 0.31, 0.29])) == [1, 2]
  assert induction_heads(torch.tensor([0.1, 0.2, 0.05])) == [1]
