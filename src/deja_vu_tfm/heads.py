"""
Scores that say what an attention head does on repeated sequences.

Each score is the average attention weight a head puts on one particular key, over a set
of queries. On a sequence with period L, a query at position q in the second copy holds
the same token as position q - L. Three keys matter:

- **Previous token**, q - 1. A previous-token head copies each token's predecessor into its
  own position, so position k comes to hold "the token before me was x".
- **Duplicate token**, q - L: the earlier occurrence of the current token.
- **Induction**, q - L + 1: the token that followed the earlier occurrence. Attending there
  and copying what it holds predicts the next token. To find it, the head matches its
  query ("I am x") against keys that say "the token before me was x", which is exactly
  what a previous-token head writes. That is the two-layer induction circuit.

Previous-token scores use every query after BOS. Duplicate and induction scores use the
queries of the second copy, q from L positions before the end to the last.
"""

import torch
from jaxtyping import Bool, Float, Int
from tiny_transformer.model import Transformer
from tiny_transformer.modes import evaluating
from torch import Tensor

from deja_vu_tfm.repeats import Repeats

SCORES = ('previous', 'duplicate', 'induction')


def diagonal_score(
  pattern: Float[Tensor, 'batch head query key'],
  offset: Int[Tensor, ' batch'],
  queries: Bool[Tensor, 'batch query'],
) -> Float[Tensor, ' head']:
  """
  Mean attention from each marked query q to key q - offset, per head.

  `offset` is per sequence, because the period is. Every marked query must have
  q - offset >= 0.
  """
  batch, heads, seq, _ = pattern.shape
  position = torch.arange(seq, device=pattern.device)
  keys = position[None, :] - offset[:, None]  # [batch, query]
  if (keys[queries] < 0).any():
    raise ValueError('a marked query would attend before the start of the sequence')
  keys = keys.clamp(min=0)
  weights = pattern.gather(3, keys[:, None, :, None].expand(batch, heads, seq, 1)).squeeze(-1)
  marked = queries[:, None, :].float()  # [batch, 1, query]
  return (weights * marked).sum(dim=(0, 2)) / marked.sum(dim=(0, 2))


def score_queries(batch: Repeats) -> dict[str, tuple[Int[Tensor, ' batch'], Bool[Tensor, 'batch query']]]:
  """For each score: the offset back to its key, and the queries it averages over."""
  position = torch.arange(batch.tokens.shape[1], device=batch.tokens.device)
  second_copy = position[None, :] >= batch.second_copy_start[:, None]
  after_bos = (position >= 1)[None, :].expand_as(second_copy)
  return {
    'previous': (torch.ones_like(batch.period), after_bos),
    'duplicate': (batch.period, second_copy),
    'induction': (batch.period - 1, second_copy),
  }


def chance(batch: Repeats) -> dict[str, float]:
  """Each score for a head that spreads its attention evenly over every key it can see."""
  position = torch.arange(batch.tokens.shape[1], device=batch.tokens.device)
  uniform = 1 / (position + 1).float()  # query q sees q + 1 keys
  return {
    name: (uniform[None, :] * queries).sum().item() / queries.sum().item()
    for name, (_, queries) in score_queries(batch).items()
  }


def scores_from_cache(cache: dict[str, Tensor], batch: Repeats, n_layers: int) -> dict[str, Float[Tensor, 'layer head']]:
  queries = score_queries(batch)
  return {
    name: torch.stack([
      diagonal_score(cache[f'blocks.{layer}.attn.pattern'], offset, marked) for layer in range(n_layers)
    ]).cpu()
    for name, (offset, marked) in queries.items()
  }


def head_scores(model: Transformer, batch: Repeats) -> dict[str, Float[Tensor, 'layer head']]:
  """Previous-token, duplicate-token and induction scores for every head, in eval mode."""
  with evaluating(model):
    _, cache = model(batch.tokens, return_cache=True)
  return scores_from_cache(cache, batch, model.cfg.n_layers)


# A head counts as an induction head when it puts at least this much of its attention on
# the induction target. Uniform attention scores about 0.03 on the default task.
INDUCTION_THRESHOLD = 0.3


def induction_heads(scores: Float[Tensor, ' head']) -> list[int]:
  """The heads of one layer that clear INDUCTION_THRESHOLD, or the single best if none do."""
  heads = [h for h, s in enumerate(scores.tolist()) if s >= INDUCTION_THRESHOLD]
  return heads or [int(scores.argmax())]
