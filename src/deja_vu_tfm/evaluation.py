"""
Run a model over many sequences in chunks, keeping only what the metrics need.

A text model's logits are large: 512 windows of 256 tokens over GPT-2's 50,257-token
vocabulary is 26 GB. So evaluation runs in chunks and keeps each target's loss, whether
the model ranked it first, and the attention patterns, then the metrics are computed once
over the whole set. Every metric is a mean over targets or queries, never a mean of chunk
means, so the chunk size cannot change a result.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from tiny_transformer.model import Patch, Transformer
from tiny_transformer.modes import evaluating
from torch import Tensor


def token_losses(
  logits: Float[Tensor, 'batch seq vocab'], tokens: Int[Tensor, 'batch seq']
) -> tuple[Float[Tensor, 'batch target'], Bool[Tensor, 'batch target']]:
  """
  Each target's loss, and whether the model ranked it first.

  Column t is the prediction of token t + 1 from tokens 0..t, as in next_token_loss.
  """
  loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction='none')
  correct = logits[:, :-1].argmax(dim=-1) == tokens[:, 1:]
  return loss, correct


def run(
  model: Transformer,
  tokens: Int[Tensor, 'batch seq'],
  *,
  chunk: int = 64,
  patch: Callable[[Tensor], dict[str, Patch]] | None = None,
  patterns: bool = True,
) -> tuple[Float[Tensor, 'batch target'], Bool[Tensor, 'batch target'], dict[str, Tensor]]:
  """
  Losses, correctness and attention patterns for every sequence, in eval mode.

  `patch`, if given, builds the patches for each chunk from that chunk's tokens, so a
  patch can depend on the inputs (path patching recomputes attention from them).
  `patterns=False` skips keeping the attention patterns, which for long text windows are
  the biggest thing kept.
  """
  device = next(model.parameters()).device
  losses, correct, kept = [], [], {}
  with evaluating(model):
    for start in range(0, len(tokens), chunk):
      part = tokens[start:start + chunk].to(device)
      logits, cache = model(part, return_cache=True, patch=patch(part) if patch else None)
      loss, right = token_losses(logits, part)
      losses.append(loss)
      correct.append(right)
      for name, value in cache.items():
        if patterns and name.endswith('attn.pattern'):
          kept.setdefault(name, []).append(value)
  return torch.cat(losses), torch.cat(correct), {name: torch.cat(parts) for name, parts in kept.items()}
