"""
How a model's weights start, which decides which copying circuit training finds.

**Weights.** tiny-transformer initialises like GPT-2: every weight and embedding normal
with std 0.02. 'torch' instead uses PyTorch's own defaults, as the small induction-head
replications do: embeddings N(0, 1), linear layers uniform with bound 1/sqrt(fan_in). At
that scale attention patterns are sharp from the first step, and position alone already
gives each head a strong preference over keys.

**Positions.**
tiny-transformer learns one position vector per position, initialised like every other
embedding (normal, std 0.02). A previous-token head then has to learn "attend one step
back" separately at every position, from vectors that start out unrelated.

Sinusoidal vectors (Vaswani et al. 2017) make that easy from the start: the vector for
position p - 1 is a fixed rotation of the vector for p, the same at every p, so a single
linear map in a head's query-key circuit can point every position at its predecessor.
Here they are only the starting point. The embeddings stay ordinary learned parameters,
so the architecture is unchanged.
"""

import math
from typing import Literal

import torch
from jaxtyping import Float
from tiny_transformer.model import Transformer
from torch import Tensor

WeightInit = Literal['gpt2', 'torch']
PositionInit = Literal['learned', 'sinusoidal']


def sinusoids(n_positions: int, d_model: int) -> Float[Tensor, 'position d_model']:
  """PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i + 1] = cos(p / 10000^(2i/d))."""
  position = torch.arange(n_positions, dtype=torch.float32)[:, None]
  frequency = torch.exp(-math.log(10_000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
  table = torch.zeros(n_positions, d_model)
  table[:, 0::2] = torch.sin(position * frequency)
  table[:, 1::2] = torch.cos(position * frequency[:d_model // 2])
  return table


def initialise(model: Transformer, weights: WeightInit, positions: PositionInit) -> None:
  """
  Re-initialise a freshly built model (before training).

  Sinusoidal positions are scaled so their root-mean-square matches the token
  embeddings', so neither positions nor tokens dominate at the start.
  """
  if weights == 'torch':
    for module in model.modules():
      reset = getattr(module, 'reset_parameters', None)
      if callable(reset):
        reset()
  if positions == 'learned':
    return
  table = sinusoids(model.cfg.max_seq_len, model.cfg.d_model)
  table = table * model.token_embed.weight.pow(2).mean().sqrt() / table.pow(2).mean().sqrt()
  with torch.no_grad():
    model.pos_embed.weight.copy_(table.to(model.pos_embed.weight.device))
