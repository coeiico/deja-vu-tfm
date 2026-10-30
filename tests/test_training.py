"""Learning checks: the re-initialised model trains every parameter, and can memorise a tiny batch."""

import pytest
import torch
from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Transformer
from tiny_transformer.objective import next_token_loss

from deja_vu_tfm.evaluation import token_losses
from deja_vu_tfm.initialisation import initialise, sinusoids
from deja_vu_tfm.repeats import RepeatTask, losses

TASK = RepeatTask(n_tokens=16, seq_len=16, min_period=3, max_period=6)
CFG = TransformerConfig(vocab_size=TASK.vocab_size, n_layers=2, d_model=32, n_heads=4, max_seq_len=16, mlp='none', norm='none')


@pytest.mark.parametrize('weights', ['gpt2', 'torch'])
def test_every_parameter_receives_a_gradient(weights: str) -> None:
  torch.manual_seed(0)
  model = Transformer(CFG)
  initialise(model, weights, 'sinusoidal')  # pyright: ignore[reportArgumentType]
  tokens = TASK.sample(8, torch.Generator().manual_seed(0)).tokens
  next_token_loss(model(tokens), tokens).backward()
  for name, p in model.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if name != 'pos_embed.weight':  # the last position's embedding feeds no target
      assert p.grad.abs().sum() > 0, name


def test_torch_init_uses_pytorch_defaults() -> None:
  torch.manual_seed(0)
  model = Transformer(CFG)
  assert model.token_embed.weight.std() < 0.03  # GPT-2's 0.02
  initialise(model, 'torch', 'learned')
  assert 0.8 < model.token_embed.weight.std() < 1.2  # nn.Embedding's N(0, 1)


def test_sinusoidal_positions_match_the_formula_and_token_scale() -> None:
  table = sinusoids(16, 32)
  assert torch.allclose(table[3, 0], torch.sin(torch.tensor(3.0)))
  assert torch.allclose(table[3, 1], torch.cos(torch.tensor(3.0)))
  torch.manual_seed(0)
  model = Transformer(CFG)
  initialise(model, 'torch', 'sinusoidal')
  rms = lambda t: t.pow(2).mean().sqrt()  # noqa: E731
  assert torch.allclose(rms(model.pos_embed.weight), rms(model.token_embed.weight))


def test_tiny_model_memorises_a_tiny_batch() -> None:
  torch.manual_seed(0)
  model = Transformer(CFG)
  initialise(model, 'torch', 'learned')
  batch = TASK.sample(4, torch.Generator().manual_seed(1))
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
  for _ in range(300):
    loss = next_token_loss(model(batch.tokens), batch.tokens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  model.eval()
  with torch.no_grad():
    token_loss, correct = token_losses(model(batch.tokens), batch.tokens)
  # Every sequence starts with BOS, so its first token cannot be told apart from the other
  # sequences' first tokens: that target stays at about ln 4. Every later target has a
  # unique prefix, and fresh tokens are unpredictable in general (ln 16 ≈ 2.8 nats), so a
  # near-zero loss on them means the model memorised these four sequences outright.
  assert token_loss[:, 1:].mean() < 0.05
  assert losses(token_loss, correct, batch)['copy_accuracy'] == 1.0
