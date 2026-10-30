"""
The synthetic task: random tokens with one repeated segment.

A sequence of `seq_len` tokens is

  BOS  f_1 ... f_m  r_1 ... r_L  r_1 ... r_L

where every f (filler) and r is drawn uniformly from [0, n_tokens), and the period L is
drawn per sequence from [min_period, max_period]. The filler pads the sequence to a fixed
length, so the repeat always ends at the last position and starts wherever L puts it.

Nothing here is predictable from the tokens' identities: they are uniform and independent.
The only predictable targets are r_2 ... r_L of the second copy, and only by copying: find
where the current token appeared before and predict the token that followed it. That is
what an induction head does.

L varies per sequence on purpose. With a fixed L, a single head could copy by attending a
fixed distance back, using position alone, and the task would no longer need induction.

The split unit is the repeated segment r. Validation and test sequences are fixed by their
own seeds, and training redraws any sequence whose segment is held out. Each sequence and
its own repeat are, by construction, always in the same split.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Bump when the sequence layout or sampling changes, so old runs are not compared as equal.
TASK_VERSION = 1


@dataclass(frozen=True)
class RepeatTask:
  n_tokens: int = 64
  """Distinct random tokens. BOS is one more id on top."""
  seq_len: int = 64
  """Tokens per sequence, BOS included."""
  min_period: int = 8
  """Shortest repeated segment."""
  max_period: int = 28
  """Longest repeated segment. Two copies must fit after BOS."""

  def __post_init__(self) -> None:
    if not 2 <= self.min_period <= self.max_period:
      raise ValueError(f'need 2 <= min_period <= max_period, got {self.min_period}, {self.max_period}')
    if 2 * self.max_period + 1 > self.seq_len:
      raise ValueError(f'two copies of {self.max_period} tokens and BOS do not fit in {self.seq_len}')

  @property
  def bos(self) -> int:
    return self.n_tokens

  @property
  def vocab_size(self) -> int:
    return self.n_tokens + 1

  def vocab(self) -> list[str]:
    return [str(t) for t in range(self.n_tokens)] + ['BOS']

  def sample(self, n: int, generator: torch.Generator, exclude: set[bytes] | None = None) -> 'Repeats':
    """Draw n sequences, redrawing any whose repeated segment is in `exclude`."""
    kept: list[Repeats] = []
    while sum(len(k.period) for k in kept) < n:
      draw = self._draw(n, generator)
      if exclude:
        keep = torch.tensor([key(s) not in exclude for s in draw.segments()], dtype=torch.bool)
        draw = Repeats(draw.tokens[keep], draw.period[keep])
      kept.append(draw)
    return Repeats(torch.cat([k.tokens for k in kept])[:n], torch.cat([k.period for k in kept])[:n])

  def _draw(self, n: int, generator: torch.Generator) -> 'Repeats':
    tokens = torch.randint(0, self.n_tokens, (n, self.seq_len), generator=generator)
    tokens[:, 0] = self.bos
    period = torch.randint(self.min_period, self.max_period + 1, (n,), generator=generator)
    # Each position of the second copy reads its token from one period earlier; every other
    # position reads itself. One gather writes the whole repeat.
    position = torch.arange(self.seq_len)
    in_second_copy = position[None, :] >= self.seq_len - period[:, None]
    source = torch.where(in_second_copy, position[None, :] - period[:, None], position[None, :])
    return Repeats(tokens.gather(1, source), period)


@dataclass(frozen=True)
class Repeats:
  """A batch of repeated sequences, with each one's period."""

  tokens: Int[Tensor, 'batch seq']
  period: Int[Tensor, ' batch']

  def __len__(self) -> int:
    return len(self.period)

  @property
  def second_copy_start(self) -> Int[Tensor, ' batch']:
    """Position of r_1 in the second copy."""
    return self.tokens.shape[1] - self.period

  def to(self, device: torch.device | str) -> 'Repeats':
    return Repeats(self.tokens.to(device), self.period.to(device))

  def __getitem__(self, index: slice) -> 'Repeats':
    return Repeats(self.tokens[index], self.period[index])

  def segments(self) -> list[Tensor]:
    """Each sequence's repeated segment r_1 ... r_L, taken from the second copy."""
    starts = self.second_copy_start.tolist()
    return [row[start:] for row, start in zip(self.tokens.cpu(), starts, strict=True)]

  def copy_mask(self) -> Bool[Tensor, 'batch seq']:
    """
    Targets that copying can predict: r_2 ... r_L of the second copy.

    r_1 of the second copy is not one of them. It is predicted from r_L of the first copy,
    and nothing at that point says the segment is about to repeat.
    """
    position = torch.arange(self.tokens.shape[1], device=self.tokens.device)
    return position[None, :] > self.second_copy_start[:, None]

  def fresh_mask(self) -> Bool[Tensor, 'batch seq']:
    """Targets nothing can predict: the filler, the first copy, and r_1 of the second."""
    position = torch.arange(self.tokens.shape[1], device=self.tokens.device)
    return (position[None, :] >= 1) & (position[None, :] <= self.second_copy_start[:, None])


@dataclass(frozen=True)
class Splits:
  """Held-out sequences, fixed by their own seeds so every run is measured on the same data."""

  val_seed: int = 1_000_003
  val_size: int = 512
  test_seed: int = 2_000_003
  test_size: int = 2048


def key(segment: Tensor) -> bytes:
  return segment.cpu().numpy().tobytes()


def held_out(task: RepeatTask, splits: Splits) -> tuple[Repeats, Repeats]:
  """The validation and test sequences. Test excludes validation segments, so they never overlap."""
  val = task.sample(splits.val_size, torch.Generator().manual_seed(splits.val_seed))
  test = task.sample(
    splits.test_size, torch.Generator().manual_seed(splits.test_seed), exclude={key(s) for s in val.segments()}
  )
  return val, test


def held_out_keys(val: Repeats, test: Repeats) -> set[bytes]:
  return {key(s) for s in val.segments()} | {key(s) for s in test.segments()}


def describe(task: RepeatTask, splits: Splits, train_seed: int) -> dict[str, Any]:
  """The data definition recorded with a run: enough to regenerate every split exactly."""
  return {'task': 'repeats', 'version': TASK_VERSION, **asdict(task), **asdict(splits), 'train_seed': train_seed}


def from_description(data: dict[str, Any]) -> tuple[RepeatTask, Splits]:
  """Rebuild the task and splits a run recorded with `describe`."""
  if data.get('task') != 'repeats' or data.get('version') != TASK_VERSION:
    raise ValueError(f'expected repeats task version {TASK_VERSION}, got {data.get("task")} version {data.get("version")}')
  task = RepeatTask(**{k: data[k] for k in RepeatTask.__dataclass_fields__})
  splits = Splits(**{k: data[k] for k in Splits.__dataclass_fields__})
  return task, splits


def losses(token_loss: Float[Tensor, 'batch target'], correct: Bool[Tensor, 'batch target'], batch: Repeats) -> dict[str, float]:
  """
  Loss on the copyable targets and on the fresh ones, and copying accuracy.

  Takes each target's loss and correctness, as evaluation.token_losses gives them.
  `copy_loss` is the measure of in-context copying. `fresh_loss` is the control: no model
  can do better than ln(n_tokens) there, so a change in it means a change in general
  behaviour rather than in copying.
  """
  copy_mask, fresh_mask = batch.copy_mask()[:, 1:], batch.fresh_mask()[:, 1:]
  if not copy_mask.any() or not fresh_mask.any():
    raise ValueError('a batch needs both copyable and fresh targets')
  return {
    'copy_loss': token_loss[copy_mask].mean().item(),
    'fresh_loss': token_loss[fresh_mask].mean().item(),
    'copy_accuracy': correct[copy_mask].float().mean().item(),
  }


def phase_change_step(history: list[dict[str, Any]], chance_loss: float) -> int | None:
  """
  The first evaluated step where copy loss falls below half of chance, ln(n_tokens).

  Half of chance is where a model has crossed from guessing to mostly copying, wherever
  its plateau sits. None if it never gets there.
  """
  for h in history:
    if h['val_copy_loss'] < chance_loss / 2:
      return h['step']
  return None
