"""The repeated-token task: layout, masks, determinism and split separation."""

import pytest
import torch

from deja_vu_tfm.evaluation import token_losses
from deja_vu_tfm.repeats import (
  Repeats,
  RepeatTask,
  Splits,
  describe,
  from_description,
  held_out,
  key,
  losses,
  phase_change_step,
)

TASK = RepeatTask(n_tokens=20, seq_len=12, min_period=2, max_period=5)


def test_second_copy_repeats_the_segment_before_it() -> None:
  batch = TASK.sample(200, torch.Generator().manual_seed(0))
  for row, period in zip(batch.tokens.tolist(), batch.period.tolist(), strict=True):
    assert row[0] == TASK.bos
    assert TASK.bos not in row[1:]
    assert row[-period:] == row[-2 * period:-period]


def test_periods_cover_the_allowed_range() -> None:
  batch = TASK.sample(500, torch.Generator().manual_seed(0))
  assert set(batch.period.tolist()) == set(range(TASK.min_period, TASK.max_period + 1))


def test_masks_on_a_hand_built_sequence() -> None:
  # BOS, filler 7 9, then the segment 3 4 5 twice: period 3, second copy starts at 6.
  batch = Repeats(torch.tensor([[20, 7, 9, 3, 4, 5, 3, 4, 5]]), torch.tensor([3]))
  assert batch.second_copy_start.tolist() == [6]
  # Targets 4 and 5 of the second copy are copyable; its first 3 is not.
  assert batch.copy_mask().tolist() == [[False] * 7 + [True] * 2]
  assert batch.fresh_mask().tolist() == [[False] + [True] * 6 + [False] * 2]
  assert batch.segments()[0].tolist() == [3, 4, 5]


def test_copy_and_fresh_masks_split_every_target() -> None:
  batch = TASK.sample(100, torch.Generator().manual_seed(1))
  copy, fresh = batch.copy_mask(), batch.fresh_mask()
  assert not (copy & fresh).any()
  assert (copy | fresh)[:, 1:].all()
  assert copy.sum(dim=1).tolist() == (batch.period - 1).tolist()


def test_losses_reward_copying_and_only_copying() -> None:
  batch = Repeats(torch.tensor([[20, 7, 9, 3, 4, 5, 3, 4, 5]]), torch.tensor([3]))
  # A model that predicts every next token perfectly: zero loss and full accuracy on both
  # masks. One that is right everywhere except the copy targets gets copy accuracy 0.
  perfect = torch.nn.functional.one_hot(torch.cat([batch.tokens[:, 1:], batch.tokens[:, :1]], dim=1), 21).float() * 50
  result = losses(*token_losses(perfect, batch.tokens), batch)
  assert result['copy_loss'] < 1e-6 and result['fresh_loss'] < 1e-6 and result['copy_accuracy'] == 1.0
  wrong = perfect.clone()
  wrong[0, 6:8] = torch.roll(wrong[0, 6:8], 1, dims=-1)
  result = losses(*token_losses(wrong, batch.tokens), batch)
  assert result['copy_accuracy'] == 0.0
  assert result['copy_loss'] > 40
  assert result['fresh_loss'] < 1e-6


def test_sampling_is_reproducible() -> None:
  a = TASK.sample(50, torch.Generator().manual_seed(0))
  b = TASK.sample(50, torch.Generator().manual_seed(0))
  assert torch.equal(a.tokens, b.tokens) and torch.equal(a.period, b.period)


def test_excluded_segments_are_never_sampled() -> None:
  # Two tokens and period 2 leave four possible segments, so collisions are certain.
  task = RepeatTask(n_tokens=2, seq_len=6, min_period=2, max_period=2)
  excluded = {key(s) for s in task.sample(3, torch.Generator().manual_seed(0)).segments()}
  drawn = task.sample(100, torch.Generator().manual_seed(1), exclude=excluded)
  assert len(drawn) == 100
  assert not any(key(s) in excluded for s in drawn.segments())


def test_validation_and_test_segments_never_overlap() -> None:
  val, test = held_out(TASK, Splits(val_size=200, test_size=200))
  assert not {key(s) for s in val.segments()} & {key(s) for s in test.segments()}


def test_description_round_trips() -> None:
  splits = Splits(val_size=10, test_size=20)
  assert from_description(describe(TASK, splits, train_seed=0)) == (TASK, splits)


def test_old_task_versions_are_rejected() -> None:
  with pytest.raises(ValueError, match='version'):
    from_description({**describe(TASK, Splits(), 0), 'version': 0})


def test_impossible_layouts_are_rejected() -> None:
  with pytest.raises(ValueError, match='fit'):
    RepeatTask(seq_len=10, min_period=2, max_period=5)
  with pytest.raises(ValueError, match='min_period'):
    RepeatTask(min_period=1)


def test_phase_change_is_the_first_step_below_half_of_chance() -> None:
  history = [{'step': s, 'val_copy_loss': loss} for s, loss in [(0, 6.2), (20, 6.0), (40, 3.2), (60, 2.9), (80, 0.1)]]
  assert phase_change_step(history, chance_loss=6.2) == 60
  assert phase_change_step(history[:3], chance_loss=6.2) is None
