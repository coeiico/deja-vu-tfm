"""The text corpus helpers, on hand-made stories: no download needed."""

import numpy as np
import pytest
import torch

from deja_vu_tfm.evaluation import token_losses
from deja_vu_tfm.repeats import RepeatTask
from deja_vu_tfm.stories import (
  EOT,
  StoryData,
  encode,
  fixed_windows,
  join_stories,
  losses,
  random_windows,
  repeat_probe,
  split_validation,
  text_vocab,
)


def test_stories_are_joined_after_end_of_text() -> None:
  stream = join_stories([[5, 6], [7], [8, 9, 10]])
  assert stream.tolist() == [EOT, 5, 6, EOT, 7, EOT, 8, 9, 10]
  assert stream.dtype == np.uint16


def test_validation_is_split_by_story_without_overlap() -> None:
  stories = [[i, i + 100] for i in range(9)]
  val, test = split_validation(stories)
  assert val + test == stories
  assert len(val) == 4 and len(test) == 5


def test_fixed_windows_tile_the_start_of_the_stream() -> None:
  stream = np.arange(20, dtype=np.uint16)
  assert fixed_windows(stream, 4, 3).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
  with pytest.raises(ValueError, match='need'):
    fixed_windows(stream, 4, 6)


def test_random_windows_are_contiguous_and_reproducible() -> None:
  stream = np.arange(50, dtype=np.uint16)
  a = random_windows(stream, 5, 30, torch.Generator().manual_seed(0))
  b = random_windows(stream, 5, 30, torch.Generator().manual_seed(0))
  assert torch.equal(a, b)
  assert (a[:, 1:] - a[:, :-1] == 1).all()
  assert a.max() < 50


def test_target_losses_align_each_position_with_the_next_token() -> None:
  tokens = torch.tensor([[1, 2, 3, 0]])
  # Certain of the right next token at positions 0 and 2, uniform at position 1.
  logits = torch.zeros(1, 4, 4)
  logits[0, 0, 2] = 100.0
  logits[0, 2, 0] = 100.0
  loss, correct = token_losses(logits, tokens)
  assert loss.shape == (1, 3)
  assert loss[0, 0].item() == pytest.approx(0.0, abs=1e-6)
  assert loss[0, 1].item() == pytest.approx(np.log(4))
  assert loss[0, 2].item() == pytest.approx(0.0, abs=1e-6)
  assert correct.tolist() == [[True, False, True]]


def test_icl_score_is_late_minus_early_loss() -> None:
  data = StoryData(seq_len=9, early=(0, 2), late=(6, 8))
  tokens = torch.zeros(2, 9, dtype=torch.long)
  # Uniform over 4 tokens early, certain of token 0 late.
  logits = torch.zeros(2, 9, 4)
  logits[:, 6:8, 0] = 100.0
  result = losses(token_losses(logits, tokens)[0], data)
  assert result['early_loss'] == pytest.approx(np.log(4))
  assert result['late_loss'] == pytest.approx(0.0, abs=1e-6)
  assert result['icl_score'] == pytest.approx(-np.log(4))


def test_repeat_probe_uses_the_most_common_tokens_and_keeps_the_repeat() -> None:
  task = RepeatTask(n_tokens=3, seq_len=10, min_period=2, max_period=4)
  batch = repeat_probe(task, 50, torch.Generator().manual_seed(0))
  assert (batch.tokens[:, 0] == 0).all()
  assert set(batch.tokens[:, 1:].unique().tolist()) == {1, 2, 3}
  for row, period in zip(batch.tokens.tolist(), batch.period.tolist(), strict=True):
    assert row[-period:] == row[-2 * period:-period]


def test_text_vocab_puts_end_of_text_first_then_tokens_by_frequency() -> None:
  stream = np.array([EOT] * 10 + [7, 7, 7, 3, 3, 9, 4], dtype=np.uint16)
  table = text_vocab(stream, 4)
  # Ties (9 and 4 appear once each) go to the lower id.
  assert table.tolist() == [EOT, 7, 3]
  assert encode(stream, table).tolist() == [0] * 10 + [1, 1, 1, 2, 2, 3, 3]
