"""
Load a trained run and watch it copy.

For a run trained on repeated tokens, this draws one repeated sequence and shows, for each
token of the second copy, what the model predicted:
  just predict default-seed0 --seed 3

For a run trained on TinyStories, it continues a prompt greedily. A prompt that repeats a
made-up phrase shows in-context copying on text:
  just predict stories-seed0 --text 'Tom saw a blue zorp. He said, "I love my blue'
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
import tyro
from tiny_transformer.generate import generate
from tiny_transformer.hub import load_model
from tiny_transformer.modes import evaluating

from deja_vu_tfm import repeats, stories


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default-seed0."""
  seed: int = 0
  """Seeds the repeated sequence (repeated-token runs)."""
  text: str = 'Once upon a time, there was a little girl named Lily.'
  """The prompt to continue (TinyStories runs)."""
  new_tokens: int = 40
  """Tokens to generate after the prompt (TinyStories runs)."""


def main(cfg: Config) -> None:
  record = json.loads((cfg.run / 'metrics.json').read_text())
  model, vocab = load_model(cfg.run)

  if record['data']['task'] == 'repeats':
    task, _ = repeats.from_description(record['data'])
    batch = task.sample(1, torch.Generator().manual_seed(cfg.seed))
    assert vocab is not None, 'repeated-token runs save their vocabulary'
    with evaluating(model):
      predicted = model(batch.tokens)[0].argmax(dim=-1)
    tokens, start, period = batch.tokens[0].tolist(), int(batch.second_copy_start.item()), int(batch.period.item())
    print(f'period {period}; the second copy starts at position {start}')
    print('first copy: ', ' '.join(vocab[t] for t in tokens[start - period:start]))
    # Position t's prediction is for token t + 1; r_1 of the second copy cannot be predicted.
    marks = [
      f'{vocab[tokens[t]]}{"" if predicted[t - 1] == tokens[t] else f"(model: {vocab[predicted[t - 1]]})"}'
      for t in range(start + 1, len(tokens))
    ]
    print('second copy:', vocab[tokens[start]], ' '.join(marks))
    right = sum(predicted[t - 1].item() == tokens[t] for t in range(start + 1, len(tokens)))
    print(f'{right} of {len(tokens) - start - 1} copyable tokens predicted')
    return

  assert vocab is not None, 'text runs save their vocabulary'
  encoding = tiktoken.get_encoding(stories.TOKENIZER)
  table = np.array(record['data']['vocab_table'])
  gpt2_ids = np.array([stories.EOT] + encoding.encode_ordinary(cfg.text))
  prompt = torch.from_numpy(stories.encode(gpt2_ids, table).astype(np.int64))[None]
  new_tokens = min(cfg.new_tokens, model.cfg.max_seq_len - prompt.shape[1])
  out = generate(model, prompt, max_new_tokens=new_tokens)[0, prompt.shape[1]:]
  print(cfg.text + '|' + ''.join(vocab[t] for t in out.tolist()))


if __name__ == '__main__':
  main(tyro.cli(Config))
