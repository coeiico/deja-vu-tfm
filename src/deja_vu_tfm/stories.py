"""
The text corpus: TinyStories, tokenized with GPT-2's tokenizer.

TinyStories (Eldan and Li, 2023) is short, simple children's stories, written by GPT-3.5
and GPT-4. Names and objects recur within a story ("Lily ... Lily"), so copying from
earlier in the context pays off, as it does in ordinary text.

- **Tokenizer.** GPT-2's BPE through tiktoken, an externally pretrained tokenizer, fixed
  and recorded by name. Its end-of-text token separates stories, and also serves as BOS
  for the repeated-token probes, so they look like the start of a new document.
- **Model vocabulary.** GPT-2 has 50,257 tokens, and an unembedding that wide would cost
  most of the compute of a two-layer model. The model sees end-of-text (id 0), the most
  common GPT-2 tokens of the training split in order of frequency (ids 1 onwards), and
  UNK for the rest. The table is fitted on training data only and saved with each run.
- **Split unit: the story.** Training uses the dataset's train split. Its validation split
  is cut by story into two halves: the first for validation, the second for test. Each
  split's stories are joined into one token stream, so a window never mixes splits.
- **Windows.** Training draws windows at random offsets of the training stream. Validation
  and test use fixed, non-overlapping windows from the start of their streams.

Token streams are cached under data/ (gitignored), keyed by dataset revision and tokenizer.
"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from deja_vu_tfm.repeats import Repeats, RepeatTask, Splits

DATASET = 'roneneldan/TinyStories'
REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
TOKENIZER = 'gpt2'
EOT = 50256
VOCAB_SIZE = 50257
DATA_VERSION = 2


@dataclass(frozen=True)
class StoryData:
  seq_len: int = 256
  """Tokens per window. Position t of a window has seen exactly t tokens of context."""
  vocab_size: int = 8192
  """Model vocabulary: end-of-text, the vocab_size - 2 most common tokens, and UNK."""
  val_windows: int = 512
  test_windows: int = 1024
  early: tuple[int, int] = (32, 64)
  """Positions averaged for the early loss of the in-context learning score."""
  late: tuple[int, int] = (192, 256)
  """Positions averaged for the late loss."""


def describe(data: StoryData, train_seed: int, streams: dict[str, np.ndarray], table: np.ndarray) -> dict[str, Any]:
  """The data definition recorded with a run: a hash of each GPT-2 token stream, and the vocabulary table."""
  return {
    'task': 'stories',
    'version': DATA_VERSION,
    'dataset': DATASET,
    'revision': REVISION,
    'tokenizer': TOKENIZER,
    **asdict(data),
    'train_seed': train_seed,
    'stream_sha256': {name: hashlib.sha256(s.tobytes()).hexdigest() for name, s in streams.items()},
    'stream_tokens': {name: len(s) for name, s in streams.items()},
    'vocab_table': table.tolist(),
  }


def from_description(record: dict[str, Any]) -> StoryData:
  if record.get('task') != 'stories' or record.get('version') != DATA_VERSION:
    raise ValueError(f'expected stories data version {DATA_VERSION}, got {record.get("task")} version {record.get("version")}')
  return StoryData(
    seq_len=record['seq_len'], vocab_size=record['vocab_size'], val_windows=record['val_windows'], test_windows=record['test_windows'],
    early=tuple(record['early']), late=tuple(record['late']),
  )


def join_stories(stories: list[list[int]]) -> np.ndarray:
  """One stream: each story preceded by end-of-text, as GPT-2 separates documents."""
  pieces = [np.array([EOT] + story, dtype=np.uint16) for story in stories]
  return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)


def split_validation(stories: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
  """Cut the dataset's validation stories into validation and test halves, by story."""
  half = len(stories) // 2
  return stories[:half], stories[half:]


def fixed_windows(stream: np.ndarray, seq_len: int, n: int) -> Int[Tensor, 'n seq']:
  """The first n non-overlapping windows of a stream."""
  if n * seq_len > len(stream):
    raise ValueError(f'{n} windows of {seq_len} need {n * seq_len} tokens; the stream has {len(stream)}')
  return torch.from_numpy(stream[:n * seq_len].astype(np.int64)).view(n, seq_len)


def random_windows(stream: np.ndarray, seq_len: int, n: int, generator: torch.Generator) -> Int[Tensor, 'n seq']:
  starts = torch.randint(0, len(stream) - seq_len + 1, (n,), generator=generator)
  return torch.stack([torch.from_numpy(stream[s:s + seq_len].astype(np.int64)) for s in starts.tolist()])


def losses(token_loss: Float[Tensor, 'batch target'], data: StoryData) -> dict[str, float]:
  """
  Loss overall, early and late in the window, and the in-context learning score.

  Takes each target's loss, as evaluation.token_losses gives it: column t is the loss of
  predicting token t + 1, having seen t + 1 tokens. The score is late loss minus early loss
  (Olsson et al. 2022 use token 500 minus token 50 of 1024). It is negative when tokens
  later in the context are easier to predict, which is what using the context, induction
  heads above all, buys.
  """
  by_position = token_loss.mean(dim=0)
  early = by_position[data.early[0]:data.early[1]].mean().item()
  late = by_position[data.late[0]:data.late[1]].mean().item()
  return {'loss': by_position.mean().item(), 'early_loss': early, 'late_loss': late, 'icl_score': late - early}


def text_vocab(train: np.ndarray, size: int) -> np.ndarray:
  """
  The GPT-2 ids the model keeps: end-of-text first, then the most common training tokens.

  Model id i stands for GPT-2 id table[i]; id len(table) is UNK. Ties in frequency go to
  the lower GPT-2 id, so the table is a function of the training stream alone.
  """
  counts = np.bincount(train, minlength=VOCAB_SIZE)
  counts[EOT] = -1
  by_frequency = np.lexsort((np.arange(VOCAB_SIZE), -counts))[:size - 2]
  return np.concatenate([[EOT], by_frequency]).astype(np.int64)


def encode(stream: np.ndarray, table: np.ndarray) -> np.ndarray:
  """GPT-2 ids to model ids; anything outside the table becomes UNK."""
  lookup = np.full(VOCAB_SIZE, len(table), dtype=np.uint16)
  lookup[table] = np.arange(len(table), dtype=np.uint16)
  return lookup[stream]


def repeat_probe(task: RepeatTask, n: int, generator: torch.Generator) -> Repeats:
  """
  Repeated random sequences in the text model's vocabulary, for scoring its heads.

  The task's tokens 0..n_tokens-1 become model ids 1..n_tokens, the n_tokens most common
  training tokens, so every one has a well-trained embedding. BOS becomes end-of-text.
  """
  batch = task.sample(n, generator)
  tokens = torch.where(batch.tokens == task.bos, 0, batch.tokens + 1)
  return Repeats(tokens, batch.period)


def load_streams(cache_dir: Path = Path('data/tinystories')) -> dict[str, np.ndarray]:
  """The train, validation and test token streams, tokenizing and caching on first use."""
  folder = cache_dir / f'{REVISION[:12]}-{TOKENIZER}'
  paths = {name: folder / f'{name}.npy' for name in ('train', 'val', 'test')}
  if all(p.exists() for p in paths.values()):
    return {name: np.load(p) for name, p in paths.items()}

  import tiktoken
  from datasets import load_dataset

  print(f'Tokenizing {DATASET} at {REVISION[:12]} with {TOKENIZER}; this happens once.')
  encoding = tiktoken.get_encoding(TOKENIZER)
  dataset = load_dataset(DATASET, revision=REVISION)
  texts = {split: [t.strip() for t in dataset[split]['text']] for split in ('train', 'validation')}
  val_stories, test_stories = split_validation(encoding.encode_ordinary_batch(texts['validation'], num_threads=16))
  streams = {
    'train': join_stories(encoding.encode_ordinary_batch(texts['train'], num_threads=16)),
    'val': join_stories(val_stories),
    'test': join_stories(test_stories),
  }
  folder.mkdir(parents=True, exist_ok=True)
  for name, stream in streams.items():
    np.save(paths[name], stream)
  return streams


def held_out_probes(task: RepeatTask, splits: Splits) -> tuple[Repeats, Repeats]:
  """Validation and test repeated-token probes for a text model, fixed by the split seeds."""
  val = repeat_probe(task, splits.val_size, torch.Generator().manual_seed(splits.val_seed))
  test = repeat_probe(task, splits.test_size, torch.Generator().manual_seed(splits.test_seed))
  return val, test
