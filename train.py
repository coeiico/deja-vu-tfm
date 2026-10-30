"""
Train a 2-layer attention-only tiny-transformer and watch in-context copying appear.

`--data repeats` (the default) trains on the synthetic repeated-token task. `--data stories`
trains on TinyStories text, and measures the same heads on repeated sequences of common
text tokens, as Olsson et al. measure models trained on text.

`just quick` runs this with --quick: a tiny CPU smoke test that must finish in under 5
minutes. `--model.n-layers 1` trains the one-layer baseline, which cannot form an
induction head.

Every `eval_every` steps the model is scored on fixed validation data: loss on the
copyable second copy of repeated sequences, loss on their fresh tokens, the
previous-token, duplicate and induction score of every head, and for text the validation
loss and in-context learning score. That history is what shows the phase change.
"""

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use('Agg')  # write PNGs without needing a display
import matplotlib.pyplot as plt
import tiktoken
import torch
import tyro
from tiny_transformer.config import TransformerConfig
from tiny_transformer.hub import count_parameters, save_model
from tiny_transformer.model import Transformer
from tiny_transformer.objective import next_token_loss
from torch import Tensor

from deja_vu_tfm import repeats, stories
from deja_vu_tfm.evaluation import run
from deja_vu_tfm.heads import chance, scores_from_cache
from deja_vu_tfm.initialisation import PositionInit, WeightInit, initialise
from deja_vu_tfm.repeats import Repeats, RepeatTask, Splits, phase_change_step
from deja_vu_tfm.stories import StoryData
from deja_vu_tfm.utils import (
  code_state,
  environment,
  get_device,
  make_run_dir,
  save_json,
  seed_everything,
)


@dataclass(frozen=True)
class ModelArgs:
  """Architecture flags. The vocabulary and sequence length come from the data."""

  n_layers: int = 2
  d_model: int = 64
  n_heads: int = 4
  mlp: Literal['gelu', 'relu', 'none'] = 'none'
  """'none' gives the attention-only model the induction-head papers study."""
  norm: Literal['layernorm', 'rmsnorm', 'none'] = 'none'
  """No normalisation by default, as in the small induction-head replications."""
  dropout: float = 0.0
  init: WeightInit = 'torch'
  """Initial weights: GPT-2's (std 0.02) or PyTorch's defaults (embeddings N(0, 1))."""
  positions: PositionInit = 'learned'
  """Initial position embeddings: random like the rest, or sinusoids. Both are learned."""


@dataclass(frozen=True)
class OptimArgs:
  steps: int = 8000
  batch_size: int = 64
  lr: float = 1e-3
  """Peak learning rate for AdamW."""
  warmup_steps: int = 0
  """Linear warm-up steps. None by default."""
  min_lr_ratio: float = 1.0
  """After warm-up, a cosine down to this fraction of the peak. 1 keeps the rate constant,
  so the step a phase change happens at is not set by the schedule."""
  weight_decay: float = 0.01
  """Applied to weight matrices and embeddings only, not to biases or norm gains."""
  grad_clip: float = 1.0
  """Clip the global gradient norm to this, so one bad batch cannot blow up the weights."""


@dataclass
class Config:
  run_name: str = 'default'
  """Outputs go to runs/<run-name>/, which must not exist yet."""
  seed: int = 0
  """Seeds Python, NumPy and PyTorch, and the stream of training sequences."""
  quick: bool = False
  """Run the tiny CPU smoke test instead of the full experiment."""
  device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
  data: Literal['repeats', 'stories'] = 'repeats'
  """Train on the synthetic repeated-token task or on TinyStories text."""
  eval_every: int = 50
  """Steps between validation checks. Small, so the phase change is not stepped over."""
  task: RepeatTask = field(default_factory=RepeatTask)
  """The repeated sequences: the training task, or with --data stories, the probe."""
  splits: Splits = field(default_factory=Splits)
  """Held-out repeated sequences."""
  stories: StoryData = field(default_factory=StoryData)
  """Text windows, used with --data stories."""
  model: ModelArgs = field(default_factory=ModelArgs)
  optim: OptimArgs = field(default_factory=OptimArgs)


def quick_config(cfg: Config) -> Config:
  """Shrink everything so the whole pipeline runs on a CPU in well under 5 minutes."""
  return replace(
    cfg,
    device='cpu',
    data='repeats',
    eval_every=100,
    splits=Splits(val_size=128, test_size=128),
    optim=replace(cfg.optim, steps=6000),
  )


def make_optimizer(model: Transformer, args: OptimArgs) -> torch.optim.AdamW:
  # Weight decay pulls weights towards zero. That suits matrices and embeddings, but biases
  # and norm gains are offsets and scales the decay would only distort (as in nanoGPT).
  decay = [p for p in model.parameters() if p.dim() >= 2]
  no_decay = [p for p in model.parameters() if p.dim() < 2]
  groups = [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
  return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


def learning_rate(step: int, args: OptimArgs) -> float:
  """Linear warm-up, then cosine decay to min_lr_ratio × the peak (as in nanoGPT)."""
  if step <= args.warmup_steps:
    return args.lr * step / args.warmup_steps
  progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
  cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 → 0
  return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)


@dataclass
class Data:
  """What training and validation need, whichever corpus the run uses."""

  vocab_size: int
  seq_len: int
  vocab: list[str] | None
  train_batch: Callable[[int, torch.Generator], Tensor]
  val_probe: Repeats
  val_text: Tensor | None
  record: dict[str, Any]


def repeat_data(cfg: Config) -> Data:
  # Validation and test sequences are fixed; training sequences are drawn fresh every step
  # from the run's seeded stream, skipping any whose repeated segment is held out.
  task = cfg.task
  val, test = repeats.held_out(task, cfg.splits)
  excluded = repeats.held_out_keys(val, test)
  return Data(
    vocab_size=task.vocab_size,
    seq_len=task.seq_len,
    vocab=task.vocab(),
    train_batch=lambda n, generator: task.sample(n, generator, excluded).tokens,
    val_probe=val,
    val_text=None,
    record={
      'data': repeats.describe(task, cfg.splits, cfg.seed),
      'evaluation': 'synthetic: validation sequences fixed by splits.val_seed; copy_loss on r_2..r_L of the second copy',
    },
  )


def story_data(cfg: Config) -> Data:
  text = cfg.stories
  raw = stories.load_streams()
  table = stories.text_vocab(raw['train'], text.vocab_size)
  streams = {name: stories.encode(stream, table) for name, stream in raw.items()}
  unk = {name: float((s == len(table)).mean()) for name, s in streams.items()}
  print(f'  vocabulary of {text.vocab_size}: UNK is {unk["train"]:.2%} of training tokens, {unk["val"]:.2%} of validation')
  if cfg.task.seq_len > text.seq_len or cfg.task.n_tokens >= len(table):
    raise ValueError('the probe must fit in the text window and draw from tokens in the vocabulary')
  val_probe, _ = stories.held_out_probes(cfg.task, cfg.splits)
  encoding = tiktoken.get_encoding(stories.TOKENIZER)
  return Data(
    vocab_size=text.vocab_size,
    seq_len=text.seq_len,
    vocab=[encoding.decode([t]) for t in table.tolist()] + ['<unk>'],
    train_batch=lambda n, generator: stories.random_windows(streams['train'], text.seq_len, n, generator),
    val_probe=val_probe,
    val_text=stories.fixed_windows(streams['val'], text.seq_len, text.val_windows),
    record={
      'data': {**stories.describe(text, cfg.seed, raw, table), 'unk_fraction': unk},
      'probe': repeats.describe(cfg.task, cfg.splits, cfg.seed),
      'evaluation': (
        'text: the first val_windows windows of the validation stream (first half of the '
        'dataset validation stories); probe: repeated sequences of the n_tokens most common '
        'training tokens, fixed by splits.val_seed'
      ),
    },
  )


def evaluate(model: Transformer, data: Data, cfg: Config) -> dict[str, Any]:
  loss, correct, patterns = run(model, data.val_probe.tokens)
  scores = scores_from_cache(patterns, data.val_probe.to(loss.device), model.cfg.n_layers)
  metrics: dict[str, Any] = {f'val_{k}': v for k, v in repeats.losses(loss, correct, data.val_probe.to(loss.device)).items()}
  if data.val_text is not None:
    text_loss, _, _ = run(model, data.val_text, chunk=32, patterns=False)
    metrics.update({f'val_text_{k}': v for k, v in stories.losses(text_loss, cfg.stories).items()})
  return {**metrics, **{k: v.tolist() for k, v in scores.items()}}


def plot_curves(history: list[dict[str, Any]], chance_loss: float, phase_step: int | None, path: Path) -> None:
  steps = [h['step'] for h in history]
  text = 'val_text_loss' in history[0]
  fig, axes = plt.subplots(1, 3 if text else 2, figsize=(18 if text else 12, 4))
  ax_loss, ax_score = axes[0], axes[1]
  ax_loss.plot(steps, [h['val_copy_loss'] for h in history], label='copyable tokens (second copy)')
  ax_loss.plot(steps, [h['val_fresh_loss'] for h in history], label='fresh tokens')
  ax_loss.axhline(chance_loss, color='grey', linestyle=':', label='ln(n_tokens)')
  ax_loss.set(xlabel='step', ylabel='repeated sequences: loss (nats)')
  ax_loss.legend()
  n_layers = len(history[0]['induction'])
  for layer in range(1, n_layers):
    for head, _ in enumerate(history[0]['induction'][layer]):
      ax_score.plot(steps, [h['induction'][layer][head] for h in history], label=f'induction {layer}.{head}')
  for head, _ in enumerate(history[0]['previous'][0]):
    ax_score.plot(steps, [h['previous'][0][head] for h in history], linestyle='--', label=f'previous-token 0.{head}')
  ax_score.set(xlabel='step', ylabel='validation score', ylim=(0, 1.02))
  ax_score.legend(fontsize=7, ncol=2)
  if text:
    ax_text = axes[2]
    ax_text.plot(steps, [h['val_text_loss'] for h in history], label='validation loss')
    ax_text.plot(steps, [h['val_text_icl_score'] for h in history], label='ICL score (late − early loss)')
    ax_text.set(xlabel='step', ylabel='TinyStories (nats)')
    ax_text.legend()
  final = history[-1]
  if phase_step is None:
    title = f'No phase change: copy loss is still {final["val_copy_loss"]:.2f} nats at step {final["step"]}'
  else:
    title = (
      f'Copy loss falls below half of ln(n_tokens) at step {phase_step}, '
      f'reaching {final["val_copy_loss"]:.3f} nats (copy accuracy {final["val_copy_accuracy"]:.1%})'
    )
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(cfg: Config) -> None:
  if cfg.quick:
    cfg = quick_config(cfg)
  seed_everything(cfg.seed)
  device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
  # TF32 matmuls on recent NVIDIA GPUs: several times faster than full float32, with a
  # 10-bit mantissa in the products that a model this small does not notice.
  torch.backends.cuda.matmul.allow_tf32 = True
  run_dir = make_run_dir(cfg.run_name)
  optim = cfg.optim
  data = repeat_data(cfg) if cfg.data == 'repeats' else story_data(cfg)

  architecture = {k: v for k, v in asdict(cfg.model).items() if k not in ('init', 'positions')}
  model_cfg = TransformerConfig(
    vocab_size=data.vocab_size, max_seq_len=data.seq_len, d_mlp=4 * cfg.model.d_model, **architecture
  )
  model = Transformer(model_cfg).to(device)
  initialise(model, cfg.model.init, cfg.model.positions)
  optimizer = make_optimizer(model, optim)
  print(f'deja-vu-tfm: {count_parameters(model):,} parameters on {device}, writing to {run_dir}')

  train_stream = torch.Generator().manual_seed(cfg.seed)
  probe = data.val_probe
  for row, period in zip(probe.tokens[:2].tolist(), probe.period[:2].tolist(), strict=True):
    shown = ' '.join(data.vocab[t] for t in row) if data.vocab else ' '.join(map(str, row))
    print(f'  probe example, period {period}:', shown)
  chance_loss = math.log(cfg.task.n_tokens)

  record: dict[str, Any] = {
    'config': asdict(cfg),
    'model_config': model_cfg.to_dict(),
    'parameters': count_parameters(model),
    **data.record,
    'chance': {'loss': chance_loss, **chance(probe)},
    'code': code_state(run_dir),
    'environment': {**environment(device), 'tf32_matmul': device.type == 'cuda'},
    'resumable': False,
    'status': 'running',
  }
  save_json(run_dir / 'metrics.json', record)

  history: list[dict[str, Any]] = []
  start = time.time()
  for step in range(0, optim.steps + 1):
    # Step 0 only evaluates the untrained model, so every curve starts from initialisation.
    if step > 0:
      lr = learning_rate(step, optim)
      for group in optimizer.param_groups:
        group['lr'] = lr
      tokens = data.train_batch(optim.batch_size, train_stream).to(device)

      logits = model(tokens)
      loss = next_token_loss(logits, tokens)
      if not torch.isfinite(loss):
        record.update(status=f'failed: non-finite loss {loss.item()} at step {step}', history=history)
        save_json(run_dir / 'metrics.json', record)
        raise SystemExit(f'Stopping: loss is {loss.item()} at step {step}.')
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip)
      optimizer.step()

    if step % cfg.eval_every == 0 or step == optim.steps:
      metrics = evaluate(model, data, cfg)
      history.append({'step': step, **metrics})
      best_induction = max((max(row) for row in metrics['induction'][1:]), default=float('nan'))
      text = f'  text {metrics["val_text_loss"]:.3f}  icl {metrics["val_text_icl_score"]:+.3f}' if data.val_text is not None else ''
      print(
        f'step {step:5d}  copy {metrics["val_copy_loss"]:.3f}  fresh {metrics["val_fresh_loss"]:.3f}  '
        f'copy acc {metrics["val_copy_accuracy"]:.3f}  best later-layer induction {best_induction:.2f}{text}'
      )

  save_model(run_dir, model, data.vocab)
  phase_step = phase_change_step(history, chance_loss)
  plot_curves(history, chance_loss, phase_step, run_dir / 'training.png')
  record.update(
    status='finished',
    seconds=round(time.time() - start, 1),
    history=history,
    phase_change_step=phase_step,
    final={k: v for k, v in history[-1].items() if k.startswith('val_')},
  )
  save_json(run_dir / 'metrics.json', record)
  print(f'phase change at step {phase_step}; done in {record["seconds"]}s: {run_dir}')


if __name__ == '__main__':
  main(tyro.cli(Config))
