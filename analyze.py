"""
Find the heads behind in-context copying in a saved run, and test that the model uses them.

Analysis reads saved weights, so it never needs to retrain. Results go to
runs/<run>/analysis/<name>/, a new folder each time, so the run's own record is never
changed. Everything is measured on the test split; head means for ablation come from
the validation split, so nothing is fitted on test.

- Baselines: the trained model against an untrained copy of the same architecture.
- Head scores: previous-token, duplicate-token and induction score for every head, with
  the score of uniform attention as chance.
- Mean ablation of every head, of the previous-token head, the induction head and both,
  of a low-scoring head in each layer (the unrelated-head control) and of each layer
  whole. The identity patch is the control that must change nothing. Each condition
  reports copying (loss and accuracy on the second copy), general loss (the fresh
  tokens, and for a text model the test text), and the induction head's score.
- Path ablation (two-layer models): the previous-token head, and an unrelated layer-0
  head as control, removed from layer 1's queries, keys or values alone.
- Composition scores from the weights, against the untrained model.
"""

import csv
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from tiny_transformer.hub import load_model
from tiny_transformer.model import Patch, Transformer
from tiny_transformer.modes import evaluating
from torch import Tensor

from deja_vu_tfm import repeats, stories
from deja_vu_tfm.circuits import (
  QKV,
  composition_scores,
  head_means,
  identity,
  mean_ablation,
  path_ablation,
)
from deja_vu_tfm.evaluation import run
from deja_vu_tfm.heads import INDUCTION_THRESHOLD, chance, induction_heads, scores_from_cache
from deja_vu_tfm.initialisation import initialise
from deja_vu_tfm.repeats import Repeats
from deja_vu_tfm.utils import (
  code_state,
  environment,
  get_device,
  make_output_dir,
  save_json,
  seed_everything,
)


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default-seed0."""
  name: str | None = None
  """Output folder under <run>/analysis/. Defaults to the current time."""
  device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
  chunk: int = 64
  """Sequences per forward pass. Changes memory use, never results."""


@dataclass
class Data:
  val_probe: Repeats
  test_probe: Repeats
  val_text: Tensor | None
  test_text: Tensor | None
  story_data: stories.StoryData | None


def load_data(record: dict[str, Any]) -> Data:
  if record['data']['task'] == 'repeats':
    task, splits = repeats.from_description(record['data'])
    val, test = repeats.held_out(task, splits)
    return Data(val, test, None, None, None)
  task, splits = repeats.from_description(record['probe'])
  val, test = stories.held_out_probes(task, splits)
  story_data = stories.from_description(record['data'])
  raw = stories.load_streams()
  for name, digest in record['data']['stream_sha256'].items():
    if hashlib.sha256(raw[name].tobytes()).hexdigest() != digest:
      raise ValueError(f'the cached {name} token stream differs from the one this run trained on')
  table = np.array(record['data']['vocab_table'])
  streams = {name: stories.encode(raw[name], table) for name in ('val', 'test')}
  # Means for ablating heads on text come from validation windows; a few hundred are plenty.
  val_text = stories.fixed_windows(streams['val'], story_data.seq_len, min(256, story_data.val_windows))
  test_text = stories.fixed_windows(streams['test'], story_data.seq_len, story_data.test_windows)
  return Data(val, test, val_text, test_text, story_data)


def text_means(model: Transformer, tokens: Tensor, chunk: int) -> dict[int, Tensor]:
  """Head means over text windows, for ablations measured on text."""
  sums: dict[int, Tensor] = {}
  with evaluating(model):
    device = next(model.parameters()).device
    for start in range(0, len(tokens), chunk):
      _, cache = model(tokens[start:start + chunk].to(device), return_cache=True)
      for layer, mean in head_means(cache, model.cfg.n_layers).items():
        sums[layer] = sums.get(layer, 0) + mean * len(tokens[start:start + chunk])
  return {layer: total / len(tokens) for layer, total in sums.items()}


def measure(
  model: Transformer, data: Data, cfg: Config, probe_patch: Any = None, text_patch: dict[str, Patch] | None = None
) -> dict[str, Any]:
  """Copying, general loss and head scores, with optional patches on the probe and the text."""
  probe = data.test_probe
  loss, correct, patterns = run(model, probe.tokens, chunk=cfg.chunk, patch=probe_patch)
  probe = probe.to(loss.device)
  result: dict[str, Any] = repeats.losses(loss, correct, probe)
  result['scores'] = {k: v.tolist() for k, v in scores_from_cache(patterns, probe, model.cfg.n_layers).items()}
  if data.test_text is not None and data.story_data is not None:
    text_loss, _, _ = run(model, data.test_text, chunk=cfg.chunk // 2, patch=(lambda _: text_patch) if text_patch else None, patterns=False)
    result.update({f'text_{k}': v for k, v in stories.losses(text_loss, data.story_data).items()})
  return result


def path_patch(model: Transformer, sender: int, means: Tensor, path: QKV) -> Callable[[Tensor], dict[str, Patch]]:
  """Per-chunk patches that remove one layer-0 head from layer 1's `path` input."""
  return lambda tokens: path_ablation(model, tokens, [sender], means, path)


def heatmap(values: Tensor, title: str, label: str, path: Path, cmap: str = 'viridis') -> None:
  fig, ax = plt.subplots(figsize=(1.6 + 0.7 * values.shape[1], 1.4 + 0.6 * values.shape[0]))
  image = ax.imshow(values.numpy(), cmap=cmap, aspect='auto')
  for i in range(values.shape[0]):
    for j in range(values.shape[1]):
      ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
  ax.set(xticks=range(values.shape[1]), yticks=range(values.shape[0]))
  fig.colorbar(image, ax=ax, label=label)
  ax.set_title(title, fontsize=9)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_patterns(patterns: dict[str, Tensor], heads: dict[str, tuple[int, int]], probe: Repeats, path: Path) -> None:
  fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 5.5))
  for ax, (role, (layer, head)) in zip(axes, heads.items(), strict=True):
    ax.imshow(patterns[f'blocks.{layer}.attn.pattern'][0, head].cpu().numpy(), cmap='Blues')
    ax.set(xlabel='key (attended to)', ylabel='query (attending)', title=f'{role} head {layer}.{head}')
  fig.suptitle(f'Attention on one test sequence, period {probe.period[0].item()}')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(cfg: Config) -> None:
  record = json.loads((cfg.run / 'metrics.json').read_text())
  device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
  out = make_output_dir(cfg.run / 'analysis' / (cfg.name or time.strftime('%Y%m%d-%H%M%S')))
  model, _ = load_model(cfg.run, device)
  data = load_data(record)
  n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

  # The untrained baseline uses the run's own seed, so it is the model training started from.
  seed_everything(record['config']['seed'])
  untrained = Transformer(model.cfg).to(device).eval()
  initialise(untrained, record['config']['model']['init'], record['config']['model']['positions'])
  trained = measure(model, data, cfg)
  results: dict[str, Any] = {
    'run': str(cfg.run),
    'code': code_state(out),
    'environment': environment(device),
    'test_probe_size': len(data.test_probe),
    'chance': {'loss': record['chance']['loss'], **chance(data.test_probe)},
    'trained': trained,
    'untrained': measure(untrained, data, cfg),
  }
  print(f'trained   copy loss {trained["copy_loss"]:.3f}, copy accuracy {trained["copy_accuracy"]:.3f}')
  print(f'untrained copy loss {results["untrained"]["copy_loss"]:.3f}')

  scores = {k: torch.tensor(v) for k, v in trained['scores'].items()}
  last = n_layers - 1
  prev = (0, int(scores['previous'][0].argmax()))
  dup = (0, int(scores['duplicate'][0].argmax()))
  ind = (last, int(scores['induction'][last].argmax()))
  inductions = [(last, h) for h in induction_heads(scores['induction'][last])]
  # Unrelated-head controls: in layer 0 the head scoring lowest as either a previous-token or
  # a duplicate-token head; in the last layer the lowest-scoring head that is not an
  # induction head, if there is one.
  layer0_job = torch.maximum(scores['previous'][0], scores['duplicate'][0])
  layer0_job[[prev[1], dup[1]]] = float('inf')
  controls = {'layer-0 control': (0, int(layer0_job.argmin()))}
  others = [h for h in range(n_heads) if (last, h) not in inductions]
  if last > 0 and others:
    controls[f'layer-{last} control'] = (last, min(others, key=lambda h: scores['induction'][last, h].item()))
  roles = {'previous-token': prev, 'duplicate-token': dup, 'induction': ind}
  results['heads'] = {
    **{name: list(head) for name, head in {**roles, **controls}.items()},
    'induction heads': [list(h) for h in inductions],
    'induction_threshold': INDUCTION_THRESHOLD,
  }
  print(f'heads: {results["heads"]}')

  def induction_score(c: dict[str, Any]) -> float:
    """The mean induction score of the induction heads under some intervention."""
    return sum(c['scores']['induction'][layer][head] for layer, head in inductions) / len(inductions)

  # Head means for ablation, from validation: repeated sequences for the probe, text windows
  # for the text loss.
  with evaluating(model):
    _, val_cache = model(data.val_probe.tokens[:256].to(device), return_cache=True)
  means = head_means(val_cache, n_layers)
  means_text = text_means(model, data.val_text, cfg.chunk // 2) if data.val_text is not None else None

  def ablate(heads: list[tuple[int, int]]) -> dict[str, Any]:
    patches = mean_ablation(heads, means)
    text = mean_ablation(heads, means_text) if means_text is not None else None
    return measure(model, data, cfg, probe_patch=lambda _: patches, text_patch=text)

  def name(h: tuple[int, int]) -> str:
    return f'{h[0]}.{h[1]}'

  # Conditions are keyed by role, so figures can find them in any seed; each also records
  # which heads it ablated, as a label.
  conditions: dict[str, Any] = {
    'none': {'label': 'no ablation', **trained},
    'identity': {'label': 'identity patch (control)', **measure(model, data, cfg, probe_patch=lambda _: identity(n_layers), text_patch=identity(n_layers))},
    'previous_head': {'label': f'previous-token head {name(prev)}', **ablate([prev])},
    'duplicate_head': {'label': f'duplicate-token head {name(dup)}', **ablate([dup])},
    'induction_heads': {'label': f'induction heads {", ".join(map(name, inductions))}', **ablate(inductions)},
    'best_induction_head': {'label': f'best induction head {name(ind)}', **ablate([ind])},
    'previous_and_induction': {'label': 'previous-token and induction heads', **ablate([prev, *inductions])},
    **{f'control_{h[0]}': {'label': f'{label} {name(h)}', **ablate([h])} for label, h in controls.items()},
    **{f'layer_{layer}': {'label': f'all of layer {layer}', **ablate([(layer, h) for h in range(n_heads)])} for layer in range(n_layers)},
  }
  singles = [[ablate([(layer, head)]) for head in range(n_heads)] for layer in range(n_layers)]
  results['ablation'] = conditions
  results['ablation_per_head'] = singles
  for c in conditions.values():
    text = f'  text loss {c["text_loss"]:.3f}' if 'text_loss' in c else ''
    print(
      f'ablate {c["label"]:36s} copy loss {c["copy_loss"]:.3f}  copy acc {c["copy_accuracy"]:.3f}  '
      f'fresh loss {c["fresh_loss"]:.3f}  induction score {induction_score(c):.2f}{text}'
    )

  if n_layers == 2:
    senders = {'previous': prev, 'duplicate': dup, 'control': controls['layer-0 control']}
    paths = {}
    reads: tuple[QKV, ...] = ('q', 'k', 'v')
    for role, sender in senders.items():
      for p in reads:
        paths[f'{role}_to_{p}'] = {
          'label': f"{role} head {name(sender)} removed from layer 1's {p.upper()}",
          **measure(model, data, cfg, probe_patch=path_patch(model, sender[1], means[0], p)),
        }
    results['path_ablation'] = paths
    for c in paths.values():
      print(f'path ablation {c["label"]:48s} copy loss {c["copy_loss"]:.3f}  copy acc {c["copy_accuracy"]:.3f}  induction score {induction_score(c):.2f}')
    results['composition'] = {
      'trained': {k: v.tolist() for k, v in composition_scores(model).items()},
      'untrained': {k: v.tolist() for k, v in composition_scores(untrained).items()},
    }

  # Diagnostic plots, titles stating the finding.
  heatmap(
    scores['induction'],
    f'{len(inductions)} head(s) in layer {last} put at least {INDUCTION_THRESHOLD:.0%} of their attention on the '
    f'induction target (chance {results["chance"]["induction"]:.1%})',
    'induction score', out / 'induction_scores.png',
  )
  heatmap(
    scores['previous'],
    f'Head {name(prev)} puts {scores["previous"][prev].item():.0%} of its attention on the previous token '
    f'(chance {results["chance"]["previous"]:.1%})',
    'previous-token score', out / 'previous_scores.png',
  )
  copy_rise = torch.tensor([[s['copy_loss'] - trained['copy_loss'] for s in row] for row in singles])
  worst = divmod(int(copy_rise.argmax()), n_heads)
  heatmap(
    copy_rise, f'Ablating head {worst[0]}.{worst[1]} raises copy loss most (+{copy_rise.max():.2f} nats)',
    'copy loss rise (nats)', out / 'ablation.png',
  )
  _, _, sample_patterns = run(model, data.test_probe.tokens[:1], chunk=1)
  plot_patterns(sample_patterns, roles, data.test_probe[:1], out / 'patterns.png')

  # The numbers behind the README figures (figure-design.md): copy a file into figures/data/
  # to show it there. Named after the run, so several runs can sit side by side.
  fig_dir = out / 'figure-data'
  fig_dir.mkdir()
  tag = cfg.run.name
  with (fig_dir / f'curves-{tag}.csv').open('w', newline='') as f:
    fields = ['step', 'copy_loss', 'fresh_loss', 'copy_accuracy', 'previous', 'duplicate', 'induction', 'best_induction']
    text = 'val_text_loss' in record['history'][0]
    fields += ['text_loss', 'icl_score'] if text else []
    writer = csv.writer(f)
    writer.writerow(fields)
    for h in record['history']:
      row = [
        h['step'], h['val_copy_loss'], h['val_fresh_loss'], h['val_copy_accuracy'],
        h['previous'][prev[0]][prev[1]], h['duplicate'][dup[0]][dup[1]], sum(h['induction'][layer][head] for layer, head in inductions) / len(inductions),
        max(h['induction'][last]),
      ]
      row += [h['val_text_loss'], h['val_text_icl_score']] if text else []
      writer.writerow(row)
  with (fig_dir / f'heads-{tag}.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['layer', 'head', 'previous', 'duplicate', 'induction', 'copy_loss_rise', 'fresh_loss_change', 'copy_accuracy_drop'])
    for layer in range(n_layers):
      for head in range(n_heads):
        s = singles[layer][head]
        writer.writerow([
          layer, head, scores['previous'][layer, head].item(), scores['duplicate'][layer, head].item(),
          scores['induction'][layer, head].item(), s['copy_loss'] - trained['copy_loss'],
          s['fresh_loss'] - trained['fresh_loss'], trained['copy_accuracy'] - s['copy_accuracy'],
        ])
  with (fig_dir / f'ablations-{tag}.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['condition', 'label', 'copy_loss', 'copy_accuracy', 'fresh_loss', 'induction_score', 'text_loss'])
    for key, c in {**conditions, **results.get('path_ablation', {})}.items():
      writer.writerow([key, c['label'], c['copy_loss'], c['copy_accuracy'], c['fresh_loss'], induction_score(c), c.get('text_loss', '')])
  if 'composition' in results:
    with (fig_dir / f'composition-{tag}.csv').open('w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['kind', 'sender', 'receiver', 'trained', 'untrained'])
      for kind in ('q', 'k', 'v'):
        for s in range(n_heads):
          for r in range(n_heads):
            writer.writerow([kind, s, r, results['composition']['trained'][kind][s][r], results['composition']['untrained'][kind][s][r]])
  keep = ('copy_loss', 'copy_accuracy', 'fresh_loss', 'text_loss', 'text_icl_score')
  save_json(fig_dir / f'summary-{tag}.json', {
    'run': tag,
    'chance': results['chance'],
    'phase_change_step': record.get('phase_change_step'),
    'steps': record['config']['optim']['steps'],
    'heads': results['heads'],
    'trained': {k: v for k, v in trained.items() if k in keep},
    'untrained': {k: v for k, v in results['untrained'].items() if k in keep},
  })
  save_json(fig_dir / f'pattern-{tag}.json', {
    'tokens': data.test_probe.tokens[0].tolist(),
    'period': data.test_probe.period[0].item(),
    'heads': {role: {'layer': layer, 'head': head, 'pattern': sample_patterns[f'blocks.{layer}.attn.pattern'][0, head].cpu().tolist()}
              for role, (layer, head) in roles.items()},
  })

  save_json(out / 'results.json', results)
  print(f'wrote {out}')


if __name__ == '__main__':
  main(tyro.cli(Config))
