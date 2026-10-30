"""
Run train.py and analyze.py on Modal: one GPU per run, every run of a sweep at once.

The scripts run unchanged, in a container built from pyproject.toml and uv.lock. runs/ and
data/ live on Modal Volumes, so results outlast the container, and `just cloud-get <run>`
copies a run back into runs/. Figures and Hugging Face uploads stay local. Read
.agents/instructions/cloud.md before changing this file.

  just cloud final                             every run in SWEEPS['final'], in parallel
  just cloud-train lr3e-3-seed0 '--optim.lr 3e-3'   one run
"""

import shlex
import subprocess
from pathlib import Path, PurePosixPath

import modal

PROJECT = 'deja-vu-tfm'
GPU = 'L4'
"""The default GPU. These models are small, so a bigger one barely speeds them up (cloud.md)."""
HOURS = 4
"""How long a run may take before Modal stops it. The TinyStories run takes 26 minutes on a 3060."""

SWEEPS: dict[str, list[tuple[str, list[str]]]] = {
  # The runs the README reports, under the names they have in runs/ and on Hugging Face.
  'final': [
    ('default-seed0', ['--seed', '0']),
    ('default-seed1', ['--seed', '1']),
    ('default-seed2', ['--seed', '2']),
    ('one-layer-seed0', ['--model.n-layers', '1']),
    ('gpt2-init-seed0', ['--model.init', 'gpt2']),
    ('stories-seed0', ['--data', 'stories', '--model.d-model', '256', '--model.n-heads', '8',
                       '--optim.steps', '20000', '--eval-every', '250', '--splits.val-size', '256']),
  ],
}

REPO = Path('/repo')
runs = modal.Volume.from_name(f'{PROJECT}-runs', create_if_missing=True)
data = modal.Volume.from_name(f'{PROJECT}-data', create_if_missing=True)
VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {'/volumes/runs': runs, '/volumes/data': data}

image = (
  modal.Image.debian_slim(python_version='3.12')
  # tiny-transformer is a git dependency, and code_state() records the commit with git. The
  # copied files belong to another user and lose their modes, which git would otherwise
  # refuse or report as edits.
  .apt_install('git')
  .run_commands("git config --global --add safe.directory '*'", 'git config --global core.fileMode false')
  .uv_sync(extra_options='--no-dev')  # exactly uv.lock's packages, without the dev tools
  .env({'PYTHONPATH': str(REPO / 'src'), 'PYTHONUNBUFFERED': '1'})
  # The working tree as it is, uncommitted edits and .git included, so metrics.json records
  # the code state just as a local run does. Only gitignored paths stay behind; .env holds
  # secrets and must never be uploaded.
  .add_local_dir('.', str(REPO), copy=True, ignore=[
    '.env', '.venv', 'runs', 'data', 'figures/build', '.trackio',
    '**/__pycache__', '.pytest_cache', '.ruff_cache', '**/__marimo__',
  ])
)
app = modal.App(PROJECT, image=image)


def link_volumes() -> None:
  # The scripts read and write runs/ and data/ relative to the repo, as they do locally.
  # .gitignore's runs/ and data/ match folders, not symlinks, so git is told to skip the
  # links too; otherwise every cloud run would record its code as edited.
  with open(REPO / '.git/info/exclude', 'a') as exclude:
    exclude.write('/runs\n/data\n')
  for name in ('runs', 'data'):
    link = REPO / name
    if not link.exists():
      link.symlink_to(f'/volumes/{name}')


def python(*args: str) -> None:
  subprocess.run(['python', *args], cwd=REPO, check=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=HOURS * 3600)
def train(name: str, flags: list[str]) -> None:
  """Train one run and analyse it, as `just train` then `just analyze <run> --name final`."""
  link_volumes()
  try:
    python('train.py', '--run-name', name, *flags)
    python('analyze.py', '--run', f'runs/{name}', '--name', 'final')
  finally:
    runs.commit()  # keep whatever the run wrote, even if it failed partway


@app.function(cpu=4, memory=16384, volumes=VOLUMES, timeout=2 * 3600)
def prepare() -> None:
  """
  Download and tokenise TinyStories into the data Volume.

  Runs once before a sweep, so parallel runs read one finished cache instead of each
  writing it at the same time.
  """
  link_volumes()
  python('-c', 'from deja_vu_tfm.stories import load_streams; load_streams()')
  data.commit()


def launch(specs: list[tuple[str, list[str]]], gpu: str) -> None:
  if any('stories' in flags for _, flags in specs):
    prepare.remote()
  # One container and one GPU per run, all at once; a failed run doesn't stop the others.
  results = train.with_options(gpu=gpu).starmap(specs, return_exceptions=True)
  for (name, _), result in zip(specs, results, strict=True):
    print(f'{name}: failed: {result}' if isinstance(result, Exception) else f'{name}: done')


@app.local_entrypoint()
def sweep(name: str, gpu: str = GPU) -> None:
  if name not in SWEEPS:
    raise SystemExit(f'no sweep named {name!r}; SWEEPS has {sorted(SWEEPS) or "none yet"}')
  launch(SWEEPS[name], gpu)


@app.local_entrypoint()
def single(name: str, flags: str = '', gpu: str = GPU) -> None:
  launch([(name, shlex.split(flags))], gpu)
