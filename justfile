set dotenv-load

hf_user := env('HF_USER', 'your-hf-username')
repo := file_name(justfile_directory())

default:
  @just --list

# Install dependencies and the pre-commit hooks
setup:
  uv sync
  uv run pre-commit install

# CPU smoke test, under 5 minutes: train through the phase change, then analyse
quick:
  #!/usr/bin/env bash
  set -euo pipefail
  name=quick-$(date +%Y%m%d-%H%M%S)
  uv run train.py --quick --run-name "$name"
  uv run analyze.py --run "runs/$name" --name smoke --device cpu

# Full training run; pass flags through, e.g. just train --run-name wd1.0-seed0
train *args:
  uv run train.py {{args}}

# Inference on a saved run
predict run *args:
  uv run predict.py --run runs/{{run}} {{args}}

# Baselines, probes, interventions and plots for a saved run
analyze run *args:
  uv run analyze.py --run runs/{{run}} {{args}}

# Build the README figures: figures/src/*.tex -> figures/*.png (.agents/instructions/figure-design.md)
figures:
  #!/usr/bin/env bash
  set -euo pipefail
  cd figures
  mkdir -p build
  shopt -s nullglob
  for script in src/*.py; do uv run python "$script"; done
  for source in src/*.tex; do
    name=$(basename "$source" .tex)
    if ! TEXINPUTS='tex//:' latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build "$source" > "build/$name.latexmk.log" 2>&1; then
      tail -n 30 "build/$name.log"
      exit 1
    fi
    pdftoppm -r 300 -png -singlefile "build/$name.pdf" "$name"
    echo "figures/$name.png"
  done

test:
  uv run pytest

# Lint and type check, as in CI (no ruff format; see .agents/instructions/python-formatting.md)
check:
  uv run ruff check .
  uv run basedpyright

# Apply Ruff's safe fixes (import order, quotes, docstring layout)
fix:
  uv run ruff check --fix .

# Exploration notebooks (marimo, saved as plain .py files)
notebook name='explore':
  uv run marimo edit notebooks/{{name}}.py

# Upload a run's weights, metrics, plots and analyses to the Hugging Face Hub, in a folder named after the run
push run:
  uv run hf upload {{hf_user}}/{{repo}} runs/{{run}} {{run}}

# Run a sweep from cloud.py on Modal, one GPU per run, e.g. just cloud final (.agents/instructions/cloud.md)
cloud sweep *args:
  uv run modal run --detach cloud.py::sweep --name {{sweep}} {{args}}

# One run on Modal, e.g. just cloud-train lr3e-3-seed0 '--optim.lr 3e-3'
cloud-train run flags='' *args:
  uv run modal run --detach cloud.py::single --name {{run}} --flags '{{flags}}' {{args}}

# Copy a run from the Modal Volume into runs/<run>/
cloud-get run:
  mkdir -p runs
  uv run modal volume get {{repo}}-runs {{run}} runs
