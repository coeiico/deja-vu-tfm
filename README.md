# deja-vu-tfm

**Induction heads.** Watching in-context copying appear as a phase change in a 2-layer attention-only transformer, and finding the heads responsible.

**Status:** not started.

## Question

Can we watch in-context copying appear as a sudden phase change, and find the exact heads responsible?

## Plan

- **Architecture:** tiny-transformer with 2 layers and `mlp=none` (attention-only).
- **Data:** random token sequences repeated once, so the second half is predictable only by copying; then TinyStories text.

## Experiments

- Loss on the repeated half over training (the phase change).
- An induction score per head.
- Locate the previous-token head (layer 0) and the induction head (layer 1).
- Ablate each.
- Show how they compose.

## Done when

- The two heads are identified, and ablating them removes the copying ability.

Plus the general checklist:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

This repo builds on [tiny-transformer](https://github.com/coeiico/tiny-transformer), installed from GitHub. To work on both at once, run `uv add --editable ../tiny-transformer`, then switch back to the git source before committing.

## Layout

```
deja-vu-tfm/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/deja_vu_tfm/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Olsson et al. 2022, In-context learning and induction heads

## License

MIT
