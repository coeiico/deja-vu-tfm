# Cloud runs on Modal

Runs can go to [Modal](https://modal.com) as well as the local GPU. `cloud.py` at the repo root
launches the same `train.py` and `analyze.py` that run locally, one container per run. This file
says when that is worth it, how `cloud.py` works and the rules it follows. What this repo in
particular needs is in `CLAUDE.md`, under Cloud Runs.

## When to use it

- Local first. Every run here is meant to fit one consumer GPU with 8–12 GB, where most take
  minutes to a few hours, and electricity costs less than any cloud.
- The main gain is parallelism. A sweep of N runs on N GPUs finishes in the time of its longest
  run and costs the same as running them one after another.
- A bigger GPU only speeds up a model big enough to fill it. Each training step has a fixed cost
  (kernel launches, Python, data) that is the same on every GPU, so tiny models (a few layers,
  d_model of 128 or less, toy MLPs) run about as fast on an H100 as on a consumer GPU. Medium
  models (deeper transformers on long sequences, ResNets, U-Nets, VAEs) run roughly 3× faster on
  an A100 than on a 12 GB consumer GPU such as an RTX 3060, and 4–6× on an H100, more once the
  code uses bf16, `torch.compile` and data held on the GPU.
- CPU-bound work (tree search, RL environments, rollouts) gains nothing from a bigger GPU. Batch
  it inside the process first, then spread it over containers only if one machine is still too
  slow (see below).
- Before a long sweep, time one short run on the cloud GPU and estimate from that, not from the
  table below.

## Setup

- `modal` is a dev dependency. Log in once per machine with `uv run modal setup`, which opens a
  browser; the token lives in `~/.modal.toml`, never in the repo.
- This repo has two Volumes, created on first use: `<repo>-runs`, which the container sees as
  `runs/`, and `<repo>-data`, seen as `data/` (downloaded and preprocessed datasets).
- Secrets go in Modal Secrets (`uv run modal secret create`), never in `cloud.py` or the image.
  `cloud.py` never uploads `.env`.

## How cloud.py works

- The image is built from `pyproject.toml` and `uv.lock` without the dev tools, so the cloud
  environment matches the local one. git is installed for git dependencies and for
  `code_state()`.
- The working tree is copied into the image with `.git` and any uncommitted edits, so
  `metrics.json` records the code state exactly as a local run does. Only gitignored paths stay
  behind.
- `train(name, flags)` runs `train.py --run-name <name> <flags>`, then `analyze.py`, then
  commits the runs Volume, even when the run fails partway.
- `SWEEPS` maps a sweep name to its runs: `(run name, flags)` pairs, with the flags `just train`
  takes. `just cloud <sweep>` starts every run at once; a failed run does not stop the others.
- `just cloud-train <run> '<flags>'` starts one run, and `just cloud-get <run>` copies a finished
  run into `runs/<run>/`. Add `--gpu H100` (or `A100-80GB`, `A10`, `L4`, `T4`) to either launch
  recipe to override the default GPU.
- Launches are detached: runs carry on when the laptop sleeps or the terminal closes. Follow
  them at modal.com/apps or with `uv run modal app logs <repo>`, and stop them with
  `uv run modal app stop <repo>`.
- Figures (`just figures`) and Hugging Face uploads (`just push`) stay local, after
  `just cloud-get`.

## Rules

- Cloud compute costs real money, so nothing runs on Modal without the user's explicit
  permission for that launch. This covers every command that starts a Modal container:
  `just cloud`, `just cloud-train`, `modal run`, `modal shell`, `modal serve` and
  `modal deploy`, test and smoke runs included. Before asking, state the GPU, the number of
  runs, the expected hours and the estimated cost. A permission covers that launch only, not
  later launches, other sweeps or a relaunch after a failure. Downloading finished results with
  `just cloud-get` starts no container.
- `train.py` and `analyze.py` stay unchanged and runnable locally; `cloud.py` only launches them.
  Whatever a cloud run does, a local run must be able to do too.
- Every run in a sweep has its own name. `make_run_dir` refuses a folder that already exists, on
  the Volume as it does locally.
- Data that several runs read (a downloaded dataset, a tokenised cache, a generated corpus) is
  built once by a `prepare()` function before the sweep starts. Parallel runs that each fill the
  same cache write the same files at the same time.
- Stages that depend on each other run in order, each `.remote()` after the last; only the runs
  within a stage go in parallel.
- A seed reproduces a run only on one kind of device. Compare only runs from the same GPU type
  in a table or figure; `metrics.json` records the GPU.
- Set `HOURS` above the longest expected run: Modal stops a run at its timeout, 24 hours at
  most. A run that could pass that needs checkpoints on the runs Volume and a way to resume.
- Work out a sweep's cost before launching it: GPUs × hours × rate. Modal also caps how many
  GPUs one account runs at once.

## GPUs and rates

From third-party trackers in October 2026; check modal.com/pricing before budgeting. Billing is
per second while a container runs.

| GPU | About $/hour | Use for |
| --- | --- | --- |
| T4 | 0.59 | a smoke test that needs CUDA |
| L4 | 0.80 | tiny models; sweeps where only parallelism matters |
| A10 | 1.10 | small CNNs and transformers |
| A100 40 GB / 80 GB | 2.10 / 2.50 | medium models |
| H100 | 3.95 | the largest runs, once the code uses bf16, compile and large batches |
| CPU | 0.047 per core, plus 0.008 per GiB | self-play, rollouts, preprocessing |

## CPU-bound work: batch first, then spread out

- **Inside one process.** Keep hundreds of games or environments as one tensor and step them
  together. Collect every position that needs the network (each game's tree-search leaf, each
  environment's observation) and evaluate them in one forward pass. This gives 10–50× and helps
  locally too.
- **Across containers.** `@app.function(cpu=8)` workers, `starmap`ped over independent shards
  (self-play games, random rollouts), write to the Volume; between rounds a GPU function trains
  on everything written so far.
- **Not for on-policy RL.** PPO updates on its newest experience after every rollout, so workers
  in separate containers would keep waiting on each other. Use one large container instead
  (`cpu=32` plus a GPU) with vectorised or GPU-native environments.

## This project's needs

What this repo's `cloud.py` needs, beyond the scaffold, is recorded in `CLAUDE.md` under Cloud
Runs: whether the cloud is worth it, the default GPU, the sweeps, any `prepare()` step, stages
that must run in order, and checkpointing. Once runs have been timed, replace the estimates
there with the measured times and the GPU actually used.
