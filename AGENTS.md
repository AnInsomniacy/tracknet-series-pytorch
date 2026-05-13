# AGENTS.md

## Scope

These instructions apply to the entire repository.

## Project Context

This repository is a PyTorch implementation of the TrackNet model series for shuttlecock trajectory tracking. It includes preprocessing, training, evaluation, inference, visualization, and synthetic tests for TrackNet V1-V5.

Keep source code, real data, and generated artifacts clearly separated:

```text
tracknet/data/       # source code for data handling
dataset/raw/         # real raw dataset, ignored by Git
dataset/processed/   # generated processed dataset, ignored by Git
outputs/train/       # local training runs and checkpoints, ignored by Git
model_results/       # tracked training logs, evaluation artifacts, and reports
```

Do not confuse `dataset/` with `tracknet/data/`.

## Environment

Use the conda environment `tracknet` unless the user explicitly requests another environment.

```bash
conda activate tracknet
```

The maintained workspace uses Python 3.11. Do not reinstall, replace, or downgrade PyTorch unless the user explicitly asks for that.

## Context Discipline

Read only the files needed for the task.

Preferred inspection targets:

- Source code under `tracknet/`.
- Configs under `configs/`.
- Tests under `tests/`.
- Project Markdown files.
- Lightweight tracked result summaries under `model_results/`.

Avoid reading dependencies, virtual environments, real datasets, processed frames, large checkpoints, or full training outputs unless they are directly required.

Prefer `rg` and `rg --files` for repository inspection.

## Training Contract

`train.batch_size` in YAML means per-rank / per-GPU batch size.

For Distributed Data Parallel:

```text
global_batch_size = train.batch_size * WORLD_SIZE
```

Current full-training contract:

```text
train.epochs = 30
train.amp = false
GPUs per model run = 2
```

Current batch-size convention:

```text
TrackNet V1: per-GPU batch 1, global batch 2
TrackNet V2: per-GPU batch 1, global batch 2
TrackNet V3 tracker: per-GPU batch 5, global batch 10
TrackNet V3 rectifier: per-GPU batch 16, global batch 32
TrackNet V4: per-GPU batch 1, global batch 2
TrackNet V5: per-GPU batch 1, global batch 2
```

Do not change batch size, epoch count, AMP, dataset roots, or GPU allocation without explicit user approval.

## GPU And Process Rules

Pin GPU jobs manually with `CUDA_VISIBLE_DEVICES`.

Standard 2-GPU training pattern:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 -m tracknet.tools.train --config configs/train_v4.yaml
```

Before launching training or evaluation jobs, inspect the current machine state:

```bash
nvidia-smi
tmux list-sessions
ps -ef | rg 'tracknet.tools.train|tracknet.tools.evaluate|torchrun|tensorboard'
```

Do not kill unrelated user processes.

Only stop TrackNet training, TrackNet evaluation, tmux sessions, or TensorBoard processes when the user explicitly requests it.

## Job Launching

Use one tmux session per long-running job.

Do not start multiple DDP training jobs at the exact same time. Start them sequentially with a short delay, then verify each job before launching the next one.

Before considering a job started, confirm:

- The tmux session exists.
- The intended GPUs are being used.
- TensorBoard has a unique port when training enables it.
- Progress has started.
- There is no immediate traceback or OOM.

This avoids TensorBoard port races, DDP initialization ambiguity, CUDA allocation conflicts, and unclear logs.

## TensorBoard

Training may launch TensorBoard automatically from rank 0.

TensorBoard starts at port `6006`; if the port is busy, the trainer searches for the next available port.

TensorBoard can outlive training because it is launched as a subprocess. Do not infer that training is still running only because a TensorBoard process is alive.

## Outputs And Checkpoints

Training outputs are written to:

```text
outputs/train/<experiment>_<timestamp>/
```

Expected run files:

```text
config.resolved.json
metrics.last.json
tensorboard/
checkpoints/last.pt
checkpoints/best.pt
checkpoints/model_best.pt
```

Use `model_best.pt` or `best.pt` for best-validation evaluation. Do not default to `last.pt` unless explicitly requested.

Delete training outputs only when explicitly requested.

## Evaluation

Evaluation configs write independent result directories under:

```text
model_results/evaluation/<model>/
```

Expected evaluation files:

```text
metrics.json
metrics.by_sequence.json
protocol.json
evaluation.resolved.json
checkpoint.json
predictions.csv
```

Use `python -m tracknet.tools.collect_evaluations` to regenerate aggregate evaluation summaries after result directories have been produced.

## Verification

For code changes, run relevant checks before claiming completion:

```bash
python -m compileall -q tracknet tests scripts
python -m pytest -q
```

For documentation-only changes, at minimum run:

```bash
git diff --check
```

For training-status or evaluation-status questions, inspect multiple signals:

- tmux output;
- process list;
- GPU state;
- TensorBoard process and logdir;
- checkpoints;
- `metrics.last.json`;
- evaluation `metrics.json` and `protocol.json`.

Do not infer runtime status from a single signal.

## Editing Rules

Keep changes scoped and consistent with existing project patterns.

Do not modify real datasets, generated processed data, local training outputs, unrelated tmux sessions, or unrelated GPU processes unless explicitly requested.

Communicate concisely in Chinese unless the user requests another language. Repository artifacts requested by the user may be written in professional English.
