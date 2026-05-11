# AGENTS.md

## Scope

These instructions apply to the entire repository.

## Project

This is a PyTorch implementation of the TrackNet model series, including preprocessing, training, evaluation, inference, and visualization for V1-V5.

Keep real data and generated artifacts separate from source code:

- Raw data: `dataset/raw/`
- Processed data: `dataset/processed/`
- Training outputs: `outputs/train/`
- Source data code: `tracknet/data/`

Do not confuse `dataset/` with `tracknet/data/`.

## Environment

Use the conda environment `tracknet` unless instructed otherwise.

```bash
conda activate tracknet
```

Python version is 3.11.

Do not reinstall or replace PyTorch unless explicitly requested.

## Context Discipline

Read source code, configs, tests, and project docs only as needed.

Do not inspect dependencies, virtual environments, raw datasets, processed frames, or training outputs unless they are directly relevant to the task.

Prefer `rg` and `rg --files` for repository inspection.

## Training Contract

`train.batch_size` in YAML means per-rank / per-GPU batch size.

For DDP:

```text
global_batch_size = train.batch_size * WORLD_SIZE
```

Current intended full-training setup:

```text
train.batch_size = 1
GPUs per model = 2
global_batch_size = 2
train.epochs = 30
train.amp = false
```

Do not change batch size, epoch count, AMP, dataset roots, or GPU allocation without explicit user approval.

## GPU Rules

Pin training jobs manually with `CUDA_VISIBLE_DEVICES`.

Standard 2-GPU DDP pattern:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 -m tracknet.tools.train --config configs/train_v4.yaml
```

Before launching training, check:

```bash
nvidia-smi
tmux list-sessions
ps -ef | rg 'tracknet.tools.train|torchrun|tensorboard'
```

Do not kill unrelated user processes.

Only stop TrackNet training, tmux sessions, or TensorBoard processes when explicitly requested.

## Launching Jobs

Use one tmux session per training job.

Do not launch multiple DDP training jobs at the exact same time. Start jobs sequentially with a short delay, and confirm each job has initialized before starting the next one.

Before launching the next job, verify:

- The tmux session exists.
- The intended GPUs are being used.
- TensorBoard has a unique port.
- Training progress has started.
- There is no immediate traceback or OOM.

This avoids TensorBoard port races, DDP initialization ambiguity, CUDA allocation conflicts, and confusing logs.

## TensorBoard

Training may launch TensorBoard automatically.

TensorBoard starts from port `6006`; if busy, the trainer searches for the next available port.

TensorBoard can outlive training because it is launched as a subprocess. Do not assume a live TensorBoard process means training is still running.

## Outputs

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

Use `model_best.pt` or `best.pt` for best-validation results. Do not default to `last.pt` unless explicitly requested.

Delete training outputs only when explicitly requested.

## Verification

For code changes, run relevant checks before claiming completion:

```bash
python -m compileall -q tracknet tests scripts
python -m pytest -q
```

For training-status questions, inspect multiple signals:

- tmux output
- process list
- GPU state
- TensorBoard events
- checkpoints
- `metrics.last.json`

Do not infer training status from a single signal.

## Editing Rules

Keep changes scoped and consistent with existing project patterns.

Do not modify real datasets, generated training outputs, unrelated tmux sessions, or unrelated GPU processes unless explicitly requested.

Communicate concisely in Chinese unless the user asks for another language.
