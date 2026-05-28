# Runbook: Full Training Experiment

## Purpose

Run a full training experiment while preserving recipe comparability,
checkpoint traceability, and post-training evaluation requirements.

## When To Use

- Starting a real training run intended to inform model selection.
- Promoting a smoke-tested architecture to a full run.
- Re-running an existing architecture with a changed training recipe.

## Prerequisites

- `git status --short` has been inspected.
- The intended source commit is recorded in the experiment log.
- `data/processed/train.chesseval` and `data/processed/validation.chesseval`
  exist.
- `WANDB_API_KEY` is available when using online W&B logging.
- The experiment log exists or will be created before training starts.
- The checkpoint name is explicit and not hardcoded in source code.

## Inputs

| Input | Description |
| --- | --- |
| Experiment name | Checkpoint and W&B run name. |
| Model variant | `cpe train --model-variant` value. |
| Training recipe | Epochs, steps, batch, lr, weight decay, scheduler, warmup, grad clip, compile mode, workers. |
| Comparison target | Previous run or baseline whose recipe should be compared. |

## Procedure

1. Create or update an experiment log under `docs/experiments/`.
2. Record a recipe comparison against the intended baseline.
3. Run targeted tests for changed model or training code.
4. Run `git diff --check`.
5. Run a smoke training job when model code, scheduler code, batch size, or
   compile mode changed.
6. Start the full training run only after the smoke gate passes.
7. Monitor validation loss, grad norm, checkpoint epoch, and obvious instability.
8. After training, run full validation and full test evaluation on the best
   checkpoint.
9. Record W&B URL, checkpoint paths, report paths, and final interpretation.

Example command shape:

```bash
WANDB_MODE=online WANDB_SILENT=true PYTHONUNBUFFERED=1 uv run cpe train <experiment-name> \
  --model-variant <variant> \
  --epochs <epochs> \
  --steps <steps-per-epoch> \
  --batch <batch-size> \
  --lr <learning-rate> \
  --wd <weight-decay> \
  --scheduler <scheduler> \
  --warmup-epochs <warmup-epochs> \
  --warmup-start-factor <warmup-start-factor> \
  --eta-min <eta-min> \
  --grad-clip <grad-clip> \
  --compile-mode <compile-mode> \
  --train-workers <train-workers> \
  --val-workers <validation-workers>
```

## Outputs

| Output | Path Or Location |
| --- | --- |
| Final checkpoint | `artifacts/checkpoints/<experiment-name>.pth` |
| Best checkpoint | `artifacts/checkpoints/<experiment-name>-best.pth` |
| Training log | `artifacts/reports/<experiment-name>.train.log` when captured |
| Experiment log | `docs/experiments/YYYY-MM-DD.NN.<slug>.md` |
| W&B run | URL or offline run directory |

## Stop Conditions

- Required data or credentials are missing.
- The checkpoint path already exists and the run is not an intentional resume.
- Smoke training OOMs, produces non-finite loss, or is too slow for the intended
  budget.
- Recipe mismatch is discovered and is not the explicit experimental variable.
- W&B or local logging would expose secrets in committed files.

## Logging Requirements

- Record the exact command.
- Record the source commit and dirty worktree status.
- Record training recipe and recipe comparison.
- Record checkpoints, W&B URL, full eval reports, and known anomalies.
- Preserve failed or non-comparable runs.

## Verification

```bash
git status --short
git diff --check
uv run python -m unittest discover -s python/tests -v
```

Run narrower tests first when appropriate, but record the final verification set.
