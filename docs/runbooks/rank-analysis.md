# Runbook: Rank Analysis

## Purpose

Inspect activation rank and spectral concentration for a saved checkpoint.

## When To Use

- A model underperforms despite apparently adequate parameter count.
- A bottleneck, funnel, fusion, or deep attention design may be collapsing useful
  representation directions.
- A training result needs diagnostic evidence beyond scalar loss.

## Prerequisites

- The checkpoint exists under `artifacts/checkpoints/`.
- `data/processed/validation.chesseval` exists.
- The model variant is supported by the rank hook registration code.

## Inputs

| Input | Description |
| --- | --- |
| Model name | Checkpoint name without `.pth`. |
| Sample count | Rank analysis sample count, if configurable. |
| Comparison target | Prior rank report or reference model. |

## Procedure

1. Confirm the checkpoint and validation dataset exist.
2. Run `cpe analyze-rank`.
3. Inspect collected layers, effective rank, participation rank, tail singular
   value, and obvious missing hooks.
4. Record findings in the experiment log.

```bash
PYTHONUNBUFFERED=1 uv run cpe analyze-rank <model-name>
```

## Outputs

| Output | Path Or Location |
| --- | --- |
| Rank report | `artifacts/reports/<model-name>.rank.txt` |

## Stop Conditions

- Checkpoint or validation data is missing.
- No relevant layers are collected.
- The target architecture changed but rank hooks were not updated.

## Logging Requirements

- Record report path, sample count, collected layer count, and key layer ranks.
- Distinguish full-rank-but-top-heavy spectra from hard rank collapse.
- Compare against a reference only when the sample count and hook semantics are
  comparable.

## Verification

```bash
uv run python -m unittest python.tests.test_analyze_rank -v
```
