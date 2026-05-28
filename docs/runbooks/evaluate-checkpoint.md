# Runbook: Evaluate Checkpoint

## Purpose

Evaluate a saved checkpoint on processed validation or test data and write a
stable JSON report.

## When To Use

- Closing a training experiment.
- Comparing a best checkpoint against a baseline.
- Producing full validation or test metrics for a model-selection decision.

## Prerequisites

- The checkpoint exists under `artifacts/checkpoints/`.
- The required processed split exists under `data/processed/`.
- The checkpoint contains or is paired with the intended model variant.
- The validation split is treated as model-selection data; the test split is
  reserved for final comparison.

## Inputs

| Input | Description |
| --- | --- |
| Model name | Checkpoint name without `.pth`. |
| Split | `validation`, `test`, or `train`. |
| Batch size | Evaluation batch size. |
| Output path | Optional explicit report path. |

## Procedure

1. Confirm the checkpoint path.
2. Confirm the split and whether full evaluation is required.
3. Run `cpe eval-dataset`.
4. Inspect the report for warnings, row counts, checkpoint epoch, and metrics.
5. Copy metrics into the experiment log.

```bash
PYTHONUNBUFFERED=1 uv run cpe eval-dataset <model-name> \
  --split validation \
  --full \
  --batch 4096 \
  --device auto \
  --output artifacts/reports/<model-name>.validation.eval.json

PYTHONUNBUFFERED=1 uv run cpe eval-dataset <model-name> \
  --split test \
  --full \
  --batch 4096 \
  --device auto \
  --output artifacts/reports/<model-name>.test.eval.json
```

## Outputs

| Output | Path Or Location |
| --- | --- |
| Validation report | `artifacts/reports/<model-name>.validation.eval.json` |
| Test report | `artifacts/reports/<model-name>.test.eval.json` |

## Stop Conditions

- Checkpoint or dataset split is missing.
- The resolved model variant is not the intended one.
- The report row count does not match the selected evaluation contract.
- Model logits or labels are non-finite.

## Logging Requirements

- Record checkpoint name, checkpoint epoch, split, row count, BCE, Brier,
  probability MAE/RMSE, CP-equivalent MAE/RMSE, ECE, and report path.
- Record validation warnings as expected when evaluating the validation split.
- Do not claim final comparison from validation-only metrics.

## Verification

```bash
python -m json.tool artifacts/reports/<model-name>.<split>.eval.json >/dev/null
```
