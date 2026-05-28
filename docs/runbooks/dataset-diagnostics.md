# Runbook: Dataset Diagnostics

## Purpose

Analyze processed or staging data quality before interpreting model results.

## When To Use

- Dataset generation or filtering changed.
- Label/material relationships look suspicious.
- A model failure bucket needs to be traced back to source positions.
- Processed data row counts or split behavior need audit evidence.

## Prerequisites

- Required raw, interim, or processed data exists under `data/`.
- The selected diagnostic command matches the data layer being inspected.
- Large diagnostics have enough disk and runtime budget.

## Inputs

| Input | Description |
| --- | --- |
| Split | `train`, `validation`, `test`, or `all` when supported. |
| Row count | Prefix/sample size or `--full`. |
| Report path | Output JSON path under `artifacts/reports/`. |
| Diagnostic report | Existing model diagnostic report for row tracing, when needed. |

## Procedure

For processed material-label diagnostics:

```bash
PYTHONUNBUFFERED=1 uv run cpe analyze-material-labels \
  --split validation \
  --full \
  --batch 8192 \
  --output artifacts/reports/<name>.json
```

For staging/source material signal diagnostics:

```bash
PYTHONUNBUFFERED=1 uv run cpe analyze-material-signal \
  --split all \
  --rows 1000000 \
  --batch 20000 \
  --output artifacts/reports/<name>.json
```

For tracing processed diagnostic rows back to staging rows:

```bash
PYTHONUNBUFFERED=1 uv run cpe trace-processed-rows <diagnostic-report-path> \
  --split validation \
  --top-worst 20 \
  --top-best 5 \
  --output artifacts/reports/<name>.json
```

## Outputs

| Output | Path Or Location |
| --- | --- |
| Material-label report | `artifacts/reports/<name>.json` |
| Material-signal report | `artifacts/reports/<name>.json` |
| Source trace report | `artifacts/reports/<name>.json` |

## Stop Conditions

- The required data layer is missing.
- Row counts do not match the intended split contract.
- Diagnostic outputs contain invalid geometry or reject rates that invalidate the
  planned experiment comparison.
- Source trace cannot map requested processed rows back to staging rows.

## Logging Requirements

- Record command, split, row count, selection method, report path, and key
  summary metrics.
- For data quality issues, record whether training data must be regenerated.
- Preserve diagnostics even when they disprove the original suspicion.

## Verification

```bash
uv run python -m unittest python.tests.test_material_label_calibration python.tests.test_material_signal python.tests.test_trace_processed_rows -v
git diff --check
```
