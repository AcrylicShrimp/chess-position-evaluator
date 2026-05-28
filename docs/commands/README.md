# Command Catalog

This catalog lists the repository entry points that a human or coding agent may
run. Use it to choose the right command, identify required runtime artifacts,
and avoid dormant prototypes.

## Status Labels

- `active`: maintained as part of the current workflow.
- `diagnostic`: maintained for investigation and analysis workflows.
- `benchmark`: maintained for measurement only.
- `experimental`: useful for local experiments, but not the primary workflow.
- `dormant`: prototype code that should not be treated as active behavior.

## Python CLI

Python commands run through the `cpe` Typer CLI from the repository root:

```bash
uv run cpe <command> ...
```

| Command | Status | Purpose | Main Inputs | Main Outputs | Owner Modules | Related Runbooks |
| --- | --- | --- | --- | --- | --- | --- |
| `cpe train` | active | Train a value model with explicit recipe settings and WandB logging. | `data/processed/train.chesseval`, `data/processed/validation.chesseval`, `WANDB_API_KEY`, model variant, scheduler, optimizer settings. | `artifacts/checkpoints/<experiment>.pth`, `artifacts/checkpoints/<experiment>-best.pth`, WandB run/artifacts. | `python/cli.py`, `python/train/entry.py`, `python/train/trainer.py` | [Full Training Experiment](../runbooks/full-training-experiment.md), [Log Experiment](../runbooks/log-experiment.md) |
| `cpe eval-dataset` | active | Evaluate a checkpoint against a processed dataset split. | Checkpoint name, split or explicit dataset path, optional model variant override. | `artifacts/reports/<model>.<split>.eval.json` unless overridden. | `python/cli.py`, `python/eval_dataset.py` | [Evaluate Checkpoint](../runbooks/evaluate-checkpoint.md) |
| `cpe eval` | active | Interactive FEN evaluation for a checkpoint. | Checkpoint name and interactive FEN input. | Terminal-side win probability output. | `python/cli.py`, `python/eval.py` | [Evaluate Checkpoint](../runbooks/evaluate-checkpoint.md) |
| `cpe battle` | active | Play against the model through the local search wrapper. | Checkpoint name and interactive moves. | Terminal gameplay output. | `python/cli.py`, `python/battle/entry.py` | None yet. |
| `cpe export-onnx` | active | Export a PyTorch checkpoint to ONNX. | Checkpoint name. | `artifacts/onnx/<model>.onnx`. | `python/cli.py`, `python/export_onnx.py` | [Evaluate Checkpoint](../runbooks/evaluate-checkpoint.md) |
| `cpe benchmark-pareto` | benchmark | Compare checkpoints by metrics, parameter count, latency, and memory. | Checkpoint names or default benchmark checkpoint registry. | `artifacts/reports/standardized-pareto-benchmark.json` unless overridden. | `python/cli.py`, `python/benchmark_pareto.py`, `python/libs/modeling/registry.py` | [Pareto Benchmark](../runbooks/pareto-benchmark.md) |
| `cpe analyze-rank` | diagnostic | Inspect activation rank and representation concentration. | Checkpoint name and `data/processed/validation.chesseval`. | Terminal rank report. | `python/cli.py`, `python/analyze_rank.py` | [Rank Analysis](../runbooks/rank-analysis.md) |
| `cpe analyze-material-labels` | diagnostic | Analyze processed labels against the fixed material-score prior. | Processed split or explicit `.chesseval` path. | `artifacts/reports/material-label-calibration.<split>.json` unless overridden. | `python/cli.py`, `python/analyze_material_label_calibration.py` | [Dataset Diagnostics](../runbooks/dataset-diagnostics.md) |
| `cpe analyze-material-signal` | diagnostic | Analyze source/staging material difference against engine centipawn targets. | `data/interim/lichess_db_eval.duckdb.tmp` or explicit staging path. | `artifacts/reports/source-material-signal.<split>-<rows>.json` unless overridden. | `python/cli.py`, `python/analyze_material_signal.py` | [Dataset Diagnostics](../runbooks/dataset-diagnostics.md) |
| `cpe diagnose-parallel-fusion` | diagnostic | Compare a parallel-fusion checkpoint against a baseline and inspect failure patterns. | Parallel checkpoint name, baseline checkpoint name, dataset split. | `artifacts/reports/<parallel>.parallel-diagnostics.<split>.<rows>.json` unless overridden. | `python/cli.py`, `python/analyze_parallel_fusion.py` | [Dataset Diagnostics](../runbooks/dataset-diagnostics.md), [Evaluate Checkpoint](../runbooks/evaluate-checkpoint.md) |
| `cpe trace-processed-rows` | diagnostic | Trace processed diagnostic row indices back to source staging FEN/cp rows. | Diagnostic JSON report and staging DuckDB. | Sibling `*.source-trace.<split>.worstN.bestN.json` report unless overridden. | `python/cli.py`, `python/trace_processed_rows.py` | [Dataset Diagnostics](../runbooks/dataset-diagnostics.md) |

## Rust Workspace

Rust commands run from the repository root.

| Crate Or Target | Status | Purpose | Main Inputs | Main Outputs | Owner Crate | Related Runbooks |
| --- | --- | --- | --- | --- | --- | --- |
| `cargo run --release -p preprocess` | active | Build processed `.chesseval` train/validation/test splits from the Lichess evaluation JSONL source. | `data/raw/lichess_db_eval.jsonl`; optional existing `data/interim/lichess_db_eval.duckdb.tmp` staging database. | `data/processed/train.chesseval`, `data/processed/validation.chesseval`, `data/processed/test.chesseval`, `data/interim/lichess_db_eval.duckdb.tmp`. | `crates/preprocess` | [Full Training Experiment](../runbooks/full-training-experiment.md), [Dataset Diagnostics](../runbooks/dataset-diagnostics.md) |
| `cargo run --release -p preprocess -- diagnose [split] [limit] [examples]` | diagnostic | Inspect strict source-row filtering and rejected example categories. | `data/interim/lichess_db_eval.duckdb.tmp`. | Terminal validation summary and optional invalid examples. | `crates/preprocess` | [Dataset Diagnostics](../runbooks/dataset-diagnostics.md) |
| `cargo run --release -p inference -- <onnx-path>` | experimental | Smoke test ONNX Runtime inference using random board tensors. | ONNX model path; defaults to an older placeholder path when omitted. | Terminal random inference probabilities. | `crates/inference` | None yet. |
| `cargo bench -p onnx-backend-benchmark` | benchmark | Benchmark ONNX Runtime latency and throughput for exported ONNX models. | `artifacts/onnx/ghost-ca-r4-256ch-6blk-best.onnx` in the current benchmark code. | Criterion benchmark output under `target/criterion/`. | `crates/onnx-backend-benchmark` | [Pareto Benchmark](../runbooks/pareto-benchmark.md) |
| `tree-search` crate | dormant | Prototype Rust-side SEE/static or quiescence search. | None in the active workflow. | None in the active workflow. | `crates/tree-search` | None; see `crates/tree-search/README.md` before reviving. |

## Update Rules

- Add or update this catalog when adding, removing, renaming, or repurposing a
  CLI command, Rust crate, binary, benchmark, or diagnostic entry point.
- Link new commands to a runbook when the workflow has more than one step or
  affects research conclusions.
- Keep checkpoint names, run IDs, and experiment-specific metrics in
  `docs/experiments/`; keep this file focused on reusable command contracts.
- Mark prototypes as `experimental` or `dormant` until their inputs, outputs, and
  verification plan are explicit.
