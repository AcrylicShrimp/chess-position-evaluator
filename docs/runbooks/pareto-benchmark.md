# Runbook: Pareto Benchmark

## Purpose

Compare saved checkpoints by quality, parameter count, forward latency,
throughput, and peak runtime memory.

## When To Use

- Choosing the next small/fast/sufficiently-strong model direction.
- Comparing existing checkpoints without spending training budget.
- Checking whether a candidate is dominated on both speed and quality.

## Prerequisites

- Target checkpoints exist under `artifacts/checkpoints/`.
- Full validation/test reports exist for quality comparison when available.
- The GPU is available if CUDA timing is the target.
- `TORCHINDUCTOR_CACHE_DIR` is set or the CLI default
  `artifacts/cache/torchinductor` is acceptable.

## Inputs

| Input | Description |
| --- | --- |
| Candidate list | Checkpoint names without `.pth`. |
| Batch size | Synthetic forward batch size. |
| Compile mode | Usually `max-autotune` for optimized CUDA timing. |
| Dtype | Usually `bf16` on CUDA. |
| Iterations | Number of measured forwards after warmup. |

## Procedure

1. Record GPU state when CUDA timing matters.
2. Run `cpe benchmark-pareto` with explicit output.
3. Parse the JSON report and sort by quality and speed.
4. Record dominated candidates, frontier candidates, and caveats in an
   experiment log.

```bash
PYTHONUNBUFFERED=1 uv run cpe benchmark-pareto \
  --batch 2048 \
  --warmup 20 \
  --iterations 50 \
  --compile-mode max-autotune \
  --device auto \
  --dtype bf16 \
  --output artifacts/reports/<benchmark-name>.json
```

## Outputs

| Output | Path Or Location |
| --- | --- |
| Pareto report | `artifacts/reports/<benchmark-name>.json` |
| Experiment log | `docs/experiments/YYYY-MM-DD.NN.<slug>.md` |
| TorchInductor cache | `artifacts/cache/torchinductor/` by default |

## Stop Conditions

- A required checkpoint is missing.
- A candidate cannot load because its model variant is unsupported.
- Timing is being compared across different devices, dtypes, batch sizes,
  compile modes, or warmed/cold compile states without recording the difference.

## Logging Requirements

- Record batch, warmup, iterations, compile mode, dtype, device, torch version,
  and cache path.
- Record parameter count, mean latency, p95 latency, examples/s, peak memory,
  validation/test BCE, and ECE for each candidate.
- State whether the benchmark measures synthetic model forward only or
  end-to-end pipeline throughput.

## Verification

```bash
python -m json.tool artifacts/reports/<benchmark-name>.json >/dev/null
git diff --check
```
