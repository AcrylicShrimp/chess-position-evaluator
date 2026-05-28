# Runbook: Log Experiment

## Purpose

Create or update a durable experiment record under `docs/experiments/`.

## When To Use

- Planning a training run, evaluation run, benchmark, ablation, smoke test, or
  diagnostic analysis.
- Recording final metrics or correcting interpretation after a run.
- Preserving failed, interrupted, mistaken, or non-comparable work.

## Prerequisites

- The experiment objective is known.
- The relevant commands, settings, artifacts, or metrics are available.
- If updating an existing log, the target log file is unambiguous.

## Inputs

| Input | Description |
| --- | --- |
| Experiment name | Human-readable name used to derive the log slug. |
| Status | Planned, Running, Completed, Failed, or Aborted. |
| Verdict | Pass, Fail, Inconclusive, Informational, or Not Evaluated. |
| Evidence | Commands, reports, metrics, checkpoints, W&B URLs, source commit. |

## Procedure

1. Check existing logs:

```bash
rg --files docs/experiments | sort
```

2. Create a new `docs/experiments/YYYY-MM-DD.NN.slug.md` file or update the
   existing target.
3. Record status, goal, hypothesis, key characteristics, setup, procedure,
   results, verdict rationale, issues, follow-up, and activity log.
4. For comparative training, record a recipe comparison before the run starts.
5. Append new evidence instead of deleting mistaken or superseded observations.

## Outputs

| Output | Path Or Location |
| --- | --- |
| Experiment log | `docs/experiments/YYYY-MM-DD.NN.slug.md` |

## Stop Conditions

- The target experiment log is ambiguous.
- Required evidence is unavailable and would lead to guessed metrics.
- A pass/fail verdict is requested without predeclared or meaningful criteria.

## Logging Requirements

- Record unknown values as `Unknown`, not guessed.
- Use repository-relative paths for artifacts.
- Include exact commands when available.
- Include final metrics and report paths.
- Preserve failed, interrupted, mistaken, and non-comparable runs.

## Verification

```bash
rg -n "^# Experiment:|^- State:|^- Verdict:|^## Results|^## Activity Log" docs/experiments/<log>.md
git diff --check
```
