# Runbooks

Runbooks are repeatable operating procedures for repository workflows. They are
written for humans and coding agents that need to run common tasks without
rediscovering the project conventions from old experiment logs.

Use a runbook when a task involves more than one command, produces artifacts, or
affects research conclusions.

## Index

| Runbook | Use When |
| --- | --- |
| [Template](template.md) | Creating a new runbook. |
| [Full Training Experiment](full-training-experiment.md) | Running a real training experiment after design and smoke checks. |
| [Evaluate Checkpoint](evaluate-checkpoint.md) | Running full validation/test evaluation for a saved checkpoint. |
| [Pareto Benchmark](pareto-benchmark.md) | Comparing saved checkpoints by quality, size, and forward latency. |
| [Rank Analysis](rank-analysis.md) | Checking activation rank or representation concentration. |
| [Dataset Diagnostics](dataset-diagnostics.md) | Inspecting material labels, staging rows, or processed-row traces. |
| [Log Experiment](log-experiment.md) | Creating or updating experiment logs under `docs/experiments/`. |

## Rules

- Keep runbooks generic enough to reuse across experiments.
- Put experiment-specific model names, checkpoint names, W&B run IDs, and metrics
  in `docs/experiments/`, not in runbooks.
- Prefer commands that run from the repository root.
- Use `uv run ...` for Python commands.
- Record all generated reports and checkpoints by repository-relative path.
- If a workflow changes research interpretation, update the relevant experiment
  log and TODO file before committing.

## Required Shape

New runbooks should follow [template.md](template.md) and include:

- purpose
- when to use
- prerequisites
- inputs
- procedure
- outputs
- stop conditions
- logging requirements
- verification
