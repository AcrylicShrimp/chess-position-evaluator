# Runbook: <Name>

## Purpose

State the workflow objective and the decision it supports.

## When To Use

List the situations where this runbook applies.

## Prerequisites

- Repository root is the working directory.
- Worktree state has been inspected with `git status --short`.
- Required data, checkpoints, credentials, and environment variables are present.

## Inputs

| Input | Description |
| --- | --- |
| `<input>` | Describe the required value or artifact. |

## Procedure

1. Record the worktree state.
2. Run the required commands in order.
3. Capture generated artifact paths.
4. Update the relevant experiment log or TODO.

```bash
# Commands go here.
```

## Outputs

| Output | Path Or Location |
| --- | --- |
| `<output>` | `<path>` |

## Stop Conditions

- Stop if required inputs are missing.
- Stop if a command fails in a way that invalidates the result.
- Stop if the observed setup differs from the comparison contract and the
  difference is not the explicit experimental variable.

## Logging Requirements

- Record commands, settings, artifacts, metrics, and warnings.
- Preserve failed, interrupted, or non-comparable runs.
- Update any active TODO status and activity log when the work moves forward.

## Verification

```bash
git diff --check
```

Record any checks that could not be run and why.
