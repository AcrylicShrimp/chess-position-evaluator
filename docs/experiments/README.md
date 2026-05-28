# Experiment Logging Notes

## Related Runbooks

- [Full Training Experiment](../runbooks/full-training-experiment.md)
- [Evaluate Checkpoint](../runbooks/evaluate-checkpoint.md)
- [Pareto Benchmark](../runbooks/pareto-benchmark.md)
- [Rank Analysis](../runbooks/rank-analysis.md)
- [Dataset Diagnostics](../runbooks/dataset-diagnostics.md)
- [Log Experiment](../runbooks/log-experiment.md)

## Comparative Training Recipe Check

Before starting any comparative training experiment, record a recipe comparison in the experiment log.

At minimum, compare:

- epochs
- steps per epoch
- batch size
- learning rate
- weight decay
- scheduler settings
- intended sample exposure

If any field differs from the comparison target, either make the difference the explicit experimental variable or mark the run as not directly comparable.

Never delete mistaken, failed, interrupted, or non-comparable experiment logs. Preserve the observed commands, artifacts, metrics, and interpretation corrections so later analysis can audit the full research path.
