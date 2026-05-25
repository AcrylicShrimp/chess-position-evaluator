# Experiment Logging Notes

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
