# Artifacts Directory

Runtime model artifacts live under this directory and are intentionally ignored
by Git.

## Layout

```text
artifacts/
├── checkpoints/  # Local training checkpoints
├── onnx/         # Local ONNX exports
└── reports/      # Local evaluation reports
```

Expected local files:

- `artifacts/checkpoints/<experiment-name>.pth`
- `artifacts/checkpoints/<experiment-name>-best.pth`
- `artifacts/onnx/<model-name>.onnx`
- optional ONNX external data files such as `artifacts/onnx/<model-name>.onnx.data`
- `artifacts/reports/<model-name>.<split>.eval.json`

Use WandB artifacts, release assets, or another explicit external artifact store
for model files that need to be shared.

Only `README.md` and `.gitkeep` files should be committed from this directory.
