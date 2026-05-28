# Chess Position Evaluator

A neural network chess position evaluator with Python training/evaluation tools,
Rust preprocessing, ONNX export, and experimental Rust inference support.

The current Python model evaluates a board from the side-to-move perspective.
Interactive tools convert that output into White and Black win probabilities for
display.

## Features

- Python CLI for training, FEN evaluation, play-against-AI, ONNX export, and
  activation-rank analysis.
- Rust preprocessing pipeline for converting the Lichess evaluation database
  into compact `.chesseval` datasets.
- Value-only CNN model with coordinate channels, Ghost shuffle blocks,
  coordinate attention, board self-attention, and an explicit board-material
  feature.
- WandB logging and optional checkpoint artifact uploads during training.
- Python and Rust inference paths for local experimentation.

## Requirements

- Python 3.12 or newer.
- `uv` for Python dependency and virtual environment management.
- Rust toolchain for preprocessing and Rust inference experiments.
- CUDA GPU recommended for training; CPU and Apple Silicon/MPS paths are
  supported where PyTorch supports them.

## Setup

```bash
git clone https://github.com/AcrylicShrimp/chess-position-evaluator.git
cd chess-position-evaluator
uv sync
```

The project uses a uv-managed virtual environment. You normally do not need to
activate it manually; use `uv run ...`.

```bash
uv run cpe --help
```

## CLI Usage

Model names are passed without the `.pth` extension. For example,
`<model-name>` resolves to:

```text
artifacts/checkpoints/<model-name>.pth
```

Evaluate FEN positions interactively:

```bash
uv run cpe eval <model-name>
```

Play against the AI:

```bash
uv run cpe battle <model-name>
```

Export a checkpoint to ONNX:

```bash
uv run cpe export-onnx <model-name>
```

Analyze activation rank:

```bash
uv run cpe analyze-rank <model-name>
```

`analyze-rank` requires `data/processed/validation.chesseval`.

Evaluate a checkpoint against a processed dataset split:

```bash
uv run cpe eval-dataset <model-name> --split validation
uv run cpe eval-dataset <model-name> --split test --full
```

Benchmark existing checkpoints for quality, parameter count, and forward
latency:

```bash
uv run cpe benchmark-pareto --output artifacts/reports/pareto.json
```

The CLI defaults `TORCHINDUCTOR_CACHE_DIR` to
`artifacts/cache/torchinductor` when the variable is not already set. This keeps
TorchInductor and Triton compile/autotune cache files project-local while still
allowing an explicit environment override.

## Data Preparation

Training data comes from the
[Lichess Evaluation Database](https://database.lichess.org/#evals).

Download the JSONL Zstandard archive and place or move it under `data/raw`.
The preprocessing code expects the decompressed file at:

```text
data/raw/lichess_db_eval.jsonl
```

Example:

```bash
mv <download-dir>/lichess_db_eval.jsonl.zst data/raw/lichess_db_eval.jsonl.zst
unzstd data/raw/lichess_db_eval.jsonl.zst
cargo run --release -p preprocess
```

The preprocessing command runs from the repository root and writes:

```text
data/processed/train.chesseval
data/processed/validation.chesseval
data/processed/test.chesseval
data/interim/lichess_db_eval.duckdb.tmp
```

The generated dataset and source JSONL files are intentionally ignored by Git.
If you need to rebuild the DuckDB staging table from a new JSONL file, remove
`data/interim/lichess_db_eval.duckdb.tmp` first.

## Training

Training requires WandB configuration. You can create a local `.env` from
`.env.example`:

```bash
cp .env.example .env
```

Set at least:

```text
WANDB_API_KEY=...
WANDB_PROJECT=chess-position-evaluator
```

Run training with explicit hyperparameters:

```bash
uv run cpe train my-experiment \
  --epochs 10 \
  --steps 1000 \
  --batch 256 \
  --lr 0.001 \
  --wd 0.0001
```

The default scheduler is cosine annealing with warm restarts. To use linear
warmup followed by cosine decay without restarts:

```bash
uv run cpe train my-experiment \
  --epochs 100 \
  --steps 1024 \
  --batch 2048 \
  --lr 0.0005 \
  --wd 0.0001 \
  --scheduler warmup-cosine \
  --warmup-epochs 5 \
  --eta-min 1e-6
```

The default model variant is `stacked-edge-gate-ffn`. Use `--model-variant
one-layer-edge-gate` for shallow dynamic edge-gated attention ablations, or
`--model-variant no-attention` for scheduler or capacity ablations against the
256-channel non-attention baseline. Use `--model-variant parallel-cnn-attn-fuse`
for the local/global branch experiment with a fixed material-prior logit plus a
positional residual head.

Useful options:

```bash
uv run cpe train --help
```

Checkpoints are written to:

```text
artifacts/checkpoints/<experiment-name>.pth
artifacts/checkpoints/<experiment-name>-best.pth
```

By default, best checkpoints are also uploaded to WandB artifacts. Disable that
with:

```bash
uv run cpe train my-experiment ... --no-upload-checkpoints
```

Resume an existing local checkpoint with:

```bash
uv run cpe train my-experiment ... --resume
```

## Model Architecture

The current value model is `ValueOnlyModel` in `python/libs/model.py`.

```text
Input board tensor
  -> Coordinate channels
  -> Convolutional trunk
  -> Ghost shuffle blocks with coordinate attention
  -> Multi-head naive board self-attention after the third trunk block
  -> ValueHead:
       - 1x1 conv path over trunk activations
       - explicit material feature from the original board tensor
       - MLP output logit
```

The `parallel-cnn-attn-fuse` variant instead splits after three shared ghost
shuffle blocks into a three-block CNN branch and a three-layer attention+FFN
branch, fuses them with a 1x1 projection, and outputs:

```text
value_logit = material_prior_logit + positional_residual_logit
```

Input channels:

- 5 metadata planes: side-to-move color and castling rights.
- 1 legal en-passant plane.
- 12 piece bitboard planes: our pieces then their pieces, ordered by
  pawn/knight/bishop/rook/queen/king.
- 2 attack-count heatmaps: our attacks and their attacks.

The board is encoded from the side-to-move perspective. For Black to move,
bitboards and heatmaps are vertically flipped so the model sees a consistent
orientation.

The model output is a raw logit for side-to-move win probability. Consumers apply
`sigmoid`; user-facing evaluation converts that side-to-move probability into
White and Black probabilities.

## Project Structure

```text
chess-position-evaluator/
├── python/
│   ├── cli.py                  # Typer CLI entrypoint
│   ├── benchmark_pareto.py     # Checkpoint Pareto benchmark
│   ├── eval.py                 # Interactive FEN evaluation
│   ├── export_onnx.py          # Checkpoint -> ONNX export
│   ├── analyze_rank.py         # Activation-rank analysis
│   ├── libs/
│   │   ├── model.py            # Model architecture
│   │   ├── encoding.py         # Board -> tensor conversion
│   │   ├── dataset.py          # .chesseval dataset reader
│   │   ├── scoring.py          # Model scoring helpers
│   │   └── movement.py         # Move encoding constants
│   ├── train/
│   │   ├── entry.py            # Training orchestration
│   │   └── trainer.py          # Training loop and WandB logging
│   ├── battle/
│   │   ├── entry.py            # Play-against-AI entrypoint
│   │   ├── negamax.py          # Search
│   │   └── compute_ordered_moves.py
│   └── tests/
├── crates/
│   ├── preprocess/             # Lichess JSONL -> .chesseval
│   ├── inference/              # ONNX Runtime inference experiment
│   ├── onnx-backend-benchmark/ # Rust ONNX benchmark
│   └── tree-search/            # Rust tree-search experiments
├── data/                       # Ignored runtime data directories
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/                  # Ignored runtime model artifacts
│   ├── cache/
│   ├── checkpoints/
│   ├── onnx/
│   └── reports/
├── docs/todos/
├── pyproject.toml
├── uv.lock
├── Cargo.toml
└── Cargo.lock
```

## Verification

Run Python tests:

```bash
uv run python -m unittest discover -s python/tests -v
```

Run a Python compile smoke check:

```bash
uv run python -m compileall -q python
```

Run Rust preprocessing or inference checks:

```bash
cargo run -p preprocess
cargo run -p inference
```

`cargo run -p preprocess` requires `data/raw/lichess_db_eval.jsonl`.
`cargo run -p inference` uses the configured ONNX artifact under
`artifacts/onnx`.

## Notes

- Older checkpoint and ONNX artifacts may use earlier architectures or input
  shapes and should be treated as historical unless re-verified.
- Dataset files, local WandB output, checkpoints, and downloaded compressed data
  are ignored by Git.
- See `data/README.md` and `artifacts/README.md` for the local runtime
  directory contracts.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Lichess](https://lichess.org/) for the open evaluation database.
- [python-chess](https://python-chess.readthedocs.io/) for chess logic.
