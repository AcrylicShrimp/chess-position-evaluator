# Chess Position Evaluator

A neural network-based chess position evaluator with a complete end-to-end training pipeline. This project demonstrates building a compact yet powerful chess engine from raw data to a playable AI.

## Motivation

This project was built to:

- Create a **strong but efficient** chess engine with a small model footprint (~27MB)
- Experience and learn the **entire deep learning pipeline** from data preprocessing to deployment
- Achieve strong playing strength even with minimal search (depth 4, branch factor 10, negamax)

## Features

- **Neural Network Evaluation**: CNN with residual blocks and squeeze-excitation for position scoring
- **Complete Training Pipeline**: From raw Lichess data to trained model
- **High-Performance Preprocessing**: Rust-based data pipeline for efficient processing
- **Training Infrastructure**: Mixed precision training, checkpointing, TensorBoard logging
- **Interactive Modes**: Position evaluation REPL and play-against-AI mode
- **Multi-Device Support**: CUDA, Apple Silicon (MPS), and CPU

## Model Architecture

The model uses a compact CNN architecture inspired by AlphaZero:

```
Input (18 channels × 8 × 8)
    ↓
Initial Conv Block (128 filters)
    ↓
6× Residual Blocks with Squeeze-Excitation
    ↓
Value Head → Win Probability (0-1)
```

**Input Representation** (18 channels):

- Side to move indicator
- Castling rights (4 channels: our/their kingside/queenside)
- En passant square
- Piece positions (12 channels: 6 piece types × 2 colors)

**Output**: Win probability from the current player's perspective

<details>
<summary>Architecture Details</summary>

- **Residual Block**: Conv3×3 → BN → ReLU → Conv3×3 → BN → SE → Skip Connection → ReLU
- **Squeeze-Excitation**: Global average pooling → FC(128→16) → ReLU → FC(16→128) → Sigmoid → Scale
- **Value Head**: Conv1×1(32) → BN → ReLU → Flatten → FC(2048→256) → ReLU → FC(256→1)

</details>

## Dataset

Training data comes from the [Lichess Evaluation Database](https://database.lichess.org/#evals), which contains millions of positions evaluated by Stockfish.

**Preprocessing pipeline**:

1. Filter evaluations with depth ≥ 24 (high-quality analysis)
2. Balance dataset: include all "nuanced" positions (cp between -150 and 150), sample equal amounts of decisive positions
3. Convert centipawns to win probability: `win_prob = 1 / (1 + 10^(-cp/400))`
4. Output to custom binary format (`.chesseval`) for fast loading

## Getting Started

### Prerequisites

- Python ≥ 3.12
- CUDA GPU (recommended), Apple Silicon, or CPU
- Rust toolchain (for preprocessing only)

### Installation

```bash
# Clone the repository
git clone https://github.com/AcrylicShrimp/chess-position-evaluator.git
cd chess-position-evaluator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**Evaluate a position:**

```bash
python src/eval.py
# Enter FEN strings to get win probability assessments
```

**Play against the AI:**

```bash
python src/battle.py
# You'll be assigned a random color and can play using algebraic notation (e.g., e4, Nf3)
```

## Training from Scratch

### 1. Download Data

Download the Lichess evaluation database from [https://database.lichess.org/#evals](https://database.lichess.org/#evals).

Extract the JSONL file and place it as `lichess_db_eval.jsonl` in the `preprocess/` directory.

### 2. Preprocess Data

```bash
cd preprocess
cargo build --release
./target/release/preprocess
```

This will generate:

- `train.chesseval` - Training set (~90% of data)
- `validation.chesseval` - Validation set (~10% of data)

Move these files to the project root directory.

### 3. Train the Model

```bash
python src/train_eval.py
```

Training configuration:

- Batch size: 8,192
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau
- Mixed precision: bfloat16 on CUDA

Checkpoints are saved to:

- `model.pth` - Latest checkpoint
- `model-best.pth` - Best validation loss

### 4. Monitor Training

```bash
tensorboard --logdir tensorboard/
```

## Benchmarks

_Coming soon: Performance benchmarks against various opponents_

| Opponent                | Win Rate | Notes |
| ----------------------- | -------- | ----- |
| Random Agent            | TBD      |       |
| Stockfish (depth 1)     | TBD      |       |
| Stockfish (ELO limited) | TBD      |       |

## Project Structure

```
chess-position-evaluator/
├── src/
│   ├── libs/              # Core library modules
│   │   ├── model.py       # Neural network architecture
│   │   ├── encoding.py    # Board → tensor conversion
│   │   ├── dataset.py     # Data loading
│   │   ├── scoring.py     # Position evaluation
│   │   └── movement.py    # Move encoding (for future policy head)
│   ├── battle/            # Game-playing components
│   │   ├── negamax.py     # Search algorithm
│   │   └── compute_ordered_moves.py  # Move ordering
│   ├── train_eval/        # Training pipeline
│   │   └── trainer.py     # Trainer class
│   ├── train_eval.py      # Training entry point
│   ├── eval.py            # Interactive evaluation
│   └── battle.py          # Play against AI
├── preprocess/            # Rust data preprocessing
│   ├── src/
│   │   ├── main.rs        # Pipeline orchestration
│   │   └── write_chesseval.rs  # Binary format writer
│   └── Cargo.toml
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Add policy head for move prediction
- [ ] Implement MCTS for stronger play
- [ ] Add formal benchmark suite
- [ ] Optimize search with transposition tables

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Lichess](https://lichess.org/) for the open evaluation database
- [python-chess](https://python-chess.readthedocs.io/) for chess logic
- AlphaZero paper for architectural inspiration
