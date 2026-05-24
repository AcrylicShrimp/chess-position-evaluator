# Cleanup Action Items

Scanned on 2026-05-25. This document tracks concrete cleanup work found during
the initial project review. Treat these items as a living checklist.

## Priority Guide

- P0: Breaks a primary workflow or can produce clearly wrong chess/eval behavior.
- P1: Blocks maintainability, packaging, reproducibility, or common developer use.
- P2: Nice-to-have cleanup that reduces confusion or future drift.

## P0: Restore Model Artifact Compatibility

- [x] Decide the canonical model input format: current 20-channel Python encoder
      or older 18-channel artifacts.
- [x] Regenerate or replace the default ONNX artifact so it matches the current
      Python model input shape.
- [x] Update Rust inference and benchmark constants to match the chosen input
      shape.
- [ ] Make Python gameplay/default checkpoint paths point at a checkpoint
      compatible with `ValueOnlyModel`.

Decision:

- The current 20-channel Python encoder is canonical.
- The compatible default model artifact is
  `ghost-ca-r4-256ch-6blk-best`.

Evidence:

- `python/libs/encoding.py` emits 20 channels.
- `python/libs/model.py` expects 20 board channels, then `AddCoords` expands to
  24 channels for the first convolution.
- `models/onnx/small-0.6337-model-best.onnx` currently declares input
  `[batch, 18, 8, 8]`.
- `models/onnx/ghost-ca-r4-256ch-6blk-best.onnx` declares input
  `[batch, 20, 8, 8]`.
- `cargo run -p inference` now uses the 20-channel ghost ONNX model.

Acceptance checks:

- `cargo run -p inference`
- ONNX graph input shape matches the selected board tensor shape.
- A smoke load of `models/checkpoints/ghost-ca-r4-256ch-6blk-best.pth` into
  `ValueOnlyModel` succeeds.

## P0: Fix Evaluation Perspective

- [x] Ensure interactive eval reports White win probability, Black win
      probability, or side-to-move win probability explicitly.
- [x] Reuse `absolute_win_prob` or equivalent logic instead of interpreting raw
      side-to-move sigmoid output as White's probability.
- [x] Add at least one test or smoke case for a Black-to-move FEN.

Evidence:

- Training labels are written from the side-to-move perspective.
- `python/libs/scoring.py` converts side-to-move output to absolute White
  probability.
- `python/eval.py` currently prints and classifies raw sigmoid output directly.

Acceptance checks:

- `cpe eval <model>` has clear output labels.
- Black-to-move positions no longer invert the explanation.
- `python/tests/test_eval_and_search.py` covers side-to-move conversion and
  absolute White-perspective descriptions.

## P0: Fix Search Terminal Scoring

- [x] Make `negamax` distinguish checkmate, stalemate, and draw outcomes.
- [x] Use `MATE_SCORE` for forced wins/losses instead of returning
      `DRAW_SCORE` for every completed game.
- [x] Add small tests around mate-in-one and stalemate positions.

Evidence:

- `python/battle/negamax.py` returns `DRAW_SCORE` whenever
  `board.outcome(claim_draw=True)` is not `None`, including checkmate.
- `MATE_SCORE` exists but is not used in terminal scoring.

Acceptance checks:

- A mate-in-one move is preferred over neutral moves.
- Stalemate/draw remains scored as neutral.
- `python/tests/test_eval_and_search.py` covers checkmate, stalemate, and a
  mate-in-one search case.

## P1: Repair CLI Packaging

- [ ] Configure Python package discovery explicitly so editable installs work.
- [ ] Make `uv run cpe --help` work from a fresh checkout.
- [ ] Decide whether command usage should be `cpe ...`,
      `.venv/bin/python python/cli.py ...`, or both.
- [ ] Consider renaming the top-level `python` package to avoid confusion with
      the standard `python` executable and packaging conventions.

Evidence:

- `.venv/bin/python python/cli.py --help` works.
- `uv run cpe --help` does not find the script in the current environment.
- `uv run --with-editable . cpe --help` fails because setuptools discovers
  multiple top-level directories in the flat layout.

Acceptance checks:

- Fresh setup can run `cpe --help`.
- CLI imports do not require manual `PYTHONPATH` tricks.

## P1: Fix Battle Entrypoint Configuration

- [ ] Use `os.environ.get("BEST_CHECKPOINT_PATH")` correctly.
- [ ] Default to `models/checkpoints/<compatible-model>.pth` instead of
      `model-best.pth` in the repository root.
- [ ] Route battle mode through the unified Typer CLI or document it clearly as
      a standalone script.

Evidence:

- `python/battle.py` checks `os.path.exists("BEST_CHECKPOINT_PATH")`, which
  tests a literal filename rather than the environment variable.
- The default checkpoint path does not match the committed model directory.

Acceptance checks:

- Battle mode starts with the documented default checkpoint.
- Setting `BEST_CHECKPOINT_PATH` overrides the default.

## P1: Update Analyze-Rank for Current Model

- [ ] Replace `model.residual_blocks` references with the current `model.blocks`
      structure.
- [ ] Hook the current value head modules by name instead of old sequential
      indices.
- [ ] Add a smoke command that fails fast when the validation dataset is absent.

Evidence:

- `python/analyze_rank.py` references `model.residual_blocks`, which does not
  exist on the current `ValueOnlyModel`.
- The value head is now an object with `conv` and `mlp`, not a flat sequential
  module.

Acceptance checks:

- `cpe analyze-rank <compatible-model>` reaches the dataset check and model hook
  setup without `AttributeError`.

## P1: Clarify Material Feature Semantics

- [ ] Decide whether the material feature should be computed from the original
      board tensor or from learned trunk activations.
- [ ] If board-material is intended, pass the original input tensor or a
      precomputed `material_diff` into `ValueHead`.
- [ ] Rename comments and arguments if the current learned-feature behavior is
      intentional.

Evidence:

- `_material_feature` documents an input shaped `[B, 20, 8, 8]`.
- `ValueOnlyModel.forward` currently passes trunk output shaped
  `[B, CHANNELS, 8, 8]` into `ValueHead`.

Acceptance checks:

- Comments match implementation.
- A small unit test verifies material-diff behavior for a known board tensor.

## P1: Refresh README

- [ ] Replace stale `src/` paths with the current `python/` and `crates/`
      layout.
- [ ] Update quick-start commands to match the chosen CLI/package setup.
- [ ] Update model architecture notes from the older 18-channel residual/SE
      description to the current 20-channel Ghost/Coordinate Attention model, or
      document the older artifacts separately.
- [ ] Replace TensorBoard references with WandB references if WandB remains the
      primary logger.

Evidence:

- `README.md` still documents `python src/eval.py`, `python src/battle.py`,
  `python src/train_eval.py`, and `preprocess/` paths.
- Current code uses `python/`, `crates/preprocess`, Typer, and WandB.

Acceptance checks:

- A new contributor can follow README commands from a clean checkout.

## P1: Add Minimal Regression Tests

- [ ] Test board encoding shape and key channel ordering.
- [ ] Test dataset row decode shape and label dtype.
- [ ] Test checkpoint compatibility for the documented default model.
- [ ] Test ONNX input shape for the documented default ONNX artifact.
- [x] Test terminal scoring in `negamax`.
- [ ] Add one CLI smoke test for `--help`.

Acceptance checks:

- A single documented test command covers Python smoke tests.
- `cargo test` remains green.

## P2: Clean Rust Tree-Search Draft Code

- [ ] Decide whether `crates/tree-search` is active engine work or experimental
      scratch code.
- [ ] If active, implement real quiescence semantics and add SEE tests.
- [ ] If experimental, mark it clearly in docs.

Evidence:

- `is_quiescent` currently returns `false` for ongoing positions both with and
  without check, so it is not yet a useful predicate.
- No other crate currently consumes `tree-search`.

Acceptance checks:

- The crate either has meaningful tests or is documented as experimental.

## P2: Repository Hygiene

- [ ] Review whether generated artifacts should stay committed directly or move
      to Git LFS / release assets / WandB artifacts.
- [ ] Expand `.gitattributes` if large model artifacts remain in git.
- [ ] Keep `target/`, `.venv/`, and generated datasets out of git.

Evidence:

- `.gitattributes` only configures `eval_model.pth` for LFS.
- Several `.pth`, `.onnx`, and `.onnx.data` artifacts are committed directly.

Acceptance checks:

- Fresh clone size and artifact policy are intentional and documented.

## Current Known-Good Commands

```bash
cargo test
cargo run -p inference
.venv/bin/python -m compileall -q python
.venv/bin/python -m unittest discover -s python/tests -v
.venv/bin/python python/cli.py --help
.venv/bin/python -c "import sys, torch; sys.path.insert(0, 'python'); from libs.model import ValueOnlyModel; print(ValueOnlyModel()(torch.randn(2, 20, 8, 8)).shape)"
```

## Current Known-Broken Commands

```bash
uv run cpe --help
uv run --with-editable . cpe --help
```
