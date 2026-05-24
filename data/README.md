# Data Directory

Runtime data lives under this directory and is intentionally ignored by Git.

## Layout

```text
data/
├── raw/        # Downloaded source files, such as lichess_db_eval.jsonl
├── interim/    # Local staging files, such as lichess_db_eval.duckdb.tmp
└── processed/  # Generated training files, such as train.chesseval
```

Expected local files:

- `data/raw/lichess_db_eval.jsonl`
- `data/interim/lichess_db_eval.duckdb.tmp`
- `data/processed/train.chesseval`
- `data/processed/validation.chesseval`

Only `README.md` and `.gitkeep` files should be committed from this directory.
