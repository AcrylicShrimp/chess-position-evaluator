from pathlib import Path

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = Path("artifacts")
CACHE_DIR = ARTIFACTS_DIR / "cache"
INDUCTOR_CACHE_DIR = CACHE_DIR / "torchinductor"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

RAW_EVALUATIONS_PATH = RAW_DATA_DIR / "lichess_db_eval.jsonl"
DUCKDB_TEMP_PATH = INTERIM_DATA_DIR / "lichess_db_eval.duckdb.tmp"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.chesseval"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation.chesseval"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.chesseval"


def checkpoint_path(model_name: str) -> Path:
    return CHECKPOINTS_DIR / f"{model_name}.pth"


def onnx_path(model_name: str) -> Path:
    return ONNX_DIR / f"{model_name}.onnx"


def evaluation_report_path(model_name: str, split: str) -> Path:
    return REPORTS_DIR / f"{model_name}.{split}.eval.json"
