import chess
import time
import duckdb
import fen2tensor
import torch


def worker_init_fn(_: int):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.connect()


class TrainData(torch.utils.data.Dataset):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

        with duckdb.connect(self.db_path, read_only=True) as conn:
            row_count = conn.sql(
                f"""
                SELECT COUNT(*) FROM train_rows
                """,
            ).fetchone()[0]

        self.len = row_count
        print(f"[✓] Using {self.len} rows total")

    def connect(self):
        self.conn = duckdb.connect(self.db_path, read_only=True)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        fen, cp, mate = self.conn.sql(
            f"""
            SELECT fen, cp, mate
            FROM train_rows
            WHERE row_idx = $row_index
            """,
            params={
                "row_index": index + 1,
            },
        ).fetchone()

        return fen2tensor.fen2tensor(fen), encode_output(fen, cp, mate)


def encode_output(fen: str, cp: float, mate: None | float) -> torch.Tensor:
    game = chess.Board(fen)

    if mate is None:
        return torch.tensor([cp * 0.01, 0], dtype=torch.float32)
    elif game.turn == chess.WHITE:
        return torch.tensor([20, 1], dtype=torch.float32)
    else:
        return torch.tensor([-20, 2], dtype=torch.float32)
