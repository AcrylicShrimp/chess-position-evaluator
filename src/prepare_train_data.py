import chess
import duckdb
import fen2tensor
import torch


class TrainData(torch.utils.data.Dataset):
    def __init__(self, lichess_db_eval_path: str, percentage: float = 0.1):
        self.path = lichess_db_eval_path
        self.conn = duckdb.connect()
        self.percentage = percentage
        self.data = []

    def reload(self):
        self.data = self.conn.sql(
            f"""
            SELECT fen, cp, mate
            FROM parquet_scan($path)
            USING SAMPLE {self.percentage * 100}%
            """,
            params={
                "path": self.path,
            },
        ).fetchall()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        fen, cp, mate = self.data[index]
        return fen2tensor.fen2tensor(fen), encode_output(fen, cp, mate)


def encode_output(fen: str, cp: float, mate: None | float) -> torch.Tensor:
    game = chess.Board(fen)

    if mate is None:
        return torch.tensor([cp * 0.01, 0], dtype=torch.float32)
    elif game.turn == chess.WHITE:
        return torch.tensor([20, 1], dtype=torch.float32)
    else:
        return torch.tensor([-20, 2], dtype=torch.float32)
