import chess
import duckdb
import fen2tensor
import torch


class TrainData(torch.utils.data.Dataset):
    def __init__(self, lichess_db_eval_path: str, percentage: float = 0.1):
        self.path = lichess_db_eval_path
        self.percentage = percentage
        self.len = 0
        self.inputs = None
        self.outputs = None

    def reload(self):
        with duckdb.connect() as conn:
            raw_data = conn.sql(
                f"""
                SELECT fen, cp, mate
                FROM parquet_scan($path)
                USING SAMPLE {self.percentage * 100}%
                """,
                params={
                    "path": self.path,
                },
            ).fetchall()

        self.len = len(raw_data)
        print(f"[✓] Loaded {self.len} rows")
        self.inputs = torch.vstack(
            [fen2tensor.fen2tensor(fen).unsqueeze(0) for fen, _, _ in raw_data]
        ).share_memory_()
        print(f"[✓] Loaded {self.inputs.shape} inputs")
        self.outputs = torch.vstack(
            [encode_output(fen, cp, mate).unsqueeze(0) for fen, cp, mate in raw_data]
        ).share_memory_()
        print(f"[✓] Loaded {self.outputs.shape} outputs")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.outputs[index]


def encode_output(fen: str, cp: float, mate: None | float) -> torch.Tensor:
    game = chess.Board(fen)

    if mate is None:
        return torch.tensor([cp * 0.01, 0], dtype=torch.float32)
    elif game.turn == chess.WHITE:
        return torch.tensor([20, 1], dtype=torch.float32)
    else:
        return torch.tensor([-20, 2], dtype=torch.float32)
