import numpy as np
import torch
from chess_evaluation_reader import read_chess_evaluation, read_chess_evaluation_length


def worker_init_fn(_: int):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.open_file()


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.path = path
        self.mm = None

        with open(path, "rb") as file:
            self.len = read_chess_evaluation_length(file)

    def open_file(self):
        self.mm = np.memmap(self.path, dtype=np.uint8, offset=8, mode="r")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.mm[index * 109 : (index + 1) * 109]
        return read_chess_evaluation(row.tobytes())
