from chess_evaluation_reader import read_chess_evaluation, read_chess_evaluation_length
import torch


def worker_init_fn(_: int):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.open_file()


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.path = path
        self.file = None

        with open(path, "rb") as file:
            self.len = read_chess_evaluation_length(file)

        print(f"[✓] Using {self.len} rows total")

    def open_file(self):
        self.file = open(self.path, "rb")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return read_chess_evaluation(self.file, index)
