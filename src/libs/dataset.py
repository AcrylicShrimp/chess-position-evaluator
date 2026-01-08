import numpy as np
import struct
import torch
from typing import BinaryIO

from libs.encoding import bitboards2tensors, boolean2tensor


class ChessEvaluationDataset(torch.utils.data.Dataset):
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


def read_chess_evaluation_length(file: BinaryIO) -> int:
    file.seek(0)
    return struct.unpack("<Q", file.read(8))[0]


def read_chess_evaluation(
    row: bytes,
) -> tuple[torch.Tensor, torch.Tensor]:
    (bitflags, player_en_passant, *pieces, win_prob) = struct.unpack("<BQ12Qf", row)

    am_i_black_color = boolean2tensor(bitflags & 1 == 1)
    our_kingside_castling_rights = boolean2tensor(bitflags & 2 == 2)
    our_queenside_castling_rights = boolean2tensor(bitflags & 4 == 4)
    their_kingside_castling_rights = boolean2tensor(bitflags & 8 == 8)
    their_queenside_castling_rights = boolean2tensor(bitflags & 16 == 16)
    bitboards = bitboards2tensors(
        [
            player_en_passant,
            *pieces,
        ]
    )

    input = torch.vstack(
        [
            am_i_black_color,
            our_kingside_castling_rights,
            our_queenside_castling_rights,
            their_kingside_castling_rights,
            their_queenside_castling_rights,
            bitboards,
        ]
    )
    label = torch.tensor([win_prob], dtype=torch.float32)

    return input, label
