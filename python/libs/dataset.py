import numpy as np
import struct
import torch
from typing import BinaryIO

from libs.encoding import bitboards2tensors, boolean2tensor

CHESS_EVALUATION_ROW_FORMAT = "<BQ12Q64Bf"
CHESS_EVALUATION_ROW_SIZE = struct.calcsize(CHESS_EVALUATION_ROW_FORMAT)


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
        if self.mm is None:
            self.open_file()

        row_start = index * CHESS_EVALUATION_ROW_SIZE
        row_end = row_start + CHESS_EVALUATION_ROW_SIZE
        row = self.mm[row_start:row_end]
        return read_chess_evaluation(row.tobytes())


def read_chess_evaluation_length(file: BinaryIO) -> int:
    file.seek(0)
    return struct.unpack("<Q", file.read(8))[0]


def read_chess_evaluation(
    row: bytes,
) -> tuple[torch.Tensor, torch.Tensor]:
    unpacked = struct.unpack(CHESS_EVALUATION_ROW_FORMAT, row)
    bitflags = unpacked[0]
    player_en_passant = unpacked[1]
    pieces = unpacked[2:14]  # 12 Q-words
    heatmap = unpacked[14:78]  # 64 bytes
    win_prob = unpacked[78]

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

    packed_heatmap = torch.tensor(heatmap, dtype=torch.uint8)
    our_heatmap = (packed_heatmap & 0x0F).view(1, 8, 8).float() / 15.0
    their_heatmap = (packed_heatmap >> 4).view(1, 8, 8).float() / 15.0

    input = torch.vstack(
        [
            am_i_black_color,
            our_kingside_castling_rights,
            our_queenside_castling_rights,
            their_kingside_castling_rights,
            their_queenside_castling_rights,
            bitboards,
            our_heatmap,
            their_heatmap,
        ]
    )
    label = torch.tensor([win_prob], dtype=torch.float32)

    return input, label
