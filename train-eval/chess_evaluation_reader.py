import numpy as np
import struct
from typing import BinaryIO
import torch


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
    bitboards = bitboard2tensors(
        [
            player_en_passant,
            *pieces,
        ]
    )

    # 4. create the input tensor
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

    # 5. create the label tensor
    label = torch.tensor([win_prob], dtype=torch.float32)

    return input, label


def boolean2tensor(boolean: bool) -> torch.Tensor:
    return torch.full((1, 8, 8), 1.0 if boolean else 0.0, dtype=torch.float32)


def bitboard2tensors(bitboards: list[int]) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array(bitboards, dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(len(bitboards), 8, 8).float()
