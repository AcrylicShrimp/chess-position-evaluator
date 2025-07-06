import numpy as np
import struct
from typing import BinaryIO
import torch


def read_chess_evaluation_length(file: BinaryIO) -> int:
    file.seek(0)
    return struct.unpack("<Q", file.read(8))[0]


def read_chess_evaluation(
    row: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    (bitflags, player_en_passant, white_attacks, black_attacks, *pieces, cp, mate) = (
        struct.unpack("<BQQQ12QfB", row)
    )

    is_white_turn = bitflags & 1 == 1
    has_white_kingside_castling_rights = bitflags & 2 == 2
    has_white_queenside_castling_rights = bitflags & 4 == 4
    has_black_kingside_castling_rights = bitflags & 8 == 8
    has_black_queenside_castling_rights = bitflags & 16 == 16

    turn = boolean2tensor(is_white_turn)
    white_kingside_castling_rights = boolean2tensor(has_white_kingside_castling_rights)
    white_queenside_castling_rights = boolean2tensor(
        has_white_queenside_castling_rights
    )
    black_kingside_castling_rights = boolean2tensor(has_black_kingside_castling_rights)
    black_queenside_castling_rights = boolean2tensor(
        has_black_queenside_castling_rights
    )

    player_en_passant_squares = bitboard2tensor(player_en_passant)
    white_attacks = bitboard2tensor(white_attacks)
    black_attacks = bitboard2tensor(black_attacks)
    white_pieces = bitboard2tensor_pieces(pieces[:6])
    black_pieces = bitboard2tensor_pieces(pieces[6:])

    # 4. create the input tensor
    input = torch.vstack(
        [
            turn,
            white_kingside_castling_rights,
            white_queenside_castling_rights,
            black_kingside_castling_rights,
            black_queenside_castling_rights,
            player_en_passant_squares,
            white_attacks,
            black_attacks,
            white_pieces,
            black_pieces,
        ]
    )

    # 5. create the label tensor
    label = torch.tensor([cp, mate], dtype=torch.float32)

    return input, label


def boolean2tensor(boolean: bool) -> torch.Tensor:
    if boolean:
        return torch.ones((1, 8, 8), dtype=torch.float32)
    else:
        return torch.zeros((1, 8, 8), dtype=torch.float32)


def bitboard2tensor(bitboard: int) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array([bitboard], dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(1, 8, 8).float()


def bitboard2tensor_pieces(bitboards: list[int]) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array(bitboards, dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(6, 8, 8).float()
