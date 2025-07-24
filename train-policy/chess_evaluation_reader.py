import chess
import numpy as np
import struct
from typing import BinaryIO
import torch


def read_chess_evaluation_length(file: BinaryIO) -> int:
    file.seek(0)
    return struct.unpack("<Q", file.read(8))[0]


def read_chess_evaluation(
    row: np.ndarray,
) -> torch.Tensor:
    (bitflags, player_en_passant, white_attacks, black_attacks, *pieces) = (
        struct.unpack("<BQQQ12Q", row)
    )

    is_white_turn = bitflags & 1 == 1
    has_white_kingside_castling_rights = bitflags & 2 == 2
    has_white_queenside_castling_rights = bitflags & 4 == 4
    has_black_kingside_castling_rights = bitflags & 8 == 8
    has_black_queenside_castling_rights = bitflags & 16 == 16

    # 1. unpack the bitflags
    turn = boolean2tensor(is_white_turn)

    # 2. unpack the castling rights
    white_kingside_castling_rights = boolean2tensor(has_white_kingside_castling_rights)
    white_queenside_castling_rights = boolean2tensor(
        has_white_queenside_castling_rights
    )
    black_kingside_castling_rights = boolean2tensor(has_black_kingside_castling_rights)
    black_queenside_castling_rights = boolean2tensor(
        has_black_queenside_castling_rights
    )

    # 3. unpack the bitboards
    bitboards = bitboard2tensors(
        [
            player_en_passant,
            white_attacks,
            black_attacks,
            *pieces,
        ]
    )

    # 4. create the input tensor
    input = torch.vstack(
        [
            turn,
            white_kingside_castling_rights,
            white_queenside_castling_rights,
            black_kingside_castling_rights,
            black_queenside_castling_rights,
            bitboards,
        ]
    )

    # 5. create the board
    piece_map = {}

    piece_list = [
        (
            (square, chess.Piece(chess.PAWN, chess.WHITE))
            for square in chess.scan_reversed(pieces[0])
        ),
        (
            (square, chess.Piece(chess.KNIGHT, chess.WHITE))
            for square in chess.scan_reversed(pieces[1])
        ),
        (
            (square, chess.Piece(chess.BISHOP, chess.WHITE))
            for square in chess.scan_reversed(pieces[2])
        ),
        (
            (square, chess.Piece(chess.ROOK, chess.WHITE))
            for square in chess.scan_reversed(pieces[3])
        ),
        (
            (square, chess.Piece(chess.QUEEN, chess.WHITE))
            for square in chess.scan_reversed(pieces[4])
        ),
        (
            (square, chess.Piece(chess.KING, chess.WHITE))
            for square in chess.scan_reversed(pieces[5])
        ),
        (
            (square, chess.Piece(chess.PAWN, chess.BLACK))
            for square in chess.scan_reversed(pieces[6])
        ),
        (
            (square, chess.Piece(chess.KNIGHT, chess.BLACK))
            for square in chess.scan_reversed(pieces[7])
        ),
        (
            (square, chess.Piece(chess.BISHOP, chess.BLACK))
            for square in chess.scan_reversed(pieces[8])
        ),
        (
            (square, chess.Piece(chess.ROOK, chess.BLACK))
            for square in chess.scan_reversed(pieces[9])
        ),
        (
            (square, chess.Piece(chess.QUEEN, chess.BLACK))
            for square in chess.scan_reversed(pieces[10])
        ),
        (
            (square, chess.Piece(chess.KING, chess.BLACK))
            for square in chess.scan_reversed(pieces[11])
        ),
    ]

    for generator in piece_list:
        for square, piece in generator:
            piece_map[square] = piece

    board = chess.Board()
    board.set_piece_map(piece_map)
    board.turn = chess.WHITE if is_white_turn else chess.BLACK

    castling_fen = ""

    if has_white_kingside_castling_rights:
        castling_fen += "K"
    if has_white_queenside_castling_rights:
        castling_fen += "Q"
    if has_black_kingside_castling_rights:
        castling_fen += "k"
    if has_black_queenside_castling_rights:
        castling_fen += "q"

    if len(castling_fen) != 0:
        board.set_castling_fen(castling_fen)

    if player_en_passant != 0:
        board.ep_square = next(chess.scan_reversed(player_en_passant))

    return input, board


def boolean2tensor(boolean: bool) -> torch.Tensor:
    return torch.full((1, 8, 8), 1.0 if boolean else 0.0, dtype=torch.float32)


def bitboard2tensors(bitboards: list[int]) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array(bitboards, dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(len(bitboards), 8, 8).float()
