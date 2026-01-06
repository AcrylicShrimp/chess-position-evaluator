import chess
import numpy as np
import torch


def boolean2tensor(boolean: bool) -> torch.Tensor:
    return torch.full((1, 8, 8), 1.0 if boolean else 0.0, dtype=torch.float32)


def bitboards2tensors(bitboards: list[int]) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array(bitboards, dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(len(bitboards), 8, 8).float()


def board2tensor(board: chess.Board) -> torch.Tensor:
    us = board.turn
    them = not us

    am_i_black_color = us == chess.BLACK
    our_kingside_castling_rights = board.has_kingside_castling_rights(us)
    our_queenside_castling_rights = board.has_queenside_castling_rights(us)
    their_kingside_castling_rights = board.has_kingside_castling_rights(them)
    their_queenside_castling_rights = board.has_queenside_castling_rights(them)

    am_i_black_color_tensor = boolean2tensor(am_i_black_color)
    our_kingside_castling_rights_tensor = boolean2tensor(our_kingside_castling_rights)
    our_queenside_castling_rights_tensor = boolean2tensor(our_queenside_castling_rights)
    their_kingside_castling_rights_tensor = boolean2tensor(
        their_kingside_castling_rights
    )
    their_queenside_castling_rights_tensor = boolean2tensor(
        their_queenside_castling_rights
    )

    en_passant_bb: list[chess.Bitboard] = (
        (1 << board.ep_square) if board.has_legal_en_passant() else 0
    )

    pieces_bb = [
        board.pieces_mask(piece_type, color)
        for color in [us, them]
        for piece_type in chess.PIECE_TYPES
    ]

    all_bitboards: list[chess.Bitboard] = [en_passant_bb, *pieces_bb]

    if am_i_black_color:
        all_bitboards = [chess.flip_vertical(x) for x in all_bitboards]

    bitboard_tensors = bitboards2tensors(all_bitboards)

    input_tensor = torch.vstack(
        [
            am_i_black_color_tensor,
            our_kingside_castling_rights_tensor,
            our_queenside_castling_rights_tensor,
            their_kingside_castling_rights_tensor,
            their_queenside_castling_rights_tensor,
            bitboard_tensors,
        ]
    )

    return input_tensor


def fen2tensor(fen: str) -> torch.Tensor:
    return board2tensor(chess.Board(fen))
