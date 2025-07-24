import chess
import numpy as np
import torch


def board2input(board: chess.Board) -> torch.Tensor:
    is_white_turn = board.turn == chess.WHITE
    has_white_kingside_castling_rights = board.has_kingside_castling_rights(chess.WHITE)
    has_white_queenside_castling_rights = board.has_queenside_castling_rights(
        chess.WHITE
    )
    has_black_kingside_castling_rights = board.has_kingside_castling_rights(chess.BLACK)
    has_black_queenside_castling_rights = board.has_queenside_castling_rights(
        chess.BLACK
    )
    player_en_passant = board.ep_square

    if player_en_passant is None:
        player_en_passant = 0

    white_attack_map, black_attack_map = build_attack_maps(board)
    piece_map = build_piece_map(board)

    return torch.vstack(
        [
            boolean2tensor(is_white_turn),
            boolean2tensor(has_white_kingside_castling_rights),
            boolean2tensor(has_white_queenside_castling_rights),
            boolean2tensor(has_black_kingside_castling_rights),
            boolean2tensor(has_black_queenside_castling_rights),
            bitboard2tensors([player_en_passant]),
            white_attack_map,
            black_attack_map,
            *piece_map,
        ],
    )


def boolean2tensor(boolean: bool) -> torch.Tensor:
    return torch.full((1, 8, 8), 1.0 if boolean else 0.0, dtype=torch.float32)


def bitboard2tensors(bitboards: list[int]) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array(bitboards, dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(len(bitboards), 8, 8).float()


def build_attack_maps(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    white_attack_map = torch.zeros(64, dtype=torch.float32)
    black_attack_map = torch.zeros(64, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            attack_map = white_attack_map
        else:
            attack_map = black_attack_map

        for attack_square in board.attacks(square):
            attack_map[attack_square] = 1

    return (white_attack_map, black_attack_map)


def build_piece_map(board: chess.Board) -> list[torch.Tensor]:
    piece_map = [torch.zeros(64, dtype=torch.float32) for _ in range(12)]

    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            piece_map[piece.piece_type - 1][square] = 1
        else:
            piece_map[piece.piece_type + 5][square] = 1

    return piece_map
