import chess
import numpy as np
import torch
from model import Model


def boolean2tensor(boolean: bool) -> torch.Tensor:
    return torch.full((1, 8, 8), 1.0 if boolean else 0.0, dtype=torch.float32)


def bitboard2tensors(bitboards: list[int]) -> torch.Tensor:
    unpacked = np.unpackbits(
        np.array(bitboards, dtype="<u8").view(np.uint8), bitorder="little"
    )
    return torch.from_numpy(unpacked).view(len(bitboards), 8, 8).float()


def get_attack_mask(board: chess.Board, color: chess.Color) -> int:
    attack_mask = 0

    for square in chess.scan_reversed(board.occupied_co[color]):
        attack_mask |= board.attacks_mask(square)

    return attack_mask


def compute_player_en_passant(board: chess.Board) -> torch.Tensor:
    for move in board.legal_moves:
        if board.is_en_passant(move):
            return bitboard2tensors([1 << move.from_square])

    return boolean2tensor(False)


def compute_white_attacks(board: chess.Board) -> chess.Bitboard:
    squares = chess.SquareSet()

    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            squares.add(square)

    return squares.mask


def compute_black_attacks(board: chess.Board) -> chess.Bitboard:
    squares = chess.SquareSet()

    for square in chess.SQUARES:
        if board.is_attacked_by(chess.BLACK, square):
            squares.add(square)

    return squares.mask


def board2tensor(board: chess.Board) -> torch.Tensor:
    turn_tensor = boolean2tensor(board.turn == chess.WHITE)
    wk_castle_tensor = boolean2tensor(
        board.has_kingside_castling_rights(chess.WHITE))
    wq_castle_tensor = boolean2tensor(
        board.has_queenside_castling_rights(chess.WHITE))
    bk_castle_tensor = boolean2tensor(
        board.has_kingside_castling_rights(chess.BLACK))
    bq_castle_tensor = boolean2tensor(
        board.has_queenside_castling_rights(chess.BLACK))

    en_passant_bb = (
        1 << board.ep_square) if board.ep_square is not None else 0
    white_attacks_bb = compute_white_attacks(board)
    black_attacks_bb = compute_black_attacks(board)
    pieces_bb = [
        board.pieces_mask(piece_type, color)
        for color in [chess.WHITE, chess.BLACK]
        for piece_type in chess.PIECE_TYPES
    ]

    all_bitboards = [en_passant_bb, white_attacks_bb,
                     black_attacks_bb, *pieces_bb]
    bitboard_tensors = bitboard2tensors(all_bitboards)

    input_tensor = torch.cat(
        [
            turn_tensor,
            wk_castle_tensor,
            wq_castle_tensor,
            bk_castle_tensor,
            bq_castle_tensor,
            bitboard_tensors,
        ],
        dim=0,
    )

    return input_tensor


def board2score(board: chess.Board, model: Model, device: torch.device) -> float:
    input_tensor = board2tensor(board)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        score = output[0, 0].cpu().sigmoid().item()
        return override_score_with_status(score, board)


def boards2scores(
    boards: list[chess.Board], model: Model, device: torch.device
) -> list[float]:
    input_tensors = torch.stack([board2tensor(board) for board in boards])
    input_tensors = input_tensors.to(device)

    with torch.no_grad():
        outputs = model(input_tensors)
        scores = outputs[:, 0].cpu().sigmoid().tolist()
        return [override_score_with_status(score, board) for score, board in zip(scores, boards)]


def override_score_with_status(score: float, board: chess.Board) -> float:
    if board.is_checkmate():
        return float("inf") if board.turn == chess.WHITE else float("-inf")
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0

    return score
