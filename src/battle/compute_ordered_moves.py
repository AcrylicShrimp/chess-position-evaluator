import chess
import torch
from battle.board2score import boards2scores
from model import Model


def compute_ordered_moves(
    board: chess.Board,
    model: Model,
    device: torch.device,
) -> list[chess.Move]:
    boards: list[chess.Board] = []
    moves: list[chess.Move] = []

    for move in board.legal_moves:
        next_board = board.copy()
        next_board.push(move)
        boards.append(next_board)
        moves.append(move)

    scores = boards2scores(boards, model, device)
    sorted_moves = sorted(
        zip(moves, scores),
        key=lambda move: move[1],
        reverse=board.turn == chess.WHITE,
    )
    return [move for move, _ in sorted_moves][:10]
