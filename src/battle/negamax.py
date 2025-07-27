import chess
import random
import torch
from battle.board2score import board2score
from battle.compute_ordered_moves import compute_ordered_moves
from model import Model


def negamax(
    board: chess.Board,
    model: Model,
    device: torch.device,
    depth: int,
    alpha: float,
    beta: float,
    color: chess.Color,
) -> float:
    outcome = board.outcome(claim_draw=True)

    if outcome is not None:
        winner = outcome.winner

        if winner is None:
            return 0
        else:
            return float("inf") if winner == color else float("-inf")

    if depth == 0:
        score = board2score(board, model, device)
        score = score if color == chess.WHITE else -score
        return score

    moves = compute_ordered_moves(board, model, device)
    value = float("-inf")

    for move in moves:
        next_board = board.copy()
        next_board.push(move)

        score = -negamax(
            next_board,
            model,
            device,
            depth - 1,
            -beta,
            -alpha,
            not color,
        )

        value = max(value, score)
        alpha = max(alpha, score)

        if beta <= alpha:
            break

    return value


def find_best_move(
    board: chess.Board, model: Model, device: torch.device, depth: int
) -> chess.Move | None:
    if board.is_game_over(claim_draw=True):
        return None

    moves = compute_ordered_moves(board, model, device)
    best_move = moves[0]

    alpha = float("-inf")
    beta = float("inf")

    for move in moves:
        next_board = board.copy()
        next_board.push(move)
        value = -negamax(
            next_board,
            model,
            device,
            depth - 1,
            -beta,
            -alpha,
            not board.turn,  # negamax uses the opposite color to the current turn
        )

        # for some randomness, add a small amount of noise to the score
        noise = random.uniform(-0.01, 0.01)
        value += noise

        if alpha < value:
            alpha = value
            best_move = move
            print(
                "best move",
                best_move,
                "alpha",
                alpha,
            )

    return best_move
