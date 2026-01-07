import chess
import random
import torch

from libs.scoring import board2score
from libs.model import EvalOnlyModel

from battle.compute_ordered_moves import (
    compute_ordered_moves,
    compute_ordered_moves_fast,
)


MATE_SCORE = 1000.0
DRAW_SCORE = 0.0


def score_to_prob(score: float) -> float:
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return 0.0
    return (score / 2) + 0.5


def prob_to_score(prob: float) -> float:
    return (prob - 0.5) * 2


transposition_table = {}


def negamax(
    board: chess.Board,
    model: EvalOnlyModel,
    device: torch.device,
    depth: int,
    alpha: float,
    beta: float,
) -> float:
    board_fen = board.fen()
    state_key = (board_fen, depth)

    if state_key in transposition_table:
        return transposition_table[state_key]

    outcome = board.outcome(claim_draw=True)

    if outcome is not None:
        return DRAW_SCORE

    if depth == 0:
        white_win_prob = board2score(board, model, device)
        score = prob_to_score(
            white_win_prob if board.turn == chess.WHITE else 1.0 - white_win_prob
        )
        return score

    moves = compute_ordered_moves_fast(board)

    if not moves and board.legal_moves.count() > 0:
        moves = list(board.legal_moves)

    max_val = float("-inf")

    for move in moves:
        board.push(move)
        score = -negamax(board, model, device, depth - 1, -beta, -alpha)
        board.pop()

        max_val = max(max_val, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    transposition_table[state_key] = max_val

    return max_val


def find_best_move(
    board: chess.Board, model: EvalOnlyModel, device: torch.device, depth: int
) -> chess.Move | None:
    if board.is_game_over(claim_draw=True):
        return None

    print(
        f"--- Thinking (Depth {depth}, Turn: {'White' if board.turn else 'Black'}) ---"
    )

    moves = compute_ordered_moves(board, model, device)
    if not moves:
        moves = list(board.legal_moves)

    best_move = moves[0]

    alpha = float("-inf")
    beta = float("inf")

    for move in moves:
        board.push(move)

        score_val = -negamax(board, model, device, depth - 1, -beta, -alpha)

        board.pop()

        noise = random.uniform(-0.005, 0.005)
        score_with_noise = score_val + noise

        win_prob = score_to_prob(score_val)
        print(f"Move: {move} | Score: {score_val:.4f} (WinProb: {win_prob*100:.1f}%)")

        if score_with_noise > alpha:
            alpha = score_with_noise
            best_move = move
            print(f"  -> New Best: {best_move} (Alpha: {alpha:.4f})")

    return best_move
