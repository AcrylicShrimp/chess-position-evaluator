import chess
import random
import torch
from battle.board2score import board2score
from battle.compute_ordered_moves import compute_ordered_moves
from dataclasses import dataclass
from enum import Enum, auto
from model import Model


class TTFlag(Enum):
    EXACT = auto()
    LOWERBOUND = auto()
    UPPERBOUND = auto()


@dataclass
class CacheEntry:
    depth: int
    score: float
    flag: TTFlag
    best_move: chess.Move | None = None


cache: dict[str, CacheEntry] = {}
cache_hits = 0
cache_reads = 0


def negamax(
    board: chess.Board,
    model: Model,
    device: torch.device,
    depth: int,
    alpha: float,
    beta: float,
    color: chess.Color,
) -> float:
    original_alpha = alpha

    global cache, cache_hits, cache_reads
    cache_reads += 1
    fen_parts = board.fen().split(" ")
    fen_key = " ".join(fen_parts[:4])

    cached_best_move: chess.Move | None = None

    if fen_key in cache:
        entry = cache[fen_key]
        cached_best_move = entry.best_move

        if entry.depth <= depth:
            cache_hits += 1

            if entry.flag == TTFlag.EXACT:
                return entry.score
            elif entry.flag == TTFlag.LOWERBOUND:
                alpha = max(alpha, entry.score)
            elif entry.flag == TTFlag.UPPERBOUND:
                beta = min(beta, entry.score)

            if beta <= alpha:
                return entry.score

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

    if cached_best_move is not None and cached_best_move in moves:
        moves.remove(cached_best_move)
        moves.insert(0, cached_best_move)

    value = float("-inf")
    best_move: chess.Move | None = None

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
            chess.BLACK if color == chess.WHITE else chess.WHITE,
        )

        if value < score:
            value = score
            best_move = move

        alpha = max(alpha, value)

        if beta <= alpha:
            break

    flag = TTFlag.EXACT

    if value <= original_alpha:
        flag = TTFlag.UPPERBOUND
    elif beta <= value:
        flag = TTFlag.LOWERBOUND

    if best_move is not None:
        new_entry = CacheEntry(depth=depth, score=value, flag=flag, best_move=best_move)
        cache[fen_key] = new_entry

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

    global cache_hits, cache_reads

    print(f"cache hits: {cache_hits}, cache reads: {cache_reads}")
    print(f"cache hit rate: {cache_hits / cache_reads:.2%}")

    return best_move
