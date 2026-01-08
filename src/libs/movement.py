import chess
import torch

TOTAL_MOVES = 4672
QUEEN_LIKE_MOVE_OFFSET = 0
KNIGHT_LIKE_MOVE_OFFSET = 3584
UNDERPROMOTION_OFFSET = 4096


def encode_moves(moves: list[tuple[chess.Piece, chess.Move, float]]) -> torch.Tensor:
    result = torch.zeros(TOTAL_MOVES, dtype=torch.float32)

    for piece, move, score in moves:
        result[move2index(piece, move)] = score

    return result


def decode_moves(
    moves: torch.Tensor, board: chess.Board
) -> list[tuple[chess.Piece, chess.Move, float]]:
    result = []
    total_score = 0.0

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)

        if piece is None:
            continue

        score = moves[move2index(piece, move)].item()
        result.append((piece, move, score))
        total_score += score

    if 1e-4 < total_score:
        result = [(piece, move, score / total_score) for piece, move, score in result]
        result = list(sorted(result, key=lambda x: x[2], reverse=True))

    return result


def move2index(piece: chess.Piece, move: chess.Move) -> int:
    if move.promotion is not None and move.promotion != chess.QUEEN:  # underpromotion
        from_file = move.from_square % 8
        to_file = move.to_square % 8

        if from_file < to_file:  # moved like a -> h
            direction_offset = 1
        elif to_file < from_file:  # moved like h -> a
            direction_offset = 2
        else:  # straight up/down
            direction_offset = 0

        if move.promotion == chess.KNIGHT:
            target_offset = 0
        elif move.promotion == chess.BISHOP:
            target_offset = 1
        else:
            target_offset = 2

        return (
            UNDERPROMOTION_OFFSET
            + move.from_square * 3 * 3
            + direction_offset * 3
            + target_offset
        )
    elif piece.piece_type == chess.KNIGHT:  # knight-like move
        from_rank = move.from_square // 8
        to_rank = move.to_square // 8

        from_file = move.from_square % 8
        to_file = move.to_square % 8

        direction_offset = {
            (-2, -1): 0,
            (-2, 1): 1,
            (-1, -2): 2,
            (-1, 2): 3,
            (1, -2): 4,
            (1, 2): 5,
            (2, -1): 6,
            (2, 1): 7,
        }[to_rank - from_rank, to_file - from_file]

        return KNIGHT_LIKE_MOVE_OFFSET + move.from_square * 8 + direction_offset
    else:  # queen-like move
        from_rank = move.from_square // 8
        to_rank = move.to_square // 8

        from_file = move.from_square % 8
        to_file = move.to_square % 8

        rank_diff = max(min(to_rank - from_rank, 1), -1)
        file_diff = max(min(to_file - from_file, 1), -1)
        direction_offset = {
            (-1, -1): 0,
            (-1, 0): 1,
            (-1, 1): 2,
            (0, -1): 3,
            (0, 1): 4,
            (1, -1): 5,
            (1, 0): 6,
            (1, 1): 7,
        }[rank_diff, file_diff]

        # up to 7 squares in any direction
        distance = max(abs(to_rank - from_rank), abs(to_file - from_file))
        distance_offset = distance - 1

        return (
            QUEEN_LIKE_MOVE_OFFSET
            + move.from_square * 8 * 7
            + direction_offset * 7
            + distance_offset
        )
