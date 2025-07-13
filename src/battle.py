import chess
import numpy as np
import os
import random
import torch
from chess_evaluation_reader import boolean2tensor, bitboard2tensors
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


def board2tensor(board: chess.Board) -> torch.Tensor:
    turn_tensor = boolean2tensor(board.turn == chess.WHITE)
    wk_castle_tensor = boolean2tensor(board.has_kingside_castling_rights(chess.WHITE))
    wq_castle_tensor = boolean2tensor(board.has_queenside_castling_rights(chess.WHITE))
    bk_castle_tensor = boolean2tensor(board.has_kingside_castling_rights(chess.BLACK))
    bq_castle_tensor = boolean2tensor(board.has_queenside_castling_rights(chess.BLACK))

    # 2. 비트보드 특징(Bitboard features)을 '정수' 형태로 모두 수집
    # 앙파상 가능한 '위치' 비트보드 (해당 위치에 1)
    en_passant_bb = (1 << board.ep_square) if board.ep_square is not None else 0

    # 공격 맵 비트보드
    white_attacks_bb = compute_white_attacks(board)
    black_attacks_bb = compute_black_attacks(board)

    # 기물 위치 비트보드 (WHITE: P, N, B, R, Q, K, BLACK: p, n, b, r, q, k 순)
    pieces_bb = [
        board.pieces_mask(piece_type, color)
        for color in [chess.WHITE, chess.BLACK]
        for piece_type in chess.PIECE_TYPES
    ]

    # 3. 수집된 모든 비트보드를 한 번에 텐서로 변환
    all_bitboards = [en_passant_bb, white_attacks_bb, black_attacks_bb, *pieces_bb]
    bitboard_tensors = bitboard2tensors(all_bitboards)  # 총 1+1+1+12 = 15개 채널

    # 4. 모든 텐서 채널을 하나로 합칩니다. torch.vstack은 여기서 torch.cat과 동일하게 동작합니다.
    input_tensor = torch.cat(
        [
            turn_tensor,  # 1 채널
            wk_castle_tensor,  # 1 채널
            wq_castle_tensor,  # 1 채널
            bk_castle_tensor,  # 1 채널
            bq_castle_tensor,  # 1 채널
            bitboard_tensors,  # 15 채널
        ],
        dim=0,  # 채널 차원을 따라 합침
    )
    # 최종 텐서 모양: (20, 8, 8)

    return input_tensor


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


def board2score(board: chess.Board, model: Model, device: torch.device) -> float:
    input_tensor = board2tensor(board)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        return output[0, 0].item()


def make_ai_move(board: chess.Board, model: Model, device: torch.device) -> chess.Move:
    moves: list[tuple[chess.Move, float]] = []

    for move in board.legal_moves:
        board_in_next_turn = board.copy()
        board_in_next_turn.push(move)
        score = board2score(board_in_next_turn, model, device)

        if board.turn == chess.BLACK:
            score = -score

        moves.append((move, score))

    sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
    best_move = sorted_moves[0]
    candidate_moves = list(
        filter(lambda x: 0 <= x[1] and (best_move[1] - x[1]) < 0.5, sorted_moves)
    )

    if len(candidate_moves) == 0:
        return best_move[0]

    return random.choice(candidate_moves)[0]


def make_player_move(board: chess.Board) -> chess.Move:
    while True:
        move = input("Enter move (san): ")

        try:
            move = board.parse_san(move)
        except ValueError:
            print("Invalid move")
            continue

        if move not in board.legal_moves:
            print("Invalid move")
            continue

        return move


def main():
    print(f"[✓] Using torch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")

    best_checkpoint_path = "best-model.pth"
    model = Model()
    model.to(device)

    if os.path.exists("BEST_CHECKPOINT_PATH"):
        best_checkpoint_path = os.environ.get("BEST_CHECKPOINT_PATH")

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[✓] Model loaded from {best_checkpoint_path}")

    best_validation_loss = checkpoint["best_validation_loss"]
    print(f"[✓] Best validation loss: {best_validation_loss:.4f}")

    board = chess.Board()
    ai_color = random.random() < 0.5

    while True:
        print(board)

        if board.is_checkmate():
            if ai_color == board.turn:
                print("You win!")
            else:
                print("AI lose!")

            break

        if board.is_stalemate():
            print("Stalemate!")
            break

        if board.is_insufficient_material():
            print("Insufficient material!")
            break

        if ai_color == board.turn:
            move = make_ai_move(board, model, device)
            print(f"AI move is: {move.uci()}")
        else:
            move = make_player_move(board)
            print(f"Your move is: {move}")

        print()
        board.push(move)


if __name__ == "__main__":
    main()
