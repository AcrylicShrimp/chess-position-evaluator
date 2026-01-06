import chess
import os
import random
import torch

from libs.scoring import board2score
from libs.model import Model

from battle.negamax import find_best_move


def make_ai_move(board: chess.Board, model: Model, device: torch.device) -> chess.Move:
    return find_best_move(board, model, device, 4)


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

    best_checkpoint_path = "model-best.pth"
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
                print("AI wins!")

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
            print(f"AI score: {board2score(board, model, device)}")
        else:
            move = make_player_move(board)
            print(f"Your move is: {move}")

        print()
        board.push(move)


if __name__ == "__main__":
    main()
