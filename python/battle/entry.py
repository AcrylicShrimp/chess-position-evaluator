import argparse
from pathlib import Path
import random

import chess
import torch

from libs.model import ValueOnlyModel
from libs.paths import checkpoint_path
from libs.scoring import board2score

from battle.negamax import find_best_move


def resolve_checkpoint_path(model_name: str) -> Path:
    if not model_name:
        raise ValueError("model_name is required")
    return checkpoint_path(model_name)


def make_ai_move(
    board: chess.Board, model: ValueOnlyModel, device: torch.device
) -> chess.Move:
    return find_best_move(board, model, device, 6)


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


def run_battle(model_name: str):
    print(f"[✓] Using torch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")

    checkpoint_path = resolve_checkpoint_path(model_name)
    model = ValueOnlyModel()
    model.to(device)

    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[✓] Model loaded from {checkpoint_path}")

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


def main():
    parser = argparse.ArgumentParser(description="Play against the AI.")
    parser.add_argument("model_name", help="Model name without .pth extension")
    args = parser.parse_args()
    run_battle(args.model_name)
