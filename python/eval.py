import chess
import torch

from libs.encoding import board2tensor
from libs.model import ValueOnlyModel, model_variant_from_checkpoint
from libs.paths import checkpoint_path


def run_eval(model_name: str):
    """Run interactive FEN evaluation with the given model."""
    print(f"[✓] Using torch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[✓] Using device: {device}")

    model_path = checkpoint_path(model_name)

    if not model_path.exists():
        print(f"Error: {model_path} not found")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model_variant = model_variant_from_checkpoint(checkpoint)
    print(f"[✓] Model variant: {model_variant}")

    model = ValueOnlyModel(model_variant=model_variant)
    model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[✓] Model loaded from {model_path}")

    if "best_validation_loss" in checkpoint:
        best_validation_loss = checkpoint["best_validation_loss"]
        print(f"[✓] Best validation loss: {best_validation_loss:.4f}")

    print()
    print("Enter FEN strings to evaluate. Press Enter with empty input to exit.")
    print()

    while True:
        try:
            fen = input("FEN: ")
        except EOFError:
            break

        fen = fen.strip()

        if fen == "":
            break

        try:
            board = chess.Board(fen)
            input_tensor = board2tensor(board)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)
                side_to_move_win_prob = torch.sigmoid(output).item()
                white_win_prob = side_to_move_prob_to_white_prob(
                    side_to_move_win_prob, board.turn
                )
                black_win_prob = 1.0 - white_win_prob
                side_to_move = "White" if board.turn == chess.WHITE else "Black"

                print(
                    f"{side_to_move} to move win probability: {side_to_move_win_prob:.2f}"
                )
                print(f"White win probability: {white_win_prob:.2f}")
                print(f"Black win probability: {black_win_prob:.2f}")
                print(describe_white_win_prob(white_win_prob))

        except Exception as e:
            print(f"Error evaluating position: {e}")

        print()


def centipawn_to_win_prob(cp: int) -> float:
    return 1.0 / (1.0 + 10.0 ** (float(-cp) / 400.0))


def side_to_move_prob_to_white_prob(
    side_to_move_win_prob: float, turn: chess.Color
) -> float:
    return (
        side_to_move_win_prob
        if turn == chess.WHITE
        else 1.0 - side_to_move_win_prob
    )


def describe_white_win_prob(white_win_prob: float) -> str:
    if white_win_prob >= centipawn_to_win_prob(300):
        return "White is winning"
    if white_win_prob >= centipawn_to_win_prob(150):
        return "White has a small advantage"
    if white_win_prob <= (1 - centipawn_to_win_prob(300)):
        return "Black is winning"
    if white_win_prob <= (1 - centipawn_to_win_prob(150)):
        return "Black has a small advantage"
    return "Both sides are equal"
