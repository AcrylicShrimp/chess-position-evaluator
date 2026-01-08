import os

import torch

from libs.encoding import fen2tensor
from libs.model import EvalOnlyModel


CHECKPOINTS_DIR = "models/checkpoints"


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

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found")
        return

    model = EvalOnlyModel()
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[✓] Model loaded from {checkpoint_path}")

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
            input_tensor = fen2tensor(fen)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)
                win_prob = torch.sigmoid(output).item()
                print(f"Win Probability: {win_prob:.2f}")

                if win_prob >= centipawn_to_win_prob(300):
                    print("White is winning")
                elif win_prob >= centipawn_to_win_prob(150):
                    print("White has a small advantage")
                elif win_prob <= (1 - centipawn_to_win_prob(300)):
                    print("Black is winning")
                elif win_prob <= (1 - centipawn_to_win_prob(150)):
                    print("Black has a small advantage")
                else:
                    print("Both sides are equal")

        except Exception as e:
            print(f"Error evaluating position: {e}")

        print()


def centipawn_to_win_prob(cp: int) -> float:
    return 1.0 / (1.0 + 10.0 ** (float(-cp) / 400.0))


