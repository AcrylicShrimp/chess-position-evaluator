import chess
import torch

from libs.encoding import board2tensor
from libs.model import Model


def board2score(board: chess.Board, model: Model, device: torch.device) -> float:
    input_tensor = board2tensor(board)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        win_prob = output[0, 0].sigmoid().item()
        return absolute_win_prob(win_prob, board.turn)


def boards2scores(
    boards: list[chess.Board], model: Model, device: torch.device
) -> list[float]:
    input_tensors = torch.stack([board2tensor(board) for board in boards])
    input_tensors = input_tensors.to(device)

    with torch.no_grad():
        outputs = model(input_tensors)
        win_probs = outputs[:, 0].sigmoid().tolist()
        return [
            absolute_win_prob(win_prob, board.turn)
            for win_prob, board in zip(win_probs, boards)
        ]


def absolute_win_prob(win_prob: float, color: chess.Color) -> float:
    return win_prob if color == chess.WHITE else 1 - win_prob
