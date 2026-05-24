import importlib
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import chess

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

eval_module = importlib.import_module("eval")
negamax_module = importlib.import_module("battle.negamax")


class EvalPerspectiveTest(unittest.TestCase):
    def test_side_to_move_probability_converts_to_white_probability(self):
        self.assertAlmostEqual(
            eval_module.side_to_move_prob_to_white_prob(0.75, chess.WHITE), 0.75
        )
        self.assertAlmostEqual(
            eval_module.side_to_move_prob_to_white_prob(0.75, chess.BLACK), 0.25
        )

    def test_white_probability_description_uses_absolute_white_perspective(self):
        self.assertEqual(
            eval_module.describe_white_win_prob(eval_module.centipawn_to_win_prob(300)),
            "White is winning",
        )
        self.assertEqual(
            eval_module.describe_white_win_prob(
                1.0 - eval_module.centipawn_to_win_prob(300)
            ),
            "Black is winning",
        )
        self.assertEqual(
            eval_module.describe_white_win_prob(0.5),
            "Both sides are equal",
        )


class NegamaxTerminalScoringTest(unittest.TestCase):
    def setUp(self):
        negamax_module.transposition_table.clear()

    def test_checkmate_is_loss_for_side_to_move(self):
        board = chess.Board()
        for san in ["f3", "e5", "g4", "Qh4#"]:
            board.push_san(san)

        self.assertTrue(board.is_checkmate())
        self.assertEqual(
            negamax_module.terminal_score(board), -negamax_module.MATE_SCORE
        )

    def test_stalemate_is_draw(self):
        board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")

        self.assertTrue(board.is_stalemate())
        self.assertEqual(negamax_module.terminal_score(board), negamax_module.DRAW_SCORE)

    def test_find_best_move_prefers_mate_in_one(self):
        board = chess.Board()
        for san in ["f3", "e5", "g4"]:
            board.push_san(san)

        legal_moves = list(board.legal_moves)

        with (
            patch.object(negamax_module, "compute_ordered_moves", return_value=legal_moves),
            patch.object(negamax_module, "board2score", return_value=0.5),
            patch.object(negamax_module.random, "uniform", return_value=0.0),
            redirect_stdout(io.StringIO()),
        ):
            move = negamax_module.find_best_move(board, model=None, device=None, depth=1)

        self.assertEqual(move, chess.Move.from_uci("d8h4"))


if __name__ == "__main__":
    unittest.main()
