import importlib
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

analysis = importlib.import_module("analyze_material_label_calibration")
dataset_module = importlib.import_module("libs.dataset")


def write_dataset_fixture(path: Path, rows: list[bytes]) -> None:
    path.write_bytes(struct.pack("<Q", len(rows)) + b"".join(rows))


def pack_row(
    *,
    win_prob: float,
    black_to_move: bool = False,
    our_piece_counts: dict[int, int] | None = None,
    their_piece_counts: dict[int, int] | None = None,
) -> bytes:
    bitflags = 0b0001_0110 | int(black_to_move)
    player_en_passant = 0
    pieces = [0] * 12
    pieces[5] = 1 << 4
    pieces[11] = 1 << 60

    next_square = 0
    for piece_index, count in (our_piece_counts or {}).items():
        for _ in range(count):
            if next_square == 4:
                next_square += 1
            pieces[piece_index] |= 1 << next_square
            next_square += 1

    next_square = 32
    for piece_index, count in (their_piece_counts or {}).items():
        for _ in range(count):
            if next_square == 60:
                next_square += 1
            pieces[6 + piece_index] |= 1 << next_square
            next_square += 1

    heatmap = [0x21] * 64
    return struct.pack(
        dataset_module.CHESS_EVALUATION_ROW_FORMAT,
        bitflags,
        player_en_passant,
        *pieces,
        *heatmap,
        win_prob,
    )


class MaterialLabelAccumulatorTest(unittest.TestCase):
    def test_prior_metrics_are_row_weighted_and_bucketed(self):
        boards = torch.zeros((3, 20, 8, 8), dtype=torch.float32)
        labels = torch.tensor([[0.5], [0.75], [0.25]], dtype=torch.float32)
        boards[1, 6, 0, 0] = 1.0
        boards[2, 12, 0, 0] = 1.0
        boards[2, 0] = 1.0

        accumulator = analysis.MaterialLabelAccumulator()
        accumulator.update(boards[:1], labels[:1])
        accumulator.update(boards[1:], labels[1:])

        metrics = accumulator.metrics()
        logits = torch.tensor(
            [
                0.0,
                analysis._MATERIAL_LOGIT_SCALE,
                -analysis._MATERIAL_LOGIT_SCALE,
            ],
            dtype=torch.float32,
        )
        expected_bce = F.binary_cross_entropy_with_logits(
            logits,
            labels.view(-1),
            reduction="mean",
        ).item()

        self.assertEqual(accumulator.rows, 3)
        self.assertAlmostEqual(metrics["bce_loss"], expected_bce, places=6)
        self.assertAlmostEqual(metrics["material_diff_mean"], 0.0, places=6)

        material_buckets = {
            item["material_diff"]: item
            for item in accumulator.material_bucket_reports()
        }
        self.assertEqual(sorted(material_buckets), [-1, 0, 1])
        self.assertEqual(material_buckets[1]["count"], 1)
        self.assertEqual(material_buckets[-1]["count"], 1)

        side_buckets = {
            item["side_to_move"]: item
            for item in accumulator.side_to_move_reports()
        }
        self.assertEqual(side_buckets["white"]["count"], 2)
        self.assertEqual(side_buckets["black"]["count"], 1)

    def test_labels_must_be_probabilities(self):
        accumulator = analysis.MaterialLabelAccumulator()
        boards = torch.zeros((1, 20, 8, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "probabilities"):
            accumulator.update(boards, torch.tensor([[1.25]]))


class MaterialLabelReportTest(unittest.TestCase):
    def test_selection_rejects_rows_with_full(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            analysis.resolve_selection(
                dataset_rows=10,
                rows=5,
                full=True,
                seed=0,
            )

    def test_run_analysis_writes_stable_report_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "sample.chesseval"
            output_path = Path(tmpdir) / "report.json"
            write_dataset_fixture(
                dataset_path,
                [
                    pack_row(win_prob=0.5),
                    pack_row(win_prob=0.8, our_piece_counts={0: 1}),
                    pack_row(
                        win_prob=0.2,
                        black_to_move=True,
                        their_piece_counts={0: 1},
                    ),
                ],
            )

            report = analysis.run_material_label_analysis(
                split="test",
                dataset_path=dataset_path,
                rows=3,
                batch_size=2,
                output_path=output_path,
            )

            loaded = json.loads(output_path.read_text())

        self.assertEqual(
            list(report.keys()),
            [
                "schema_version",
                "data",
                "run",
                "material_prior",
                "metrics",
                "piece_geometry",
                "valid_geometry_metrics",
                "invalid_geometry_metrics",
                "material_buckets",
                "absolute_material_buckets",
                "side_to_move_buckets",
                "calibration_bins",
                "warnings",
            ],
        )
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["data"]["evaluated_rows"], 3)
        self.assertEqual(loaded["warnings"], [])
        self.assertEqual(loaded["piece_geometry"]["rows"], 3)
        self.assertTrue(math.isfinite(loaded["metrics"]["prior_calibration_ece"]))


if __name__ == "__main__":
    unittest.main()
