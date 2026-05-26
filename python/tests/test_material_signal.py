import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

analysis = importlib.import_module("analyze_material_signal")


def create_staging_fixture(path: Path, rows: list[tuple[str, int]]) -> None:
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE rows(fen VARCHAR, cp BIGINT)")
        conn.executemany("INSERT INTO rows VALUES (?, ?)", rows)


class MaterialSignalAccumulatorTest(unittest.TestCase):
    def test_material_signal_accumulates_side_relative_cp(self):
        accumulator = analysis.MaterialSignalAccumulator()
        accumulator.update_source_row(
            "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1",
            900,
        )
        accumulator.update_source_row(
            "4k3/8/8/8/8/8/8/Q3K3 b - - 0 1",
            900,
        )

        metrics = accumulator.metrics()
        self.assertEqual(metrics["accepted_rows"], 2)
        self.assertEqual(metrics["rejected_rows"], 0)
        self.assertEqual(metrics["material_cp_sign_match_rows"], 2)

        buckets = {
            item["material_diff"]: item
            for item in accumulator.signed_material_reports()
        }
        self.assertEqual(sorted(buckets), [-9, 9])
        self.assertEqual(buckets[9]["relative_cp_mean"], 900)
        self.assertEqual(buckets[-9]["relative_cp_mean"], -900)
        self.assertAlmostEqual(
            buckets[9]["fixed_prior_prob"],
            analysis.centipawn_to_probability(900),
        )

    def test_standard_position_rejects_missing_king(self):
        accumulator = analysis.MaterialSignalAccumulator()
        accumulator.update_source_row("8/8/8/8/8/8/8/4K3 w - - 0 1", 0)

        metrics = accumulator.metrics()
        self.assertEqual(metrics["accepted_rows"], 0)
        self.assertEqual(metrics["rejected_rows"], 1)
        self.assertEqual(
            metrics["primary_reject_reasons"],
            {"black_king_count": 1},
        )

    def test_probability_round_trip(self):
        for cp in (-400, -100, 0, 100, 400):
            probability = analysis.centipawn_to_probability(cp)
            self.assertAlmostEqual(
                analysis.probability_to_centipawn(probability),
                cp,
                places=6,
            )


class MaterialSignalReportTest(unittest.TestCase):
    def test_selection_rejects_rows_with_full(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            analysis.resolve_selection(
                source_rows_total=10,
                split="all",
                rows=5,
                full=True,
            )

    def test_run_analysis_writes_stable_report_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_path = Path(tmpdir) / "sample.duckdb"
            output_path = Path(tmpdir) / "report.json"
            create_staging_fixture(
                staging_path,
                [
                    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
                    ("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1", 900),
                    ("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1", 900),
                ],
            )

            report = analysis.run_material_signal_analysis(
                split="all",
                staging_path=staging_path,
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
                "target",
                "material",
                "metrics",
                "signed_material_buckets",
                "absolute_material_buckets",
                "side_to_move_material_buckets",
                "warnings",
            ],
        )
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["data"]["evaluated_rows"], 3)
        self.assertEqual(loaded["metrics"]["accepted_rows"], 3)
        self.assertEqual(loaded["metrics"]["rejected_rows"], 0)
        self.assertTrue(
            math.isfinite(
                loaded["metrics"]["signed_material_vs_relative_cp_pearson"]
            )
        )


if __name__ == "__main__":
    unittest.main()
