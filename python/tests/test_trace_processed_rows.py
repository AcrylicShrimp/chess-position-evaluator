import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

trace_rows = importlib.import_module("trace_processed_rows")


class TraceProcessedRowsReportInputTest(unittest.TestCase):
    def test_load_trace_requests_deduplicates_report_indices(self):
        report = {
            "top_examples": {
                "parallel_worst": [
                    {"dataset_index": 3, "label": 0.1},
                    {"dataset_index": 1, "label": 0.2},
                ],
                "parallel_best": [
                    {"dataset_index": 3, "label": 0.3},
                    {"dataset_index": 2, "label": 0.4},
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(json.dumps(report))
            requests = trace_rows.load_trace_requests_from_report(
                path,
                top_worst=2,
                top_best=2,
            )

        self.assertEqual(
            [request.dataset_index for request in requests],
            [1, 2, 3],
        )
        self.assertEqual(
            {request.source_group for request in requests},
            {"parallel_worst", "parallel_best"},
        )

    def test_trace_to_report_row_preserves_model_diagnostics(self):
        request = trace_rows.TraceRequest(
            dataset_index=7,
            source_group="parallel_worst",
            report_item={
                "label": 0.5,
                "material_diff": 2.0,
                "baseline_prob": 0.4,
                "parallel_prob": 0.9,
                "parallel_minus_baseline_bce": 1.25,
            },
        )
        trace = trace_rows.SourceTrace(
            request=request,
            source_row_offset=9,
            source_row_number=1009,
            fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            cp=0,
            side_to_move="white",
            relative_cp=0,
            label_probability=0.5,
            material_diff=2,
        )

        row = trace_rows.trace_to_report_row(trace)

        self.assertEqual(row["dataset_index"], 7)
        self.assertEqual(row["source_row_number"], 1009)
        self.assertEqual(row["label_probability_delta"], 0.0)
        self.assertEqual(row["material_diff_delta"], 0.0)
        self.assertEqual(row["parallel_prob"], 0.9)
        self.assertEqual(row["parallel_minus_baseline_bce"], 1.25)

    def test_load_trace_requests_requires_at_least_one_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(json.dumps({"top_examples": {}}))

            with self.assertRaisesRegex(ValueError, "at least one"):
                trace_rows.load_trace_requests_from_report(
                    path,
                    top_worst=0,
                    top_best=0,
                )


if __name__ == "__main__":
    unittest.main()
