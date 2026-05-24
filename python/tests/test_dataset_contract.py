import importlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

dataset_module = importlib.import_module("libs.dataset")


def write_dataset_fixture(path: Path, rows: list[bytes]) -> None:
    path.write_bytes(struct.pack("<Q", len(rows)) + b"".join(rows))


def pack_row(win_prob: float) -> bytes:
    bitflags = 0b0001_0111
    player_en_passant = 0
    pieces = [0] * 12
    pieces[0] = 1
    pieces[11] = 1 << 63
    heatmap = [0x21] * 64
    return struct.pack(
        dataset_module.CHESS_EVALUATION_ROW_FORMAT,
        bitflags,
        player_en_passant,
        *pieces,
        *heatmap,
        win_prob,
    )


class DatasetContractTest(unittest.TestCase):
    def test_read_chess_evaluation_row_shape_and_label_dtype(self):
        input_tensor, label = dataset_module.read_chess_evaluation(pack_row(0.75))

        self.assertEqual(input_tensor.shape, torch.Size([20, 8, 8]))
        self.assertEqual(input_tensor.dtype, torch.float32)
        self.assertEqual(label.shape, torch.Size([1]))
        self.assertEqual(label.dtype, torch.float32)
        self.assertAlmostEqual(label.item(), 0.75, places=6)

    def test_dataset_lazy_opens_file_for_single_process_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "sample.chesseval"
            write_dataset_fixture(dataset_path, [pack_row(0.25), pack_row(0.5)])

            dataset = dataset_module.ChessEvaluationDataset(str(dataset_path))

            self.assertEqual(len(dataset), 2)
            self.assertIsNone(dataset.mm)

            input_tensor, label = dataset[0]

            self.assertIsNotNone(dataset.mm)
            self.assertEqual(input_tensor.shape, torch.Size([20, 8, 8]))
            self.assertEqual(label.dtype, torch.float32)
            self.assertAlmostEqual(label.item(), 0.25, places=6)

    def test_dataloader_with_zero_workers_returns_batched_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "sample.chesseval"
            write_dataset_fixture(dataset_path, [pack_row(0.25), pack_row(0.5)])

            dataset = dataset_module.ChessEvaluationDataset(str(dataset_path))
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                num_workers=0,
            )

            input_batch, label_batch = next(iter(loader))

            self.assertEqual(input_batch.shape, torch.Size([2, 20, 8, 8]))
            self.assertEqual(input_batch.dtype, torch.float32)
            self.assertEqual(label_batch.shape, torch.Size([2, 1]))
            self.assertEqual(label_batch.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
