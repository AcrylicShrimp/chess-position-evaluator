import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from libs.dataset import ChessEvaluationDataset
from libs.model import (
    _MATERIAL_LOGIT_SCALE,
    _MATERIAL_VALUES,
    _material_diff_from_board,
)
from libs.paths import (
    REPORTS_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)


DEFAULT_ROWS = 1_000_000
DEFAULT_BATCH_SIZE = 8192
CALIBRATION_BINS = 10


@dataclass(frozen=True)
class AnalysisSelection:
    dataset_rows: int
    evaluated_rows: int
    selection: str
    seed: int


class SummaryStats:
    def __init__(self) -> None:
        self.count = 0
        self.label_sum = 0.0
        self.label_sq_sum = 0.0
        self.prior_prob_sum = 0.0
        self.prior_prob_sq_sum = 0.0
        self.residual_sum = 0.0
        self.residual_abs_sum = 0.0
        self.residual_sq_sum = 0.0
        self.bce_sum = 0.0
        self.brier_sum = 0.0
        self.material_diff_sum = 0.0
        self.material_diff_sq_sum = 0.0
        self.material_diff_min: float | None = None
        self.material_diff_max: float | None = None

    def update(
        self,
        *,
        material_diff: torch.Tensor,
        prior_probs: torch.Tensor,
        labels: torch.Tensor,
        bce: torch.Tensor,
        brier: torch.Tensor,
    ) -> None:
        material_diff = material_diff.float().view(-1)
        prior_probs = prior_probs.float().view(-1)
        labels = labels.float().view(-1)
        bce = bce.float().view(-1)
        brier = brier.float().view(-1)

        count = labels.numel()
        if count == 0:
            return

        residual = labels - prior_probs
        local_min = float(material_diff.min().item())
        local_max = float(material_diff.max().item())

        self.count += count
        self.label_sum += float(labels.sum().item())
        self.label_sq_sum += float(labels.square().sum().item())
        self.prior_prob_sum += float(prior_probs.sum().item())
        self.prior_prob_sq_sum += float(prior_probs.square().sum().item())
        self.residual_sum += float(residual.sum().item())
        self.residual_abs_sum += float(residual.abs().sum().item())
        self.residual_sq_sum += float(residual.square().sum().item())
        self.bce_sum += float(bce.sum().item())
        self.brier_sum += float(brier.sum().item())
        self.material_diff_sum += float(material_diff.sum().item())
        self.material_diff_sq_sum += float(material_diff.square().sum().item())
        self.material_diff_min = (
            local_min
            if self.material_diff_min is None
            else min(self.material_diff_min, local_min)
        )
        self.material_diff_max = (
            local_max
            if self.material_diff_max is None
            else max(self.material_diff_max, local_max)
        )

    def report(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "material_diff_mean": None,
                "material_diff_std": None,
                "material_diff_min": None,
                "material_diff_max": None,
                "label_mean": None,
                "label_std": None,
                "prior_prob_mean": None,
                "residual_mean": None,
                "residual_abs_mean": None,
                "residual_rmse": None,
                "bce_loss": None,
                "brier_score": None,
            }

        label_mean = self.label_sum / self.count
        material_diff_mean = self.material_diff_sum / self.count
        residual_sq_mean = self.residual_sq_sum / self.count
        return {
            "count": self.count,
            "material_diff_mean": material_diff_mean,
            "material_diff_std": _std_from_sums(
                self.material_diff_sum,
                self.material_diff_sq_sum,
                self.count,
            ),
            "material_diff_min": self.material_diff_min,
            "material_diff_max": self.material_diff_max,
            "label_mean": label_mean,
            "label_std": _std_from_sums(
                self.label_sum,
                self.label_sq_sum,
                self.count,
            ),
            "prior_prob_mean": self.prior_prob_sum / self.count,
            "residual_mean": self.residual_sum / self.count,
            "residual_abs_mean": self.residual_abs_sum / self.count,
            "residual_rmse": math.sqrt(max(residual_sq_mean, 0.0)),
            "bce_loss": self.bce_sum / self.count,
            "brier_score": self.brier_sum / self.count,
        }


class MaterialLabelAccumulator:
    def __init__(self, bins: int = CALIBRATION_BINS) -> None:
        self.bins = bins
        self.global_stats = SummaryStats()
        self.valid_geometry_stats = SummaryStats()
        self.invalid_geometry_stats = SummaryStats()
        self.baseline_bce_sum = 0.0
        self.baseline_brier_sum = 0.0
        self.material_buckets: dict[int, SummaryStats] = {}
        self.absolute_material_buckets: dict[int, SummaryStats] = {}
        self.side_to_move_buckets = {
            "white": SummaryStats(),
            "black": SummaryStats(),
        }
        self.calibration_counts = [0] * bins
        self.calibration_prior_sums = [0.0] * bins
        self.calibration_label_sums = [0.0] * bins
        self.invalid_piece_overlap_rows = 0
        self.invalid_piece_total_rows = 0
        self.invalid_king_count_rows = 0
        self.invalid_pawn_count_rows = 0
        self.material_abs_gt_39_rows = 0

    @property
    def rows(self) -> int:
        return self.global_stats.count

    def update(self, boards: torch.Tensor, labels: torch.Tensor) -> None:
        if boards.dim() != 4 or boards.shape[1:] != (20, 8, 8):
            raise ValueError(
                "boards must be shaped [batch, 20, 8, 8] for material analysis"
            )

        labels = labels.float().view(-1)
        if boards.shape[0] != labels.numel():
            raise ValueError(
                f"board and label batch sizes differ: {boards.shape[0]} vs {labels.numel()}"
            )

        if labels.numel() == 0:
            return

        if not torch.isfinite(boards).all():
            raise ValueError("dataset boards contain non-finite values")

        if not torch.isfinite(labels).all():
            raise ValueError("dataset labels contain non-finite values")

        if torch.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("dataset labels must be probabilities in [0, 1]")

        boards = boards.float()
        material_diff = _material_diff_from_board(boards, _MATERIAL_VALUES)
        prior_logits = material_diff * _MATERIAL_LOGIT_SCALE
        prior_probs = torch.sigmoid(prior_logits)
        bce = F.binary_cross_entropy_with_logits(
            prior_logits,
            labels,
            reduction="none",
        )
        brier = (prior_probs - labels).square()
        valid_geometry_mask = self._update_piece_geometry(boards, material_diff)

        self.global_stats.update(
            material_diff=material_diff,
            prior_probs=prior_probs,
            labels=labels,
            bce=bce,
            brier=brier,
        )
        self._update_stats_from_mask(
            self.valid_geometry_stats,
            valid_geometry_mask,
            material_diff,
            prior_probs,
            labels,
            bce,
            brier,
        )
        self._update_stats_from_mask(
            self.invalid_geometry_stats,
            ~valid_geometry_mask,
            material_diff,
            prior_probs,
            labels,
            bce,
            brier,
        )
        self.baseline_bce_sum += float(
            F.binary_cross_entropy_with_logits(
                torch.zeros_like(labels),
                labels,
                reduction="sum",
            ).item()
        )
        self.baseline_brier_sum += float(
            (torch.full_like(labels, 0.5) - labels).square().sum().item()
        )
        self._update_calibration(prior_probs, labels)
        self._update_bucket_map(
            self.material_buckets,
            torch.round(material_diff).to(torch.int64),
            material_diff,
            prior_probs,
            labels,
            bce,
            brier,
        )
        self._update_bucket_map(
            self.absolute_material_buckets,
            torch.round(material_diff.abs()).to(torch.int64),
            material_diff,
            prior_probs,
            labels,
            bce,
            brier,
        )
        self._update_side_to_move_buckets(
            boards,
            material_diff,
            prior_probs,
            labels,
            bce,
            brier,
        )

    def metrics(self) -> dict[str, float | int | None]:
        if self.rows == 0:
            raise ValueError("cannot compute metrics with zero evaluated rows")

        metrics = self.global_stats.report()
        metrics.update(
            {
                "material_abs_mean": self._material_abs_mean(),
                "prior_calibration_ece": self.prior_calibration_ece(),
                "baseline_bce_loss_0_5": self.baseline_bce_sum / self.rows,
                "baseline_brier_score_0_5": self.baseline_brier_sum / self.rows,
            }
        )
        return metrics

    def material_bucket_reports(self) -> list[dict[str, Any]]:
        return [
            {"material_diff": key, **stats.report()}
            for key, stats in sorted(self.material_buckets.items())
        ]

    def absolute_material_bucket_reports(self) -> list[dict[str, Any]]:
        return [
            {"abs_material_diff": key, **stats.report()}
            for key, stats in sorted(self.absolute_material_buckets.items())
        ]

    def side_to_move_reports(self) -> list[dict[str, Any]]:
        return [
            {"side_to_move": side, **self.side_to_move_buckets[side].report()}
            for side in ("white", "black")
        ]

    def piece_geometry_report(self) -> dict[str, float | int]:
        invalid_rows = self.invalid_geometry_stats.count
        return {
            "rows": self.rows,
            "valid_rows": self.valid_geometry_stats.count,
            "invalid_rows": invalid_rows,
            "invalid_ratio": invalid_rows / self.rows if self.rows else 0.0,
            "invalid_piece_overlap_rows": self.invalid_piece_overlap_rows,
            "invalid_piece_total_rows": self.invalid_piece_total_rows,
            "invalid_king_count_rows": self.invalid_king_count_rows,
            "invalid_pawn_count_rows": self.invalid_pawn_count_rows,
            "material_abs_gt_39_rows": self.material_abs_gt_39_rows,
        }

    def valid_geometry_metrics(self) -> dict[str, float | int | None]:
        return self.valid_geometry_stats.report()

    def invalid_geometry_metrics(self) -> dict[str, float | int | None]:
        return self.invalid_geometry_stats.report()

    def calibration_bins(self) -> list[dict[str, float | int | None]]:
        result = []
        for index in range(self.bins):
            lower = index / self.bins
            upper = (index + 1) / self.bins
            count = self.calibration_counts[index]
            if count == 0:
                prior_mean = None
                label_mean = None
                abs_gap = None
            else:
                prior_mean = self.calibration_prior_sums[index] / count
                label_mean = self.calibration_label_sums[index] / count
                abs_gap = abs(prior_mean - label_mean)

            result.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "prior_prob_mean": prior_mean,
                    "label_mean": label_mean,
                    "abs_gap": abs_gap,
                }
            )

        return result

    def prior_calibration_ece(self) -> float:
        if self.rows == 0:
            raise ValueError("cannot compute calibration with zero evaluated rows")

        total = 0.0
        for bin_data in self.calibration_bins():
            count = bin_data["count"]
            gap = bin_data["abs_gap"]
            if count == 0 or gap is None:
                continue
            total += (count / self.rows) * gap
        return total

    def _material_abs_mean(self) -> float:
        total = 0.0
        for key, stats in self.absolute_material_buckets.items():
            total += key * stats.count
        return total / self.rows

    def _update_calibration(
        self,
        prior_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        indices = torch.clamp(
            (prior_probs.float().view(-1) * self.bins).long(),
            max=self.bins - 1,
        )
        labels = labels.float().view(-1)
        prior_probs = prior_probs.float().view(-1)

        for bin_index in range(self.bins):
            mask = indices == bin_index
            count = int(mask.sum().item())
            if count == 0:
                continue

            self.calibration_counts[bin_index] += count
            self.calibration_prior_sums[bin_index] += float(
                prior_probs[mask].sum().item()
            )
            self.calibration_label_sums[bin_index] += float(labels[mask].sum().item())

    def _update_bucket_map(
        self,
        buckets: dict[int, SummaryStats],
        keys: torch.Tensor,
        material_diff: torch.Tensor,
        prior_probs: torch.Tensor,
        labels: torch.Tensor,
        bce: torch.Tensor,
        brier: torch.Tensor,
    ) -> None:
        for key_tensor in torch.unique(keys, sorted=True):
            key = int(key_tensor.item())
            mask = keys == key
            buckets.setdefault(key, SummaryStats()).update(
                material_diff=material_diff[mask],
                prior_probs=prior_probs[mask],
                labels=labels[mask],
                bce=bce[mask],
                brier=brier[mask],
            )

    def _update_side_to_move_buckets(
        self,
        boards: torch.Tensor,
        material_diff: torch.Tensor,
        prior_probs: torch.Tensor,
        labels: torch.Tensor,
        bce: torch.Tensor,
        brier: torch.Tensor,
    ) -> None:
        am_i_black = boards[:, 0, 0, 0] > 0.5
        for side, mask in (
            ("white", ~am_i_black),
            ("black", am_i_black),
        ):
            if not bool(mask.any().item()):
                continue
            self.side_to_move_buckets[side].update(
                material_diff=material_diff[mask],
                prior_probs=prior_probs[mask],
                labels=labels[mask],
                bce=bce[mask],
                brier=brier[mask],
            )

    def _update_piece_geometry(
        self,
        boards: torch.Tensor,
        material_diff: torch.Tensor,
    ) -> torch.Tensor:
        our_pieces = boards[:, 6:12]
        their_pieces = boards[:, 12:18]
        all_pieces = boards[:, 6:18]

        overlap_mask = all_pieces.sum(dim=1).amax(dim=(1, 2)) > 1.5
        our_total = our_pieces.sum(dim=(1, 2, 3))
        their_total = their_pieces.sum(dim=(1, 2, 3))
        piece_total_mask = (our_total > 16.0) | (their_total > 16.0)
        king_count_mask = (
            our_pieces[:, 5].sum(dim=(1, 2)) != 1.0
        ) | (their_pieces[:, 5].sum(dim=(1, 2)) != 1.0)
        pawn_count_mask = (
            our_pieces[:, 0].sum(dim=(1, 2)) > 8.0
        ) | (their_pieces[:, 0].sum(dim=(1, 2)) > 8.0)
        material_abs_mask = material_diff.abs() > 39.0
        invalid_mask = (
            overlap_mask
            | piece_total_mask
            | king_count_mask
            | pawn_count_mask
            | material_abs_mask
        )

        self.invalid_piece_overlap_rows += int(overlap_mask.sum().item())
        self.invalid_piece_total_rows += int(piece_total_mask.sum().item())
        self.invalid_king_count_rows += int(king_count_mask.sum().item())
        self.invalid_pawn_count_rows += int(pawn_count_mask.sum().item())
        self.material_abs_gt_39_rows += int(material_abs_mask.sum().item())
        return ~invalid_mask

    def _update_stats_from_mask(
        self,
        stats: SummaryStats,
        mask: torch.Tensor,
        material_diff: torch.Tensor,
        prior_probs: torch.Tensor,
        labels: torch.Tensor,
        bce: torch.Tensor,
        brier: torch.Tensor,
    ) -> None:
        if not bool(mask.any().item()):
            return

        stats.update(
            material_diff=material_diff[mask],
            prior_probs=prior_probs[mask],
            labels=labels[mask],
            bce=bce[mask],
            brier=brier[mask],
        )


def _std_from_sums(value_sum: float, value_sq_sum: float, count: int) -> float:
    if count <= 0:
        return 0.0
    mean = value_sum / count
    variance = max(value_sq_sum / count - mean * mean, 0.0)
    return math.sqrt(variance)


def resolve_dataset_path(split: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    if split == "train":
        return TRAIN_DATA_PATH

    if split == "validation":
        return VALIDATION_DATA_PATH

    if split == "test":
        return TEST_DATA_PATH

    raise ValueError(f"unsupported split: {split}")


def resolve_selection(
    dataset_rows: int,
    rows: int | None,
    full: bool,
    seed: int,
) -> AnalysisSelection:
    if full and rows is not None:
        raise ValueError("--rows and --full are mutually exclusive")

    if dataset_rows < 0:
        raise ValueError("dataset row count cannot be negative")

    if full:
        return AnalysisSelection(
            dataset_rows=dataset_rows,
            evaluated_rows=dataset_rows,
            selection="full",
            seed=seed,
        )

    requested_rows = rows if rows is not None else DEFAULT_ROWS
    if requested_rows <= 0:
        raise ValueError("--rows must be greater than zero")

    return AnalysisSelection(
        dataset_rows=dataset_rows,
        evaluated_rows=min(requested_rows, dataset_rows),
        selection="deterministic-prefix",
        seed=seed,
    )


def default_report_path(split: str) -> Path:
    return REPORTS_DIR / f"material-label-calibration.{split}.json"


def build_report(
    *,
    split: str,
    dataset_path: Path,
    selection: AnalysisSelection,
    batch_size: int,
    workers: int,
    duration_seconds: float,
    accumulator: MaterialLabelAccumulator,
) -> dict[str, Any]:
    warnings = []
    if split == "validation":
        warnings.append("validation_split_is_model_selection_data")

    return {
        "schema_version": 1,
        "data": {
            "split": split,
            "dataset_path": str(dataset_path),
            "dataset_rows": selection.dataset_rows,
            "evaluated_rows": selection.evaluated_rows,
            "selection": selection.selection,
            "seed": selection.seed,
        },
        "run": {
            "batch_size": batch_size,
            "workers": workers,
            "torch_version": torch.__version__,
            "duration_seconds": duration_seconds,
        },
        "material_prior": {
            "piece_values": [float(value) for value in _MATERIAL_VALUES.tolist()],
            "logit_scale": _MATERIAL_LOGIT_SCALE,
            "scale_description": "material_diff * log(10) / 4",
        },
        "metrics": accumulator.metrics(),
        "piece_geometry": accumulator.piece_geometry_report(),
        "valid_geometry_metrics": accumulator.valid_geometry_metrics(),
        "invalid_geometry_metrics": accumulator.invalid_geometry_metrics(),
        "material_buckets": accumulator.material_bucket_reports(),
        "absolute_material_buckets": accumulator.absolute_material_bucket_reports(),
        "side_to_move_buckets": accumulator.side_to_move_reports(),
        "calibration_bins": accumulator.calibration_bins(),
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, allow_nan=False)
    output_path.write_text(report_json + "\n")


def run_material_label_analysis(
    split: str = "validation",
    dataset_path: Path | None = None,
    rows: int | None = None,
    full: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    workers: int = 0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch must be greater than zero")
    if workers < 0:
        raise ValueError("--workers must be greater than or equal to zero")

    resolved_dataset_path = resolve_dataset_path(split, dataset_path)
    resolved_output_path = output_path or default_report_path(split)

    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"{resolved_dataset_path} not found")

    dataset = ChessEvaluationDataset(str(resolved_dataset_path))
    selection = resolve_selection(len(dataset), rows, full, seed)
    subset = Subset(dataset, range(selection.evaluated_rows))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    print(f"Torch: {torch.__version__}")
    print(f"Dataset: {resolved_dataset_path}")
    print(f"Rows: {selection.evaluated_rows} / {selection.dataset_rows}")
    print(f"Selection: {selection.selection}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"Material prior scale: {_MATERIAL_LOGIT_SCALE:.12f}")

    accumulator = MaterialLabelAccumulator()
    start_time = time.perf_counter()

    for boards, labels in loader:
        accumulator.update(boards, labels)

    duration_seconds = time.perf_counter() - start_time
    report = build_report(
        split=split,
        dataset_path=resolved_dataset_path,
        selection=selection,
        batch_size=batch_size,
        workers=workers,
        duration_seconds=duration_seconds,
        accumulator=accumulator,
    )
    write_report(report, resolved_output_path)

    metrics = report["metrics"]
    piece_geometry = report["piece_geometry"]
    valid_metrics = report["valid_geometry_metrics"]
    print()
    print("Material-label calibration summary")
    print(f"  prior_bce_loss: {metrics['bce_loss']:.6f}")
    print(f"  prior_brier_score: {metrics['brier_score']:.6f}")
    print(f"  prior_calibration_ece: {metrics['prior_calibration_ece']:.6f}")
    if valid_metrics["count"]:
        print(f"  valid_geometry_prior_bce_loss: {valid_metrics['bce_loss']:.6f}")
    print(f"  label_mean: {metrics['label_mean']:.6f}")
    print(f"  prior_prob_mean: {metrics['prior_prob_mean']:.6f}")
    print(f"  residual_mean: {metrics['residual_mean']:.6f}")
    print(f"  residual_abs_mean: {metrics['residual_abs_mean']:.6f}")
    print(
        "  material_diff_range: "
        f"{metrics['material_diff_min']:.1f}..{metrics['material_diff_max']:.1f}"
    )
    print(
        "  invalid_piece_geometry_rows: "
        f"{piece_geometry['invalid_rows']} "
        f"({piece_geometry['invalid_ratio'] * 100.0:.4f}%)"
    )
    print(f"Report: {resolved_output_path}")

    return report
