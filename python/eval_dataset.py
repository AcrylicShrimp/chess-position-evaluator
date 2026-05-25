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
from libs.model import ValueOnlyModel
from libs.paths import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    checkpoint_path,
    evaluation_report_path,
)


DEFAULT_ROWS = 1_000_000
DEFAULT_BATCH_SIZE = 4096
CALIBRATION_BINS = 10
CP_EPSILON = 1e-6


@dataclass
class EvaluationSelection:
    dataset_rows: int
    evaluated_rows: int
    selection: str
    seed: int


class EvaluationAccumulator:
    def __init__(self, bins: int = CALIBRATION_BINS):
        self.bins = bins
        self.rows = 0
        self.bce_sum = 0.0
        self.brier_sum = 0.0
        self.prob_abs_sum = 0.0
        self.prob_sq_sum = 0.0
        self.cp_abs_sum = 0.0
        self.cp_sq_sum = 0.0
        self.baseline_bce_sum = 0.0
        self.baseline_brier_sum = 0.0
        self.bin_counts = [0] * bins
        self.bin_pred_sums = [0.0] * bins
        self.bin_target_sums = [0.0] * bins

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.float().view(-1)
        labels = labels.float().view(-1)

        if logits.shape != labels.shape:
            raise ValueError(
                f"logits and labels must have the same shape; got {logits.shape} and {labels.shape}"
            )

        if logits.numel() == 0:
            return

        if not torch.isfinite(logits).all():
            raise ValueError("model produced non-finite logits")

        if not torch.isfinite(labels).all():
            raise ValueError("dataset produced non-finite labels")

        if torch.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("dataset labels must be probabilities in [0, 1]")

        probs = torch.sigmoid(logits)
        prob_diff = probs - labels
        pred_cp = probability_to_centipawn(probs)
        target_cp = probability_to_centipawn(labels)
        cp_diff = pred_cp - target_cp
        baseline_logits = torch.zeros_like(labels)
        baseline_diff = torch.full_like(labels, 0.5) - labels

        row_count = logits.numel()
        self.rows += row_count
        self.bce_sum += F.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        ).item()
        self.brier_sum += torch.sum(prob_diff.square()).item()
        self.prob_abs_sum += torch.sum(prob_diff.abs()).item()
        self.prob_sq_sum += torch.sum(prob_diff.square()).item()
        self.cp_abs_sum += torch.sum(cp_diff.abs()).item()
        self.cp_sq_sum += torch.sum(cp_diff.square()).item()
        self.baseline_bce_sum += F.binary_cross_entropy_with_logits(
            baseline_logits, labels, reduction="sum"
        ).item()
        self.baseline_brier_sum += torch.sum(baseline_diff.square()).item()

        indices = torch.clamp((probs * self.bins).long(), max=self.bins - 1)
        for bin_index in range(self.bins):
            mask = indices == bin_index
            count = int(mask.sum().item())
            if count == 0:
                continue

            self.bin_counts[bin_index] += count
            self.bin_pred_sums[bin_index] += torch.sum(probs[mask]).item()
            self.bin_target_sums[bin_index] += torch.sum(labels[mask]).item()

    def metrics(self) -> dict[str, float]:
        if self.rows == 0:
            raise ValueError("cannot compute metrics with zero evaluated rows")

        calibration_ece = 0.0
        for bin_data in self.calibration_bins():
            count = bin_data["count"]
            if count == 0:
                continue
            calibration_ece += (count / self.rows) * bin_data["abs_gap"]

        return {
            "bce_loss": self.bce_sum / self.rows,
            "brier_score": self.brier_sum / self.rows,
            "prob_mae": self.prob_abs_sum / self.rows,
            "prob_rmse": math.sqrt(self.prob_sq_sum / self.rows),
            "cp_equivalent_mae": self.cp_abs_sum / self.rows,
            "cp_equivalent_rmse": math.sqrt(self.cp_sq_sum / self.rows),
            "calibration_ece": calibration_ece,
            "baseline_bce_loss_0_5": self.baseline_bce_sum / self.rows,
            "baseline_brier_score_0_5": self.baseline_brier_sum / self.rows,
        }

    def calibration_bins(self) -> list[dict[str, float | int | None]]:
        result = []
        for bin_index in range(self.bins):
            lower = bin_index / self.bins
            upper = (bin_index + 1) / self.bins
            count = self.bin_counts[bin_index]

            if count == 0:
                pred_mean = None
                target_mean = None
                abs_gap = None
            else:
                pred_mean = self.bin_pred_sums[bin_index] / count
                target_mean = self.bin_target_sums[bin_index] / count
                abs_gap = abs(pred_mean - target_mean)

            result.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "pred_mean": pred_mean,
                    "target_mean": target_mean,
                    "abs_gap": abs_gap,
                }
            )

        return result


def probability_to_centipawn(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(CP_EPSILON, 1.0 - CP_EPSILON)
    return 400.0 * torch.log10(probability / (1.0 - probability))


def resolve_dataset_path(split: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    if split == "train":
        return TRAIN_DATA_PATH

    if split == "validation":
        return VALIDATION_DATA_PATH

    raise ValueError(f"unsupported split: {split}")


def resolve_selection(
    dataset_rows: int,
    rows: int | None,
    full: bool,
    seed: int,
) -> EvaluationSelection:
    if full and rows is not None:
        raise ValueError("--rows and --full are mutually exclusive")

    if dataset_rows < 0:
        raise ValueError("dataset row count cannot be negative")

    if full:
        return EvaluationSelection(
            dataset_rows=dataset_rows,
            evaluated_rows=dataset_rows,
            selection="full",
            seed=seed,
        )

    requested_rows = rows if rows is not None else DEFAULT_ROWS
    if requested_rows <= 0:
        raise ValueError("--rows must be greater than zero")

    return EvaluationSelection(
        dataset_rows=dataset_rows,
        evaluated_rows=min(requested_rows, dataset_rows),
        selection="deterministic-prefix",
        seed=seed,
    )


def select_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda device was requested but is not available")

    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps device was requested but is not available")

    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device_name}")

    return torch.device(normalized)


def build_report(
    *,
    model_name: str,
    checkpoint: dict[str, Any],
    model_path: Path,
    split: str,
    dataset_path: Path,
    selection: EvaluationSelection,
    device: torch.device,
    batch_size: int,
    duration_seconds: float,
    accumulator: EvaluationAccumulator,
) -> dict[str, Any]:
    warnings = []
    if split == "validation":
        warnings.append("validation_split_is_model_selection_data")

    return {
        "schema_version": 1,
        "model": {
            "name": model_name,
            "checkpoint_path": str(model_path),
            "checkpoint_epoch": json_safe_number(checkpoint.get("epoch")),
            "checkpoint_best_validation_loss": json_safe_number(
                checkpoint.get("best_validation_loss")
            ),
        },
        "data": {
            "split": split,
            "dataset_path": str(dataset_path),
            "dataset_rows": selection.dataset_rows,
            "evaluated_rows": selection.evaluated_rows,
            "selection": selection.selection,
            "seed": selection.seed,
        },
        "run": {
            "device": str(device),
            "batch_size": batch_size,
            "torch_version": torch.__version__,
            "duration_seconds": duration_seconds,
        },
        "metrics": accumulator.metrics(),
        "calibration_bins": accumulator.calibration_bins(),
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, allow_nan=False)
    output_path.write_text(report_json + "\n")


def json_safe_number(value: Any) -> float | int | None:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return None


def run_eval_dataset(
    model_name: str,
    split: str = "validation",
    dataset_path: Path | None = None,
    rows: int | None = None,
    full: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    device_name: str = "auto",
    output_path: Path | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch must be greater than zero")

    model_path = checkpoint_path(model_name)
    resolved_dataset_path = resolve_dataset_path(split, dataset_path)
    resolved_output_path = output_path or evaluation_report_path(
        model_name, split
    )

    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"{resolved_dataset_path} not found")

    device = select_device(device_name)

    print(f"Torch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Dataset: {resolved_dataset_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model = ValueOnlyModel()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dataset = ChessEvaluationDataset(str(resolved_dataset_path))
    selection = resolve_selection(len(dataset), rows, full, seed)
    subset = Subset(dataset, range(selection.evaluated_rows))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Rows: {selection.evaluated_rows} / {selection.dataset_rows}")
    print(f"Selection: {selection.selection}")
    print(f"Batch size: {batch_size}")

    accumulator = EvaluationAccumulator()
    start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            accumulator.update(logits, labels)

    duration_seconds = time.perf_counter() - start_time

    report = build_report(
        model_name=model_name,
        checkpoint=checkpoint,
        model_path=model_path,
        split=split,
        dataset_path=resolved_dataset_path,
        selection=selection,
        device=device,
        batch_size=batch_size,
        duration_seconds=duration_seconds,
        accumulator=accumulator,
    )
    write_report(report, resolved_output_path)

    metrics = report["metrics"]
    print()
    print("Evaluation summary")
    print(f"  bce_loss: {metrics['bce_loss']:.6f}")
    print(f"  brier_score: {metrics['brier_score']:.6f}")
    print(f"  prob_mae: {metrics['prob_mae']:.6f}")
    print(f"  prob_rmse: {metrics['prob_rmse']:.6f}")
    print(f"  cp_equivalent_mae: {metrics['cp_equivalent_mae']:.2f}")
    print(f"  cp_equivalent_rmse: {metrics['cp_equivalent_rmse']:.2f}")
    print(f"  calibration_ece: {metrics['calibration_ece']:.6f}")
    print(f"Report: {resolved_output_path}")

    return report
