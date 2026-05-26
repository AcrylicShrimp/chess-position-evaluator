import json
import math
import platform
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import duckdb

from libs.paths import DUCKDB_TEMP_PATH, REPORTS_DIR


DEFAULT_ROWS = 1_000_000
DEFAULT_BATCH_SIZE = 20_000
TABLE_NAME = "rows"
TRAIN_SET_RATIO = 0.9
VALIDATION_SET_RATIO = 0.05
MATERIAL_CP_PER_POINT = 100.0
CP_THRESHOLDS = (100, 300, 800, 1500)
MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
MAX_ABS_MATERIAL_DIFF = 39
BACK_RANKS = chess.BB_RANK_1 | chess.BB_RANK_8


@dataclass(frozen=True)
class SourceSelection:
    source_rows_total: int
    split: str
    split_offset: int
    split_rows: int
    evaluated_rows: int
    selection: str


@dataclass
class RunningStats:
    count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, value: float) -> None:
        self.count += 1
        self.value_sum += value
        self.value_sq_sum += value * value
        self.minimum = value if self.minimum is None else min(
            self.minimum, value)
        self.maximum = value if self.maximum is None else max(
            self.maximum, value)

    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.value_sum / self.count

    def std(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.value_sum / self.count
        variance = max(self.value_sq_sum / self.count - mean * mean, 0.0)
        return math.sqrt(variance)

    def report(self, prefix: str) -> dict[str, float | int | None]:
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": self.mean(),
            f"{prefix}_std": self.std(),
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
        }


@dataclass
class CorrelationStats:
    count: int = 0
    x_sum: float = 0.0
    y_sum: float = 0.0
    x_sq_sum: float = 0.0
    y_sq_sum: float = 0.0
    xy_sum: float = 0.0

    def update(self, x: float, y: float) -> None:
        self.count += 1
        self.x_sum += x
        self.y_sum += y
        self.x_sq_sum += x * x
        self.y_sq_sum += y * y
        self.xy_sum += x * y

    def pearson(self) -> float | None:
        if self.count < 2:
            return None

        n = float(self.count)
        numerator = n * self.xy_sum - self.x_sum * self.y_sum
        x_var = n * self.x_sq_sum - self.x_sum * self.x_sum
        y_var = n * self.y_sq_sum - self.y_sum * self.y_sum
        denominator = math.sqrt(max(x_var, 0.0) * max(y_var, 0.0))
        if denominator == 0.0:
            return None
        return numerator / denominator


@dataclass
class CpBucket:
    count: int = 0
    cp_sum: float = 0.0
    cp_sq_sum: float = 0.0
    abs_cp_sum: float = 0.0
    abs_cp_sq_sum: float = 0.0
    label_prob_sum: float = 0.0
    label_entropy_sum: float = 0.0
    prior_bce_sum: float = 0.0
    positive_cp_count: int = 0
    negative_cp_count: int = 0
    zero_cp_count: int = 0
    cp_counts: Counter[int] = field(default_factory=Counter)
    abs_cp_threshold_counts: Counter[int] = field(default_factory=Counter)

    def update(self, relative_cp: int, material_diff: int) -> None:
        label_prob = centipawn_to_probability(relative_cp)
        prior_prob = material_prior_probability(material_diff)
        abs_cp = abs(relative_cp)

        self.count += 1
        self.cp_sum += relative_cp
        self.cp_sq_sum += relative_cp * relative_cp
        self.abs_cp_sum += abs_cp
        self.abs_cp_sq_sum += abs_cp * abs_cp
        self.label_prob_sum += label_prob
        self.label_entropy_sum += binary_entropy(label_prob)
        self.prior_bce_sum += binary_cross_entropy(prior_prob, label_prob)
        self.cp_counts[relative_cp] += 1

        if relative_cp > 0:
            self.positive_cp_count += 1
        elif relative_cp < 0:
            self.negative_cp_count += 1
        else:
            self.zero_cp_count += 1

        for threshold in CP_THRESHOLDS:
            if abs_cp > threshold:
                self.abs_cp_threshold_counts[threshold] += 1

    def report_signed(self, material_diff: int) -> dict[str, Any]:
        material_cp = material_diff * MATERIAL_CP_PER_POINT
        label_mean = self._mean(self.label_prob_sum)
        return {
            "material_diff": material_diff,
            "count": self.count,
            "label_mean": label_mean,
            "cp_from_label_mean": probability_to_centipawn(label_mean),
            "relative_cp_mean": self._mean(self.cp_sum),
            "relative_cp_std": self._std(self.cp_sum, self.cp_sq_sum),
            "relative_cp_quantiles": self._quantile_report(),
            "abs_relative_cp_mean": self._mean(self.abs_cp_sum),
            "abs_relative_cp_std": self._std(self.abs_cp_sum, self.abs_cp_sq_sum),
            "label_entropy_mean": self._mean(self.label_entropy_sum),
            "fixed_material_cp": material_cp,
            "fixed_prior_prob": material_prior_probability(material_diff),
            "fixed_prior_bce": self._mean(self.prior_bce_sum),
            "residual_cp_mean": self._mean(self.cp_sum) - material_cp,
            "residual_cp_std": self._std(self.cp_sum, self.cp_sq_sum),
            "residual_cp_quantiles": self._quantile_report(offset=-material_cp),
            "positive_cp_rate": rate(self.positive_cp_count, self.count),
            "negative_cp_rate": rate(self.negative_cp_count, self.count),
            "zero_cp_rate": rate(self.zero_cp_count, self.count),
            "abs_cp_thresholds": self._threshold_report(),
        }

    def report_absolute(self, abs_material_diff: int) -> dict[str, Any]:
        return {
            "abs_material_diff": abs_material_diff,
            "count": self.count,
            "label_mean": self._mean(self.label_prob_sum),
            "relative_cp_mean": self._mean(self.cp_sum),
            "relative_cp_std": self._std(self.cp_sum, self.cp_sq_sum),
            "relative_cp_quantiles": self._quantile_report(),
            "abs_relative_cp_mean": self._mean(self.abs_cp_sum),
            "abs_relative_cp_std": self._std(self.abs_cp_sum, self.abs_cp_sq_sum),
            "abs_relative_cp_quantiles": self._abs_quantile_report(),
            "label_entropy_mean": self._mean(self.label_entropy_sum),
            "positive_cp_rate": rate(self.positive_cp_count, self.count),
            "negative_cp_rate": rate(self.negative_cp_count, self.count),
            "zero_cp_rate": rate(self.zero_cp_count, self.count),
            "abs_cp_thresholds": self._threshold_report(),
        }

    def _mean(self, value_sum: float) -> float | None:
        if self.count == 0:
            return None
        return value_sum / self.count

    def _std(self, value_sum: float, value_sq_sum: float) -> float | None:
        if self.count == 0:
            return None
        mean = value_sum / self.count
        variance = max(value_sq_sum / self.count - mean * mean, 0.0)
        return math.sqrt(variance)

    def _quantile_report(self, offset: float = 0.0) -> dict[str, float | int | None]:
        return {
            "q10": quantile_from_counts(self.cp_counts, 0.10, offset=offset),
            "q25": quantile_from_counts(self.cp_counts, 0.25, offset=offset),
            "q50": quantile_from_counts(self.cp_counts, 0.50, offset=offset),
            "q75": quantile_from_counts(self.cp_counts, 0.75, offset=offset),
            "q90": quantile_from_counts(self.cp_counts, 0.90, offset=offset),
        }

    def _abs_quantile_report(self) -> dict[str, int | None]:
        abs_counts: Counter[int] = Counter()
        for value, count in self.cp_counts.items():
            abs_counts[abs(value)] += count
        return {
            "q10": quantile_from_counts(abs_counts, 0.10),
            "q25": quantile_from_counts(abs_counts, 0.25),
            "q50": quantile_from_counts(abs_counts, 0.50),
            "q75": quantile_from_counts(abs_counts, 0.75),
            "q90": quantile_from_counts(abs_counts, 0.90),
        }

    def _threshold_report(self) -> list[dict[str, float | int]]:
        return [
            {
                "abs_cp_greater_than": threshold,
                "count": self.abs_cp_threshold_counts[threshold],
                "rate": rate(self.abs_cp_threshold_counts[threshold], self.count),
            }
            for threshold in CP_THRESHOLDS
        ]


class MaterialSignalAccumulator:
    def __init__(self) -> None:
        self.source_rows = 0
        self.accepted_rows = 0
        self.rejected_rows = 0
        self.reject_reasons: Counter[str] = Counter()
        self.material_stats = RunningStats()
        self.abs_material_stats = RunningStats()
        self.cp_stats = RunningStats()
        self.abs_cp_stats = RunningStats()
        self.label_prob_stats = RunningStats()
        self.label_entropy_stats = RunningStats()
        self.signed_material_to_cp = CorrelationStats()
        self.abs_material_to_abs_cp = CorrelationStats()
        self.signed_buckets: dict[int, CpBucket] = {}
        self.absolute_buckets: dict[int, CpBucket] = {}
        self.side_material_buckets: dict[tuple[str, int], CpBucket] = {}
        self.material_nonzero_rows = 0
        self.sign_match_rows = 0
        self.sign_mismatch_rows = 0
        self.sign_neutral_rows = 0

    def update_source_row(self, fen: str, cp: int) -> None:
        self.source_rows += 1
        try:
            board = chess.Board(fen)
        except ValueError:
            self._reject("invalid_fen")
            return

        reason = standard_position_reject_reason(board)
        if reason is not None:
            self._reject(reason)
            return

        us = board.turn
        them = not us
        relative_cp = int(cp if us == chess.WHITE else -cp)
        material_diff = material_score(board, us) - material_score(board, them)
        abs_material_diff = abs(material_diff)
        abs_cp = abs(relative_cp)
        label_prob = centipawn_to_probability(relative_cp)
        label_entropy = binary_entropy(label_prob)

        self.accepted_rows += 1
        self.material_stats.update(material_diff)
        self.abs_material_stats.update(abs_material_diff)
        self.cp_stats.update(relative_cp)
        self.abs_cp_stats.update(abs_cp)
        self.label_prob_stats.update(label_prob)
        self.label_entropy_stats.update(label_entropy)
        self.signed_material_to_cp.update(material_diff, relative_cp)
        self.abs_material_to_abs_cp.update(abs_material_diff, abs_cp)

        self.signed_buckets.setdefault(material_diff, CpBucket()).update(
            relative_cp,
            material_diff,
        )
        self.absolute_buckets.setdefault(abs_material_diff, CpBucket()).update(
            relative_cp,
            material_diff,
        )
        side = "black" if us == chess.BLACK else "white"
        self.side_material_buckets.setdefault((side, material_diff), CpBucket()).update(
            relative_cp,
            material_diff,
        )

        if material_diff != 0:
            self.material_nonzero_rows += 1
            sign_product = material_diff * relative_cp
            if sign_product > 0:
                self.sign_match_rows += 1
            elif sign_product < 0:
                self.sign_mismatch_rows += 1
            else:
                self.sign_neutral_rows += 1

    def metrics(self) -> dict[str, Any]:
        return {
            "source_rows": self.source_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "rejected_ratio": rate(self.rejected_rows, self.source_rows),
            "primary_reject_reasons": dict(sorted(self.reject_reasons.items())),
            **self.material_stats.report("material_diff"),
            **self.abs_material_stats.report("abs_material_diff"),
            **self.cp_stats.report("relative_cp"),
            **self.abs_cp_stats.report("abs_relative_cp"),
            **self.label_prob_stats.report("label_probability"),
            **self.label_entropy_stats.report("label_entropy"),
            "signed_material_vs_relative_cp_pearson": self.signed_material_to_cp.pearson(),
            "abs_material_vs_abs_relative_cp_pearson": self.abs_material_to_abs_cp.pearson(),
            "material_nonzero_rows": self.material_nonzero_rows,
            "material_cp_sign_match_rows": self.sign_match_rows,
            "material_cp_sign_mismatch_rows": self.sign_mismatch_rows,
            "material_cp_sign_neutral_rows": self.sign_neutral_rows,
            "material_cp_sign_match_rate_excluding_neutral": rate(
                self.sign_match_rows,
                self.sign_match_rows + self.sign_mismatch_rows,
            ),
        }

    def signed_material_reports(self) -> list[dict[str, Any]]:
        return [
            bucket.report_signed(material_diff)
            for material_diff, bucket in sorted(self.signed_buckets.items())
        ]

    def absolute_material_reports(self) -> list[dict[str, Any]]:
        return [
            bucket.report_absolute(abs_material_diff)
            for abs_material_diff, bucket in sorted(self.absolute_buckets.items())
        ]

    def side_to_move_reports(self) -> list[dict[str, Any]]:
        return [
            {
                "side_to_move": side,
                **bucket.report_signed(material_diff),
            }
            for (side, material_diff), bucket in sorted(
                self.side_material_buckets.items()
            )
        ]

    def _reject(self, reason: str) -> None:
        self.rejected_rows += 1
        self.reject_reasons[reason] += 1


def centipawn_to_probability(cp: float) -> float:
    logit = cp * math.log(10.0) / 400.0
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def probability_to_centipawn(probability: float | None) -> float | None:
    if probability is None:
        return None
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    return 400.0 * math.log10(probability / (1.0 - probability))


def material_prior_probability(material_diff: int) -> float:
    return centipawn_to_probability(material_diff * MATERIAL_CP_PER_POINT)


def binary_cross_entropy(prediction: float, target: float) -> float:
    prediction = min(max(prediction, 1e-12), 1.0 - 1e-12)
    return -(
        target * math.log(prediction)
        + (1.0 - target) * math.log(1.0 - prediction)
    )


def binary_entropy(probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def quantile_from_counts(
    counts: Counter[int],
    quantile: float,
    *,
    offset: float = 0.0,
) -> float | int | None:
    total = sum(counts.values())
    if total == 0:
        return None

    target_index = int(quantile * (total - 1))
    cumulative = 0
    for value, count in sorted(counts.items()):
        cumulative += count
        if cumulative > target_index:
            result = value + offset
            if float(result).is_integer():
                return int(result)
            return result
    return None


def material_score(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in MATERIAL_VALUES.items()
    )


def standard_position_reject_reason(board: chess.Board) -> str | None:
    white_occupancy = board.occupied_co[chess.WHITE]
    black_occupancy = board.occupied_co[chess.BLACK]
    if white_occupancy & black_occupancy:
        return "overlapping_occupancy"

    for color, color_name in (
        (chess.WHITE, "white"),
        (chess.BLACK, "black"),
    ):
        piece_count = sum(len(board.pieces(piece_type, color))
                          for piece_type in chess.PIECE_TYPES)
        king_count = len(board.pieces(chess.KING, color))
        pawn_count = len(board.pieces(chess.PAWN, color))
        pawns = board.pieces_mask(chess.PAWN, color)

        if king_count != 1:
            return f"{color_name}_king_count"
        if piece_count > 16:
            return f"{color_name}_piece_total"
        if pawn_count > 8:
            return f"{color_name}_pawn_count"
        if pawns & BACK_RANKS:
            return f"{color_name}_pawn_back_rank"

    material_diff = material_score(
        board, chess.WHITE) - material_score(board, chess.BLACK)
    if abs(material_diff) > MAX_ABS_MATERIAL_DIFF:
        return "extreme_material"

    return None


def resolve_selection(
    *,
    source_rows_total: int,
    split: str,
    rows: int | None,
    full: bool,
) -> SourceSelection:
    if full and rows is not None:
        raise ValueError("--rows and --full are mutually exclusive")
    if source_rows_total < 0:
        raise ValueError("source row count cannot be negative")

    train_rows = int(source_rows_total * TRAIN_SET_RATIO)
    validation_rows = int(source_rows_total * VALIDATION_SET_RATIO)
    test_rows = source_rows_total - train_rows - validation_rows
    split_offsets = {
        "train": (0, train_rows),
        "validation": (train_rows, validation_rows),
        "test": (train_rows + validation_rows, test_rows),
        "all": (0, source_rows_total),
    }
    if split not in split_offsets:
        raise ValueError(f"unsupported split: {split}")

    split_offset, split_rows = split_offsets[split]
    if full:
        evaluated_rows = split_rows
        selection = "full"
    else:
        requested_rows = rows if rows is not None else DEFAULT_ROWS
        if requested_rows <= 0:
            raise ValueError("--rows must be greater than zero")
        evaluated_rows = min(requested_rows, split_rows)
        selection = "deterministic-prefix"

    return SourceSelection(
        source_rows_total=source_rows_total,
        split=split,
        split_offset=split_offset,
        split_rows=split_rows,
        evaluated_rows=evaluated_rows,
        selection=selection,
    )


def default_report_path(split: str, rows: int | None, full: bool) -> Path:
    suffix = "full" if full else f"{rows or DEFAULT_ROWS}"
    return REPORTS_DIR / f"source-material-signal.{split}-{suffix}.json"


def build_report(
    *,
    staging_path: Path,
    selection: SourceSelection,
    batch_size: int,
    duration_seconds: float,
    accumulator: MaterialSignalAccumulator,
) -> dict[str, Any]:
    warnings = []
    if selection.split == "validation":
        warnings.append("validation_split_is_model_selection_data")
    if selection.evaluated_rows == 0:
        warnings.append("empty_selection")

    return {
        "schema_version": 1,
        "data": {
            "staging_path": str(staging_path),
            "table": TABLE_NAME,
            "split": selection.split,
            "source_rows_total": selection.source_rows_total,
            "split_offset": selection.split_offset,
            "split_rows": selection.split_rows,
            "evaluated_rows": selection.evaluated_rows,
            "selection": selection.selection,
        },
        "run": {
            "batch_size": batch_size,
            "python_version": platform.python_version(),
            "duckdb_version": duckdb.__version__,
            "python_chess_version": chess.__version__,
            "duration_seconds": duration_seconds,
        },
        "target": {
            "source_cp_perspective": "white-relative cp from staging rows",
            "analysis_cp_perspective": "side-to-move-relative cp",
            "mate_clamp_cp": 2000,
            "probability_transform": "1 / (1 + 10^(-cp / 400))",
            "note": "The source does not include real game outcomes.",
        },
        "material": {
            "piece_values": {
                "pawn": 1,
                "knight": 3,
                "bishop": 3,
                "rook": 5,
                "queen": 9,
                "king": 0,
            },
            "material_cp_per_point": MATERIAL_CP_PER_POINT,
            "max_abs_material_diff": MAX_ABS_MATERIAL_DIFF,
            "strict_filter": [
                "one king per side",
                "at most 16 pieces per side",
                "at most 8 pawns per side",
                "no pawns on back ranks",
                "abs(white_material - black_material) <= 39",
            ],
        },
        "metrics": accumulator.metrics(),
        "signed_material_buckets": accumulator.signed_material_reports(),
        "absolute_material_buckets": accumulator.absolute_material_reports(),
        "side_to_move_material_buckets": accumulator.side_to_move_reports(),
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        report, indent=2, allow_nan=False) + "\n")


def run_material_signal_analysis(
    *,
    split: str = "all",
    staging_path: Path | None = None,
    rows: int | None = None,
    full: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch must be greater than zero")

    resolved_staging_path = staging_path or DUCKDB_TEMP_PATH
    if not resolved_staging_path.exists():
        raise FileNotFoundError(f"{resolved_staging_path} not found")

    with duckdb.connect(str(resolved_staging_path), read_only=True) as conn:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if TABLE_NAME not in tables:
            raise RuntimeError(
                f"{resolved_staging_path} does not contain table '{TABLE_NAME}'"
            )
        source_rows_total = int(
            conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        )
        selection = resolve_selection(
            source_rows_total=source_rows_total,
            split=split,
            rows=rows,
            full=full,
        )
        resolved_output_path = output_path or default_report_path(
            split, rows, full)

        print(f"DuckDB: {duckdb.__version__}")
        print(f"Staging: {resolved_staging_path}")
        print(f"Rows: {selection.evaluated_rows} / {selection.split_rows}")
        print(f"Split: {selection.split}")
        print(f"Selection: {selection.selection}")
        print(f"Batch size: {batch_size}")

        start_time = time.perf_counter()
        accumulator = MaterialSignalAccumulator()
        cursor = conn.execute(
            f"SELECT fen, cp FROM {TABLE_NAME} LIMIT ? OFFSET ?",
            [selection.evaluated_rows, selection.split_offset],
        )
        while True:
            source_rows = cursor.fetchmany(batch_size)
            if not source_rows:
                break
            for fen, cp in source_rows:
                accumulator.update_source_row(fen, int(cp))

    duration_seconds = time.perf_counter() - start_time
    report = build_report(
        staging_path=resolved_staging_path,
        selection=selection,
        batch_size=batch_size,
        duration_seconds=duration_seconds,
        accumulator=accumulator,
    )
    write_report(report, resolved_output_path)

    metrics = report["metrics"]
    print()
    print("Material signal summary")
    print(f"  accepted_rows: {metrics['accepted_rows']}")
    print(
        "  rejected_rows: "
        f"{metrics['rejected_rows']} ({metrics['rejected_ratio'] * 100.0:.4f}%)"
    )
    print(
        "  signed_material_vs_relative_cp_pearson: "
        f"{metrics['signed_material_vs_relative_cp_pearson']:.6f}"
    )
    print(
        "  abs_material_vs_abs_relative_cp_pearson: "
        f"{metrics['abs_material_vs_abs_relative_cp_pearson']:.6f}"
    )
    print(
        "  material_cp_sign_match_rate_excluding_neutral: "
        f"{metrics['material_cp_sign_match_rate_excluding_neutral']:.6f}"
    )
    print(f"Report: {resolved_output_path}")

    return report
