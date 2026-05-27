from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import duckdb

from analyze_material_signal import (
    TABLE_NAME,
    centipawn_to_probability,
    material_score,
    resolve_selection,
    standard_position_reject_reason,
)
from libs.paths import DUCKDB_TEMP_PATH, REPORTS_DIR


DEFAULT_BATCH_SIZE = 20_000
LABEL_MATCH_TOLERANCE = 1e-5
MATERIAL_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TraceRequest:
    dataset_index: int
    source_group: str
    report_item: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceTrace:
    request: TraceRequest
    source_row_offset: int
    source_row_number: int
    fen: str
    cp: int
    side_to_move: str
    relative_cp: int
    label_probability: float
    material_diff: int


def is_preprocess_accepted_board(board: chess.Board) -> bool:
    return (
        board.status() == chess.STATUS_VALID
        and standard_position_reject_reason(board) is None
    )


def report_trace_path(report_path: Path, split: str, top_worst: int, top_best: int) -> Path:
    return (
        REPORTS_DIR
        / f"{report_path.stem}.source-trace.{split}.worst{top_worst}.best{top_best}.json"
    )


def load_trace_requests_from_report(
    report_path: Path,
    *,
    top_worst: int,
    top_best: int,
) -> list[TraceRequest]:
    if top_worst < 0 or top_best < 0:
        raise ValueError("--top-worst and --top-best must be non-negative")
    if top_worst == 0 and top_best == 0:
        raise ValueError(
            "at least one of --top-worst or --top-best must be positive")

    report = json.loads(report_path.read_text())
    top_examples = report.get("top_examples", {})

    requests: list[TraceRequest] = []
    seen: set[int] = set()
    for group_name, limit in (
        ("parallel_worst", top_worst),
        ("parallel_best", top_best),
    ):
        for item in top_examples.get(group_name, [])[:limit]:
            dataset_index = int(item["dataset_index"])
            if dataset_index in seen:
                continue
            seen.add(dataset_index)
            requests.append(
                TraceRequest(
                    dataset_index=dataset_index,
                    source_group=group_name,
                    report_item=item,
                )
            )

    return sorted(requests, key=lambda request: request.dataset_index)


def trace_processed_rows(
    *,
    staging_path: Path,
    split: str,
    requests: list[TraceRequest],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[str, Any], list[SourceTrace]]:
    if batch_size <= 0:
        raise ValueError("--batch must be greater than zero")
    if not requests:
        raise ValueError("no rows requested")
    if not staging_path.exists():
        raise FileNotFoundError(f"{staging_path} not found")

    targets = {request.dataset_index: request for request in requests}
    max_target = max(targets)
    if min(targets) < 0:
        raise ValueError("dataset indices must be non-negative")

    traces: list[SourceTrace] = []
    rejected_rows_before_last_target = 0

    with duckdb.connect(str(staging_path), read_only=True) as conn:
        if not table_exists(conn, TABLE_NAME):
            raise RuntimeError(
                f"{staging_path} does not contain table '{TABLE_NAME}'")

        source_rows_total = int(
            conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        )
        selection = resolve_selection(
            source_rows_total=source_rows_total,
            split=split,
            rows=None,
            full=True,
        )

        accepted_index = -1
        source_offset = 0
        while source_offset < selection.split_rows and len(traces) < len(targets):
            remaining = selection.split_rows - source_offset
            limit = min(batch_size, remaining)
            rows = conn.execute(
                f"SELECT fen, cp FROM {TABLE_NAME} LIMIT ? OFFSET ?",
                [limit, selection.split_offset + source_offset],
            ).fetchall()
            if not rows:
                break

            for local_offset, (fen, cp) in enumerate(rows):
                source_row_offset = source_offset + local_offset
                try:
                    board = chess.Board(fen)
                except ValueError:
                    rejected_rows_before_last_target += 1
                    continue

                if not is_preprocess_accepted_board(board):
                    rejected_rows_before_last_target += 1
                    continue

                accepted_index += 1
                if accepted_index not in targets:
                    if accepted_index > max_target:
                        break
                    continue

                request = targets[accepted_index]
                side_to_move = "black" if board.turn == chess.BLACK else "white"
                them = not board.turn
                relative_cp = int(cp if board.turn == chess.WHITE else -cp)
                material_diff = material_score(board, board.turn) - material_score(
                    board, them
                )
                traces.append(
                    SourceTrace(
                        request=request,
                        source_row_offset=source_row_offset,
                        source_row_number=selection.split_offset + source_row_offset,
                        fen=fen,
                        cp=int(cp),
                        side_to_move=side_to_move,
                        relative_cp=relative_cp,
                        label_probability=centipawn_to_probability(
                            relative_cp),
                        material_diff=material_diff,
                    )
                )

                if len(traces) == len(targets):
                    break

            source_offset += len(rows)

    if len(traces) != len(targets):
        found = {trace.request.dataset_index for trace in traces}
        missing = sorted(set(targets) - found)
        raise RuntimeError(
            f"could not trace processed dataset indices: {missing}")
    validate_traces_against_report(traces)

    metadata = {
        "source_rows_total": source_rows_total,
        "split": split,
        "split_offset": selection.split_offset,
        "split_rows": selection.split_rows,
        "max_requested_dataset_index": max_target,
        "source_rows_scanned": source_offset,
        "rejected_rows_before_last_target": rejected_rows_before_last_target,
        "mapping_contract": (
            "processed split index is the zero-based accepted-row index within "
            "the split source window after applying python-chess STATUS_VALID "
            "and the repository strict preprocessing filter"
        ),
    }
    return metadata, sorted(traces, key=lambda trace: trace.request.dataset_index)


def validate_traces_against_report(traces: list[SourceTrace]) -> None:
    mismatches = []
    for trace in traces:
        item = trace.request.report_item
        if not item:
            continue

        label = item.get("label")
        if (
            label is not None
            and abs(trace.label_probability - float(label)) > LABEL_MATCH_TOLERANCE
        ):
            mismatches.append(
                f"idx={trace.request.dataset_index} label "
                f"{trace.label_probability} != {label}"
            )

        material_diff = item.get("material_diff")
        if (
            material_diff is not None
            and abs(trace.material_diff - float(material_diff))
            > MATERIAL_MATCH_TOLERANCE
        ):
            mismatches.append(
                f"idx={trace.request.dataset_index} material "
                f"{trace.material_diff} != {material_diff}"
            )

    if mismatches:
        preview = "; ".join(mismatches[:5])
        raise RuntimeError(
            f"source trace did not match diagnostic rows: {preview}")


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        > 0
    )


def build_trace_report(
    *,
    diagnostic_report_path: Path,
    staging_path: Path,
    split: str,
    top_worst: int,
    top_best: int,
    metadata: dict[str, Any],
    traces: list[SourceTrace],
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "diagnostic_report_path": str(diagnostic_report_path),
        "staging_path": str(staging_path),
        "split": split,
        "top_worst": top_worst,
        "top_best": top_best,
        "run": {
            "duration_seconds": duration_seconds,
        },
        "mapping": metadata,
        "rows": [trace_to_report_row(trace) for trace in traces],
    }


def trace_to_report_row(trace: SourceTrace) -> dict[str, Any]:
    report_item = trace.request.report_item or {}
    label_from_report = report_item.get("label")
    material_from_report = report_item.get("material_diff")
    return {
        "dataset_index": trace.request.dataset_index,
        "source_group": trace.request.source_group,
        "source_row_offset": trace.source_row_offset,
        "source_row_number": trace.source_row_number,
        "fen": trace.fen,
        "cp": trace.cp,
        "side_to_move": trace.side_to_move,
        "relative_cp": trace.relative_cp,
        "label_probability": trace.label_probability,
        "material_diff": trace.material_diff,
        "report_label": label_from_report,
        "report_material_diff": material_from_report,
        "label_probability_delta": (
            None
            if label_from_report is None
            else trace.label_probability - float(label_from_report)
        ),
        "material_diff_delta": (
            None
            if material_from_report is None
            else trace.material_diff - float(material_from_report)
        ),
        "baseline_prob": report_item.get("baseline_prob"),
        "parallel_prob": report_item.get("parallel_prob"),
        "baseline_bce": report_item.get("baseline_bce"),
        "parallel_bce": report_item.get("parallel_bce"),
        "parallel_minus_baseline_bce": report_item.get(
            "parallel_minus_baseline_bce"
        ),
    }


def write_trace_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        report, indent=2, allow_nan=False) + "\n")


def run_trace_processed_rows(
    *,
    diagnostic_report_path: Path,
    split: str = "validation",
    staging_path: Path | None = None,
    top_worst: int = 10,
    top_best: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not diagnostic_report_path.exists():
        raise FileNotFoundError(f"{diagnostic_report_path} not found")

    resolved_staging_path = staging_path or DUCKDB_TEMP_PATH
    requests = load_trace_requests_from_report(
        diagnostic_report_path,
        top_worst=top_worst,
        top_best=top_best,
    )

    start_time = time.perf_counter()
    metadata, traces = trace_processed_rows(
        staging_path=resolved_staging_path,
        split=split,
        requests=requests,
        batch_size=batch_size,
    )
    duration_seconds = time.perf_counter() - start_time
    report = build_trace_report(
        diagnostic_report_path=diagnostic_report_path,
        staging_path=resolved_staging_path,
        split=split,
        top_worst=top_worst,
        top_best=top_best,
        metadata=metadata,
        traces=traces,
        duration_seconds=duration_seconds,
    )
    output = output_path or report_trace_path(
        diagnostic_report_path,
        split,
        top_worst,
        top_best,
    )
    write_trace_report(report, output)

    print(f"DuckDB: {duckdb.__version__}")
    print(f"Staging: {resolved_staging_path}")
    print(f"Diagnostic report: {diagnostic_report_path}")
    print(f"Split: {split}")
    print(f"Rows traced: {len(traces)}")
    print(f"Source rows scanned: {metadata['source_rows_scanned']}")
    print(f"Report: {output}")
    for row in report["rows"][: min(len(report["rows"]), 5)]:
        print(
            "  "
            f"idx={row['dataset_index']} "
            f"group={row['source_group']} "
            f"cp={row['cp']} "
            f"rel_cp={row['relative_cp']} "
            f"mat={row['material_diff']} "
            f"parallel_prob={row['parallel_prob']} "
            f"fen={row['fen']}"
        )

    return report
