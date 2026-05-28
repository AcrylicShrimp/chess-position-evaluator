import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from libs.model import ValueOnlyModel, model_variant_from_checkpoint
from libs.modeling.registry import default_benchmark_model_names
from libs.paths import checkpoint_path, evaluation_report_path


DEFAULT_PARETO_MODELS = default_benchmark_model_names()


@dataclass(frozen=True)
class BenchmarkConfig:
    model_names: tuple[str, ...]
    batch_size: int
    warmup_steps: int
    iterations: int
    compile_mode: str
    device_name: str
    dtype_name: str
    output_path: Path


def select_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized in {"cpu", "cuda", "mps"}:
        return torch.device(normalized)

    raise ValueError(f"unsupported device: {device_name}")


def select_autocast_dtype(dtype_name: str) -> torch.dtype | None:
    normalized = dtype_name.lower()
    if normalized in {"none", "float32", "fp32"}:
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16

    raise ValueError(f"unsupported dtype: {dtype_name}")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluation_summary(model_name: str, split: str) -> dict[str, Any] | None:
    report = load_json(evaluation_report_path(model_name, split))
    if report is None:
        return None

    data = report.get("data", {})
    metrics = report.get("metrics", {})
    run = report.get("run", {})
    return {
        "report_path": str(evaluation_report_path(model_name, split)),
        "rows": data.get("evaluated_rows"),
        "bce_loss": metrics.get("bce_loss"),
        "brier_score": metrics.get("brier_score"),
        "prob_mae": metrics.get("prob_mae"),
        "cp_equivalent_mae": metrics.get("cp_equivalent_mae"),
        "calibration_ece": metrics.get("calibration_ece"),
        "duration_seconds": run.get("duration_seconds"),
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for empty values")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward_latency(
    model: torch.nn.Module,
    board: torch.Tensor,
    material_diff: torch.Tensor,
    *,
    warmup_steps: int,
    iterations: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    autocast_enabled = autocast_dtype is not None and device.type in {
        "cuda",
        "cpu",
    }
    autocast_device = device.type

    def run_once() -> torch.Tensor:
        with torch.inference_mode():
            with torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype or torch.float32,
                enabled=autocast_enabled,
            ):
                return model(board, material_diff)

    synchronize(device)
    first_start = time.perf_counter()
    run_once()
    synchronize(device)
    first_forward_ms = (time.perf_counter() - first_start) * 1000.0

    for _ in range(warmup_steps):
        run_once()
    synchronize(device)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_once()
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(latencies)
    return {
        "first_forward_ms": first_forward_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "iterations": iterations,
        "warmup_steps": warmup_steps,
        "examples_per_second": board.shape[0] * 1000.0 / mean_ms,
    }


def benchmark_model(
    model_name: str,
    config: BenchmarkConfig,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    path = checkpoint_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    variant = model_variant_from_checkpoint(checkpoint)
    model = ValueOnlyModel(model_variant=variant)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    if config.compile_mode != "none":
        model = torch.compile(model, mode=config.compile_mode)

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    board = torch.randn(
        config.batch_size,
        20,
        8,
        8,
        generator=generator,
        device=device,
    )
    material_diff = torch.zeros(config.batch_size, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timing = measure_forward_latency(
        model,
        board,
        material_diff,
        warmup_steps=config.warmup_steps,
        iterations=config.iterations,
        device=device,
        autocast_dtype=autocast_dtype,
    )

    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)

    result = {
        "name": model_name,
        "variant": variant,
        "checkpoint_path": str(path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_validation_loss": checkpoint.get(
            "best_validation_loss"
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "timing": timing,
        "peak_memory_bytes": peak_memory,
        "validation": evaluation_summary(model_name, "validation"),
        "test": evaluation_summary(model_name, "test"),
    }

    del model
    del checkpoint
    del board
    del material_diff
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def run_pareto_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    device = select_device(config.device_name)
    autocast_dtype = select_autocast_dtype(config.dtype_name)

    if config.compile_mode != "none" and device.type not in {"cuda", "cpu"}:
        raise ValueError(
            "torch.compile benchmark is only supported on cpu/cuda here; "
            "use --compile-mode none for this device"
        )

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for model_name in config.model_names:
        print(f"[benchmark] {model_name}", flush=True)
        results.append(
            benchmark_model(
                model_name,
                config,
                device,
                autocast_dtype,
            )
        )

    completed = time.strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "schema_version": 1,
        "started_at": started,
        "completed_at": completed,
        "config": {
            "models": list(config.model_names),
            "batch_size": config.batch_size,
            "warmup_steps": config.warmup_steps,
            "iterations": config.iterations,
            "compile_mode": config.compile_mode,
            "requested_device": config.device_name,
            "device": str(device),
            "dtype": config.dtype_name,
            "torchinductor_cache_dir": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR"
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "results": results,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"[benchmark] wrote {config.output_path}", flush=True)
    return report
