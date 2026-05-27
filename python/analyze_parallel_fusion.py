from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from eval_dataset import (
    EvaluationAccumulator,
    json_safe_number,
    resolve_dataset_path,
    select_device,
)
from libs.dataset import ChessEvaluationDataset
from libs.model import (
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
    ValueOnlyModel,
    _material_diff_from_board,
    model_variant_from_checkpoint,
)
from libs.paths import REPORTS_DIR, checkpoint_path


DEFAULT_ROWS = 200_000
DEFAULT_BATCH_SIZE = 2048
TOP_EXAMPLES = 20
LABEL_BUCKETS = tuple(i / 10 for i in range(11))
ABS_MATERIAL_BUCKETS = (0.0, 1.0, 2.0, 3.0, 5.0, 9.0, 15.0, math.inf)


@dataclass
class ModelBundle:
    name: str
    path: Path
    checkpoint: dict[str, Any]
    variant: str
    model: ValueOnlyModel


class TensorStats:
    def __init__(self) -> None:
        self.rows = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.abs_sum = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float()
        count = values.numel()
        if count == 0:
            return

        self.rows += count
        self.sum += values.sum().item()
        self.sum_sq += values.square().sum().item()
        self.abs_sum += values.abs().sum().item()
        self.min_value = min(self.min_value, values.min().item())
        self.max_value = max(self.max_value, values.max().item())

    def report(self) -> dict[str, float | int | None]:
        if self.rows == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "mean_abs": None,
                "rms": None,
                "min": None,
                "max": None,
            }

        mean = self.sum / self.rows
        variance = max(0.0, self.sum_sq / self.rows - mean * mean)
        return {
            "count": self.rows,
            "mean": mean,
            "std": math.sqrt(variance),
            "mean_abs": self.abs_sum / self.rows,
            "rms": math.sqrt(self.sum_sq / self.rows),
            "min": self.min_value,
            "max": self.max_value,
        }


class BucketAccumulator:
    def __init__(self, edges: tuple[float, ...]) -> None:
        if len(edges) < 2:
            raise ValueError("bucket accumulator requires at least two edges")
        self.edges = edges
        self.counts = [0 for _ in range(len(edges) - 1)]
        self.baseline_loss_sums = [0.0 for _ in range(len(edges) - 1)]
        self.parallel_loss_sums = [0.0 for _ in range(len(edges) - 1)]

    def update(
        self,
        values: torch.Tensor,
        baseline_losses: torch.Tensor,
        parallel_losses: torch.Tensor,
    ) -> None:
        values_cpu = values.detach().float().cpu()
        baseline_cpu = baseline_losses.detach().float().cpu()
        parallel_cpu = parallel_losses.detach().float().cpu()

        for value, baseline_loss, parallel_loss in zip(
            values_cpu.tolist(),
            baseline_cpu.tolist(),
            parallel_cpu.tolist(),
            strict=True,
        ):
            index = self._bucket_index(value)
            self.counts[index] += 1
            self.baseline_loss_sums[index] += baseline_loss
            self.parallel_loss_sums[index] += parallel_loss

    def _bucket_index(self, value: float) -> int:
        for index in range(len(self.edges) - 1):
            lower = self.edges[index]
            upper = self.edges[index + 1]
            is_last = index == len(self.edges) - 2
            if lower <= value < upper or (is_last and value <= upper):
                return index
        if value < self.edges[0]:
            return 0
        return len(self.edges) - 2

    def report(self) -> list[dict[str, float | int | str | None]]:
        rows = []
        for index, count in enumerate(self.counts):
            baseline_loss = None
            parallel_loss = None
            delta = None
            if count:
                baseline_loss = self.baseline_loss_sums[index] / count
                parallel_loss = self.parallel_loss_sums[index] / count
                delta = parallel_loss - baseline_loss

            upper = self.edges[index + 1]
            rows.append(
                {
                    "lower": self.edges[index],
                    "upper": "inf" if math.isinf(upper) else upper,
                    "count": count,
                    "baseline_bce": baseline_loss,
                    "parallel_bce": parallel_loss,
                    "parallel_minus_baseline_bce": delta,
                }
            )
        return rows


class TopExamples:
    def __init__(self, limit: int = TOP_EXAMPLES) -> None:
        self.limit = limit
        self.sequence = 0
        self.worst: list[tuple[float, int, dict[str, float | int]]] = []
        self.best: list[tuple[float, int, dict[str, float | int]]] = []

    def update(
        self,
        *,
        start_index: int,
        labels: torch.Tensor,
        material_diff: torch.Tensor,
        baseline_logits: torch.Tensor,
        parallel_logits: torch.Tensor,
        baseline_losses: torch.Tensor,
        parallel_losses: torch.Tensor,
    ) -> None:
        labels_cpu = labels.detach().float().view(-1).cpu()
        material_cpu = material_diff.detach().float().view(-1).cpu()
        baseline_prob_cpu = torch.sigmoid(
            baseline_logits.detach().float().view(-1).cpu()
        )
        parallel_prob_cpu = torch.sigmoid(
            parallel_logits.detach().float().view(-1).cpu()
        )
        baseline_loss_cpu = baseline_losses.detach().float().view(-1).cpu()
        parallel_loss_cpu = parallel_losses.detach().float().view(-1).cpu()

        for offset in range(labels_cpu.numel()):
            delta = (
                parallel_loss_cpu[offset].item()
                - baseline_loss_cpu[offset].item()
            )
            item = {
                "dataset_index": start_index + offset,
                "label": labels_cpu[offset].item(),
                "material_diff": material_cpu[offset].item(),
                "baseline_prob": baseline_prob_cpu[offset].item(),
                "parallel_prob": parallel_prob_cpu[offset].item(),
                "baseline_bce": baseline_loss_cpu[offset].item(),
                "parallel_bce": parallel_loss_cpu[offset].item(),
                "parallel_minus_baseline_bce": delta,
            }
            self._push_worst(delta, item)
            self._push_best(delta, item)

    def _push_worst(self, delta: float, item: dict[str, float | int]) -> None:
        self.sequence += 1
        entry = (delta, self.sequence, item)
        if len(self.worst) < self.limit:
            heapq.heappush(self.worst, entry)
            return
        if delta > self.worst[0][0]:
            heapq.heapreplace(self.worst, entry)

    def _push_best(self, delta: float, item: dict[str, float | int]) -> None:
        key = -delta
        self.sequence += 1
        entry = (key, self.sequence, item)
        if len(self.best) < self.limit:
            heapq.heappush(self.best, entry)
            return
        if key > self.best[0][0]:
            heapq.heapreplace(self.best, entry)

    def report(self) -> dict[str, list[dict[str, float | int]]]:
        return {
            "parallel_worst": [
                item for _, _, item in sorted(
                    self.worst,
                    key=lambda entry: entry[0],
                    reverse=True,
                )
            ],
            "parallel_best": [
                item for _, _, item in sorted(
                    self.best,
                    key=lambda entry: entry[2]["parallel_minus_baseline_bce"],
                )
            ],
        }


def load_model_bundle(
    model_name: str,
    *,
    device: torch.device,
    expected_variant: str | None = None,
) -> ModelBundle:
    path = checkpoint_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    variant = model_variant_from_checkpoint(checkpoint)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"{model_name} variant is {variant!r}, expected {expected_variant!r}"
        )

    model = ValueOnlyModel(model_variant=variant)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return ModelBundle(
        name=model_name,
        path=path,
        checkpoint=checkpoint,
        variant=variant,
        model=model,
    )


def parallel_branch_activations(
    model: ValueOnlyModel,
    inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.model_variant != MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE:
        raise ValueError("parallel diagnostics require parallel-cnn-attn-fuse")

    out = model.add_coords(inputs)
    out = model.initial_block(out)
    shared = model.trunk.shared_blocks(out)
    local = model.trunk.local_blocks(shared)
    global_ = model.trunk.global_blocks(shared)
    return {
        "shared": shared,
        "local": local,
        "global": global_,
    }


def parallel_logits_from_activations(
    model: ValueOnlyModel,
    activations: dict[str, torch.Tensor],
    material_diff: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = activations["local"]
    global_ = activations["global"]
    if mode == "full":
        fused_input = torch.cat([local, global_], dim=1)
        head_material = material_diff
    elif mode == "local_only":
        fused_input = torch.cat([local, torch.zeros_like(global_)], dim=1)
        head_material = material_diff
    elif mode == "global_only":
        fused_input = torch.cat([torch.zeros_like(local), global_], dim=1)
        head_material = material_diff
    elif mode == "no_material":
        fused_input = torch.cat([local, global_], dim=1)
        head_material = torch.zeros_like(material_diff)
    elif mode == "material_only":
        fused_input = torch.zeros_like(torch.cat([local, global_], dim=1))
        head_material = material_diff
    else:
        raise ValueError(f"unsupported parallel diagnostic mode: {mode}")

    fused = model.trunk.fuse(fused_input)
    logits = model.value_head(fused, head_material)
    return logits, fused


def update_activation_stats(
    stats: dict[str, TensorStats],
    activations: dict[str, torch.Tensor],
) -> None:
    for name, tensor in activations.items():
        stats[name].update(tensor)


def update_cosine_stats(
    cosine_stats: TensorStats,
    local: torch.Tensor,
    global_: torch.Tensor,
) -> None:
    local_flat = local.detach().float().flatten(start_dim=1)
    global_flat = global_.detach().float().flatten(start_dim=1)
    cosine = F.cosine_similarity(local_flat, global_flat, dim=1)
    cosine_stats.update(cosine)


def per_sample_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits.float().view(-1),
        labels.float().view(-1),
        reduction="none",
    )


def default_output_path(parallel_model_name: str, split: str, rows: int) -> Path:
    return (
        REPORTS_DIR
        / f"{parallel_model_name}.parallel-diagnostics.{split}.{rows}.json"
    )


def run_parallel_fusion_diagnostics(
    *,
    parallel_model_name: str,
    baseline_model_name: str,
    split: str = "validation",
    dataset_path: Path | None = None,
    rows: int = DEFAULT_ROWS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device_name: str = "auto",
    output_path: Path | None = None,
) -> dict[str, Any]:
    if rows <= 0:
        raise ValueError("--rows must be greater than zero")
    if batch_size <= 0:
        raise ValueError("--batch must be greater than zero")

    resolved_dataset_path = resolve_dataset_path(split, dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"{resolved_dataset_path} not found")

    device = select_device(device_name)
    parallel = load_model_bundle(
        parallel_model_name,
        device=device,
        expected_variant=MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
    )
    baseline = load_model_bundle(baseline_model_name, device=device)

    dataset = ChessEvaluationDataset(str(resolved_dataset_path))
    evaluated_rows = min(rows, len(dataset))
    if evaluated_rows == 0:
        raise ValueError("dataset has no rows to evaluate")

    subset = Subset(dataset, range(evaluated_rows))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Torch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Parallel model: {parallel.path}")
    print(f"Baseline model: {baseline.path}")
    print(f"Dataset: {resolved_dataset_path}")
    print(f"Rows: {evaluated_rows} / {len(dataset)}")
    print(f"Batch size: {batch_size}")

    mode_names = (
        "parallel_full",
        "parallel_local_only",
        "parallel_global_only",
        "parallel_no_material",
        "parallel_material_only",
        "baseline_full",
    )
    accumulators = {
        mode_name: EvaluationAccumulator()
        for mode_name in mode_names
    }
    activation_stats = {
        "shared": TensorStats(),
        "local": TensorStats(),
        "global": TensorStats(),
        "fused": TensorStats(),
    }
    cosine_stats = TensorStats()
    label_buckets = BucketAccumulator(LABEL_BUCKETS)
    abs_material_buckets = BucketAccumulator(ABS_MATERIAL_BUCKETS)
    top_examples = TopExamples()
    parallel_worse_rows = 0
    loss_delta_sum = 0.0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            material_diff = _material_diff_from_board(
                inputs,
                parallel.model.material_weights,
            )

            baseline_full = baseline.model(inputs, material_diff)
            activations = parallel_branch_activations(parallel.model, inputs)
            full_logits, fused = parallel_logits_from_activations(
                parallel.model,
                activations,
                material_diff,
                mode="full",
            )
            local_logits, _ = parallel_logits_from_activations(
                parallel.model,
                activations,
                material_diff,
                mode="local_only",
            )
            global_logits, _ = parallel_logits_from_activations(
                parallel.model,
                activations,
                material_diff,
                mode="global_only",
            )
            no_material_logits, _ = parallel_logits_from_activations(
                parallel.model,
                activations,
                material_diff,
                mode="no_material",
            )
            material_only_logits, _ = parallel_logits_from_activations(
                parallel.model,
                activations,
                material_diff,
                mode="material_only",
            )

            accumulators["baseline_full"].update(baseline_full, labels)
            accumulators["parallel_full"].update(full_logits, labels)
            accumulators["parallel_local_only"].update(local_logits, labels)
            accumulators["parallel_global_only"].update(global_logits, labels)
            accumulators["parallel_no_material"].update(
                no_material_logits, labels
            )
            accumulators["parallel_material_only"].update(
                material_only_logits, labels
            )
            update_activation_stats(
                activation_stats,
                {**activations, "fused": fused},
            )
            update_cosine_stats(
                cosine_stats,
                activations["local"],
                activations["global"],
            )

            baseline_losses = per_sample_bce(baseline_full, labels)
            parallel_losses = per_sample_bce(full_logits, labels)
            deltas = parallel_losses - baseline_losses
            loss_delta_sum += deltas.sum().item()
            parallel_worse_rows += int((deltas > 0).sum().item())

            label_buckets.update(
                labels.view(-1), baseline_losses, parallel_losses)
            abs_material_buckets.update(
                material_diff.abs(),
                baseline_losses,
                parallel_losses,
            )
            top_examples.update(
                start_index=batch_index * batch_size,
                labels=labels,
                material_diff=material_diff,
                baseline_logits=baseline_full,
                parallel_logits=full_logits,
                baseline_losses=baseline_losses,
                parallel_losses=parallel_losses,
            )

    duration_seconds = time.perf_counter() - start_time
    output = output_path or default_output_path(
        parallel_model_name,
        split,
        evaluated_rows,
    )

    mode_metrics = {
        mode_name: accumulators[mode_name].metrics()
        for mode_name in mode_names
    }
    report = {
        "schema_version": 1,
        "models": {
            "parallel": {
                "name": parallel.name,
                "variant": parallel.variant,
                "checkpoint_path": str(parallel.path),
                "checkpoint_epoch": json_safe_number(
                    parallel.checkpoint.get("epoch")
                ),
                "checkpoint_best_validation_loss": json_safe_number(
                    parallel.checkpoint.get("best_validation_loss")
                ),
            },
            "baseline": {
                "name": baseline.name,
                "variant": baseline.variant,
                "checkpoint_path": str(baseline.path),
                "checkpoint_epoch": json_safe_number(
                    baseline.checkpoint.get("epoch")
                ),
                "checkpoint_best_validation_loss": json_safe_number(
                    baseline.checkpoint.get("best_validation_loss")
                ),
            },
        },
        "data": {
            "split": split,
            "dataset_path": str(resolved_dataset_path),
            "dataset_rows": len(dataset),
            "evaluated_rows": evaluated_rows,
            "selection": "deterministic-prefix",
        },
        "run": {
            "device": str(device),
            "batch_size": batch_size,
            "torch_version": torch.__version__,
            "duration_seconds": duration_seconds,
        },
        "metrics_by_mode": mode_metrics,
        "parallel_vs_baseline": {
            "mean_bce_delta": loss_delta_sum / evaluated_rows,
            "parallel_worse_fraction": parallel_worse_rows / evaluated_rows,
            "parallel_worse_rows": parallel_worse_rows,
        },
        "activation_stats": {
            name: stats.report()
            for name, stats in activation_stats.items()
        },
        "local_global_cosine": cosine_stats.report(),
        "label_buckets": label_buckets.report(),
        "abs_material_buckets": abs_material_buckets.report(),
        "top_examples": top_examples.report(),
        "warnings": [
            "branch ablations zero one branch after training and therefore measure intervention sensitivity, not independently trained branch quality",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print()
    print("Parallel fusion diagnostics")
    for mode_name in mode_names:
        metrics = mode_metrics[mode_name]
        print(
            f"  {mode_name}: "
            f"bce={metrics['bce_loss']:.6f} "
            f"ece={metrics['calibration_ece']:.6f} "
            f"cp_mae={metrics['cp_equivalent_mae']:.2f}"
        )
    print(
        "  parallel_minus_baseline_bce: "
        f"{report['parallel_vs_baseline']['mean_bce_delta']:.6f}"
    )
    print(
        "  parallel_worse_fraction: "
        f"{report['parallel_vs_baseline']['parallel_worse_fraction']:.6f}"
    )
    print(f"Report: {output}")

    return report
