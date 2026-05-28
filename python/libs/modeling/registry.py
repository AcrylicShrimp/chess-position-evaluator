from dataclasses import dataclass
from typing import Literal

from libs.modeling.constants import (
    MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
    MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION,
    MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION,
    MODEL_VARIANT_NO_ATTENTION,
    MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL,
    MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
)

ModelVariantStatus = Literal[
    "active",
    "baseline",
    "experimental",
    "dormant",
]


@dataclass(frozen=True)
class ModelVariantSpec:
    id: str
    family: str
    status: ModelVariantStatus
    description: str
    material_behavior: str
    expected_params: int | None
    uses_material_diff: bool
    benchmark_default: bool


@dataclass(frozen=True)
class BenchmarkCheckpointSpec:
    checkpoint_name: str
    variant_id: str
    role: str


MODEL_VARIANTS: dict[str, ModelVariantSpec] = {
    MODEL_VARIANT_STACKED_EDGE_GATE_FFN: ModelVariantSpec(
        id=MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
        family="stacked-attention",
        status="active",
        description="Six-block 256-channel trunk with stacked edge-gated board attention.",
        material_behavior="material_diff feature in value head",
        expected_params=457685,
        uses_material_diff=True,
        benchmark_default=True,
    ),
    MODEL_VARIANT_ONE_LAYER_EDGE_GATE: ModelVariantSpec(
        id=MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
        family="stacked-attention",
        status="experimental",
        description="Six-block 256-channel trunk with one edge-gated board attention layer.",
        material_behavior="material_diff feature in value head",
        expected_params=223573,
        uses_material_diff=True,
        benchmark_default=True,
    ),
    MODEL_VARIANT_NO_ATTENTION: ModelVariantSpec(
        id=MODEL_VARIANT_NO_ATTENTION,
        family="cnn-baseline",
        status="baseline",
        description="Six-block 256-channel CNN trunk without board attention.",
        material_behavior="material_diff feature in value head",
        expected_params=155861,
        uses_material_diff=True,
        benchmark_default=True,
    ),
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE: ModelVariantSpec(
        id=MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
        family="parallel-fusion",
        status="dormant",
        description="Shared CNN stem with parallel CNN and attention branches fused by 1x1 projection.",
        material_behavior="material probability inputs in value head",
        expected_params=589397,
        uses_material_diff=True,
        benchmark_default=False,
    ),
    MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD: ModelVariantSpec(
        id=MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
        family="parallel-fusion",
        status="experimental",
        description="Parallel CNN and attention branches aligned into a shared space before addition.",
        material_behavior="material probability inputs in value head",
        expected_params=590421,
        uses_material_diff=True,
        benchmark_default=False,
    ),
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL: ModelVariantSpec(
        id=MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
        family="parallel-fusion",
        status="experimental",
        description="Parallel CNN and attention branch fusion without material-score value inputs.",
        material_behavior="no material_diff input to value head",
        expected_params=589205,
        uses_material_diff=False,
        benchmark_default=True,
    ),
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL: ModelVariantSpec(
        id=MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
        family="parallel-fusion",
        status="experimental",
        description="Parallel fusion with k-only dynamic edge gating and no material-score value inputs.",
        material_behavior="no material_diff input to value head",
        expected_params=585941,
        uses_material_diff=False,
        benchmark_default=True,
    ),
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL: ModelVariantSpec(
        id=MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL,
        family="parallel-fusion",
        status="experimental",
        description="Parallel k-edge attention with late local/global evidence maps and no material-score inputs.",
        material_behavior="no material_diff input to value head",
        expected_params=463065,
        uses_material_diff=False,
        benchmark_default=False,
    ),
    MODEL_VARIANT_FUNNEL_CNN_ATTENTION: ModelVariantSpec(
        id=MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
        family="funnel-attention",
        status="experimental",
        description="CNN channel funnel ending in six edge-gated attention layers.",
        material_behavior="material_diff feature in value head",
        expected_params=474197,
        uses_material_diff=True,
        benchmark_default=True,
    ),
    MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION: ModelVariantSpec(
        id=MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION,
        family="funnel-attention",
        status="experimental",
        description="CNN channel funnel with interleaved attention and CNN refresh blocks.",
        material_behavior="material_diff feature in value head",
        expected_params=336509,
        uses_material_diff=True,
        benchmark_default=False,
    ),
    MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION: ModelVariantSpec(
        id=MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION,
        family="funnel-attention",
        status="active",
        description="CNN channel funnel with paired attention layers and CNN refresh stages.",
        material_behavior="material_diff feature in value head",
        expected_params=490877,
        uses_material_diff=True,
        benchmark_default=True,
    ),
}

SUPPORTED_MODEL_VARIANTS = tuple(MODEL_VARIANTS.keys())

DEFAULT_BENCHMARK_CHECKPOINTS = (
    BenchmarkCheckpointSpec(
        checkpoint_name=(
            "ghost-ca-r4-256ch-mhattn4x16-edgegate3-ffn64-mid1-cleaneddata-"
            "warmupcosine5-lr7p5e4-100e-best"
        ),
        variant_id=MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
        role="best stacked attention cleaned-data checkpoint",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name=(
            "ghost-ca-r4-256ch-mhattn4x16-edgegate3-ffn64-mid1-cleaneddata-"
            "warmupcosine5-lr5e4-100e-best"
        ),
        variant_id=MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
        role="stacked attention lr5e4 comparison checkpoint",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name="ghost-ca-r4-256ch-noattn-warmupcosine5-lr5e4-100e-best",
        variant_id=MODEL_VARIANT_NO_ATTENTION,
        role="no-attention baseline",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name="ghost-ca-r4-256ch-edgegate1-warmupcosine5-lr5e4-100e-best",
        variant_id=MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
        role="one-layer edge-gate attention ablation",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name=(
            "ghost-ca-r4-funnel-cnn224-160-128-attn6-edgegate-"
            "warmupcosine5-lr5e4-100e-best"
        ),
        variant_id=MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
        role="funnel attention checkpoint",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name=(
            "ghost-ca-r4-funnel-cnn224-160-128-attn6-refresh3-edgegate-"
            "warmupcosine5-lr7p5e4-100e-best"
        ),
        variant_id=MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION,
        role="funnel depth-refresh checkpoint",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name=(
            "ghost-ca-r4-256ch-parallel-cnn3-attn3ffn64-fuse1x1-nomaterial-"
            "warmupcosine5-lr5e4-100e-best"
        ),
        variant_id=MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
        role="parallel fusion without material inputs",
    ),
    BenchmarkCheckpointSpec(
        checkpoint_name=(
            "ghost-ca-r4-256ch-parallel-cnn3-attn3ffn64-kedge-fuse1x1-"
            "nomaterial-warmupcosine5-lr5e4-100e-best"
        ),
        variant_id=MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
        role="k-edge parallel fusion without material inputs",
    ),
)


def get_model_variant_spec(variant_id: str) -> ModelVariantSpec:
    try:
        return MODEL_VARIANTS[variant_id]
    except KeyError as exc:
        allowed = allowed_model_variant_text()
        raise ValueError(
            f"unsupported model variant {variant_id!r}; expected one of: {allowed}"
        ) from exc


def allowed_model_variant_text() -> str:
    return ", ".join(SUPPORTED_MODEL_VARIANTS)


def default_benchmark_model_names() -> tuple[str, ...]:
    return tuple(spec.checkpoint_name for spec in DEFAULT_BENCHMARK_CHECKPOINTS)
