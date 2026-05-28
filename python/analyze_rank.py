"""
Analyze activation rank to determine if model channel capacity is appropriate.

Usage:
    cpe analyze-rank <model-name>

Interpretation:
    - Rapid decay to ~0 (eff_rank << channels): Over-parameterized
    - Gradual decay reaching ~0: Good capacity
    - Plateau, doesn't reach ~0: Under-parameterized
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from libs.dataset import ChessEvaluationDataset
from libs.model import (
    ATTENTION_AFTER_BLOCK,
    ValueOnlyModel,
    model_variant_from_checkpoint,
)
from libs.paths import VALIDATION_DATA_PATH, checkpoint_path


DATASET_PATH = VALIDATION_DATA_PATH
NUM_SAMPLES = 2000
BATCH_SIZE = 256
RANK_THRESHOLD = 0.01  # 1% of max singular value


def run_analyze_rank(model_name: str):
    """Analyze activation rank for the given model."""
    model_path = checkpoint_path(model_name)

    try:
        validate_input_paths(model_path, DATASET_PATH)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples: {NUM_SAMPLES}")
    print()

    # Load model
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)
    model_variant = model_variant_from_checkpoint(checkpoint)
    print(f"Model variant: {model_variant}")
    model = ValueOnlyModel(model_variant=model_variant)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Load dataset
    dataset = ChessEvaluationDataset(DATASET_PATH)
    dataset.open_file()

    # Limit to NUM_SAMPLES
    indices = list(range(min(NUM_SAMPLES, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

    # Register hooks and collect activations
    activations = register_hooks(model)

    print("Collecting activations...")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = model(inputs)

    print(f"Collected {len(activations)} layers")
    print()

    # Analyze each layer
    for name, acts in activations.items():
        # Concatenate all batches: (total_samples, C, H, W)
        all_acts = torch.cat(acts, dim=0)
        singular_values, eff_rank = compute_svd_stats(all_acts)
        print_ascii_plot(name, singular_values, eff_rank)
        print()


def validate_input_paths(model_path: Path, dataset_path: Path) -> None:
    """Fail before model or dataset setup when required artifacts are absent."""
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found")


def register_hooks(model: ValueOnlyModel) -> dict:
    """Register forward hooks to capture activations at key layers."""
    activations = {}

    def make_hook(name):
        activations[name] = []

        def hook(module, input, output):
            # Detach and move to CPU to save memory
            activations[name].append(output.detach().cpu())

        return hook

    # Hook initial_block output
    model.initial_block.register_forward_hook(make_hook("initial_block"))

    if hasattr(model, "blocks"):
        # Hook each current sequential trunk block output.
        for i, block in enumerate(model.blocks):
            block.register_forward_hook(make_hook(f"block_{i}"))
            if i + 1 == ATTENTION_AFTER_BLOCK:
                model.board_attention.register_forward_hook(
                    make_hook("board_attention")
                )
    elif hasattr(model, "trunk") and hasattr(model.trunk, "wide_blocks"):
        for i, block in enumerate(model.trunk.wide_blocks):
            block.register_forward_hook(make_hook(f"wide_block_{i}"))
        model.trunk.compress_wide_to_mid.register_forward_hook(
            make_hook("compress_wide_to_mid")
        )
        for i, block in enumerate(model.trunk.mid_blocks):
            block.register_forward_hook(make_hook(f"mid_block_{i}"))
        model.trunk.compress_mid_to_attention.register_forward_hook(
            make_hook("compress_mid_to_attention")
        )
        for i, block in enumerate(model.trunk.narrow_blocks):
            block.register_forward_hook(make_hook(f"narrow_block_{i}"))
        if hasattr(model.trunk, "attention_blocks"):
            for i, block in enumerate(model.trunk.attention_blocks.layers):
                block.register_forward_hook(make_hook(f"attention_block_{i}"))
        if hasattr(model.trunk, "interleaved_blocks"):
            for i, block in enumerate(model.trunk.interleaved_blocks):
                block.attention.register_forward_hook(
                    make_hook(f"interleaved_attention_{i}")
                )
                block.refresh.register_forward_hook(
                    make_hook(f"interleaved_refresh_{i}")
                )
        if hasattr(model.trunk, "depth_refresh_blocks"):
            for i, block in enumerate(model.trunk.depth_refresh_blocks):
                for j, attention_block in enumerate(block.attention_blocks):
                    attention_block.register_forward_hook(
                        make_hook(f"depth_refresh_attention_{i}_{j}")
                    )
                block.refresh.register_forward_hook(
                    make_hook(f"depth_refresh_{i}")
                )
    elif hasattr(model, "trunk"):
        for i, block in enumerate(model.trunk.shared_blocks):
            block.register_forward_hook(make_hook(f"shared_block_{i}"))
        for i, block in enumerate(model.trunk.local_blocks):
            block.register_forward_hook(make_hook(f"local_block_{i}"))
        for i, block in enumerate(model.trunk.global_blocks.layers):
            block.register_forward_hook(make_hook(f"global_block_{i}"))
        if hasattr(model.trunk, "fuse"):
            model.trunk.fuse.register_forward_hook(make_hook("fuse"))
        if hasattr(model.trunk, "local_evidence"):
            model.trunk.local_evidence.register_forward_hook(
                make_hook("local_evidence")
            )
        if hasattr(model.trunk, "global_evidence"):
            model.trunk.global_evidence.register_forward_hook(
                make_hook("global_evidence")
            )
    else:
        raise ValueError(
            "unsupported model structure for activation rank analysis")

    # Hook named value head modules.
    if hasattr(model.value_head, "conv"):
        model.value_head.conv.register_forward_hook(
            make_hook("value_head_conv"))
    model.value_head.mlp.register_forward_hook(make_hook("value_head_mlp"))

    return activations


def compute_svd_stats(activations: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Compute SVD statistics for activation tensor.

    Args:
        activations: Tensor of shape (N, C, H, W) for conv2d or (N, features) for linear

    Returns:
        (singular_values, effective_rank)
    """
    if activations.dim() == 4:
        # Conv2d: (N, C, H, W) -> (N*H*W, C)
        c = activations.shape[1]
        reshaped = activations.permute(0, 2, 3, 1).reshape(-1, c)
    elif activations.dim() == 2:
        # Linear: (N, features) -> use directly
        reshaped = activations
    else:
        raise ValueError(f"Unexpected tensor dim: {activations.dim()}")

    # Compute SVD (only need singular values)
    # Use float32 for stability
    reshaped = reshaped.float()
    _, s, _ = torch.linalg.svd(reshaped, full_matrices=False)

    # Normalize singular values
    s_normalized = s / s[0]

    # Compute effective rank: count values above threshold
    eff_rank = int((s_normalized > RANK_THRESHOLD).sum().item())

    return s_normalized, eff_rank


def print_ascii_plot(name: str, singular_values: torch.Tensor, eff_rank: int):
    """Print ASCII bar chart of singular value spectrum."""
    n_values = len(singular_values)
    bar_width = 40

    print(f"{name} (eff_rank={eff_rank}/{n_values}):")

    # Show first 10, around threshold, and last few
    indices_to_show = set()

    # First 10
    indices_to_show.update(range(min(10, n_values)))

    # Around effective rank threshold
    if eff_rank > 0:
        indices_to_show.update(
            range(max(0, eff_rank - 2), min(n_values, eff_rank + 3)))

    # Last 3
    indices_to_show.update(range(max(0, n_values - 3), n_values))

    indices_to_show = sorted(indices_to_show)

    prev_idx = -1
    for idx in indices_to_show:
        if prev_idx >= 0 and idx > prev_idx + 1:
            print("  ...")

        val = singular_values[idx].item()
        bar_len = int(val * bar_width)
        bar = "\u2588" * bar_len

        # Mark threshold crossing
        marker = ""
        if idx == eff_rank - 1:
            marker = "  \u2190 threshold"

        print(f"{idx + 1:3d}: {bar:<{bar_width}} {val:.4f}{marker}")
        prev_idx = idx
