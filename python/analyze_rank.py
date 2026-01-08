"""
Analyze activation rank to determine if model channel capacity is appropriate.

Usage:
    cpe analyze-rank <model-name>

Interpretation:
    - Rapid decay to ~0 (eff_rank << channels): Over-parameterized
    - Gradual decay reaching ~0: Good capacity
    - Plateau, doesn't reach ~0: Under-parameterized
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

from libs.dataset import ChessEvaluationDataset
from libs.model import EvalOnlyModel


CHECKPOINTS_DIR = "models/checkpoints"
DATASET_PATH = "validation.chesseval"
NUM_SAMPLES = 2000
BATCH_SIZE = 256
RANK_THRESHOLD = 0.01  # 1% of max singular value


def run_analyze_rank(model_name: str):
    """Analyze activation rank for the given model."""
    model_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        sys.exit(1)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found")
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
    model = EvalOnlyModel()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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


def register_hooks(model: EvalOnlyModel) -> dict:
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

    # Hook each residual block output
    for i, block in enumerate(model.residual_blocks):
        model.residual_blocks[i].register_forward_hook(make_hook(f"residual_block_{i}"))

    # Hook value_head linear layer (index 4: Linear 128->128)
    model.value_head[4].register_forward_hook(make_hook("value_head_linear"))

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
        indices_to_show.update(range(max(0, eff_rank - 2), min(n_values, eff_rank + 3)))

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


