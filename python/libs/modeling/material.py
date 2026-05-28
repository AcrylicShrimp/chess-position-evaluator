import math

import torch


# Order matches chess.PIECE_TYPES (pawn, knight, bishop, rook, queen, king)
_MATERIAL_VALUES = torch.tensor(
    [1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=torch.float32)
_MATERIAL_ALPHA = 5.0
_MATERIAL_LOGIT_SCALE = math.log(10.0) / 4.0
_MATERIAL_FEATURE_COUNT = 3


def _material_diff_from_board(
    board: torch.Tensor,
    material_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute raw material difference (our material - enemy material).

    Args:
        board: Board tensor shaped [B, 20, 8, 8] before coords are added.
        material_weights: Piece values shaped [6].

    Returns:
        Tensor of shape [B].
    """
    if not torch.jit.is_tracing() and (board.dim() != 4 or board.shape[1] != 20):
        raise ValueError(
            "material diff requires a board tensor shaped [B, 20, 8, 8]"
        )

    # Channels: 0-4 meta, 5 en-passant, 6-11 ours, 12-17 theirs, 18-19 heatmaps.
    our_pieces = board[:, 6:12]
    enemy_pieces = board[:, 12:18]
    weights = material_weights.to(dtype=board.dtype, device=board.device).view(
        1, 6, 1, 1
    )
    our_score = (our_pieces * weights).sum(dim=(1, 2, 3))
    enemy_score = (enemy_pieces * weights).sum(dim=(1, 2, 3))
    return our_score - enemy_score


def _material_feature(
    x: torch.Tensor,
    material_weights: torch.Tensor,
    material_scale: float,
    material_diff: torch.Tensor | None = None,
    alpha: float = _MATERIAL_ALPHA,
) -> torch.Tensor:
    """
    Compute (or inject) a normalized material difference feature.

    Args:
        x: Board tensor shaped [B, 20, 8, 8] when material_diff is not provided.
        material_weights: Piece values shaped [6].
        material_scale: Scalar multiplier.
        material_diff: Optional precomputed raw material diff (my - enemy),
            shaped [B] or [B, 1]. If None, it is derived from x.
        alpha: Normalization factor for tanh.

    Returns:
        Tensor of shape [B, 1] to concatenate after flattening.
    """
    batch = x.shape[0]

    if material_diff is None:
        diff = _material_diff_from_board(x, material_weights)
    else:
        diff = material_diff.to(device=x.device, dtype=x.dtype).reshape(batch)

    normalized = torch.tanh(diff / alpha)
    scaled = material_scale * normalized
    return scaled.unsqueeze(1)


def _material_prior_logit(
    material_diff: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    diff = material_diff.to(device=device, dtype=dtype).reshape(-1, 1)
    return diff * _MATERIAL_LOGIT_SCALE


def _material_input_features(
    x: torch.Tensor,
    material_diff: torch.Tensor,
    alpha: float = _MATERIAL_ALPHA,
) -> torch.Tensor:
    batch = x.shape[0]
    diff = material_diff.to(device=x.device, dtype=x.dtype).reshape(batch)
    signed_material = torch.tanh(diff / alpha)
    abs_material = torch.tanh(diff.abs() / alpha)
    material_prior_prob = torch.sigmoid(diff * _MATERIAL_LOGIT_SCALE)
    return torch.stack(
        (signed_material, abs_material, material_prior_prob),
        dim=1,
    )
