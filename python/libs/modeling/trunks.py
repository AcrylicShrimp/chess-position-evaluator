import torch

from libs.modeling.attention import BoardAttentionBlock, BoardAttentionStack
from libs.modeling.constants import (
    ATTENTION_AFTER_BLOCK,
    ATTENTION_FFN_HIDDEN,
    ATTENTION_HEAD_DIM,
    ATTENTION_HEADS,
    ATTENTION_LAYERS,
    BLOCKS,
    CHANNELS,
    EDGE_GATE_K_ONLY,
    EDGE_GATE_QK,
    FUNNEL_ATTENTION_CHANNELS,
    FUNNEL_ATTENTION_LAYERS,
    FUNNEL_ATTENTION_PER_REFRESH,
    FUNNEL_BLOCKS_PER_STAGE,
    FUNNEL_INITIAL_CHANNELS,
    FUNNEL_INTERLEAVED_STAGES,
    FUNNEL_MID_CHANNELS,
    FUNNEL_REFRESH_STAGES,
)
from libs.modeling.primitives import ChannelProjection, GhostShuffleBlock


class ParallelCnnAttentionTrunk(torch.nn.Module):
    def __init__(self, edge_gate_mode: str = EDGE_GATE_QK):
        super().__init__()
        self.shared_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(ATTENTION_AFTER_BLOCK)
            ]
        )
        self.local_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(BLOCKS - ATTENTION_AFTER_BLOCK)
            ]
        )
        self.global_blocks = BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
            edge_gate_mode=edge_gate_mode,
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS * 2, CHANNELS, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_blocks(x)
        local = self.local_blocks(shared)
        global_ = self.global_blocks(shared)
        return self.fuse(torch.cat([local, global_], dim=1))


class ParallelCnnAttentionAlignedAddTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(ATTENTION_AFTER_BLOCK)
            ]
        )
        self.local_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(BLOCKS - ATTENTION_AFTER_BLOCK)
            ]
        )
        self.global_blocks = BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
        )
        self.local_aligner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Conv2d(CHANNELS, CHANNELS, kernel_size=1, bias=False),
        )
        self.global_aligner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Conv2d(CHANNELS, CHANNELS, kernel_size=1, bias=False),
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_blocks(x)
        local = self.local_blocks(shared)
        global_ = self.global_blocks(shared)
        local_aligned = self.local_aligner(local)
        global_aligned = self.global_aligner(global_)
        return self.fuse(local_aligned + global_aligned)


class ParallelCnnAttentionKEdgeLateEvidenceTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(ATTENTION_AFTER_BLOCK)
            ]
        )
        self.local_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(BLOCKS - ATTENTION_AFTER_BLOCK)
            ]
        )
        self.global_blocks = BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
            edge_gate_mode=EDGE_GATE_K_ONLY,
        )
        self.local_evidence = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )
        self.global_evidence = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_blocks(x)
        local = self.local_blocks(shared)
        global_ = self.global_blocks(shared)
        local_evidence = self.local_evidence(local)
        global_evidence = self.global_evidence(global_)
        return torch.cat([local_evidence, global_evidence], dim=1)


class FunnelCnnAttentionTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wide_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_INITIAL_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_wide_to_mid = ChannelProjection(
            FUNNEL_INITIAL_CHANNELS,
            FUNNEL_MID_CHANNELS,
        )
        self.mid_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_MID_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_mid_to_attention = ChannelProjection(
            FUNNEL_MID_CHANNELS,
            FUNNEL_ATTENTION_CHANNELS,
        )
        self.narrow_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_ATTENTION_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.attention_blocks = BoardAttentionStack(
            FUNNEL_ATTENTION_LAYERS,
            FUNNEL_ATTENTION_CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wide_blocks(x)
        x = self.compress_wide_to_mid(x)
        x = self.mid_blocks(x)
        x = self.compress_mid_to_attention(x)
        x = self.narrow_blocks(x)
        return self.attention_blocks(x)


class AttentionCnnRefreshBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = BoardAttentionBlock(
            channels,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
        )
        self.refresh = GhostShuffleBlock(channels, ratio=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return self.refresh(x)


class FunnelInterleavedAttentionTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wide_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_INITIAL_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_wide_to_mid = ChannelProjection(
            FUNNEL_INITIAL_CHANNELS,
            FUNNEL_MID_CHANNELS,
        )
        self.mid_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_MID_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_mid_to_attention = ChannelProjection(
            FUNNEL_MID_CHANNELS,
            FUNNEL_ATTENTION_CHANNELS,
        )
        self.narrow_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_ATTENTION_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.interleaved_blocks = torch.nn.ModuleList(
            [
                AttentionCnnRefreshBlock(FUNNEL_ATTENTION_CHANNELS)
                for _ in range(FUNNEL_INTERLEAVED_STAGES)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wide_blocks(x)
        x = self.compress_wide_to_mid(x)
        x = self.mid_blocks(x)
        x = self.compress_mid_to_attention(x)
        x = self.narrow_blocks(x)
        for block in self.interleaved_blocks:
            x = block(x)
        return x


class AttentionPairCnnRefreshBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention_blocks = torch.nn.ModuleList(
            [
                BoardAttentionBlock(
                    channels,
                    ATTENTION_HEADS,
                    ATTENTION_HEAD_DIM,
                    ATTENTION_FFN_HIDDEN,
                )
                for _ in range(FUNNEL_ATTENTION_PER_REFRESH)
            ]
        )
        self.refresh = GhostShuffleBlock(channels, ratio=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        return self.refresh(x)


class FunnelDepthRefreshAttentionTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wide_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_INITIAL_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_wide_to_mid = ChannelProjection(
            FUNNEL_INITIAL_CHANNELS,
            FUNNEL_MID_CHANNELS,
        )
        self.mid_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_MID_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_mid_to_attention = ChannelProjection(
            FUNNEL_MID_CHANNELS,
            FUNNEL_ATTENTION_CHANNELS,
        )
        self.narrow_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_ATTENTION_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.depth_refresh_blocks = torch.nn.ModuleList(
            [
                AttentionPairCnnRefreshBlock(FUNNEL_ATTENTION_CHANNELS)
                for _ in range(FUNNEL_REFRESH_STAGES)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wide_blocks(x)
        x = self.compress_wide_to_mid(x)
        x = self.mid_blocks(x)
        x = self.compress_mid_to_attention(x)
        x = self.narrow_blocks(x)
        for block in self.depth_refresh_blocks:
            x = block(x)
        return x


def _run_trunk_blocks(
    x: torch.Tensor,
    blocks: torch.nn.Sequential,
    board_attention: torch.nn.Module,
) -> torch.Tensor:
    for index, block in enumerate(blocks):
        x = block(x)
        if index + 1 == ATTENTION_AFTER_BLOCK:
            x = board_attention(x)
    return x
