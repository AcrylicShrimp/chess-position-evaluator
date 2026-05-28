CHANNELS = 256
BLOCKS = 6
ATTENTION_AFTER_BLOCK = 3
MODEL_VARIANT_STACKED_EDGE_GATE_FFN = "stacked-edge-gate-ffn"
MODEL_VARIANT_ONE_LAYER_EDGE_GATE = "one-layer-edge-gate"
MODEL_VARIANT_NO_ATTENTION = "no-attention"
MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE = "parallel-cnn-attn-fuse"
MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD = "parallel-cnn-attn-aligned-add"
MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL = (
    "parallel-cnn-attn-fuse-no-material"
)
MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL = (
    "parallel-cnn-attn-kedge-fuse-no-material"
)
MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL = (
    "parallel-cnn-attn-kedge-lateevidence-no-material"
)
MODEL_VARIANT_FUNNEL_CNN_ATTENTION = "funnel-cnn224-160-128-attn6-edgegate"
MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION = (
    "funnel-cnn224-160-128-interleave-attn3-edgegate"
)
MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION = (
    "funnel-cnn224-160-128-attn6-refresh3-edgegate"
)
EDGE_GATE_QK = "qk"
EDGE_GATE_Q_ONLY = "q-only"
EDGE_GATE_K_ONLY = "k-only"
ATTENTION_HEADS = 4
ATTENTION_HEAD_DIM = 16
ATTENTION_DIM = ATTENTION_HEADS * ATTENTION_HEAD_DIM
ATTENTION_LAYERS = 3
ATTENTION_FFN_HIDDEN = 64
FUNNEL_INITIAL_CHANNELS = 224
FUNNEL_MID_CHANNELS = 160
FUNNEL_ATTENTION_CHANNELS = 128
FUNNEL_BLOCKS_PER_STAGE = 2
FUNNEL_ATTENTION_LAYERS = 6
FUNNEL_INTERLEAVED_STAGES = 3
FUNNEL_REFRESH_STAGES = 3
FUNNEL_ATTENTION_PER_REFRESH = 2
BOARD_ATTENTION_SIZE = 8
BOARD_ATTENTION_RELATIONS = (
    "same_square",
    "same_rank",
    "same_file",
    "same_diagonal",
    "same_anti_diagonal",
    "knight_move",
    "king_adjacent",
    "our_pawn_attack_geometry",
    "their_pawn_attack_geometry",
)
