"""One config for model, training, augmentation, and evaluation. No magic numbers
in the training loop — every knob lives here. All values are starting points."""

from dataclasses import dataclass, field

from .preprocess.landmarks import LANDMARK_SETS


@dataclass
class AugmentConfig:
    mask_total_frac: float = 0.4     # temporal masking: spans totaling up to this frame fraction
    mask_span_min: int = 4
    mask_span_max: int = 16
    rotation_deg: float = 5.0        # affine jitter about the origin (canonical space)
    scale: float = 0.05
    translation: float = 0.02
    node_dropout_p: float = 0.05
    noise_sigma: float = 0.005


@dataclass
class Config:
    # model (Kaggle 1st-place cnn_transformer pattern, ~2M param budget; docs/research/task3a)
    n_classes: int = 246
    t_frames: int = 160
    landmark_version: str = "v1"
    n_nodes: int = 65  # derived from landmark_version in __post_init__
    n_channels: int = 10
    dim: int = 192
    n_conv_blocks: int = 3
    conv_kernel: int = 17
    n_transformer_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    drop_path: float = 0.2
    embed_dim: int = 512
    loss: str = "ce"                 # "ce" | "arcface"
    arcface_s: float = 30.0
    arcface_m: float = 0.3
    # training
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 10               # early stopping on val top-1
    amp: bool = True
    seed: int = 42
    num_workers: int = 2
    # evaluation
    n_val_participants: int = 4
    n_random_impostors: int = 20
    far_target: float = 0.05
    cluster_overlap_flag: float = 0.5
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def __post_init__(self):
        self.n_nodes = LANDMARK_SETS[self.landmark_version].n_nodes
