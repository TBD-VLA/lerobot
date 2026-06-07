from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("tbdvla")
@dataclass
class TBDVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    vlm_checkpoint: str = "Qwen/Qwen3-VL-2B-Instruct"
    num_vlm_layers: int = -1
    gradient_checkpointing: bool = False
    compile_model: bool = False
    precision: str = "bfloat16"
    # "eager" (default historical), "sdpa", or "flex_attention". The custom 4D
    # block-diffusion mask is dense, so "flex_attention" wraps it via score_mod
    # inside a Triton-compiled kernel — same numerics, faster than eager math
    # attention.
    attn_implementation: str = "sdpa"
    use_state: bool = True
    n_bins: int = 512
    state_dropout_p: float = 0.0
    max_task_tokens: int = 64
    use_prefix_prediction_loss: bool = False

    # Inference hyperparameters
    block_temporal_size: int = 4  # number of temporal steps per block
    n_diffusion_steps: int = 2
    expectation_sample: bool = True
    gripper_dims: tuple | None = (-1,) # specicy when using sticky grippers
    latency_timestep: int = 0  # number of blocks to inpaint from previous generation

    # Image parameters. Incoming images are resized to `image_resolution`
    # (skipped if already at that size) and then cropped to `crop_shape`.
    image_resolution: tuple[int, int] = (
        256,
        256,
    )
    crop_shape: tuple[int, int] | None = None

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def validate_features(self) -> None:
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(0, self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None