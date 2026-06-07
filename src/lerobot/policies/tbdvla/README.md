# Installation
```
git clone https://github.com/TBD-VLA/lerobot.git
cd lerobot
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[libero]"
uv pip install -U transformers
```

# TBD-VLA Policy

Block Discrete Denoising Diffusion for Vision-Language-Action models using a Qwen3-VL VLM backbone.

## Architecture

- **VLM Backbone**: `Qwen/Qwen3-VL-2B-Instruct` (configurable via `--policy.vlm_checkpoint`)
- **Autoregression**: Autoregressive between temporal action chunks
- **Diffusion**: Masked diffusion (parallel decoding) within each temporal action chunk

## Files

| File | Description |
|---|---|
| `configuration_tbdvla.py` | `TBDVLAConfig` dataclass with all hyperparameters |
| `modeling_tbdvla.py` | `TBDVLAPolicy` — model, training loss, and inference |
| `processor_tbdvla.py` | Pre/post-processing pipelines (normalization, device transfer) |
| `__init__.py` | Exports `TBDVLAConfig`, `TBDVLAPolicy`, `make_tbdvla_pre_post_processors` |

## Training

Training and evaluation are run separately. Train first, then evaluate checkpoints.

### Single GPU

```bash
python src/lerobot/scripts/lerobot_train.py \
  --policy.type=tbdvla \
  --output_dir=/$OUTPUT_DIR \
  --dataset.repo_id=sean1295/libero_all \
  --job_name=tbdvla_experiment \
  --steps=150000 \
  --batch_size=4 \
  --save_freq=20000 \
  --log_freq=1000 \
  --policy.device=cuda \
  --policy.n_bins=512 \
  --policy.block_temporal_size=4 \
  --policy.n_diffusion_steps=2 \
  --policy.gripper_dims=[-1] \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

### Multi-GPU (4 GPUs)

```bash
accelerate launch --multi_gpu --num_processes=4 \
  src/lerobot/scripts/lerobot_train.py \
  --wandb.enable=false \
  --num_workers=4 \
  --policy.type=tbdvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=sean1295/libero_all \
  --steps=150000 \
  --save_freq=20000 \
  --log_freq=1000 \
  --batch_size=16 \
  --policy.device=cuda \
  --policy.n_bins=512 \
  --policy.block_temporal_size=4 \
  --policy.n_diffusion_steps=2 \
  --policy.gripper_dims=[-1] \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16
```

## Evaluation

Evaluate a saved checkpoint against the LIBERO environment after training is complete.

### Evaluate a specific checkpoint

```bash
uv run python src/lerobot/scripts/lerobot_eval.py \
  --policy.path=$CKPT_DIR \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.n_action_steps=8 \
  --policy.n_diffusion_steps=2  
```

### LIBERO task variants

| `--env.task` | Description |
|---|---|
| `libero_10` | Long horizon tasks |
| `libero_spatial` | Spatial reasoning tasks |
| `libero_object` | Object manipulation tasks |
| `libero_goal` | Goal varying tasks |

## TBD-VLA Parameters

### Model Architecture

| Parameter | Description | Default |
|---|---|---|
| `--policy.vlm_checkpoint` | Qwen3-VL model ID | `Qwen/Qwen3-VL-2B-Instruct` |
| `--policy.num_vlm_layers` | Number of VLM layers to use (-1 = all) | -1 |

### Diffusion / Block Denoising 

| Parameter | Description | Default |
|---|---|---|
| `--policy.block_temporal_size` | Temporal steps per block | 4 |
| `--policy.n_diffusion_steps` | Number of denoising steps at inference | 2 |
| `--policy.chunk_size` | Action chunk length (multipliers of block_temporal_size) | 16 |


### Training Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `--policy.n_bins` | Number of action discretization bins | 512 |
| `--policy.n_action_steps` | Steps executed per inference (must be <= chunk_size) | 16 |
| `--policy.n_obs_steps` | Number of observation steps (only 1 supported) | 1 |
| `--policy.max_task_tokens` | Max task/language tokens fed to the VLM | 64 |
| `--policy.use_state` | Include proprioceptive state input | true |
| `--policy.state_dropout_p` | Dropout probability for state input | 0.0 |
| `--policy.image_resolution` | Resize images to this resolution before cropping (skipped if already that size) | 256,256 |
| `--policy.crop_shape` | Image crop dimensions (e.g., `224,224`) | None |
| `--policy.gradient_checkpointing` | Enable gradient checkpointing (saves VRAM) | false |
| `--policy.precision` | Training precision (`float16`, `bfloat16`, `float32`) | `bfloat16` |
| `--policy.attn_implementation` | Attention backend (`eager`, `sdpa`, `flex_attention`) | `sdpa` |
| `--policy.optimizer_lr` | AdamW learning rate (applied to all parameters) | 1e-4 |
| `--policy.optimizer_betas` | Adam betas | (0.95, 0.999) |
| `--policy.optimizer_weight_decay` | Weight decay | 0.01 |
| `--policy.scheduler_name` | LR scheduler type | `cosine` |
| `--policy.scheduler_warmup_steps` | Warmup steps | 500 |
| `--policy.grad_clip_norm` | Gradient clipping norm | 1.0 |

### Inference Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `--policy.gripper_dims` | Gripper dimension indices (for sticky (binary) grippers. Gripper values become either -1 or 1) | [-1] |
| `--policy.expectation_sample` | Use expectation-based sampling | true |
| `--policy.compile_model` | Wrap the VLM forward in `torch.compile` (faster inference, one-time compile cost) | false |
| `--policy.latency_timestep` | Compensation timestep using Real-Time Chunking | 0 |

## VLM Backbones

Set any Qwen3-VL checkpoint via `--policy.vlm_checkpoint`. The default is `Qwen/Qwen3-VL-2B-Instruct`. Larger Qwen3-VL variants increase capacity at the cost of more VRAM.

## VLA Checkpoints 🤗

### [LIBERO Checkpoint](https://huggingface.co/sean1295/tbdvla_libero)
