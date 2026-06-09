<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

# TBD-VLA
[![Project Website](https://img.shields.io/badge/Project-Website-2ea44f?style=for-the-badge)](https://tbd-vla.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.07895-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2606.07895)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)


This is LeRobot implementation for Block Discrete Denoising Diffusion for Vision-Language-Action models using a Qwen3-VL VLM backbone.

# Installation
```
git clone https://github.com/TBD-VLA/lerobot.git
cd lerobot
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[libero]"
uv pip install -U transformers
uv pip install -U accelerate
```
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

## Evaluation

Evaluate a saved checkpoint against the LIBERO environment after training is complete.

### Evaluate specific checkpoints

```bash
uv run python src/lerobot/scripts/lerobot_eval.py \
  --policy.path=$CKPT_DIR \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.n_action_steps=12 \
  --policy.n_diffusion_steps=2  
```

### VLA Checkpoints 🤗

#### [sean1295/tbdvla_libero](https://huggingface.co/sean1295/tbdvla_libero)

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
| `--policy.n_action_steps` | Steps executed per inference (must be <= chunk_size) | 12 |
| `--policy.gripper_dims` | Gripper dimension indices (for sticky (binary) grippers. Gripper values become either -1 or 1) | [-1] |
| `--policy.expectation_sample` | Use expectation-based sampling | true |
| `--policy.compile_model` | Wrap the VLM forward in `torch.compile` (faster inference, one-time compile cost) | false |
| `--policy.latency_timestep` | Compensation timestep using Real-Time Chunking | 0 |

## VLM Backbones

Set any Qwen3-VL checkpoint via `--policy.vlm_checkpoint`. The default is `Qwen/Qwen3-VL-2B-Instruct`. Larger Qwen3-VL variants increase capacity at the cost of more VRAM.

## BibTex
```bibtex
@article{lee2026tbdvlatemporalblockdiffusion,
      title={TBD-VLA: Temporal Block Diffusion Vision Language Action Model},
      author={Lee, Sung-Wook and Kang, Xuhui and Kuo, Yen-Ling},
      journal={arXiv preprint},
      year={2026},
}
```