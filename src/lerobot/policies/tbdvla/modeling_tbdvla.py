#!/usr/bin/env python
"""
lerobot.policies.tbdvla.modeling_tbdvla
Block Diffusion for TBDVLA — BD3-LM-style training and inference.

Adapts the Block Discrete Denoising Diffusion Language Model (BD3-LM) approach
from Arriola et al. (ICLR 2025) to the VLA action-token setting.

Uses Qwen3-VL as the VLM backbone.
"""

import logging
import math
from collections import deque

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Resize
from transformers import AutoProcessor
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)

# Qwen3-VL omits the `_supports_flex_attn` flag even though its attention
# modules dispatch through ALL_ATTENTION_FUNCTIONS. Opt every submodule class
# in so the whole model can initialize under flex_attention.
for _cls in (
    Qwen3VLPreTrainedModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
):
    _cls._supports_flex_attn = True


def _compile_friendly_deepstack_process(
    self, hidden_states: Tensor, visual_pos_masks: Tensor, visual_embeds: Tensor
) -> Tensor:
    # Add visual_embeds at the masked positions without boolean-mask assignment.
    # Mask assignment lowers to a data-dependent nonzero() that torch.compile
    # cannot trace (graph breaks + recompiles); scatter into a dense [B, S, H]
    # tensor via cumsum-derived indices instead, which is static-shape.
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

    B, S, H = hidden_states.shape
    flat_mask_long = visual_pos_masks.reshape(-1).to(torch.long)
    n_visual = visual_embeds.shape[0]
    # Index of each token among the masked entries (row-major). Unmasked
    # entries get a junk index, zeroed out below; clamp keeps it in range.
    flat_idx = torch.cumsum(flat_mask_long, dim=0) - 1
    flat_idx = flat_idx.clamp(min=0, max=max(n_visual - 1, 0))

    gathered = visual_embeds.index_select(0, flat_idx)
    gathered = gathered * flat_mask_long.unsqueeze(-1).to(gathered.dtype)
    return hidden_states + gathered.reshape(B, S, H)


Qwen3VLTextModel._deepstack_process = _compile_friendly_deepstack_process


# With Qwen3-VL's head_dim=128, the default flex_attention block sizes
# (BLOCK_M=BLOCK_N=128) can exceed the GPU shared-memory budget. Wrap the
# forward to request smaller Triton blocks so the kernel always fits.
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _ALL_ATTN
from transformers.integrations.flex_attention import (
    flex_attention_forward as _hf_flex_attention_forward,
)

def _flex_attention_forward_smaller_blocks(*args, **kwargs):
    if not kwargs.get("kernel_options"):
        kwargs["kernel_options"] = {"BLOCK_M": 64, "BLOCK_N": 64}
    return _hf_flex_attention_forward(*args, **kwargs)

_ALL_ATTN["flex_attention"] = _flex_attention_forward_smaller_blocks

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.tbdvla.configuration_tbdvla import TBDVLAConfig
from lerobot.utils.constants import ACTION, OBS_STATE


PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
EPS = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# KV cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_kv_seq_length(past_key_values) -> int:
    """Return the cached sequence length from any HuggingFace KV cache format."""
    if isinstance(past_key_values, (Cache, DynamicCache)):
        return past_key_values.get_seq_length()

    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()

    if hasattr(past_key_values, "self_attention_cache"):
        return _get_kv_seq_length(past_key_values.self_attention_cache)

    if isinstance(past_key_values, (tuple, list)):
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)):
            return first_layer[0].shape[2]

    raise TypeError(
        f"Cannot determine KV cache length from type {type(past_key_values)}. "
        "Expected Cache, DynamicCache, or tuple-of-tuples."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Policy wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TBDVLAPolicy(PreTrainedPolicy):
    """
    TBDVLA with Block Diffusion (BD3-LM-style) training and inference.
    Action-dim shift variant with anchor block for uniform block prediction.
    """
    config_class = TBDVLAConfig
    name = "tbdvla"

    def __init__(self, config: TBDVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = Qwen3VLBlockDiffusion(config)
        self.reset()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._prev_decoded_blocks = None

    def get_optim_params(self):
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # Decode a fresh action chunk, carrying the previous chunk's tail for
        # latency inpainting (see generate_actions / latency_timestep).
        actions, decoded_blocks = self.model.generate_actions(
            batch,
            n_steps=self.config.n_diffusion_steps,
            prev_decoded_blocks=self._prev_decoded_blocks,
        )
        self._prev_decoded_blocks = decoded_blocks
        deploy_latency = getattr(self, "_deploy_latency", 0)
        n_steps = self.config.n_action_steps
        # The chunk holds n_action_steps + latency_timestep slots; skip the
        # first `offset` inpainted slots. With latency_timestep=0 this is [:n_steps].
        offset = min(deploy_latency, self.config.latency_timestep)
        actions = actions[:, offset : offset + n_steps]
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        loss_dict = self.model.forward(batch)
        loss = loss_dict.pop("loss")
        return loss, loss_dict

    @torch.no_grad()
    def compute_val_mse(self, batch: dict[str, Tensor]) -> dict[str, float]:
        self.eval()
        model = self.model
        action_dim = self.config.action_feature.shape[0]
        n_eval_steps = self.config.n_action_steps + self.config.latency_timestep

        gt = batch[ACTION][:, :n_eval_steps, :action_dim]

        _, decoded_blocks = model.generate_actions(
            batch,
            n_steps=self.config.n_diffusion_steps + self.config.latency_timestep,
        )

        # Reconstruct the full (untrimmed) continuous actions from decoded_blocks
        device = gt.device
        all_ids = torch.cat(decoded_blocks, dim=1)
        batch_size = all_ids.shape[0]
        n_decoded_timesteps = all_ids.shape[1] // model.action_dim
        disc = all_ids.reshape(batch_size, n_decoded_timesteps, model.action_dim)
        disc_local = (disc - model.action_bin_start).clamp(0, self.config.n_bins - 1)
        pred_full = model._bin_centers(device)[disc_local]

        pred = pred_full[:, :n_eval_steps, :action_dim]
        return {"val_mse": F.mse_loss(pred, gt).item()}

# ══════════════════════════
# Core Block Diffusion Model 
# ══════════════════════════

class Qwen3VLBlockDiffusion(nn.Module):
    """
    TBDVLA with Block Diffusion — anchor block variant.
    Uses Qwen3-VL as the VLM backbone.

    Token layout (training, doubled / concatenated):
        [prefix] [anchor] [noisy_blk_0 .. noisy_blk_{B-1}]
                            [clean_blk_0 .. clean_blk_{B-1}] [suffix]

    The anchor is a block of block_size MASK tokens prepended before
    the noisy blocks. It is always fully masked and serves as the
    predictor for block 0, making all blocks structurally uniform.

    Each block is exactly block_size (= block_temporal_size * action_dim) tokens.

    Prediction shift: clean block b predicts noisy block b+1.
    For block 0: the anchor predicts noisy block 0.

    Attention:
      - Anchor: sees prefix (causally), bidirectional within self,
                  sees noisy block 0 (like clean→noisy next block)
      - Noisy block b: sees prefix, clean blocks 0..b-1,
                       bidirectional within self
      - Clean block b: sees prefix, clean blocks 0..b-1,
                       bidirectional within self, noisy block b+1
    No noisy or clean block attends to the anchor: it only predicts block 0,
    and inference drops it once block 0 is decoded.
    """

    def __init__(self, config: TBDVLAConfig):
        super().__init__()
        self.config = config
        self.precision = PRECISION.get(config.precision, torch.float32)

        # ── VLM backbone — Qwen3-VL ──
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_checkpoint,
            torch_dtype=self.precision,
            device_map=config.device,
            attn_implementation=config.attn_implementation,
        )

        if config.gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if getattr(config, "freeze_vision_encoder", False):
            for p in self.vlm.model.visual.parameters():
                p.requires_grad = False

        # ── Layer truncation ──
        self.num_vlm_layers = getattr(config, "num_vlm_layers", -1)
        if self.num_vlm_layers > 0:
            lang_model = self.vlm.model.language_model
            total_layers = len(lang_model.layers)
            assert self.num_vlm_layers <= total_layers, (
                f"num_vlm_layers ({self.num_vlm_layers}) exceeds "
                f"total layers ({total_layers})"
            )

            print(
                f"Truncating language model: {total_layers} → {self.num_vlm_layers} layers"
            )

            lang_model.layers = lang_model.layers[:self.num_vlm_layers]

            if hasattr(lang_model.config, "num_hidden_layers"):
                lang_model.config.num_hidden_layers = self.num_vlm_layers

        if getattr(config, "compile_model", False):
            # Compile the whole vlm.forward (not per layer) to fuse across
            # layers. dynamic=True absorbs the few distinct sequence shapes seen
            # per inference (prefix + one per decoded block) without retracing.
            self.vlm.forward = torch.compile(
                self.vlm.forward,
                mode="default",
                fullgraph=False,
                dynamic=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            config.vlm_checkpoint,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "right"

        # ── Action dimensions ──
        self.action_horizon = config.chunk_size
        self.action_dim = config.action_feature.shape[0]
        self.n_action_tokens = self.action_horizon * self.action_dim

        # ── Block diffusion parameters ──
        assert config.latency_timestep < config.chunk_size, (
            f"latency_timestep ({config.latency_timestep}) must be less than "
            f"chunk_size ({config.chunk_size})"
        )
        assert config.n_action_steps + config.latency_timestep <= config.chunk_size, (
            f"n_action_steps ({config.n_action_steps}) + latency_timestep "
            f"({config.latency_timestep}) must be <= chunk_size ({config.chunk_size}), "
            "otherwise there aren't enough overflow timesteps in the previous "
            "chunk to inpaint the new one."
        )

        self.block_size = config.block_temporal_size * self.action_dim
        self.n_blocks = self.n_action_tokens // self.block_size

        self.image_keys = config.image_features.keys()
        self.image_resolution = tuple(config.image_resolution)
        self.resize_fn = Resize(self.image_resolution, antialias=True)
        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.center_crop_fn = CenterCrop(config.crop_shape)

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        # ── Special tokens ──
        original_vocab_size = len(self.processor.tokenizer)
        bin_tokens = [f"<ACT_BIN_{i}>" for i in range(config.n_bins)]
        new_tokens = ["<STATE_SLOT>", "<MASK_ACT>", "<ACT_SLOT>"] + bin_tokens
        for tok in new_tokens:
            assert tok not in self.processor.tokenizer.get_vocab(), (
                f"Token {tok} already exists."
            )
        self.processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)

        self.state_slot_token_id = self.processor.tokenizer.convert_tokens_to_ids("<STATE_SLOT>")
        self.mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<MASK_ACT>")
        self.slot_token_id = self.processor.tokenizer.convert_tokens_to_ids("<ACT_SLOT>")
        self.action_bin_start = original_vocab_size + 3

        self.slot_token_str = self.processor.tokenizer.decode([self.slot_token_id])
        self.state_slot_token_str = self.processor.tokenizer.decode([self.state_slot_token_id])

        self._init_special_embeddings()

        # Inference-time caches of things that depend only on the prompt layout
        # (see _build_prompt / _build_inputs / _get_position_ids_from_inputs).
        self._prompt_cache: dict[tuple, str] = {}
        self._layout_cache: dict[tuple, dict] = {}

    def _init_special_embeddings(self):
        print("Vocab is initialized!")
        embed = self.vlm.get_input_embeddings().weight
        head = self.vlm.lm_head.weight
        hidden_dim = embed.shape[1]
        n_bins = self.config.n_bins

        with torch.no_grad():
            ref_norm = embed[:self.action_bin_start - 3].norm(dim=-1).mean().item()

            # Sinusoidal bin embeddings
            bin_phase = torch.linspace(0, math.pi, n_bins)
            n_freq = hidden_dim // 2
            freq_indices = 2 * torch.arange(0, n_freq, dtype=torch.float32) + 1
            amplitudes = 1.0 / freq_indices
            angles = bin_phase.unsqueeze(1) * freq_indices.unsqueeze(0)

            sinusoidal = torch.zeros(n_bins, hidden_dim)
            sinusoidal[:, 0::2] = torch.sin(angles) * amplitudes
            sinusoidal[:, 1::2] = torch.cos(angles) * amplitudes
            sinusoidal = sinusoidal / sinusoidal.norm(dim=-1, keepdim=True) * ref_norm
            sinusoidal = sinusoidal.to(device=embed.device, dtype=embed.dtype)

            for i in range(n_bins):
                tid = self.action_bin_start + i
                embed[tid] = sinusoidal[i]
                head[tid] = sinusoidal[i].clone()

            # Random init for special tokens
            for tid in [self.mask_token_id, self.slot_token_id, self.state_slot_token_id]:
                vec = torch.randn(hidden_dim, device=embed.device, dtype=embed.dtype)
                embed[tid] = vec / vec.norm() * ref_norm
                hvec = torch.randn(hidden_dim, device=head.device, dtype=head.dtype)
                head[tid] = hvec / hvec.norm() * ref_norm

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def prepare_images(self, batch: dict) -> dict:
        images = {}
        present = [k for k in self.image_keys if k in batch]
        if not present:
            raise ValueError("No image features found in batch.")
        for key in self.image_keys:
            if key in present:
                images[key] = batch[key]
        return images

    def _crop_batch(self, imgs: Tensor) -> Tensor:
        """Crop a (B, C, H, W) batch to ``config.crop_shape``.

        Eval uses a deterministic center crop. Training uses an independent
        random crop per sample (vectorised — torchvision's RandomCrop would
        share one window across the whole batch).
        """
        if not self.training:
            return self.center_crop_fn(imgs)

        b, c, h, w = imgs.shape
        th, tw = self.config.crop_shape
        if h == th and w == tw:
            return imgs

        tops = torch.randint(0, h - th + 1, (b,), device=imgs.device)
        lefts = torch.randint(0, w - tw + 1, (b,), device=imgs.device)
        rows = tops[:, None] + torch.arange(th, device=imgs.device)  # (B, th)
        cols = lefts[:, None] + torch.arange(tw, device=imgs.device)  # (B, tw)
        bidx = torch.arange(b, device=imgs.device)[:, None, None, None]
        cidx = torch.arange(c, device=imgs.device)[None, :, None, None]
        return imgs[bidx, cidx, rows[:, None, :, None], cols[:, None, None, :]]  # (B, C, th, tw)

    def _discretise_actions(self, actions: Tensor, device) -> Tensor:
        n = self.config.n_bins
        step = 2.0 * (1.0 + EPS) / n
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS - step, n, device=device)
        disc = torch.bucketize(actions, bins) - 1
        return disc.reshape(disc.shape[0], -1) + self.action_bin_start

    def _bin_centers(self, device) -> Tensor:
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_bins + 1, device=device)
        return 0.5 * (bins[:-1] + bins[1:])

    def _normalize_lang_text(self, lang_text, batch_size: int) -> list[str]:
        if isinstance(lang_text, str):
            return [lang_text] * batch_size
        if isinstance(lang_text, tuple):
            return list(lang_text)
        return lang_text

    # ──────────────────────────────────────────────────────────────────────────
    # Position ID helpers for Qwen3-VL M-RoPE (3D: temporal, height, width)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_text_position_ids(self, seq_len: int, batch_size: int, device: torch.device) -> Tensor:
        """Build 3D M-RoPE position IDs for text-only sequences."""
        pos_1d = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return pos_1d.unsqueeze(0).expand(3, -1, -1).clone()

    def _get_position_ids_from_inputs(self, model_inputs: dict, batch_size: int) -> Tensor:
        """
        Get 3D M-RoPE position IDs from model inputs.
        Returns: (3, batch_size, seq_len) tensor.

        At inference the positions depend only on the prompt layout (token types,
        padding, image grids), so they are computed once per layout and reused.
        """
        layout = model_inputs.get("layout")
        if layout is not None and "position_ids" in layout:
            return layout["position_ids"].clone()

        input_ids = model_inputs["input_ids"]
        device = input_ids.device
        seq_len = input_ids.shape[1]

        image_grid_thw = model_inputs.get("image_grid_thw", None)
        video_grid_thw = model_inputs.get("video_grid_thw", None)
        attention_mask = model_inputs.get("attention_mask", None)
        mm_token_type_ids = model_inputs.get("mm_token_type_ids", None)

        has_vision = image_grid_thw is not None or video_grid_thw is not None

        if has_vision:
            mrope_pos, _ = self.vlm.model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            position_ids = mrope_pos
        else:
            position_ids = self._build_text_position_ids(seq_len, batch_size, device)

        if layout is not None:
            layout["position_ids"] = position_ids.clone()
        return position_ids

    # ──────────────────────────────────────────────────────────────────────────
    # _build_inputs — task slot approach, no decode round-trip
    # ──────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, lang_text: str, state_dim: int, n_images: int, n_slots: int) -> str:
        """Chat-formatted prompt of one sample. A pure function of its arguments, so it is cached."""
        key = (lang_text, state_dim, n_images, n_slots)
        prompt = self._prompt_cache.get(key)
        if prompt is not None:
            return prompt

        task_cleaned = lang_text.lower().strip().replace("_", " ")
        task_token_ids = self.processor.tokenizer.encode(
            task_cleaned, add_special_tokens=False,
        )

        # Truncate if a cap is configured.
        max_task_tokens = self.config.max_task_tokens
        if max_task_tokens is not None and len(task_token_ids) > max_task_tokens:
            task_token_ids = task_token_ids[:max_task_tokens]
            task_cleaned = self.processor.tokenizer.decode(
                task_token_ids, skip_special_tokens=True,
            )

        state_str = self.state_slot_token_str * state_dim
        if self.config.use_state:
            prefix = f"State: {state_str}, Task: {task_cleaned}, Actions: "
        else:
            prefix = f"Task: {task_cleaned}, Actions: "

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": prefix}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.slot_token_str * n_slots}],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False,
        )
        self._prompt_cache[key] = prompt
        return prompt

    def _build_inputs(
        self,
        states: Tensor,
        images: dict,
        lang_text: list[str] | str,
        action_ids: Tensor | None,
        n_slots: int | None = None,
    ) -> tuple[dict, Tensor, list[int]]:
        device = states.device
        batch_size = states.shape[0]
        states = states.clamp(-1.0, 1.0)
        lang_text = self._normalize_lang_text(lang_text, batch_size)

        if n_slots is None:
            n_slots = self.n_action_tokens

        prompts = [
            self._build_prompt(text, states.shape[-1], len(images), n_slots) for text in lang_text
        ]

        # ── Prepare images ──
        # Resize (skipped when already at the target size) and crop are applied
        # per key on the whole (B, C, H, W) batch, so each transform launches
        # one kernel instead of one per image.
        processed = []
        for imgs in images.values():
            if tuple(imgs.shape[-2:]) != self.image_resolution:
                imgs = self.resize_fn(imgs)
            if self.do_crop:
                imgs = self._crop_batch(imgs)
            processed.append(imgs)

        # Regroup into the per-sample list of views the processor expects.
        images_reshaped = [
            list(views) for views in zip(*(p.unbind(0) for p in processed), strict=True)
        ]

        # The text side of the processor output (token ids, padding, slot and
        # state positions) depends only on the prompts and image sizes, never on
        # the state or pixel values. At inference it is built once per layout and
        # reused, so each step runs only the image processor and locates the
        # slots without device-to-host syncs.
        layout_key = (tuple(prompts), tuple(tuple(p.shape[-2:]) for p in processed))
        layout = None if self.training else self._layout_cache.get(layout_key)
        if layout is None:
            proc_out = self.processor(
                images=images_reshaped,
                text=prompts,
                do_rescale=False,
                do_resize=True,
                padding=True,
                return_tensors="pt",
            )
            text_inputs = {k: v for k, v in proc_out.items() if k not in ("pixel_values", "image_grid_thw")}
            image_inputs = {k: v for k, v in proc_out.items() if k in ("pixel_values", "image_grid_thw")}

            # ── Locate action / state slots (on the CPU token ids) ──
            input_ids = text_inputs["input_ids"]
            slot_mask = input_ids == self.slot_token_id
            slot_counts = slot_mask.sum(dim=1)
            assert (slot_counts == n_slots).all(), (
                f"Expected {n_slots} ACT_SLOT tokens, got {slot_counts.tolist()}"
            )
            grid = None
            if not self.training:  # image grids known on the CPU, for the CUDA-graph path
                grid = tuple(tuple(int(x) for x in row) for row in image_inputs["image_grid_thw"].tolist())
                merge = self.vlm.model.visual.spatial_merge_size
                n_image_tokens = int((input_ids == self.vlm.config.image_token_id).sum())
                assert n_image_tokens == sum(t * h * w for t, h, w in grid) // merge**2, (
                    f"{n_image_tokens} image tokens for image grids {grid}"
                )
            layout = {
                "grid": grid,  # (t, h, w) per image, on the CPU
                "text": {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in text_inputs.items()},
                "slot_positions": slot_mask.nonzero()[:, 1].reshape(batch_size, n_slots).to(device),
                "state_positions": (
                    (input_ids == self.state_slot_token_id).nonzero()[:, 1].reshape(batch_size, -1).to(device)
                    if self.config.use_state else None
                ),
            }
            if not self.training:
                self._layout_cache[layout_key] = layout
        else:
            image_inputs = self.processor.image_processor(
                images=images_reshaped, do_rescale=False, do_resize=True, return_tensors="pt",
            )

        model_inputs = dict(layout["text"])
        model_inputs.update({k: v.to(device) for k, v in image_inputs.items()})
        if "pixel_values" in model_inputs:
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to(self.precision)
        if not self.training:
            model_inputs["layout"] = layout

        input_ids = model_inputs["input_ids"].clone()  # cached tensor: fill a copy
        slot_positions = layout["slot_positions"]

        if action_ids is None:
            fill = torch.full(
                (batch_size, n_slots),
                fill_value=self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
        else:
            fill = action_ids.to(device)

        input_ids.scatter_(1, slot_positions, fill)

        # ── Fill state slots ──
        if self.config.use_state:
            n = self.config.n_bins
            step = 2.0 * (1.0 + EPS) / n
            bins = torch.linspace(-1.0 - EPS, 1.0 + EPS - step, n, device=device)
            disc_states = (
                torch.bucketize(states.clamp(-1, 1), bins) - 1
            ) + self.action_bin_start

            drop_p = self.config.state_dropout_p
            if self.training and drop_p > 0.0:
                drop_mask = torch.rand(batch_size, device=device) < drop_p
                disc_states = torch.where(
                    drop_mask[:, None].expand_as(disc_states),
                    torch.full_like(disc_states, self.pad_token_id),
                    disc_states,
                )

            input_ids.scatter_(1, layout["state_positions"], disc_states)

        model_inputs["input_ids"] = input_ids
        return model_inputs, slot_positions

    # ──────────────────────────────────────────────────────────────────────────
    # Block-diffusion training attention mask (anchor variant)
    # ──────────────────────────────────────────────────────────────────────────

    def _make_block_diffusion_training_mask(
        self,
        attention_mask,
        anchor_positions,
        noisy_slot_positions,
        clean_slot_positions,
    ):
        """Build the block-diffusion training attention mask (anchor variant).

        Categorizes each token (prefix=0, anchor=1, noisy=2, clean=3, suffix=4)
        and applies the allowed-pair table via broadcast — no per-sample loop.

        The anchor block acts as "clean block -1": sees prefix, bidirectional
        within itself, sees noisy block 0 (clean→noisy next-block pattern).
        Noisy and clean blocks do not see it (inference drops it after block 0).

        Returns additive mask (BS, 1, L, L) with 0 for allowed, -inf for blocked.
        """
        BS, L = attention_mask.shape
        device = attention_mask.device
        blk_sz = self.block_size
        n_blocks = self.n_blocks

        # cat: 0=prefix, 1=anchor, 2=noisy, 3=clean, 4=suffix
        cat = torch.zeros(BS, L, dtype=torch.long, device=device)
        blk = torch.zeros(BS, L, dtype=torch.long, device=device)

        block_ids = torch.arange(n_blocks, device=device).repeat_interleave(blk_sz)
        block_ids_b = block_ids.unsqueeze(0).expand(BS, -1)

        cat.scatter_(1, anchor_positions, 1)
        cat.scatter_(1, noisy_slot_positions, 2)
        cat.scatter_(1, clean_slot_positions, 3)
        blk.scatter_(1, noisy_slot_positions, block_ids_b)
        blk.scatter_(1, clean_slot_positions, block_ids_b)

        # Suffix = strictly after the last clean position
        last_clean = clean_slot_positions.max(dim=1).values  # (BS,)
        pos_idx = torch.arange(L, device=device)
        is_after_clean = pos_idx.unsqueeze(0) > last_clean.unsqueeze(1)
        cat = torch.where(is_after_clean, torch.full_like(cat, 4), cat)

        cat_i = cat.unsqueeze(2)  # (BS, L, 1)
        cat_j = cat.unsqueeze(1)  # (BS, 1, L)
        blk_i = blk.unsqueeze(2)
        blk_j = blk.unsqueeze(1)
        pos_i = pos_idx.view(1, L, 1)
        pos_j = pos_idx.view(1, 1, L)
        causal = pos_j <= pos_i

        is_pre_i = cat_i == 0
        is_pre_j = cat_j == 0
        is_sen_i = cat_i == 1
        is_sen_j = cat_j == 1
        is_noi_i = cat_i == 2
        is_noi_j = cat_j == 2
        is_cln_i = cat_i == 3
        is_cln_j = cat_j == 3
        is_suf_i = cat_i == 4

        allowed = (
            (is_pre_i & is_pre_j & causal)
            | (is_sen_i & is_pre_j)
            | (is_sen_i & is_sen_j)
            | (is_sen_i & is_noi_j & (blk_j == 0))
            | (is_noi_i & is_pre_j)
            | (is_noi_i & is_noi_j & (blk_i == blk_j))
            | (is_noi_i & is_cln_j & (blk_i > blk_j))
            | (is_cln_i & is_pre_j)
            | (is_cln_i & is_noi_j & (blk_i + 1 == blk_j))
            | (is_cln_i & is_cln_j & (blk_i >= blk_j))
            | (is_suf_i & causal)
        )

        pad = attention_mask.bool()
        allowed = allowed & pad.unsqueeze(1) & pad.unsqueeze(2)

        additive = torch.where(
            allowed,
            torch.zeros((), device=device, dtype=self.precision),
            torch.full((), float("-inf"), device=device, dtype=self.precision),
        )
        return additive.unsqueeze(1)

    # ──────────────────────────────────────────────────────────────────────────
    # Prefix LM loss (vectorized)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_prefix_lm_loss(self, all_logits, input_ids, anchor_positions):
        """Vectorized prefix LM loss — autoregressive CE over the text prefix.

        Predicts every token from the start marker ("State: " or "Task: ")
        up to (but not including) the first anchor slot. With the default
        layout this covers State values + Task description + "Actions: "
        + the assistant-turn glue tokens.

        Uses a static-shape mask so torch.compile can trace through it without
        a graph break.
        """
        device = input_ids.device
        BS, L = input_ids.shape
        V = all_logits.shape[-1]

        marker_str = "State: " if self.config.use_state else "Task: "
        marker_ids = self.processor.tokenizer.encode(
            marker_str, add_special_tokens=False,
        )
        marker_len = len(marker_ids)
        marker_tensor = torch.tensor(marker_ids, device=device, dtype=torch.long)

        if L < marker_len + 1:
            return torch.tensor(0.0, device=device)

        windows = input_ids.unfold(1, marker_len, 1)  # (BS, L-ml+1, ml)
        marker_match = (windows == marker_tensor.view(1, 1, -1)).all(dim=2)
        has_match = marker_match.any(dim=1)
        # argmax returns 0 if no match; guard via has_match in the validity mask.
        prefix_start = marker_match.long().argmax(dim=1)
        prefix_end = anchor_positions[:, 0]  # exclusive
        valid = has_match & (prefix_end > prefix_start)

        pos = torch.arange(L, device=device).unsqueeze(0)
        target_mask = (
            (pos >= prefix_start.unsqueeze(1))
            & (pos < prefix_end.unsqueeze(1))
            & valid.unsqueeze(1)
        )  # (BS, L) — True at prefix-token positions in input_ids

        # Shift by one to align with causal LM logits: logits[:, :-1] predict
        # input_ids[:, 1:].
        shifted_logits = all_logits[:, :-1, :].float()       # (BS, L-1, V)
        shifted_targets = input_ids[:, 1:]                   # (BS, L-1)
        shifted_mask = target_mask[:, 1:].to(shifted_logits.dtype)  # (BS, L-1)

        per_token = F.cross_entropy(
            shifted_logits.reshape(-1, V),
            shifted_targets.reshape(-1),
            reduction="none",
        ).reshape(BS, L - 1)

        denom = shifted_mask.sum().clamp(min=1.0)
        return (per_token * shifted_mask).sum() / denom

    # ──────────────────────────────────────────────────────────────────────────
    # Training forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> dict:
        """
        BD3-LM training with anchor block and concatenated doubled layout.

        Sequence layout (concatenated):
            [prefix] [anchor] [noisy_blk_0 .. noisy_blk_{B-1}]
                                [clean_blk_0 .. clean_blk_{B-1}] [suffix]

        Prefix text layout (state before task so task prediction sees
        full observation context):
            "State: {state_str}, Task: {task_desc}, Actions: "

        The anchor is block_size MASK tokens placed right before the noisy
        blocks. It is a structural placeholder that serves as the predictor
        for block 0, making all blocks uniform.

        Position IDs:
          - Anchor: continues sequentially from prefix
          - Clean block b: aliases to noisy block b's positions

        Loss:
          Action loss (action_dim shift):
            - Predictor positions = [anchor] + [clean_blk_0 .. clean_blk_{B-2}]
            - Targets = ground-truth tokens of blocks 0..B-1
            - Loss mask = corruption mask (only masked positions in target block)
          Prefix LM loss (optional, controlled by use_prefix_prediction_loss):
            - Causal next-token prediction on the full text prefix
              (state values + task description + "Actions: " + glue tokens)
            - Encourages the model to reason about state and task given images
        """
        device = batch[OBS_STATE].device
        images = self.prepare_images(batch)
        actions = batch[ACTION].clamp(-1.0, 1.0)
        batch_size = actions.shape[0]
        blk_sz = self.block_size

        # 1. Discretise all clean actions → (B, n_action_tokens)
        clean_ids = self._discretise_actions(actions, device)

        # 2. Reshape into blocks: (B, n_blocks, block_size)
        clean_blocks = clean_ids.reshape(batch_size, self.n_blocks, blk_sz)

        # 3. Sample mask ratio per block and apply masking
        mask_ratio = torch.rand(batch_size, self.n_blocks, 1, device=device)
        mask_probs = mask_ratio.expand_as(clean_blocks)
        is_masked_blocks = torch.rand_like(clean_blocks.float()) < mask_probs

        # 3a. Train-time inpainting: freeze the first `latency_timestep`
        #     timesteps as clean (never masked, never contributing to loss).
        #     Mirrors the inference-time inpainting geometry exactly.
        n_inpaint = self.config.latency_timestep
        if n_inpaint > 0:
            bts = self.config.block_temporal_size
            n_full_inpaint_blocks = n_inpaint // bts
            n_partial_timesteps = n_inpaint % bts
            n_partial_tokens = n_partial_timesteps * self.action_dim

            # frozen_blocks[b, t] = True iff token t of block b is inpainted
            # (and therefore must not be masked and must not contribute to loss).
            frozen_blocks = torch.zeros(
                self.n_blocks, blk_sz, dtype=torch.bool, device=device,
            )
            if n_full_inpaint_blocks > 0:
                frozen_blocks[:n_full_inpaint_blocks, :] = True
            if n_partial_tokens > 0 and n_full_inpaint_blocks < self.n_blocks:
                frozen_blocks[n_full_inpaint_blocks, :n_partial_tokens] = True

            # Broadcast over batch: (1, n_blocks, blk_sz) -> (B, n_blocks, blk_sz)
            frozen_blocks_b = frozen_blocks.unsqueeze(0).expand_as(clean_blocks)
            is_masked_blocks = is_masked_blocks & ~frozen_blocks_b

        noisy_blocks = torch.where(is_masked_blocks, self.mask_token_id, clean_blocks)

        noisy_ids = noisy_blocks.reshape(batch_size, -1)

        # 3b. Handle padded actions
        if "action_is_pad" in batch:
            action_is_pad = batch["action_is_pad"]
            pad_mask_tokens = action_is_pad.repeat_interleave(self.action_dim, dim=1)
            pad_mask_blocks = pad_mask_tokens.reshape(batch_size, self.n_blocks, blk_sz)

            noisy_blocks = torch.where(pad_mask_blocks, self.mask_token_id, noisy_blocks)
            noisy_ids = noisy_blocks.reshape(batch_size, -1)

            is_masked_blocks = is_masked_blocks & ~pad_mask_blocks

        # 4. Build anchor (all MASK tokens) + noisy + clean for doubled layout
        anchor_ids = torch.full(
            (batch_size, blk_sz),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        # Layout: [anchor | noisy_blk_0..B-1 | clean_blk_0..B-1]
        all_action_ids = torch.cat([anchor_ids, noisy_ids, clean_ids], dim=1)
        n_total_slots = blk_sz + 2 * self.n_action_tokens

        # 5. Build the full input sequence
        model_inputs, all_slot_positions = self._build_inputs(
            states=batch[OBS_STATE],
            images=images,
            lang_text=batch.get("task", ""),
            action_ids=all_action_ids,
            n_slots=n_total_slots,
        )

        # 6. Extract anchor, noisy, and clean slot positions
        anchor_positions = all_slot_positions[:, :blk_sz]
        noisy_slot_positions = all_slot_positions[:, blk_sz:blk_sz + self.n_action_tokens]
        clean_slot_positions = all_slot_positions[:, blk_sz + self.n_action_tokens:]

        # 7. Position aliasing: clean block b gets noisy block b's positions
        #    Anchor positions are already sequential from prefix (no aliasing needed)
        position_ids = self._get_position_ids_from_inputs(model_inputs, batch_size)

        noisy_pos_expanded = noisy_slot_positions.unsqueeze(0).expand(3, -1, -1)
        noisy_pos_vals = position_ids.gather(2, noisy_pos_expanded)
        clean_pos_expanded = clean_slot_positions.unsqueeze(0).expand(3, -1, -1)
        position_ids.scatter_(2, clean_pos_expanded, noisy_pos_vals)

        # 8. Build block-diffusion attention mask
        bidir_mask = self._make_block_diffusion_training_mask(
            model_inputs["attention_mask"],
            anchor_positions,
            noisy_slot_positions,
            clean_slot_positions,
        )

        # 9. Forward pass
        outputs = self.vlm.forward(
            input_ids=model_inputs["input_ids"],
            attention_mask=bidir_mask,
            position_ids=position_ids,
            pixel_values=model_inputs.get("pixel_values", None),
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            use_cache=False,
        )

        # 10. Compute action loss with action_dim shift
        V = outputs.logits.shape[-1]

        clean_blocks_pos = clean_slot_positions.reshape(batch_size, self.n_blocks, blk_sz)

        # Predictor for block b: anchor for b=0, clean_block[b-1] for b>=1
        # This is now uniform: [anchor, clean_0, ..., clean_{B-2}]
        predictor_pos_flat = torch.cat(
            [anchor_positions, clean_blocks_pos[:, :-1, :].reshape(batch_size, -1)],
            dim=1,
        )

        # Gather in original dtype (bf16), then upcast just the gathered slice
        slot_logits = outputs.logits.gather(
            dim=1,
            index=predictor_pos_flat.unsqueeze(-1).expand(-1, -1, V),
        ).float()

        targets = clean_ids.to(device)

        token_loss = F.cross_entropy(
            slot_logits.reshape(-1, V),
            targets.reshape(-1),
            reduction="none",
        ).reshape(batch_size, self.n_blocks, blk_sz)

        loss_mask = is_masked_blocks.float()
        action_loss = (token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        # 11. Compute prefix causal LM loss
        #     With layout "State: ..., Task: {desc}, Actions: ...", every
        #     prefix token (state values + task description + "Actions: ")
        #     is causally conditioned on the preceding tokens (and images).
        #     The block-diffusion mask already makes prefix↔prefix attention
        #     strictly causal, so we just add a CE term over those positions.

        if self.config.use_prefix_prediction_loss:
            prefix_lm_loss = self._compute_prefix_lm_loss(
                outputs.logits,
                model_inputs["input_ids"],
                anchor_positions,
            )
            loss = action_loss + prefix_lm_loss
        else:
            loss = action_loss
            prefix_lm_loss = torch.tensor(0.0, device=device)

        return {
            "loss": loss,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════════════

    def _denoise_block(
        self,
        block_logits_fn,
        batch_size: int,
        n_steps: int,
        device: torch.device,
        fixed_prefix: Tensor | None = None,
        n_fixed: int = 0,
    ) -> Tensor:
        """Run masked diffusion unmasking for a single block.

        Written without device-to-host syncs (no boolean indexing, no `.any()`
        on device tensors), so the next model call is queued while the GPU is
        still busy with the previous one.
        """
        blk_sz = self.block_size
        n_steps = min(n_steps, blk_sz - n_fixed)
        n_bins = self.config.n_bins
        lo = self.action_bin_start

        current_ids = torch.full(
            (batch_size, blk_sz),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        if n_fixed > 0 and fixed_prefix is not None:
            current_ids[:, :n_fixed] = fixed_prefix

        is_free = torch.ones(blk_sz, dtype=torch.bool, device=device)
        is_free[:n_fixed] = False
        bin_indices = torch.arange(n_bins, device=device)
        pos_in_sort = torch.arange(blk_sz, device=device).unsqueeze(0)
        gripper_cols = None
        if self.config.expectation_sample and self.config.gripper_dims:
            dims = torch.arange(blk_sz, device=device) % self.action_dim
            gripper_cols = torch.zeros(blk_sz, dtype=torch.bool, device=device)
            for d in self.config.gripper_dims:
                gripper_cols |= dims == d % self.action_dim

        for step in range(n_steps, 0, -1):
            still_masked = (current_ids == self.mask_token_id) & is_free

            logits = block_logits_fn(current_ids)
            action_probs = logits[..., lo:lo + n_bins].softmax(dim=-1)
            if self.config.expectation_sample:
                expected_idx = (action_probs * bin_indices.to(action_probs.dtype)).sum(dim=-1)
                candidates = expected_idx.round().long().clamp(0, n_bins - 1) + lo
                if gripper_cols is not None:
                    candidates = torch.where(gripper_cols, action_probs.argmax(dim=-1) + lo, candidates)
            else:
                candidates = action_probs.argmax(dim=-1) + lo
            confidence = self._unmask_confidence(action_probs, device).masked_fill(~still_masked, float("-inf"))

            # Round down so the schedule is back-loaded — fewer tokens unmasked
            # in early (high-uncertainty) steps, more in later steps. E.g. 14
            # tokens over 4 steps -> (3, 3, 4, 4); over 3 steps -> (4, 5, 5).
            # The final step (step == 1) divides by 1, so it always clears the
            # remainder and decoding completes.
            n_to_unmask = (still_masked.sum(dim=1).float() / step).floor().long().clamp(min=1)

            # Unmask the n_to_unmask most confident masked positions: rank them,
            # then scatter the per-rank decision back to positions.
            sorted_conf, sorted_idx = confidence.sort(dim=1, descending=True)
            unmask_sorted = (pos_in_sort < n_to_unmask.unsqueeze(1)) & (sorted_conf > float("-inf"))
            unmask = torch.zeros_like(still_masked).scatter(1, sorted_idx, unmask_sorted)
            current_ids = torch.where(unmask, candidates, current_ids)

        # Every free token is unmasked by the last step: with n_steps <= n_free,
        # at least `step` tokens are still masked going into each step, and
        # step 1 takes all that remain. Checked on the device (no host sync).
        torch._assert_async(~((current_ids == self.mask_token_id) & is_free).any())
        return current_ids

    def _graph_decoder_for(self, model_inputs: dict):
        """The CUDA-graph decoder if `config.cuda_graphs` applies to these inputs, else None."""
        layout = model_inputs.get("layout")
        enabled = self.config.cuda_graphs
        if enabled is None:
            enabled = self.config.compile_model
        if (
            not enabled
            or self.training
            or layout is None
            or model_inputs["input_ids"].device.type != "cuda"
            or len(set(layout["grid"])) != 1  # the captured vision path needs equal-size views
        ):
            return None
        if getattr(self, "_graph_decoder", None) is None:
            from lerobot.policies.tbdvla.cuda_graphs import CudaGraphDecoder

            self._graph_decoder = CudaGraphDecoder(self)
        return self._graph_decoder

    def _unmask_confidence(self, action_probs, device):
        """Confidence score for unmasking order."""
        return action_probs.max(dim=-1).values

    @torch.no_grad()
    def _compute_prefix_kv_cache(self, states, images, lang_text):
        """
        Compute KV cache for the shared prefix (everything before the
        action slots).

        With the anchor variant, the prefix is cleanly separated from
        action blocks — no tail subtraction needed. We cache up to the
        minimum prefix_end across the batch, ensuring we never cut into
        image pad tokens.

        Returns:
            prefix_kv: KV cache for tokens 0..prefix_cache_len-1
            prefix_cache_len: number of tokens in the cached prefix
            per_sample_prefix_end: (B,) tensor of per-sample prefix_end values
            model_inputs: full model inputs dict
            slot_positions: (B, n_action_tokens) slot positions
            position_ids: (3, B, full_seq_len) M-RoPE position IDs
        """
        model_inputs, slot_positions = self._build_inputs(
            states=states,
            images=images,
            lang_text=lang_text,
            action_ids=None,
        )

        batch_size = states.shape[0]
        device = states.device
        input_ids = model_inputs["input_ids"]

        per_sample_prefix_end = slot_positions[:, 0]  # (B,)
        prefix_cache_len = per_sample_prefix_end.min().item()

        # Ensure we never slice through image pad tokens
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_pad_mask = input_ids == image_pad_id
        if image_pad_mask.any():
            positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            last_image_pad_per_sample = (image_pad_mask * positions).max(dim=1).values + 1
            min_safe_prefix = last_image_pad_per_sample.max().item()
            prefix_cache_len = max(prefix_cache_len, min_safe_prefix)

        position_ids = self._get_position_ids_from_inputs(model_inputs, batch_size)
        graph_decoder = self._graph_decoder_for(model_inputs)
        if graph_decoder is not None:
            prefix_kv = graph_decoder.prefix(model_inputs, prefix_cache_len, position_ids)
            return prefix_kv, prefix_cache_len, per_sample_prefix_end, model_inputs, slot_positions, position_ids

        # Standard causal mask for the prefix
        causal_4d = torch.tril(torch.ones(prefix_cache_len, prefix_cache_len, device=device))
        causal_4d_mask = causal_4d.masked_fill(causal_4d == 0, float("-inf"))
        causal_4d_mask = causal_4d_mask.masked_fill(causal_4d == 1, 0.0)
        causal_4d_mask = causal_4d_mask.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1, -1
        ).to(self.precision)

        prefix_position_ids = position_ids[:, :, :prefix_cache_len]

        outputs = self.vlm.forward(
            input_ids=input_ids[:, :prefix_cache_len],
            attention_mask=causal_4d_mask,
            position_ids=prefix_position_ids,
            pixel_values=model_inputs.get("pixel_values", None),
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            use_cache=True,
        )

        return (
            outputs.past_key_values,
            prefix_cache_len,
            per_sample_prefix_end,
            model_inputs,
            slot_positions,
            position_ids,
        )

    # Roles of the suffix tokens that follow the shared cached prefix at inference.
    _EXTRA, _ANCHOR, _NOISY, _CLEAN = 0, 1, 2, 3

    def _init_block_decoding(
        self,
        prefix_kv,
        prefix_cache_len: int,
        per_sample_prefix_end: Tensor,
        position_ids_full: Tensor,
        model_inputs: dict,
    ) -> dict:
        """Start decoding a chunk on top of the shared prefix cache.

        Inference uses the *training* layout and mask. Training gives every block
        two token groups with different attention:
            noisy_j  -> prefix, noisy_j, clean_{<j}
            clean_j  -> prefix, clean_{<=j}, noisy_{j+1}
            anchor   -> prefix, anchor, noisy_0
        and block j is predicted at its predictor positions (anchor for j = 0,
        else clean_{j-1}). Nothing depends on a block later than its own
        successor, so once block j is decoded the pair
        (predictor_j, noisy_j := decoded_j) is final and its keys/values can stay
        in the cache. Every model call therefore carries only the live pair
        [predictor_n | noisy_n]; the pair finished by the previous block is
        committed in the same call as the next block's first denoising step.

        The anchor only predicts block 0 and no other block attends to it, so its
        pair is never committed, which keeps 2 * block_size fewer tokens of KV per
        sample.

        Cache layout: [prefix | extra-prefix (left-padded per sample)
                       | (clean_0, noisy_1) | (clean_1, noisy_2) | ... ].
        Order is irrelevant to attention: visibility comes from the explicit 4D
        mask, positions from RoPE (clean_j aliases noisy_j's positions, as in
        training).

        Returns the decoding state consumed by `_forward_block_inference`.
        """
        device = model_inputs["input_ids"].device
        prefix_ends = per_sample_prefix_end.tolist()
        batch_size = len(prefix_ends)
        state = {
            "kv": prefix_kv,
            "prefix_len": prefix_cache_len,
            # Last prefix position per sample, (3, B): action positions continue from it.
            "last_pos": torch.stack(
                [position_ids_full[:, s, pe - 1] for s, pe in enumerate(prefix_ends)], dim=1
            ),
            # Role / block id / validity of the suffix columns already in the cache.
            "cat": torch.zeros(0, dtype=torch.long, device=device),
            "bid": torch.zeros(0, dtype=torch.long, device=device),
            "valid": torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            "n_committed": 0,  # blocks whose pair has been settled (committed, or dropped for block 0)
        }

        # Prefix tokens past the shared cached prefix (prompts differ in length),
        # left-padded across the batch.
        max_extra = max(pe - prefix_cache_len for pe in prefix_ends)
        if max_extra > 0:
            ids, pos, valid = [], [], []
            for s, pe in enumerate(prefix_ends):
                left_pad = max_extra - (pe - prefix_cache_len)
                extra = model_inputs["input_ids"][s, prefix_cache_len:pe]
                extra_pos = position_ids_full[:, s, prefix_cache_len:pe]
                if left_pad > 0:
                    pad = torch.full((left_pad,), self.pad_token_id, dtype=torch.long, device=device)
                    extra = torch.cat([pad, extra])
                    # Dummy positions for the left-pad (masked out)
                    pad_pos = position_ids_full[:, s, prefix_cache_len:prefix_cache_len + left_pad]
                    extra_pos = torch.cat([pad_pos, extra_pos], dim=1)
                ids.append(extra)
                pos.append(extra_pos)
                valid.append(torch.arange(max_extra, device=device) >= left_pad)
            self._forward_suffix_rows(
                state,
                torch.stack(ids),
                torch.stack(pos, dim=1),
                torch.full((max_extra,), self._EXTRA, dtype=torch.long, device=device),
                torch.full((max_extra,), -1, dtype=torch.long, device=device),
                torch.stack(valid),
                keep=max_extra,
            )
        return state

    def _forward_suffix_rows(
        self, state: dict, ids: Tensor, pos: Tensor, rcat: Tensor, rbid: Tensor, rvalid: Tensor, keep: int
    ) -> Tensor:
        """Run rows (role `rcat`, block id `rbid`) against the cache and leave the
        first `keep` of them in it. Returns logits (B, n_rows, V)."""
        cx, ca, cn, cc = self._EXTRA, self._ANCHOR, self._NOISY, self._CLEAN
        kv, prefix_len = state["kv"], state["prefix_len"]
        device = ids.device
        batch_size, n_rows = ids.shape

        ccat = torch.cat([state["cat"], rcat])
        cbid = torch.cat([state["bid"], rbid])
        cvalid = torch.cat([state["valid"], rvalid], dim=1)
        ri, cj = rcat[:, None], ccat[None, :]
        bi, bj = rbid[:, None], cbid[None, :]
        n_cached = state["cat"].numel()
        col_idx = torch.arange(n_cached + n_rows, device=device)[None, :]
        row_idx = (n_cached + torch.arange(n_rows, device=device))[:, None]
        # Same allowed-pair table as `_make_block_diffusion_training_mask`
        # (the extra-prefix tokens are prefix: causal among themselves, visible to all).
        allowed = (
            ((ri == cx) & (cj == cx) & (col_idx <= row_idx))
            | ((ri == ca) & ((cj == cx) | (cj == ca) | ((cj == cn) & (bj == 0))))
            | ((ri == cn) & ((cj == cx) | ((cj == cn) & (bi == bj)) | ((cj == cc) & (bj < bi))))
            | ((ri == cc) & ((cj == cx) | ((cj == cn) & (bj == bi + 1)) | ((cj == cc) & (bj <= bi))))
        )
        allowed = allowed[None] & cvalid[:, None, :] & rvalid[:, :, None]
        # Left-pad rows would otherwise be masked everywhere (softmax over an all
        # -inf row is NaN, see the training mask). Let each see itself: nothing
        # attends to it and its logits are never read.
        allowed = allowed | ((col_idx == row_idx)[None] & ~rvalid[:, :, None])

        if getattr(kv, "graph_runner", None) is not None:  # static cache of the CUDA-graph path
            logits = self._graph_decoder.suffix_rows(state, ids, pos, allowed, rvalid)
            if keep:
                state["cat"] = torch.cat([state["cat"], rcat[:keep]])
                state["bid"] = torch.cat([state["bid"], rbid[:keep]])
                state["valid"] = torch.cat([state["valid"], rvalid[:, :keep]], dim=1)
            return logits

        kv_len = _get_kv_seq_length(kv)
        neg = float("-inf")
        attn = torch.zeros((batch_size, 1, n_rows, kv_len + n_rows), device=device, dtype=self.precision)
        attn[:, 0, :, :prefix_len] = torch.where(rvalid[:, :, None], 0.0, neg).to(self.precision)
        attn[:, 0, :, prefix_len:] = torch.where(allowed, 0.0, neg).to(self.precision)

        try:
            outputs = self.vlm.forward(
                input_ids=ids,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=kv,
                cache_position=torch.arange(kv_len, kv_len + n_rows, device=device),
                use_cache=False,
            )
        finally:
            # DynamicLayer.update() appends to the cache even under use_cache=False:
            # keep the committed rows (they come first), drop the rest.
            if hasattr(kv, "layers"):
                for layer in kv.layers:
                    if getattr(layer, "is_initialized", False):
                        layer.crop(kv_len + keep)
        if keep:
            state["cat"] = torch.cat([state["cat"], rcat[:keep]])
            state["bid"] = torch.cat([state["bid"], rbid[:keep]])
            state["valid"] = torch.cat([state["valid"], rvalid[:, :keep]], dim=1)
        return outputs.logits

    def _block_pair_rows(self, state: dict, j: int, decoded_blocks: list[Tensor], noisy_tokens: Tensor):
        """Rows [predictor of block j | noisy_j]: token ids, M-RoPE positions, roles, block ids."""
        blk_sz = self.block_size
        device = noisy_tokens.device
        if j == 0:
            predictor, role, pred_bid = torch.full_like(noisy_tokens, self.mask_token_id), self._ANCHOR, -1
        else:
            predictor, role, pred_bid = decoded_blocks[j - 1], self._CLEAN, j - 1

        def positions(slot):  # slot 0 = anchor, slot b + 1 = block b (noisy and clean alike)
            offsets = 1 + slot * blk_sz + torch.arange(blk_sz, device=device)
            return state["last_pos"][:, :, None] + offsets.view(1, 1, -1)

        ids = torch.cat([predictor, noisy_tokens], dim=1)
        pos = torch.cat([positions(j), positions(j + 1)], dim=2)
        rcat = torch.cat([
            torch.full((blk_sz,), role, dtype=torch.long, device=device),
            torch.full((blk_sz,), self._NOISY, dtype=torch.long, device=device),
        ])
        rbid = torch.cat([
            torch.full((blk_sz,), pred_bid, dtype=torch.long, device=device),
            torch.full((blk_sz,), j, dtype=torch.long, device=device),
        ])
        return ids, pos, rcat, rbid

    def _forward_block_inference(
        self,
        state: dict,
        decoded_blocks: list[Tensor],
        current_noisy_ids: Tensor,
    ) -> Tensor:
        """
        Forward pass for one denoising step of block n = len(decoded_blocks).

        Sends [pairs of finished blocks not yet in the cache | predictor_n | noisy_n]
        after the KV cache (see `_init_block_decoding`); the finished pairs stay
        in the cache, the live pair is dropped again. Block 0's pair (the anchor)
        is never committed.

        Returns predictor logits: (B, block_size, V).
        """
        blk_sz = self.block_size
        n = len(decoded_blocks)
        pending = range(max(state["n_committed"], 1), n)
        parts = [self._block_pair_rows(state, j, decoded_blocks, decoded_blocks[j]) for j in pending]
        parts.append(self._block_pair_rows(state, n, decoded_blocks, current_noisy_ids))
        ids = torch.cat([p[0] for p in parts], dim=1)
        pos = torch.cat([p[1] for p in parts], dim=2)
        rcat = torch.cat([p[2] for p in parts])
        rbid = torch.cat([p[3] for p in parts])
        rvalid = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        keep = 2 * blk_sz * len(pending)
        logits = self._forward_suffix_rows(state, ids, pos, rcat, rbid, rvalid, keep)
        state["n_committed"] = n
        return logits[:, keep:keep + blk_sz].float()

    @torch.no_grad()
    def generate_actions(self, batch, n_steps=2, prev_decoded_blocks=None):
        """Generate actions via block-by-block masked diffusion."""
        device = next(self.vlm.parameters()).device
        batch_size = batch["observation.state"].shape[0]
        images = self.prepare_images(batch)
        blk_sz = self.block_size
        bts = self.config.block_temporal_size
        n_inpaint = self.config.latency_timestep

        states = batch["observation.state"]
        lang_text = batch.get("task", "")

        steps_per_block = min(blk_sz, n_steps)

        # ── Compute prefix KV cache (once) ──
        (
            prefix_kv,
            prefix_cache_len,
            per_sample_prefix_end,
            model_inputs,
            slot_positions,
            position_ids_full,
        ) = self._compute_prefix_kv_cache(
            states=states,
            images=images,
            lang_text=lang_text,
        )

        # ── Inpainting geometry ──
        # Matches the training-time geometry in `forward`:
        #   - blocks [0, n_full_inpaint_blocks) are fully inpainted
        #   - block n_full_inpaint_blocks is partially inpainted in its first
        #     n_partial_tokens positions (if n_partial_tokens > 0)
        #
        # The overhang from the previous chunk holds, in order:
        #   [timestep n_action_steps .. n_action_steps + n_inpaint - 1]
        # flattened to tokens. We slice it into:
        #   - overlap_full:    (B, n_full_inpaint_blocks, blk_sz)     full blocks
        #   - overlap_partial: (B, n_partial_tokens)                   partial prefix
        has_inpaint = (
            prev_decoded_blocks is not None
            and n_inpaint > 0
            and len(prev_decoded_blocks) > 0
        )

        n_full_inpaint_blocks = 0
        n_partial_tokens = 0
        overlap_full = None
        overlap_partial = None

        if has_inpaint:
            n_inpaint_tokens = n_inpaint * self.action_dim
            n_full_inpaint_blocks = n_inpaint // bts
            n_partial_timesteps = n_inpaint % bts
            n_partial_tokens = n_partial_timesteps * self.action_dim

            # Stride between successive chunks = n_action_steps. The inpainted
            # portion of the new chunk corresponds to the "committed but not yet
            # executed" tail of the previous chunk, i.e. previous-chunk timesteps
            # [n_action_steps, n_action_steps + latency_timestep).
            shift_tokens = self.config.n_action_steps * self.action_dim
            prev_tokens = torch.cat(prev_decoded_blocks, dim=1)
            overlap_tokens = prev_tokens[:, shift_tokens:]
            n_avail = overlap_tokens.shape[1]

            if n_avail < n_inpaint_tokens:
                # Previous chunk's tail is too short to inpaint from; fall back
                # to decoding the new chunk from scratch.
                has_inpaint = False
                n_full_inpaint_blocks = 0
                n_partial_tokens = 0
            else:
                overlap_tokens = overlap_tokens[:, :n_inpaint_tokens]
                n_full_overlap_tokens = n_full_inpaint_blocks * blk_sz

                if n_full_inpaint_blocks > 0:
                    overlap_full = overlap_tokens[:, :n_full_overlap_tokens].reshape(
                        batch_size, n_full_inpaint_blocks, blk_sz,
                    )
                if n_partial_tokens > 0:
                    overlap_partial = overlap_tokens[
                        :, n_full_overlap_tokens:n_full_overlap_tokens + n_partial_tokens
                    ]

        decoded_blocks = []
        decode_state = self._init_block_decoding(
            prefix_kv,
            prefix_cache_len,
            per_sample_prefix_end,
            position_ids_full,
            model_inputs,
        )

        def _block_logits_fn(noisy_ids):
            return self._forward_block_inference(
                state=decode_state,
                decoded_blocks=decoded_blocks,
                current_noisy_ids=noisy_ids,
            )

        def _denoise_or_inpaint(blk_idx, block_logits_fn):
            # Fully inpainted block: copy overhang directly, no model call.
            if has_inpaint and blk_idx < n_full_inpaint_blocks:
                return overlap_full[:, blk_idx, :]

            if (
                has_inpaint
                and blk_idx == n_full_inpaint_blocks
                and n_partial_tokens > 0
            ):
                n_free = blk_sz - n_partial_tokens
                adjusted_steps = max(1, math.ceil(steps_per_block * n_free / blk_sz))
                return self._denoise_block(
                    block_logits_fn=block_logits_fn,
                    batch_size=batch_size,
                    n_steps=adjusted_steps,
                    device=device,
                    fixed_prefix=overlap_partial,
                    n_fixed=n_partial_tokens,
                )

            # Free block: standard denoising.
            return self._denoise_block(
                block_logits_fn=block_logits_fn,
                batch_size=batch_size,
                n_steps=steps_per_block,
                device=device,
            )

        # ── Decide how many blocks to generate ──
        # Must cover:
        #   (a) what we'll execute this step:          n_action_steps timesteps
        #   (b) overhang for the next call to inpaint: n_inpaint timesteps
        n_needed_timesteps = self.config.n_action_steps + n_inpaint
        n_blocks_needed = math.ceil(n_needed_timesteps / bts)

        # If we're inpainting, we must at least cover all inpainted blocks
        # (trivially satisfied by the line above given the config assertion, but
        # kept for safety).
        if has_inpaint:
            n_blocks_needed = max(
                n_blocks_needed,
                n_full_inpaint_blocks + (1 if n_partial_tokens > 0 else 0),
            )
        n_blocks_needed = min(n_blocks_needed, self.n_blocks)

        # ── Decode blocks ──
        # In-painted blocks need no model call; their pair enters the cache with
        # the next block that does.
        for blk_idx in range(n_blocks_needed):
            block_result = _denoise_or_inpaint(
                blk_idx=blk_idx, block_logits_fn=_block_logits_fn,
            )
            decoded_blocks.append(block_result)

        # ── Decode to continuous ──
        all_ids = torch.cat(decoded_blocks, dim=1)
        n_decoded_timesteps = n_blocks_needed * bts
        disc = all_ids.reshape(batch_size, n_decoded_timesteps, self.action_dim)
        disc_local = (disc - self.action_bin_start).clamp(0, self.config.n_bins - 1)
        continuous_actions = self._bin_centers(device)[disc_local]
        # Keep enough slots for the caller to slice
        # `[deploy_latency : deploy_latency + n_action_steps]` under RTC.
        n_keep = min(self.config.n_action_steps + n_inpaint, n_decoded_timesteps)
        continuous_actions = continuous_actions[:, :n_keep, :]

        return continuous_actions, decoded_blocks