"""CUDA-graph inference for TBDVLA (``TBDVLAConfig.cuda_graphs``).

At small batch sizes a model call is a long sequence of small kernels and is bound by the CPU-side cost of launching
them. This module replays the two kinds of model calls of `generate_actions` from captured CUDA graphs:

  * prefix: vision tower + prompt. The prompt is right-padded to a 32-token bucket and its keys/values are written
    straight into a static KV cache.
  * block steps: the rows of `_forward_suffix_rows` are padded to a fixed count and attend to the whole static cache;
    columns past the valid length are masked, so "committing" rows only advances the valid length.

Every shape is fixed per (batch size, prompt bucket), so each pair is captured once and replayed afterwards. Graph
capture needs a sync-free forward, so while capturing the Qwen3-VL vision tower takes everything it derives from
`image_grid_thw` (rotary / position embeddings, cu_seqlens, per-image split sizes) from caches keyed by the grid known
on the CPU, and runs its block-diagonal attention as one batched call (all TBDVLA views have the same size). Outside
capture the Qwen3-VL code is unchanged.

Results match the regular path up to bf16 rounding (attention runs over a longer, masked key range).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from transformers import cache_utils
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

PREFIX_BUCKET = 32  # prompt lengths are padded to a multiple of this

# ── Static-grid Qwen3-VL vision path (active only while capturing) ──────────────────────────────────────────

_STATIC_GRID: list[tuple | None] = [None]  # ((t, h, w), ...) of the images being encoded, or None
_GRID_CACHE: dict[tuple, Tensor] = {}


@contextlib.contextmanager
def static_vision(vision_model, grid: tuple):
    """Within the block, Qwen3-VL's vision code reads its grid-derived tensors from caches for `grid`."""
    if ("rot", grid) not in _GRID_CACHE:
        dev = vision_model.patch_embed.proj.weight.device
        grid_t = torch.tensor(grid, dtype=torch.long, device=dev)
        _GRID_CACHE[("rot", grid)] = _orig_rot_pos_emb(vision_model, grid_t)
        _GRID_CACHE[("pos", grid)] = _orig_fast_pos_embed(vision_model, grid_t)
        cu = torch.repeat_interleave(grid_t[:, 1] * grid_t[:, 2], grid_t[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        _GRID_CACHE[("cu", grid)] = F.pad(cu, (1, 0), value=0)
    _STATIC_GRID[0] = grid
    try:
        yield
    finally:
        _STATIC_GRID[0] = None


_orig_rot_pos_emb = qwen3_vl.Qwen3VLVisionModel.rot_pos_emb
_orig_fast_pos_embed = qwen3_vl.Qwen3VLVisionModel.fast_pos_embed_interpolate
_orig_vision_forward = qwen3_vl.Qwen3VLVisionModel.forward
_orig_vision_attention = qwen3_vl.Qwen3VLVisionAttention.forward
_orig_get_image_features = qwen3_vl.Qwen3VLModel.get_image_features
_orig_get_placeholder_mask = qwen3_vl.Qwen3VLModel.get_placeholder_mask


def _vision_forward(self, hidden_states, grid_thw, **kwargs):
    grid = _STATIC_GRID[0]
    if grid is None:
        return _orig_vision_forward(self, hidden_states, grid_thw, **kwargs)
    kwargs.pop("return_dict", None)
    hidden_states = self.patch_embed(hidden_states) + _GRID_CACHE[("pos", grid)]
    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = _GRID_CACHE[("rot", grid)].reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = _GRID_CACHE[("cu", grid)]
    deepstack_features = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        if layer_num in self.deepstack_visual_indexes:
            merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
            deepstack_features.append(merger(hidden_states))
    return qwen3_vl.BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=self.merger(hidden_states),
        deepstack_features=deepstack_features,
    )


def _vision_attention(
    self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs
):
    grid = _STATIC_GRID[0]
    if grid is None or len(set(grid)) != 1 or qwen3_vl.is_flash_attention_requested(self.config):
        return _orig_vision_attention(
            self, hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs
        )
    # Equal-size images: the per-image (block-diagonal) attention is one batched call.
    seq_len = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = qwen3_vl.apply_rotary_pos_emb_vision(q, k, cos, sin)
    n_images = len(grid)
    q, k, v = (
        t.reshape(n_images, seq_len // n_images, self.num_heads, -1).transpose(1, 2) for t in (q, k, v)
    )
    attention_interface = qwen3_vl.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, qwen3_vl.eager_attention_forward
    )
    out = attention_interface(
        self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0, is_causal=False, **kwargs
    )[0]
    return self.proj(out.reshape(seq_len, -1).contiguous())


def _get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
    grid = _STATIC_GRID[0]
    if grid is None:
        return _orig_get_image_features(self, pixel_values, image_grid_thw, **kwargs)
    kwargs.pop("return_dict", None)
    out = self.visual(
        pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw, return_dict=True, **kwargs
    )
    merge = self.visual.spatial_merge_size
    out.pooler_output = torch.split(out.pooler_output, [t * h * w // merge**2 for t, h, w in grid])
    return out


def _get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
    if _STATIC_GRID[0] is None or input_ids is None:
        return _orig_get_placeholder_mask(self, input_ids, inputs_embeds, image_features, video_features)
    # The image-token count is checked on the CPU token ids when the layout is built.
    image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    return image_mask, video_mask


qwen3_vl.Qwen3VLVisionModel.forward = _vision_forward
qwen3_vl.Qwen3VLVisionAttention.forward = _vision_attention
qwen3_vl.Qwen3VLModel.get_image_features = _get_image_features
qwen3_vl.Qwen3VLModel.get_placeholder_mask = _get_placeholder_mask


# ── Static KV cache ──────────────────────────────────────────────────────────────────────────────────────────


class _IndexedLayer(cache_utils.CacheLayerMixin):
    """Preallocated keys/values; `update` writes in place at the positions in `write_idx`."""

    is_compileable = False
    is_sliding = False

    def __init__(self, keys: Tensor, values: Tensor, write_idx: Tensor):
        super().__init__()
        self.keys, self.values, self.write_idx = keys, values, write_idx
        self.is_initialized = True

    def lazy_initialization(self, *args, **kwargs):
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        self.keys.index_copy_(2, self.write_idx, key_states)
        self.values.index_copy_(2, self.write_idx, value_states)
        return self.keys, self.values

    def get_seq_length(self, *args, **kwargs) -> int:
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        return self.keys.shape[-2]

    def get_mask_sizes(self, *args, **kwargs) -> tuple[int, int]:
        return self.keys.shape[-2], 0


class StaticKVCache(cache_utils.Cache):
    def __init__(self, layers: list[_IndexedLayer], runner: _GraphRunner):
        super().__init__(layers=layers)
        self.graph_runner = runner


class _GraphRunner:
    """Static KV cache of `cache_len` columns for `batch_size` samples, plus the captured prefix / block graphs."""

    def __init__(self, model, forward, batch_size: int, cache_len: int, n_rows: int):
        text_config = model.vlm.config.text_config
        head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
        )
        device, dtype = model.vlm.device, model.precision
        self.model, self.forward = model, forward
        self.batch_size, self.cache_len, self.n_rows = batch_size, cache_len, n_rows
        shape = (batch_size, text_config.num_key_value_heads, cache_len, head_dim)
        self.keys = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(text_config.num_hidden_layers)
        ]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.row_idx = torch.arange(n_rows, device=device)
        self.write_idx = self.row_idx.clone()
        self.cache = StaticKVCache(
            [_IndexedLayer(k, v, self.write_idx) for k, v in zip(self.keys, self.values, strict=True)], self
        )
        self.block_inputs = {
            "input_ids": torch.zeros(batch_size, n_rows, dtype=torch.long, device=device),
            "position_ids": torch.zeros(3, batch_size, n_rows, dtype=torch.long, device=device),
            "attention_mask": torch.zeros(batch_size, 1, n_rows, cache_len, dtype=dtype, device=device),
            "cache_position": self.row_idx.clone(),
        }
        self.block_graph = None
        self.prefix_graphs: dict[tuple, tuple] = {}

    @staticmethod
    def _capture(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm-up (compilation, cuBLAS workspaces) off the capture
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        return graph, out

    def run_block(self, input_ids, position_ids, attention_mask, write_start: int) -> Tensor:
        inputs = self.block_inputs
        inputs["input_ids"].copy_(input_ids)
        inputs["position_ids"].copy_(position_ids)
        inputs["attention_mask"].copy_(attention_mask)
        self.write_idx.copy_(self.row_idx + write_start)
        inputs["cache_position"].copy_(self.row_idx + write_start)
        if self.block_graph is None:
            self.block_graph, self.block_logits = self._capture(
                lambda: self.forward(**inputs, past_key_values=self.cache, use_cache=False).logits
            )
        self.block_graph.replay()
        return self.block_logits

    def run_prefix(self, input_ids, position_ids, attention_mask, pixel_values, image_grid_thw, grid: tuple):
        key = (input_ids.shape[1], tuple(pixel_values.shape), grid)
        if key not in self.prefix_graphs:
            inputs = {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "pixel_values": pixel_values.clone(),
                "image_grid_thw": image_grid_thw.clone(),
                "cache_position": torch.arange(input_ids.shape[1], device=input_ids.device),
            }

            def call():
                for layer in self.cache.layers:  # the prefix writes positions [0, bucket)
                    layer.write_idx = inputs["cache_position"]
                try:
                    with static_vision(self.model.vlm.model.visual, grid):
                        return self.forward(
                            **inputs, past_key_values=self.cache, use_cache=False, logits_to_keep=1
                        )
                finally:
                    for layer in self.cache.layers:
                        layer.write_idx = self.write_idx

            self.prefix_graphs[key] = (self._capture(call)[0], inputs)
        graph, inputs = self.prefix_graphs[key]
        inputs["input_ids"].copy_(input_ids)
        inputs["position_ids"].copy_(position_ids)
        inputs["attention_mask"].copy_(attention_mask)
        inputs["pixel_values"].copy_(pixel_values)
        graph.replay()


class CudaGraphDecoder:
    """Holds one `_GraphRunner` per (batch size, cache length bucket)."""

    def __init__(self, model):
        self.model = model
        raw_forward = type(model.vlm).forward.__get__(model.vlm)
        if model.config.compile_model:
            # Static shapes: one compiled graph per captured shape (a few per run), which the replay then reuses.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
            self.forward = torch.compile(raw_forward, mode="default", fullgraph=False, dynamic=False)
        else:
            self.forward = raw_forward
        blk = model.block_size
        # Rows of one call: the live pair and the finished pairs it commits (one, or k - 1 after k in-painted blocks).
        n_inpainted = model.config.latency_timestep // model.config.block_temporal_size
        self.n_rows = 2 * blk * max(2, n_inpainted)
        # Room past the prompt: per-sample prompt-length differences (task text is capped at max_task_tokens),
        # committed clean blocks 0..n_blocks-2, and one call's rows.
        self.max_extra = model.config.max_task_tokens or 64
        self.suffix_len = self.max_extra + blk * (model.n_blocks - 1) + self.n_rows
        self.runners: dict[tuple, _GraphRunner] = {}

    def runner(self, batch_size: int, prompt_len: int) -> _GraphRunner:
        bucket = -(-prompt_len // PREFIX_BUCKET) * PREFIX_BUCKET
        cache_len = -(-(bucket + self.suffix_len) // 64) * 64
        key = (batch_size, cache_len)
        if key not in self.runners:
            self.runners[key] = _GraphRunner(self.model, self.forward, batch_size, cache_len, self.n_rows)
        return self.runners[key]

    def prefix(self, model_inputs: dict, prefix_len: int, position_ids: Tensor) -> StaticKVCache:
        """Run the prompt [0, prefix_len) into a static cache (the graph path of `_compute_prefix_kv_cache`)."""
        input_ids = model_inputs["input_ids"]
        batch_size, device, dtype = input_ids.shape[0], input_ids.device, self.model.precision
        runner = self.runner(batch_size, prefix_len)
        bucket = -(-prefix_len // PREFIX_BUCKET) * PREFIX_BUCKET
        ids = input_ids[:, :bucket].clone()
        ids[:, prefix_len:] = self.model.pad_token_id
        causal = torch.tril(torch.ones(bucket, bucket, dtype=torch.bool, device=device))
        mask = torch.full(
            (batch_size, 1, bucket, runner.cache_len), float("-inf"), device=device, dtype=dtype
        )
        mask[:, 0, :, :bucket] = torch.where(causal, 0.0, float("-inf")).to(dtype)
        runner.run_prefix(
            ids,
            position_ids[:, :, :bucket],
            mask,
            model_inputs["pixel_values"],
            model_inputs["image_grid_thw"],
            model_inputs["layout"]["grid"],
        )
        return runner.cache

    @staticmethod
    def suffix_rows(state: dict, ids: Tensor, pos: Tensor, allowed: Tensor, rvalid: Tensor) -> Tensor:
        """The graph path of `_forward_suffix_rows`: rows `ids` (positions `pos`) against the static cache in
        `state["kv"]`, with `allowed` (B, n_rows, n_cached + n_rows) the visibility of the suffix columns.
        Returns logits (B, n_rows, V)."""
        runner = state["kv"].graph_runner
        batch_size, n_rows = ids.shape
        prefix_len = state["prefix_len"]
        kv_len = prefix_len + state["cat"].numel()
        if n_rows > runner.n_rows or kv_len + runner.n_rows > runner.cache_len:
            raise ValueError(
                f"CUDA-graph cache too small: {n_rows} rows at {kv_len} in {runner.cache_len} columns"
            )
        device, dtype, neg = ids.device, runner.model.precision, float("-inf")
        mask = torch.full((batch_size, 1, runner.n_rows, runner.cache_len), neg, device=device, dtype=dtype)
        mask[:, 0, :n_rows, :prefix_len] = torch.where(rvalid[:, :, None], 0.0, neg).to(dtype)
        mask[:, 0, :n_rows, prefix_len : kv_len + n_rows] = torch.where(allowed, 0.0, neg).to(dtype)
        pad_rows = torch.arange(n_rows, runner.n_rows, device=device)
        mask[:, 0, pad_rows, kv_len + pad_rows] = 0  # padding rows only see themselves; nothing reads them
        ids_p = torch.full(
            (batch_size, runner.n_rows), runner.model.pad_token_id, dtype=torch.long, device=device
        )
        ids_p[:, :n_rows] = ids
        pos_p = torch.zeros((3, batch_size, runner.n_rows), dtype=torch.long, device=device)
        pos_p[:, :, :n_rows] = pos
        return runner.run_block(ids_p, pos_p, mask, kv_len)[:, :n_rows]
