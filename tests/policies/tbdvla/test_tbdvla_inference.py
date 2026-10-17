#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TBDVLA inference pieces that need no VLM weights: the sync-free block denoiser and the block commit schedule."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from tests.utils import skip_if_package_missing

N_BINS, ACTION_DIM, BTS = 32, 7, 4
BLOCK = BTS * ACTION_DIM
BIN_START, MASK_ID, VOCAB = 100, 5, 100 + N_BINS + 3


def _reference_denoise_block(
    self,
    block_logits_fn,
    batch_size: int,
    n_steps: int,
    device: torch.device,
    fixed_prefix: Tensor | None = None,
    n_fixed: int = 0,
) -> Tensor:
    """`_denoise_block` before the sync-free rewrite (origin/main @ f2bc9941)."""
    blk_sz = self.block_size
    n_free = blk_sz - n_fixed
    n_steps = min(n_steps, n_free)

    current_ids = torch.full(
        (batch_size, blk_sz),
        fill_value=self.mask_token_id,
        dtype=torch.long,
        device=device,
    )

    if n_fixed > 0 and fixed_prefix is not None:
        current_ids[:, :n_fixed] = fixed_prefix

    action_slice = slice(
        self.action_bin_start,
        self.action_bin_start + self.config.n_bins,
    )

    for step in range(n_steps, 0, -1):
        is_free = torch.ones(blk_sz, dtype=torch.bool, device=device)
        if n_fixed > 0:
            is_free[:n_fixed] = False

        still_masked = (current_ids == self.mask_token_id) & is_free[None, :]
        if not still_masked.any():
            break

        logits = block_logits_fn(current_ids)
        action_logits = logits[..., action_slice]

        action_probs = action_logits.softmax(dim=-1)
        if self.config.expectation_sample:
            bin_indices = torch.arange(self.config.n_bins, device=device, dtype=action_probs.dtype)
            expected_idx = (action_probs * bin_indices).sum(dim=-1)
            candidates = expected_idx.round().long().clamp(0, self.config.n_bins - 1) + self.action_bin_start

            if self.config.gripper_dims:
                argmax_candidates = action_probs.argmax(dim=-1) + self.action_bin_start
                for d in self.config.gripper_dims:
                    dim_idx = d % self.action_dim
                    dim_mask = torch.arange(blk_sz, device=device) % self.action_dim == dim_idx
                    candidates[:, dim_mask] = argmax_candidates[:, dim_mask]
        else:
            candidates = action_probs.argmax(dim=-1) + self.action_bin_start
        confidence = self._unmask_confidence(action_probs, device)

        new_ids = current_ids.clone()

        n_still_per_sample = still_masked.sum(dim=1)
        # Round down so the schedule is back-loaded — fewer tokens unmasked
        # in early (high-uncertainty) steps, more in later steps. E.g. 14
        # tokens over 4 steps -> (3, 3, 4, 4); over 3 steps -> (4, 5, 5).
        # The final step (step == 1) divides by 1, so it always clears the
        # remainder and decoding completes.
        n_to_unmask = (n_still_per_sample.float() / step).floor().long().clamp(min=1)

        conf_masked = confidence.clone()
        conf_masked[~still_masked] = float("-inf")

        _, sorted_idx = conf_masked.sort(dim=1, descending=True)
        pos_in_sort = torch.arange(blk_sz, device=device).unsqueeze(0)
        unmask_mask = pos_in_sort < n_to_unmask.unsqueeze(1)
        unmask_mask = unmask_mask & (conf_masked.gather(1, sorted_idx) > float("-inf"))

        selected_positions = sorted_idx[unmask_mask]
        batch_indices = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand_as(sorted_idx)[unmask_mask]
        )
        new_ids[batch_indices, selected_positions] = candidates[batch_indices, selected_positions]

        current_ids = new_ids

    # Safety net: force-unmask remaining
    is_free = torch.ones(blk_sz, dtype=torch.bool, device=device)
    if n_fixed > 0:
        is_free[:n_fixed] = False
    still_masked = (current_ids == self.mask_token_id) & is_free[None, :]
    if still_masked.any():
        fb_logits = block_logits_fn(current_ids)
        fb_probs = fb_logits[..., action_slice].softmax(dim=-1)
        if self.config.expectation_sample:
            bin_indices = torch.arange(self.config.n_bins, device=device, dtype=fb_probs.dtype)
            fallback = (fb_probs * bin_indices).sum(dim=-1).round().long().clamp(
                0, self.config.n_bins - 1
            ) + self.action_bin_start
            if self.config.gripper_dims:
                argmax_fallback = fb_probs.argmax(dim=-1) + self.action_bin_start
                for d in self.config.gripper_dims:
                    dim_idx = d % self.action_dim
                    dim_mask = torch.arange(blk_sz, device=device) % self.action_dim == dim_idx
                    fallback[:, dim_mask] = argmax_fallback[:, dim_mask]
        else:
            fallback = fb_probs.argmax(dim=-1) + self.action_bin_start
        current_ids = torch.where(still_masked, fallback, current_ids)

    return current_ids


def _stub(expectation_sample=True, gripper_dims=(-1,)):
    from lerobot.policies.tbdvla.modeling_tbdvla import Qwen3VLBlockDiffusion

    stub = SimpleNamespace(
        block_size=BLOCK,
        mask_token_id=MASK_ID,
        action_bin_start=BIN_START,
        action_dim=ACTION_DIM,
        config=SimpleNamespace(
            n_bins=N_BINS, expectation_sample=expectation_sample, gripper_dims=gripper_dims
        ),
    )
    stub._unmask_confidence = lambda probs, device: Qwen3VLBlockDiffusion._unmask_confidence(
        stub, probs, device
    )
    return stub, Qwen3VLBlockDiffusion


@skip_if_package_missing("transformers")
@pytest.mark.parametrize("expectation_sample, gripper_dims", [(True, (-1,)), (True, None), (False, (-1,))])
@pytest.mark.parametrize("n_steps, n_fixed", [(2, 0), (3, 0), (4, 9), (2, 27), (5, BLOCK)])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_denoise_block_matches_reference(expectation_sample, gripper_dims, n_steps, n_fixed, batch_size):
    stub, cls = _stub(expectation_sample, gripper_dims)
    gen = torch.Generator().manual_seed(n_steps * 100 + n_fixed + batch_size)
    calls = []

    def logits_fn(current_ids):
        # Deterministic in the ids it is given, so both implementations see the same logits iff they agree.
        calls.append(current_ids.clone())
        g = torch.Generator().manual_seed(int(current_ids.sum()) % 2**31)
        return torch.randn(current_ids.shape[0], BLOCK, VOCAB, generator=g) * 3

    fixed = (
        torch.randint(BIN_START, BIN_START + N_BINS, (batch_size, n_fixed), generator=gen)
        if n_fixed
        else None
    )
    new = cls._denoise_block(stub, logits_fn, batch_size, n_steps, torch.device("cpu"), fixed, n_fixed)
    new_calls, calls[:] = list(calls), []
    ref = _reference_denoise_block(stub, logits_fn, batch_size, n_steps, torch.device("cpu"), fixed, n_fixed)
    assert torch.equal(new, ref)
    assert len(new_calls) == len(calls) and all(
        torch.equal(a, b) for a, b in zip(new_calls, calls, strict=True)
    )
    assert not (new == MASK_ID).any()


def _commit_stub():
    """Stub whose model call records (roles, block ids, keep), commits like the real cache and returns logits
    tagged with each row's (role, block id)."""
    from lerobot.policies.tbdvla.modeling_tbdvla import Qwen3VLBlockDiffusion as M

    stub = SimpleNamespace(
        block_size=BLOCK,
        mask_token_id=MASK_ID,
        _EXTRA=M._EXTRA,
        _ANCHOR=M._ANCHOR,
        _NOISY=M._NOISY,
        _CLEAN=M._CLEAN,
    )
    stub._block_pair_rows = lambda *a: M._block_pair_rows(stub, *a)
    seen = []

    def fake_suffix_rows(state, ids, pos, rcat, rbid, rvalid, keep):
        seen.append((rcat.tolist(), rbid.tolist(), keep))
        if keep:
            state["cat"] = torch.cat([state["cat"], rcat[:keep]])
            state["bid"] = torch.cat([state["bid"], rbid[:keep]])
        logits = torch.zeros(ids.shape[0], ids.shape[1], VOCAB)
        logits[..., 0], logits[..., 1] = rcat.float(), rbid.float()  # tag each row's logits with its role / block
        return logits

    stub._forward_suffix_rows = fake_suffix_rows
    state = {
        "last_pos": torch.zeros(3, 1, dtype=torch.long),
        "n_committed": 0,
        "cat": torch.zeros(0, dtype=torch.long),
        "bid": torch.zeros(0, dtype=torch.long),
    }
    return M, stub, state, seen


def _rows(*groups):
    """Expected (roles, block ids) for groups of (role, block id), one block each."""
    return [r for r, _ in groups for _ in range(BLOCK)], [b for _, b in groups for _ in range(BLOCK)]


@skip_if_package_missing("transformers")
def test_block_commit_schedule_drops_anchor():
    """Block 0's (anchor) pair is never committed. For j >= 1, block j+1's first call commits clean_{j-1} only:
    noisy_j is sent as its context and dropped (nothing later attends to it)."""
    cls, stub, state, seen = _commit_stub()
    c, n, a = cls._CLEAN, cls._NOISY, cls._ANCHOR
    decoded = []
    for blk in range(4):  # 4 blocks x 2 denoising steps
        for _ in range(2):
            out = cls._forward_block_inference(stub, state, decoded, torch.full((1, BLOCK), MASK_ID))
            # The returned logits are the live predictor's: the anchor for block 0, else clean_{blk-1}.
            assert (out[..., 0] == (a if blk == 0 else c)).all() and (out[..., 1] == blk - 1).all()
        decoded.append(torch.full((1, BLOCK), BIN_START))

    keeps = [k for _, _, k in seen]
    assert keeps == [0, 0, 0, 0, BLOCK, 0, BLOCK, 0]
    assert seen[0][:2] == _rows((a, -1), (n, 0))
    assert seen[2][:2] == _rows((c, 0), (n, 1))
    assert seen[4][:2] == _rows((c, 0), (n, 1), (c, 1), (n, 2))
    assert seen[6][:2] == _rows((c, 1), (n, 2), (c, 2), (n, 3))
    committed = list(zip(state["cat"].tolist(), state["bid"].tolist(), strict=True))
    assert committed == [(c, 0)] * BLOCK + [(c, 1)] * BLOCK  # only clean blocks, never the anchor


@skip_if_package_missing("transformers")
def test_commit_after_inpainted_blocks():
    """Several finished pairs in one call (in-painted blocks 0..2): their clean halves come first and are kept."""
    cls, stub, state, seen = _commit_stub()
    decoded = [torch.full((1, BLOCK), BIN_START) for _ in range(3)]
    out = cls._forward_block_inference(stub, state, decoded, torch.full((1, BLOCK), MASK_ID))
    c, n = cls._CLEAN, cls._NOISY
    assert (out[..., 0] == c).all() and (out[..., 1] == 2).all()  # the live predictor clean_2
    assert seen[0] == (*_rows((c, 0), (c, 1), (n, 1), (n, 2), (c, 2), (n, 3)), 2 * BLOCK)
    committed = list(zip(state["cat"].tolist(), state["bid"].tolist(), strict=True))
    assert committed == [(c, 0)] * BLOCK + [(c, 1)] * BLOCK
