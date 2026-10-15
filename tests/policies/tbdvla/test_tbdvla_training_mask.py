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

"""TBDVLA block-diffusion training attention mask (needs no VLM weights)."""

from types import SimpleNamespace

import torch

from tests.utils import skip_if_package_missing

BLOCK, N_BLOCKS, PREFIX, SUFFIX = 3, 4, 5, 2


def _training_mask():
    """Allowed (L, L) for [prefix | anchor | noisy_0..B-1 | clean_0..B-1 | suffix], plus the slot positions."""
    from lerobot.policies.tbdvla.modeling_tbdvla import Qwen3VLBlockDiffusion

    stub = SimpleNamespace(block_size=BLOCK, n_blocks=N_BLOCKS, precision=torch.float32)
    n_slots = N_BLOCKS * BLOCK
    anchor = torch.arange(PREFIX, PREFIX + BLOCK)
    noisy = torch.arange(PREFIX + BLOCK, PREFIX + BLOCK + n_slots)
    clean = noisy + n_slots
    attention_mask = torch.ones(1, PREFIX + BLOCK + 2 * n_slots + SUFFIX, dtype=torch.long)
    mask = Qwen3VLBlockDiffusion._make_block_diffusion_training_mask(
        stub, attention_mask, anchor[None], noisy[None], clean[None]
    )
    return mask[0, 0] == 0, anchor, noisy.view(N_BLOCKS, BLOCK), clean.view(N_BLOCKS, BLOCK)


@skip_if_package_missing("transformers")
def test_only_the_anchor_attends_to_the_anchor():
    allowed, anchor, noisy, clean = _training_mask()
    assert not allowed[noisy.flatten()][:, anchor].any()
    assert not allowed[clean.flatten()][:, anchor].any()
    # The anchor itself still predicts block 0: it sees itself and noisy block 0 only.
    assert allowed[anchor][:, anchor].all()
    assert allowed[anchor][:, noisy[0]].all()
    assert not allowed[anchor][:, noisy[1:].flatten()].any()
    assert not allowed[anchor][:, clean.flatten()].any()


def _sees(allowed, rows, cols) -> bool:
    sub = allowed[rows][:, cols]
    assert sub.all() or not sub.any()  # whole blocks only
    return bool(sub.all())


@skip_if_package_missing("transformers")
def test_block_visibility():
    allowed, _, noisy, clean = _training_mask()
    for i in range(N_BLOCKS):
        for j in range(N_BLOCKS):
            assert _sees(allowed, noisy[i], noisy[j]) == (j == i)
            assert _sees(allowed, noisy[i], clean[j]) == (j < i)
            assert _sees(allowed, clean[i], clean[j]) == (j <= i)
            assert _sees(allowed, clean[i], noisy[j]) == (j == i + 1)
