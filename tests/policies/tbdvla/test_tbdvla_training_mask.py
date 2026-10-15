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


def _training_mask(clean_blocks_attend_anchor: bool):
    """Allowed (L, L) for [prefix | anchor | noisy_0..B-1 | clean_0..B-1 | suffix], plus the slot positions."""
    from lerobot.policies.tbdvla.modeling_tbdvla import Qwen3VLBlockDiffusion

    stub = SimpleNamespace(
        block_size=BLOCK,
        n_blocks=N_BLOCKS,
        precision=torch.float32,
        config=SimpleNamespace(clean_blocks_attend_anchor=clean_blocks_attend_anchor),
    )
    n_slots = N_BLOCKS * BLOCK
    anchor = torch.arange(PREFIX, PREFIX + BLOCK)
    noisy = torch.arange(PREFIX + BLOCK, PREFIX + BLOCK + n_slots)
    clean = noisy + n_slots
    attention_mask = torch.ones(1, PREFIX + BLOCK + 2 * n_slots + SUFFIX, dtype=torch.long)
    mask = Qwen3VLBlockDiffusion._make_block_diffusion_training_mask(
        stub, attention_mask, anchor[None], noisy[None], clean[None]
    )
    return mask[0, 0] == 0, anchor, noisy, clean


@skip_if_package_missing("transformers")
def test_default_mask_lets_clean_blocks_attend_anchor():
    allowed, anchor, noisy, clean = _training_mask(True)
    assert allowed[clean][:, anchor].all()
    assert not allowed[noisy][:, anchor].any()


@skip_if_package_missing("transformers")
def test_clean_blocks_attend_anchor_false_only_hides_anchor_from_clean_blocks():
    default, anchor, noisy, clean = _training_mask(True)
    allowed, _, _, _ = _training_mask(False)
    assert not allowed[clean][:, anchor].any()
    assert not allowed[noisy][:, anchor].any()
    assert allowed[anchor][:, anchor].all()  # the anchor still predicts block 0
    expected = default.clone()
    expected[clean[:, None], anchor[None, :]] = False
    assert torch.equal(allowed, expected)
