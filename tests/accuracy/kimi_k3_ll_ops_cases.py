# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3 SiTU DeepEP LL numerical checks on live HCU."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.hcu


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("a live HCU/ROCm device is required")
    return torch.device("cuda", 0)


def _pack_int4_high_low(weight: torch.Tensor) -> torch.Tensor:
    nibbles = weight.to(torch.int16) & 0xF
    pairs = nibbles.view(*weight.shape[:-1], -1, 2)
    return ((pairs[..., 0] << 4) | pairs[..., 1]).to(torch.int8)


def _unpack_int4_high_low(packed: torch.Tensor) -> torch.Tensor:
    values = packed.view(torch.uint8)
    high = ((values >> 4) & 0xF).to(torch.int16)
    low = (values & 0xF).to(torch.int16)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    return torch.stack((high, low), dim=-1).flatten(-2).contiguous()


def _situ(gateup: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    gate, up = gateup.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def test_kimi_ll_situ_masked_w4a8_matches_dequantized_reference() -> None:
    from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc
    from lightop.activation import fuse_situ_mul_quant_ep
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8_deepgemm_runtime import (
        KimiK3LLExperts,
    )

    device = _device()
    experts_count, max_tokens, hidden, intermediate = 2, 32, 7168, 1536
    counts = torch.tensor([32, 16], device=device, dtype=torch.int32)
    generator = torch.Generator(device=device).manual_seed(20261011)
    activation = torch.randint(
        -8,
        8,
        (experts_count, max_tokens, hidden),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    activation_scale = torch.rand(
        (experts_count, max_tokens, 1),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).add_(0.01)
    logical_w13 = torch.randint(
        -8,
        8,
        (experts_count, 2 * intermediate, hidden),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    logical_w2 = torch.randint(
        -8,
        8,
        (experts_count, hidden, intermediate),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    checkpoint_w13 = _pack_int4_high_low(logical_w13)
    checkpoint_w2 = _pack_int4_high_low(logical_w2)
    raw_w13 = checkpoint_w13.clone()
    raw_w2 = checkpoint_w2.clone()
    scale_w13 = torch.rand(
        (experts_count, 2 * intermediate, 1),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).add_(0.01)
    scale_w2 = torch.rand(
        (experts_count, hidden, 1),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).add_(0.01)

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(checkpoint_w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(checkpoint_w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(scale_w13, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(scale_w2, requires_grad=False)
    experts = object.__new__(KimiK3LLExperts)
    experts._deepgemm_w13 = None
    experts._deepgemm_w2 = None
    experts.process_weights_after_loading(layer)

    assert experts._deepgemm_w13.untyped_storage().data_ptr() == (
        layer.w13_weight.untyped_storage().data_ptr()
    )
    assert experts._deepgemm_w2.untyped_storage().data_ptr() == (
        layer.w2_weight.untyped_storage().data_ptr()
    )

    workspace1 = torch.empty(
        (experts_count, max_tokens, 2 * intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    expected_m = 16
    scale_w13_hipc = layer.w13_weight_scale
    scale_w2_hipc = layer.w2_weight_scale
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (activation, activation_scale),
        (experts._deepgemm_w13, scale_w13_hipc),
        workspace1,
        counts,
        expected_m,
    )
    quantized_situ, situ_scale = fuse_situ_mul_quant_ep(
        workspace1,
        counts,
        situ_beta=1.25,
        situ_linear_beta=2.0,
        expect_m=expected_m,
    )
    output = torch.empty(
        (experts_count, max_tokens, hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (quantized_situ, situ_scale),
        (experts._deepgemm_w2, scale_w2_hipc),
        output,
        counts,
        expected_m,
    )

    unpacked_w13 = _unpack_int4_high_low(raw_w13)
    unpacked_w2 = _unpack_int4_high_low(raw_w2)
    for expert_id, count in enumerate(counts.tolist()):
        first_reference = (
            activation[expert_id, :count].float()
            * activation_scale[expert_id, :count]
        ) @ (
            unpacked_w13[expert_id].float()
            * (scale_w13_hipc[expert_id] * 16.0)
        ).T
        torch.testing.assert_close(
            workspace1[expert_id, :count].float(),
            first_reference,
            rtol=3e-2,
            atol=0.1,
        )

        situ_reference = _situ(
            workspace1[expert_id, :count], beta=1.25, linear_beta=2.0
        )
        quantized_reference = (
            quantized_situ[expert_id, :count].float()
            * situ_scale[expert_id, :count]
        )
        torch.testing.assert_close(
            quantized_reference,
            situ_reference,
            rtol=5e-2,
            atol=0.1,
        )
        output_reference = quantized_reference @ (
            unpacked_w2[expert_id].float()
            * (scale_w2_hipc[expert_id] * 16.0)
        ).T
        torch.testing.assert_close(
            output[expert_id, :count].float(),
            output_reference,
            rtol=4e-2,
            atol=0.25,
        )
        assert torch.isfinite(output[expert_id, :count]).all()
