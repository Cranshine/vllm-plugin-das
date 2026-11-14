# SPDX-License-Identifier: Apache-2.0
"""CPU wiring regressions for the K3.1 fused rmsnorm+quant decode path.

Device kernels are test doubles: the fused-norm module and the HCU linear
prequant contract are fakes so the patch wiring runs on CPU only.
"""

import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from vllm_hcu.platforms import envs as henvs
from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_reference_paths


@pytest.fixture
def fake_moe_runner(monkeypatch):
    """Stand in for the real MoE runner module (avoids its torch-op
    registrations; this suite exercises rmsnorm wiring only)."""
    runner_module = ModuleType(
        "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
    )
    runner_module.MoERunner = type("Runner", (), {})
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    return runner_module


class _FakeRMSNorm(nn.Module):
    """Plain RMSNorm double matching the wheel module's contract."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        return x * self.weight


class _FakeLinear(nn.Module):
    """Column-parallel double exposing the HCU prequant contract."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.prequant_calls = 0

    def _forward_with_hcu_quanted(self, x, x_and_scale_quanted):
        if x_and_scale_quanted is not None:
            self.prequant_calls += 1
        return x @ self.weight.t(), None

    def forward(self, x):
        return self._forward_with_hcu_quanted(x, None)


class _FakeFusedRMSNormAndQuant(nn.Module):
    """FusedRMSNormAndQuant double: normalizes in place, yields (q, s)."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None, quant_dtype=torch.int8,
                update_input=True):
        assert update_input, "decoder path relies on in-place normalization"
        x.mul_(2.0)  # stands in for the normalized write-back
        q = (x * 0.25).to(torch.int8)
        s = torch.full((x.shape[0], 1), 0.5, dtype=torch.float32)
        return q, s, None


class _InnerMLA(nn.Module):
    """Inner attention backend double recording wrapper-level calls."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.calls = []

    def forward(self, q, kv_c_normed, k_pe, output_shape=None):
        self.calls.append(output_shape)
        return torch.zeros(output_shape)


class _FakeKimiK3MLAWrapper(nn.Module):
    """Minimal K3 MLA wrapper shaped like the production one.

    The production wrapper carries the projections themselves (see
    test_kimi_k3_reference_adapters.py) plus an inner ``mla_attn`` backend.
    """

    def __init__(self, hidden_size=8):
        super().__init__()
        self.q_lora_rank = 4
        self.kv_lora_rank = 4
        self.qk_rope_head_dim = 1
        self.qk_head_dim = 5
        self.qk_nope_head_dim = 4
        self.v_head_dim = 1
        self.num_heads = 2
        self.fused_qkv_a_proj = _FakeLinear(
            hidden_size, self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        )
        self.q_a_layernorm = _FakeRMSNorm(self.q_lora_rank)
        self.q_b_proj = _FakeLinear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim
        )
        self.kv_a_layernorm = _FakeRMSNorm(self.kv_lora_rank)
        self.o_proj = _FakeLinear(self.num_heads * self.v_head_dim, hidden_size)
        self.rotary_emb = None
        self.indexer = None
        self.is_sparse = False
        self.skip_topk = False
        self.mla_attn = _InnerMLA(self.num_heads * self.v_head_dim)


class _FakeGenericMLAWrapper(nn.Module):
    """Stand-in for the generic wrapper DSA layers route through."""

    def forward(self, positions, hidden_states, *args, **kwargs):
        raise AssertionError(
            "DSA layers must not receive the fused prequant argument"
        )


class _FakeKimiMLAAttention(nn.Module):
    def __init__(self, mla_attn):
        super().__init__()
        self.q_lora_rank = 4
        self.mla_attn = mla_attn


class _FakeKimiMoE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gate = type("Gate", (), {})()
        self.gate.weight = nn.Parameter(torch.zeros(4, 8))
        self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(4))
        self.gate.allow_cublas_router_gemm = True
        self.experts = type("Experts", (), {})()
        self.experts._shared_experts = None


class _FakeKimiRoutedOutputTransform(nn.Module):
    pass


class _FakeKimiDecoderLayer(nn.Module):
    """Decoder double shaped like the wheel layer's attn-res contract."""

    def __init__(self, layer_idx=0, generic=False, attn_res=True):
        super().__init__()
        self.hidden_size = 8
        self.use_attn_residuals = attn_res
        wrapper = _FakeGenericMLAWrapper() if generic else _FakeKimiK3MLAWrapper()
        self.self_attn = _FakeKimiMLAAttention(wrapper)
        self.input_layernorm = _FakeRMSNorm(8)
        self.post_attention_layernorm = _FakeRMSNorm(8)
        self.self_attention_res_proj = _FakeLinear(8, 1)
        self.self_attention_res_norm = _FakeRMSNorm(8)
        self.mlp_res_proj = _FakeLinear(8, 1)
        self.mlp_res_norm = _FakeRMSNorm(8)
        self.is_block_write_layer = False
        self.block_write_idx = 0
        self.prev_valid_blocks = 0
        self.mlp = nn.Identity()

    def _run_self_attn(self, positions, hidden_states):
        attn_output = torch.empty_like(hidden_states)
        self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            output=attn_output,
        )
        return attn_output

    def forward_attn_residual(self, positions, hidden_states, block_residual):
        prefix_sum = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._run_self_attn(positions, hidden_states)
        prefix_sum = prefix_sum + hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual


def _fake_target():
    target = ModuleType(patch_kimi_k3_reference_paths.TARGET_MODULE)
    target.KimiRoutedOutputTransform = _FakeKimiRoutedOutputTransform
    target.KimiK3MultiHeadLatentAttentionWrapper = _FakeKimiK3MLAWrapper
    target.KimiMoE = _FakeKimiMoE
    target.KimiMLAAttention = _FakeKimiMLAAttention
    target.KimiDecoderLayer = _FakeKimiDecoderLayer
    return target


def _enable_hcu(monkeypatch, fused=True):
    monkeypatch.setattr(henvs, "custom_ops_enabled", lambda: True)
    monkeypatch.setattr(henvs, "k3_sp_moe_enabled", lambda: False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", fused)


def test_fused_rms_quant_disabled_keeps_plain_norm(monkeypatch, fake_moe_runner):
    _enable_hcu(monkeypatch, fused=False)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    layer = target.KimiDecoderLayer()
    x = torch.randn(3, 8)
    prefix, block_residual = layer.forward_attn_residual(
        torch.zeros(3, dtype=torch.long), x, torch.zeros(2, 3, 8)
    )
    assert isinstance(layer.input_layernorm, _FakeRMSNorm)
    assert layer.self_attn.mla_attn.fused_qkv_a_proj.prequant_calls == 0
    assert prefix.shape == (3, 8)
    assert block_residual.shape == (2, 3, 8)


def test_fused_rms_quant_swaps_norm_and_threads_prequant(monkeypatch, fake_moe_runner):
    _enable_hcu(monkeypatch, fused=True)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    layer = target.KimiDecoderLayer()
    fused = layer.input_layernorm
    assert isinstance(fused, _FakeFusedRMSNormAndQuant)
    assert layer._hcu_fused_rms_quant is True
    # post_attention_layernorm stays a plain norm (MoE side kept unfused).
    assert isinstance(layer.post_attention_layernorm, _FakeRMSNorm)

    wrapper = layer.self_attn.mla_attn
    x = torch.ones(3, 8)
    prefix, _ = layer.forward_attn_residual(
        torch.zeros(3, dtype=torch.long), x, torch.zeros(2, 3, 8)
    )
    # the prequant pair reached the fused QKV projection once
    assert wrapper.fused_qkv_a_proj.prequant_calls == 1
    # hidden_states was normalized in place (fake doubles the values)
    assert torch.allclose(x, torch.full((3, 8), 2.0))
    assert prefix.shape == (3, 8)


def test_fused_rms_quant_skips_dsa_and_generic_wrapper(monkeypatch, fake_moe_runner):
    _enable_hcu(monkeypatch, fused=True)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    # DSA layer: same stock attention class but generic wrapper instance.
    dsa_layer = target.KimiDecoderLayer(generic=True)
    assert isinstance(dsa_layer.input_layernorm, _FakeRMSNorm)
    assert not getattr(dsa_layer, "_hcu_fused_rms_quant", False)


def test_fused_rms_quant_skips_plain_residual_layers(monkeypatch, fake_moe_runner):
    _enable_hcu(monkeypatch, fused=True)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    # Non attn-res layer (plain 2-tuple RMSNorm residual contract).
    plain = target.KimiDecoderLayer(attn_res=False)
    assert isinstance(plain.input_layernorm, _FakeRMSNorm)
    assert not getattr(plain, "_hcu_fused_rms_quant", False)


def test_fused_rms_quant_respects_env_gates(monkeypatch, fake_moe_runner):
    _enable_hcu(monkeypatch, fused=True)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    layer = target.KimiDecoderLayer()
    assert isinstance(layer.input_layernorm, _FakeFusedRMSNormAndQuant)

    # custom-ops gate off: plain norm stays.
    monkeypatch.setattr(henvs, "custom_ops_enabled", lambda: False)
    layer_no_ops = target.KimiDecoderLayer()
    assert isinstance(layer_no_ops.input_layernorm, _FakeRMSNorm)

    # SP-MoE on: the input-norm fusion only touches the attention side, so
    # it stays enabled (the MoE-side post_attention_layernorm is not fused).
    monkeypatch.setattr(henvs, "custom_ops_enabled", lambda: True)
    monkeypatch.setattr(henvs, "k3_sp_moe_enabled", lambda: True)
    layer_sp = target.KimiDecoderLayer()
    assert isinstance(layer_sp.input_layernorm, _FakeFusedRMSNormAndQuant)


def test_decode_clean_logits_flag_lazy_resolution(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS", raising=False)
    assert henvs.VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS is False
    monkeypatch.setenv("VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS", "1")
    assert henvs.VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS is True
    monkeypatch.setenv("VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS", "0")
    assert henvs.VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS is False
