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


@pytest.fixture
def fake_k31_module(monkeypatch):
    """Stand in for vllm_hcu.models.kimi_k31 (skips the heavy vllm model
    import); the identity of KimiK31FusedQuantMLAWrapper is what the patch
    isinstance whitelist checks, so the stub class is the real subject."""
    k31_module = ModuleType("vllm_hcu.models.kimi_k31")
    k31_module.KimiK31FusedQuantMLAWrapper = _FakeK31FusedMLAWrapper
    monkeypatch.setitem(sys.modules, k31_module.__name__, k31_module)
    return k31_module


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
    """Stand-in for the plain generic wrapper (no prequant support).

    DSA layers must route through :class:`_FakeK31FusedMLAWrapper` when the
    fusion gate is on; this class exists to prove the whitelist still skips
    wrappers that cannot carry the pair.
    """

    def forward(self, positions, hidden_states, *args, **kwargs):
        raise AssertionError(
            "plain generic wrappers must not receive the fused prequant arg"
        )


class _FakeK31FusedMLAWrapper(nn.Module):
    """K3.1 DSA fused wrapper double: submodules + explicit-pair capsule."""

    def __init__(self, hidden_size=8, is_sparse=True):
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
        self.is_sparse = is_sparse
        self.skip_topk = False
        self.fuse_qkv_rmsnorm = False
        self.dcp_q_replicate = False
        self.g_proj = None
        self.mla_attn = _InnerMLA(self.num_heads * self.v_head_dim)
        self.forward_kwargs = []

    def forward(
        self, positions, hidden_states, llama_4_scaling=None,
        x_and_scale_quanted=None,
    ):
        if x_and_scale_quanted is not None:
            self.fused_qkv_a_proj._forward_with_hcu_quanted(
                hidden_states, x_and_scale_quanted
            )
        self.forward_kwargs.append(
            {"x_and_scale_quanted": x_and_scale_quanted}
        )
        return torch.zeros(hidden_states.shape[0], 8)


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

    def __init__(self, layer_idx=0, generic=False, attn_res=True, wrapper=None):
        super().__init__()
        self.hidden_size = 8
        self.use_attn_residuals = attn_res
        if wrapper is None:
            wrapper = (
                _FakeGenericMLAWrapper() if generic else _FakeKimiK3MLAWrapper()
            )
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
    monkeypatch.setattr(henvs, "fused_rms_quant_enabled", lambda: fused)


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


def test_fused_rms_quant_skips_dsa_and_generic_wrapper(
    monkeypatch, fake_moe_runner, fake_k31_module
):
    _enable_hcu(monkeypatch, fused=True)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    # DSA layer whose wrapper is the PLAIN generic class: the isinstance
    # whitelist must still reject it even with the fusion env enabled.
    dsa_layer = target.KimiDecoderLayer(generic=True)
    assert isinstance(dsa_layer.input_layernorm, _FakeRMSNorm)
    assert not getattr(dsa_layer, "_hcu_fused_rms_quant", False)


def test_fused_rms_quant_dsa_subclass_threads_prequant(
    monkeypatch, fake_moe_runner, fake_k31_module
):
    _enable_hcu(monkeypatch, fused=True)
    monkeypatch.setattr(
        "vllm_hcu.ops.fuse_rms_norm_quant.FusedRMSNormAndQuant",
        _FakeFusedRMSNormAndQuant,
    )
    target = _fake_target()
    assert patch_kimi_k3_reference_paths.apply_to_module(target)

    dsa_layer = target.KimiDecoderLayer(wrapper=_FakeK31FusedMLAWrapper())
    assert isinstance(dsa_layer.input_layernorm, _FakeFusedRMSNormAndQuant)
    assert dsa_layer._hcu_fused_rms_quant is True

    wrapper = dsa_layer.self_attn.mla_attn
    prefix, _ = dsa_layer.forward_attn_residual(
        torch.zeros(3, dtype=torch.long), torch.ones(3, 8), torch.zeros(2, 3, 8)
    )
    # The fused pair reached the DSA wrapper through the explicit kwarg and
    # hit the fused QKV projection exactly once.
    last = wrapper.forward_kwargs[-1]
    assert last["x_and_scale_quanted"] is not None
    assert wrapper.fused_qkv_a_proj.prequant_calls == 1
    assert prefix.shape == (3, 8)


def test_envs_fused_rms_quant_gate_truth_table(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_USE_FUSED_RMS_QUANT", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    assert henvs.fused_rms_quant_enabled() is False
    monkeypatch.setenv("VLLM_HCU_USE_FUSED_RMS_QUANT", "1")
    assert henvs.fused_rms_quant_enabled() is True
    # master switch off => gate off regardless of the child switch
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert henvs.fused_rms_quant_enabled() is False
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.setenv("VLLM_HCU_USE_FUSED_RMS_QUANT", "False")
    assert henvs.fused_rms_quant_enabled() is False


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

    # combined gate off (master switch closed): plain norm stays.
    monkeypatch.setattr(henvs, "fused_rms_quant_enabled", lambda: False)
    layer_no_ops = target.KimiDecoderLayer()
    assert isinstance(layer_no_ops.input_layernorm, _FakeRMSNorm)

    # SP-MoE on: the input-norm fusion only touches the attention side, so
    # it stays enabled (the MoE-side post_attention_layernorm is not fused;
    # the SP forward threads the pair through _run_self_attn itself).
    monkeypatch.setattr(henvs, "fused_rms_quant_enabled", lambda: True)
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


# ---------------------------------------------------------------------------
# The real KimiK31FusedQuantMLAWrapper (audited forward clone + drift guard).
# These tests import the actual kimi_k31 model module.
# ---------------------------------------------------------------------------


class _CallableMLA:
    """Upstream's self.mla_attn is a callable MLAAttention module."""

    def __init__(self):
        self.calls = 0

    def __call__(self, q, kv_c_normed, k_pe, output_shape=None,
                 q_dcp_replicated=None):
        self.calls += 1
        assert output_shape is not None
        return torch.zeros(output_shape)


def _build_real_k31_wrapper():
    from vllm_hcu.models.kimi_k31 import KimiK31FusedQuantMLAWrapper

    wrapper = object.__new__(KimiK31FusedQuantMLAWrapper)
    # The real __init__ needs a full MLAModules stack; bypass it and bring
    # up the nn.Module machinery explicitly so submodule assignment works
    # (same transplant pattern KimiK31DSAAttention uses on foreign
    # instances).
    nn.Module.__init__(wrapper)
    wrapper.q_lora_rank = 4
    wrapper.kv_lora_rank = 4
    wrapper.qk_rope_head_dim = 1
    wrapper.qk_head_dim = 5
    wrapper.qk_nope_head_dim = 4
    wrapper.v_head_dim = 1
    wrapper.num_heads = 2
    wrapper.fused_qkv_a_proj = _FakeLinear(8, 9)
    wrapper.q_a_layernorm = _FakeRMSNorm(4)
    wrapper.q_b_proj = _FakeLinear(4, 10)
    wrapper.kv_a_layernorm = _FakeRMSNorm(4)
    wrapper.o_proj = _FakeLinear(2, 8)
    wrapper.rotary_emb = None
    wrapper.indexer = None
    wrapper.indexer_rope_emb = None
    wrapper.is_sparse = True
    wrapper.skip_topk = False
    wrapper.fuse_qkv_rmsnorm = False
    wrapper.dcp_q_replicate = False
    wrapper.g_proj = None
    wrapper.mla_attn = _CallableMLA()
    return wrapper


def test_real_k31_fused_wrapper_threads_pair_and_hits_proj_once():
    wrapper = _build_real_k31_wrapper()
    hidden = torch.randn(3, 8)
    pair = (
        torch.zeros(3, 8, dtype=torch.int8),
        torch.full((3, 1), 0.5),
    )
    out = wrapper.forward(
        torch.zeros(3, dtype=torch.long), hidden, x_and_scale_quanted=pair
    )
    assert wrapper.fused_qkv_a_proj.prequant_calls == 1
    assert wrapper.mla_attn.calls == 1
    assert out.shape == (3, 8)


def test_real_k31_fused_wrapper_falls_back_without_pair():
    wrapper = _build_real_k31_wrapper()
    hidden = torch.randn(3, 8)
    # No pair: pure upstream behavior (the upstream forward runs our fakes
    # fine -- bare proj call, same splits, same attention call).
    out = wrapper.forward(torch.zeros(3, dtype=torch.long), hidden)
    assert wrapper.fused_qkv_a_proj.prequant_calls == 0
    assert out.shape == (3, 8)


def test_real_k31_fused_wrapper_falls_back_on_lightly_cp(monkeypatch):
    """Lightly-CP instances must never take the single-rank fast path.

    worker.op_opt.mla.lightly_cp_wrapper rewrites the base class forward to
    all-gather kv_c_normed/k_pe under CP; the fused fast path has no such
    gather, so a CP-enabled wrapper must fall back to the patched base
    forward (dropping the pair) even when the decoder supplies one.
    """
    from types import SimpleNamespace

    wrapper = _build_real_k31_wrapper()
    wrapper._hcu_feature_config = SimpleNamespace(enable_lightly_cp=True)
    hidden = torch.randn(3, 8)
    pair = (
        torch.zeros(3, 8, dtype=torch.int8),
        torch.full((3, 1), 0.5),
    )
    out = wrapper.forward(
        torch.zeros(3, dtype=torch.long), hidden, x_and_scale_quanted=pair
    )
    # Fast path skipped: the pair never reached the QKV projection.
    assert wrapper.fused_qkv_a_proj.prequant_calls == 0
    assert out.shape == (3, 8)


@pytest.fixture
def _reset_signature_guard():
    from vllm_hcu.models import kimi_k31

    original = kimi_k31._upstream_mla_forward_verified
    kimi_k31._upstream_mla_forward_verified = False
    yield
    # Restore the pre-test truth: hardcoding True here would mask a real
    # signature drift for every later consumer in the same process.
    kimi_k31._upstream_mla_forward_verified = original


def test_real_k31_signature_guard_passes_pinned_upstream(_reset_signature_guard):
    from vllm_hcu.models import kimi_k31

    kimi_k31._verify_upstream_mla_forward_once()
    assert kimi_k31._upstream_mla_forward_verified is True


def test_real_k31_signature_guard_rejects_upstream_drift(
    monkeypatch, _reset_signature_guard
):
    from vllm_hcu.models import kimi_k31

    class _DriftedUpstream(nn.Module):
        def forward(self, positions, hidden_states, new_required_arg):
            raise AssertionError("drifted upstream must never run")

    monkeypatch.setattr(
        kimi_k31, "MultiHeadLatentAttentionWrapper", _DriftedUpstream
    )
    with pytest.raises(RuntimeError, match="signature changed"):
        kimi_k31._verify_upstream_mla_forward_once()
    assert kimi_k31._upstream_mla_forward_verified is False
