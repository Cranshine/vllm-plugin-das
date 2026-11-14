# SPDX-License-Identifier: Apache-2.0
"""Preserve the audited v0.25.1 Kimi-K3 kernel and TP reduction paths."""
from __future__ import annotations
from types import ModuleType
import functools
import torch
from vllm_hcu.platforms import envs as henvs
from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.reference_paths"
TARGETS = (f"{TARGET_MODULE}.KimiDecoderLayer.forward_attn_residual",)
_MARKER = "_vllm_hcu_kimi_reference_paths_applied"

def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
) -> torch.Tensor:
    if num_valid_blocks <= 0:
        return prefix_sum

    from boltops.generic import attn_res
    return attn_res(
        prefix_sum,
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        num_valid_blocks,
        norm.variance_epsilon,
    )

def forward_attn_residual(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_sum = hidden_states
    hidden_states = _apply_attn_res(
        prefix_sum,
        block_residual,
        self.self_attention_res_proj,
        self.self_attention_res_norm,
        self.prev_valid_blocks,
    )

    if self.is_block_write_layer:
        block_residual[:, self.block_write_idx, :].copy_(prefix_sum)
        prefix_sum = None

    if getattr(self, "_hcu_fused_rms_quant", False):
        # FusedRMSNormAndQuant normalizes ``hidden_states`` in place and
        # returns the int8/scale pair for the QKV projection, replacing the
        # rmsnorm kernel + per-token quant pair (default update_input=True
        # keeps the bf16 normed values the indexer and output gate read).
        q_quanted, s_quanted, _ = self.input_layernorm(hidden_states)
        hidden_states = self._run_self_attn(positions, hidden_states, q_quanted, s_quanted)
    else:
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._run_self_attn(positions, hidden_states)
    if prefix_sum is not None:
        prefix_sum = prefix_sum + hidden_states
    else:
        prefix_sum = hidden_states

    mlp_valid_blocks = self.prev_valid_blocks + (
        1 if self.is_block_write_layer else 0
    )
    hidden_states = _apply_attn_res(
        prefix_sum,
        block_residual,
        self.mlp_res_proj,
        self.mlp_res_norm,
        mlp_valid_blocks,
    )

    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    prefix_sum = prefix_sum + hidden_states
    return prefix_sum, block_residual

def _reference_mla_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
    x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    q_c = None
    if self.q_lora_rank is not None:
        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None
        assert self.q_b_proj is not None
        qkv_output = self.fused_qkv_a_proj._forward_with_hcu_quanted(
            hidden_states, x_and_scale_quanted
        )
        qkv_lora = qkv_output[0] if isinstance(qkv_output, tuple) else qkv_output
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        assert self.kv_a_proj_with_mqa is not None
        assert self.q_proj is not None
        kv_output = self.kv_a_proj_with_mqa._forward_with_hcu_quanted(
            hidden_states, x_and_scale_quanted
        )
        kv_lora = kv_output[0] if isinstance(kv_output, tuple) else kv_output
        q_output = self.q_proj._forward_with_hcu_quanted(
            hidden_states, x_and_scale_quanted
        )
        q = q_output[0] if isinstance(q_output, tuple) else q_output

    kv_c, k_pe = kv_lora.split(
        [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    kv_c_normed = self.kv_a_layernorm(kv_c)
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if self.rotary_emb is not None:
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )
    if self.indexer and self.is_sparse and not self.skip_topk:
        self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
    if llama_4_scaling is not None:
        q *= llama_4_scaling

    attn_out = self.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
    )
    g_proj = getattr(self, "g_proj", None)
    if g_proj is not None:
        attn_out = attn_out * g_proj(hidden_states)[0].sigmoid()
    return self.o_proj(attn_out)[0]


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    decoder = require_class(target, "KimiDecoderLayer", f"{TARGET_MODULE}.KimiDecoderLayer")
    transform = require_class(target, "KimiRoutedOutputTransform", f"{TARGET_MODULE}.KimiRoutedOutputTransform")
    from vllm_hcu.model_executor.layers.fused_moe.moe_runner import MoERunner
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_situ import SituAndMul
    from vllm_hcu.ops.fuse_rms_norm_quant import FusedRMSNormAndQuant
    target.SituAndMul = SituAndMul
    mla_wrapper = target.KimiK3MultiHeadLatentAttentionWrapper
    original_mla_forward = mla_wrapper.forward

    @functools.wraps(original_mla_forward)
    def hcu_mla_forward(self, positions, hidden_states, llama_4_scaling=None,
                        x_and_scale_quanted=None):
        if getattr(self, "dcp_q_replicate", False):
            return original_mla_forward(self, positions, hidden_states, llama_4_scaling)
        return _reference_mla_forward(
            self, positions, hidden_states, llama_4_scaling, x_and_scale_quanted
        )

    mla_wrapper.forward = hcu_mla_forward
    # The stock KimiMLAAttention.forward hands the wrapper only positions and
    # hidden states. Extend it so the fused rmsnorm+quant pair produced by the
    # decoder layer reaches the QKV projection without a re-quant. DSA layers
    # transplant their state onto this same stock class; forward stays
    # byte-compatible (the extra argument is optional and only consumed by the
    # K3 MLA wrapper). Minimal audit fixtures may omit the class entirely.
    attention_cls = getattr(target, "KimiMLAAttention", None)
    if attention_cls is not None and hasattr(attention_cls, "forward"):
        original_attention_forward = attention_cls.forward

        @functools.wraps(original_attention_forward)
        def hcu_mla_attention_forward(self, positions, hidden_states, output,
                                      x_and_scale_quanted=None):
            if x_and_scale_quanted is None:
                output[:] = self.mla_attn(positions, hidden_states)
            else:
                output[:] = self.mla_attn(
                    positions, hidden_states, x_and_scale_quanted=x_and_scale_quanted
                )

        attention_cls.forward = hcu_mla_attention_forward
    # The 028 sharded latent-tail runner uses a different collective/GEMM
    # sequence. The HCU runner fuses latent+shared partials before RMSNorm.
    target.ROCmLatentMoERunner = MoERunner
    transform._hcu_fuse_shared_and_routed_tp_all_reduce = True
    original_residual_forward = decoder.forward_attn_residual

    @functools.wraps(original_residual_forward)
    def hcu_residual_forward(self, *args, **kwargs):
        if not henvs.custom_ops_enabled():
            return original_residual_forward(self, *args, **kwargs)
        return forward_attn_residual(self, *args, **kwargs)

    # The upstream helper accepts fused norm/delta arguments that the BoltOps
    # helper does not. Restore the complete upstream forward when disabled.
    decoder.forward_attn_residual = hcu_residual_forward

    original_run_self_attn = getattr(decoder, "_run_self_attn", None)

    if original_run_self_attn is not None and callable(original_run_self_attn):
        @functools.wraps(original_run_self_attn)
        def hcu_run_self_attn(self, positions, hidden_states, q_quanted=None,
                              s_quanted=None):
            attn_output = torch.empty_like(hidden_states)
            if q_quanted is None:
                self.self_attn(
                    hidden_states=hidden_states,
                    positions=positions,
                    output=attn_output,
                )
            elif getattr(self, "_hcu_prequant_via_pin", False):
                # K3.1 DSA layers route through the stock generic MLA
                # wrapper, whose forward signature stays (positions,
                # hidden_states). Pin the fused pair on the target
                # projection; the wrapper's bare proj(hidden) call consumes
                # it inside LinearBase._forward_with_hcu_quanted.
                proj = getattr(
                    getattr(self.self_attn, "mla_attn", None),
                    "fused_qkv_a_proj", None,
                )
                if proj is not None:
                    proj._hcu_next_quanted = (q_quanted, s_quanted)
                self.self_attn(
                    hidden_states=hidden_states,
                    positions=positions,
                    output=attn_output,
                )
            else:
                self.self_attn(
                    hidden_states=hidden_states,
                    positions=positions,
                    output=attn_output,
                    x_and_scale_quanted=(q_quanted, s_quanted),
                )
            return attn_output

        decoder._run_self_attn = hcu_run_self_attn

    def _enable_fused_rms_quant(layer) -> None:
        """Swap the QKV-feeding norms for FusedRMSNormAndQuant in place.

        MLA layers qualify through either wrapper: the audited K3 wrapper
        threads the pair via an extra forward kwarg, and the K3.1 DSA
        layers — which on the Pro model are every full-attention layer —
        route through the stock generic ``MultiHeadLatentAttentionWrapper``.
        That generic forward signature stays untouched; the pair travels
        pinned on the projection (``_hcu_next_quanted``) and is consumed by
        its bare ``proj(hidden)`` call in
        ``LinearBase._forward_with_hcu_quanted``. KDA layers have no MLA
        QKV projection and stay excluded. The RMSNorm weight Parameter
        object is shared with the fused module, so ``load_weights`` keeps
        populating the same storage. The swap is limited to attn-res blocks
        (K3.1 Pro), whose HCU ``forward_attn_residual`` consumes the fused
        pair; the plain ``forward`` residual path keeps its 2-tuple RMSNorm
        contract.
        """
        if not (
            henvs.VLLM_HCU_USE_FUSED_RMS_QUANT and henvs.custom_ops_enabled()
        ):
            return
        if not getattr(layer, "use_attn_residuals", False):
            return
        attention_cls = getattr(target, "KimiMLAAttention", None)
        attention = getattr(layer, "self_attn", None)
        if attention_cls is None or not isinstance(attention, attention_cls):
            return
        mla_attn = getattr(attention, "mla_attn", None)
        if isinstance(mla_attn, target.KimiK3MultiHeadLatentAttentionWrapper):
            layer._hcu_prequant_via_pin = False
        else:
            # Generic wrapper (K3.1 DSA): requires a HCU projection to
            # carry the pinned pair.
            proj = getattr(mla_attn, "fused_qkv_a_proj", None)
            if proj is None or not callable(
                getattr(proj, "_forward_with_hcu_quanted", None)
            ):
                return
            layer._hcu_prequant_via_pin = True
        if getattr(attention, "q_lora_rank", None) is None:
            return
        # Only input_layernorm is fused: it feeds the QKV projection, whose
        # HCU linear contract accepts the prequantized int8/scale pair.
        # post_attention_layernorm feeds the router gate (bf16) plus the MoE
        # dispatch quant and keeps its plain RMSNorm forward — that is also
        # why SP-MoE stays compatible: the K3 SP-MoE path replaces only the
        # post-attention MoE side, never the attention input norm. Norm calls
        # in ``forward_attn_residual`` and the module swap below stay in
        # lockstep.
        norm = getattr(layer, "input_layernorm", None)
        if norm is None or isinstance(norm, FusedRMSNormAndQuant):
            return
        fused = FusedRMSNormAndQuant(
            norm.weight.shape[-1], eps=norm.variance_epsilon
        )
        fused.weight = norm.weight
        layer.input_layernorm = fused
        layer._hcu_fused_rms_quant = True

    original_decoder_init = decoder.__init__

    @functools.wraps(original_decoder_init)
    def hcu_decoder_init(self, *args, **kwargs):
        original_decoder_init(self, *args, **kwargs)
        _enable_fused_rms_quant(self)

    decoder.__init__ = hcu_decoder_init

    moe = require_class(target, "KimiMoE", f"{TARGET_MODULE}.KimiMoE")
    original_init = moe.__init__

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # 0251 uses the BF16 projection followed by an FP32 cast on ROCm.
        # 028 enables BF16->FP32 hipBLASLt GEMM for all ROCm routers.
        self.gate.allow_cublas_router_gemm = False
        # Keep the same checkpoint rounding and LightOp gate specialization
        # as 0251. Preserve the Parameter object referenced by the router.
        self.gate.e_score_correction_bias.data = (
            self.gate.e_score_correction_bias.data.to(self.gate.weight.dtype)
        )
        shared = getattr(self.experts, "_shared_experts", None)
        if shared is not None:
            # 0251's TP path is single-stream. Do not introduce auxiliary
            # stream graph capture or reorder shared-expert GEMMs.
            shared._stream = None

    moe.__init__ = hcu_init
    setattr(target, _MARKER, True)
    return True

def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
