# SPDX-License-Identifier: Apache-2.0
"""Use the same external Conv1D kernels as the 0251 Kimi KDA path."""
from __future__ import annotations
import os
import torch
from types import ModuleType
from ._common import load_exact_module

TARGET_MODULE = "vllm.models.kimi_k3.amd.kda"
PATCH_ID = "worker.op_opt.kimi_k3.kda_reference_conv"
TARGETS = (f"{TARGET_MODULE}.causal_conv1d_fn", f"{TARGET_MODULE}.causal_conv1d_update",
           f"{TARGET_MODULE}.chunk_kda_with_fused_gate")
_MARKER = "_vllm_hcu_kimi_reference_conv_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    original_fn, original_update = target.causal_conv1d_fn, target.causal_conv1d_update
    original_chunk = target.chunk_kda_with_fused_gate

    def hcu_fn(x, weight, bias, *, activation, conv_states, has_initial_state,
               cache_indices, query_start_loc, metadata):
        if os.environ.get("VLLM_KIMI_CONV1D_BACKEND", "1") != "1":
            return original_fn(x, weight, bias, activation=activation,
                               conv_states=conv_states, has_initial_state=has_initial_state,
                               cache_indices=cache_indices, query_start_loc=query_start_loc,
                               metadata=metadata)
        from causal_conv1d import causal_conv1d_fn_hcu
        # 0251's default prefill path derives these exact sequence lengths.
        return causal_conv1d_fn_hcu(
            x, weight, bias, activation=activation, initial_states=conv_states,
            has_initial_state=has_initial_state, cache_indices=cache_indices,
            query_start_loc=query_start_loc, seq_lens_cpu=query_start_loc.diff().tolist())

    def hcu_update(x, conv_state, weight, bias=None, **kwargs):
        if (os.environ.get("VLLM_KIMI_CONV1D_BACKEND", "1") != "1"
                or kwargs.get("num_accepted_tokens") is not None
                or kwargs.get("query_start_loc") is not None):
            return original_update(x, conv_state, weight, bias, **kwargs)
        from causal_conv1d import causal_conv1d_update
        # The external op returns its own dense output, like 0251. KDA uses
        # this return value, so do not add a copy into Triton's scratch buffer.
        return causal_conv1d_update(
            x, conv_state, weight, bias, activation=kwargs.get("activation"),
            conv_state_indices=kwargs.get("conv_state_indices"))

    target.causal_conv1d_fn, target.causal_conv1d_update = hcu_fn, hcu_update

    def hcu_chunk(**kwargs):
        q = kwargs["q"]
        if (os.environ.get("VLLM_KIMI_FLASHKDA_BACKEND", "1") != "1"
                or kwargs.get("lower_bound") is None or q.shape[-1] != 128):
            return original_chunk(**kwargs)
        from flash_kda import fwd
        initial_state = kwargs["initial_state"]
        output = torch.empty_like(kwargs["v"])
        final_state = torch.empty_like(initial_state)
        fwd(q.contiguous(), kwargs["k"].contiguous(), kwargs["v"].contiguous(),
            kwargs["raw_g"].contiguous(), kwargs["raw_beta"].contiguous(),
            float(q.shape[-1] ** -0.5), output, kwargs["A_log"].contiguous(),
            kwargs["g_bias"].view(q.shape[-2], q.shape[-1]).contiguous(),
            float(kwargs["lower_bound"]), initial_state=initial_state.contiguous(),
            final_state=final_state, cu_seqlens=kwargs["cu_seqlens"].to(torch.int64).contiguous())
        return output, final_state

    target.chunk_kda_with_fused_gate = hcu_chunk
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
