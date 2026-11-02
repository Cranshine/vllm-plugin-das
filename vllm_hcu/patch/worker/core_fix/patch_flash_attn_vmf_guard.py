# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shrink the FA funnel max_seqlen_q so the gfx938 epilogue row guard compiles in.

Source conviction (flash-attention-das, NOT modified by this patch):

- ``csrc/flash_attn/fwd_launch_template.h:153``:
  ``is_even_MN = !is_varlen && seqlen_k % kBlockN == 0 && seqlen_q % kBlockM == 0``
  disarms the epilogue row guard whenever the launcher receives a seqlen_q
  that is a multiple of kBlockM=128.
- ``csrc/flash_attn/fwd_epilogue_gfx938.h:39-45`` (O rows) and
  ``csrc/flash_attn/fwd_epilogue.h:135-143`` (LSE rows): with the guard
  compiled out, the kernel stores full kBlockM=128 rows unconditionally.  A
  launcher seqlen_q=128 over a 64-row packed O walks +66 KiB past the slab
  and VMFaults (12 q-heads x d192 x bf16).

The dist funnel (vllm ``vllm/vllm_flash_attn/flash_attn_interface.py::
flash_attn_varlen_func``, fa_version=4 branch) forwards ``max_seqlen_q``
verbatim to the vendor launcher, so any padded/rounded caller value disarms
the guard.  R34 patched that funnel file in place; this adapter provides the
same protection from plugin code only, as a runtime monkey-patch armed by
the exact-import coordinator *before* any dist backend binds the funnel
symbol (module-completion ordering guarantees every later
``from vllm.vllm_flash_attn import flash_attn_varlen_func`` binds the
wrapper).

Guard rule (pure host-side shape arithmetic, capture-safe by construction):

- Fast path: untouched calls proceed unmodified after a python-int shape
  check.  No tensor ops, no allocation, no device synchronization, so the
  branch is legal inside full cudagraph capture and the decision bakes into
  the graph exactly like the constant launcher value it inspects.
- Shrink path: when the packed query rows provably cannot reach the
  launcher budget (``total_q - batch + 1 < max_seqlen_q``, the pigeonhole
  bound of per-request query lengths, reached only when every sibling
  request carries a single row), replace the budget with that bound — or
  ``bound + 127`` when the bound itself is a kBlockM multiple, keeping the
  budget strictly below the inflated value while never dropping below any
  real per-request length.  Both replacements satisfy
  ``max_seqlen_q % 128 != 0``, which compiles the epilogue guard back in:
  grid m-dim can only stay equal or grow (``ceil(shrunk/128) >=
  ceil(true_max/128)`` since shrunk >= true_max), extra m-blocks compute
  against the same metadata and masked rows store nothing.
- Everything else (bound already equal to the budget with sibling rows at
  one, or a bound that is a multiple with true_max pinned to it) is left
  untouched and only counted, because the write-OOB provably cannot fire
  when the budget equals the worst-case real row count.

The wrapper closes the three R34 coverage gaps: it acts before allocation
regardless of ``out`` ownership, applies to every fa_version branch, and
lives in vllm_hcu instead of editing the dist file.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.vllm_flash_attn.flash_attn_interface"
PATCH_ID = "worker.core_fix.flash_attn_vmf_guard"
_MARKER = "_vllm_hcu_fa4_vmf_guard_applied"
_WRAPPER_MARKER = "_vllm_hcu_fa4_vmf_guard_wrapper"

_K_BLOCK_M = 128

logger = logging.getLogger("vllm_hcu.patch.flash_attn_vmf_guard")

# Audited upstream signature (vllm-project/vllm flash_attn_interface.py,
# fetched 2026-11-02): all thirty-eight parameters are POSITIONAL_OR_KEYWORD
# in this exact order.  The funnel is the only seam that both owns the
# fa_version selector and reaches the vendor launcher for every call flavor.
_POSITIONAL_PARAMS: tuple[str, ...] = (
    "q",
    "k",
    "v",
    "max_seqlen_q",
    "cu_seqlens_q",
    "max_seqlen_k",
    "cu_seqlens_k",
    "seqused_k",
    "q_v",
    "dropout_p",
    "softmax_scale",
    "causal",
    "window_size",
    "softcap",
    "alibi_slopes",
    "deterministic",
    "return_attn_probs",
    "block_table",
    "return_softmax_lse",
    "out",
    "scheduler_metadata",
    "q_descale",
    "k_descale",
    "v_descale",
    "num_splits",
    "output_scale",
    "fa_version",
    "s_aux",
    "cp_world_size",
    "cp_rank",
    "cp_tot_seqused_k",
    "mask_mod",
    "block_sparse_tensors",
    "aux_tensors",
    "aux_tensor_leading_dims",
    "dynamic_causal",
    "gather_kv_indices",
    "gather_kv_valid_length",
)

# Host-side counters only: dict/int mutations are capture-legal, and the
# wrapper never reads them for control flow.  guard_counters() exposes the
# state for readiness probes without any device synchronization.
_counters: dict[str, int] = {
    "total": 0,
    "shrunk": 0,
    "passthrough": 0,
    "errors": 0,
}


def guard_counters() -> Mapping[str, int]:
    """Return the funnel wrapper counters for readiness/diagnostics probes."""

    return dict(_counters)


def _shrunken_budget(total_rows: int, batch: int, budget: int) -> int | None:
    """Return a lower launcher budget when rows provably underfill it.

    ``budget`` arrives as a nonzero multiple of ``_K_BLOCK_M`` (the only
    shape that disarms the epilogue guard).  The pigeonhole bound of the
    worst per-request query length is ``total_rows - batch + 1``; every
    sibling request contributes at least one row.  The replacement stays
    below ``budget`` and never drops below any real request length.
    """

    if batch < 1 or total_rows < batch:
        return None
    row_bound = total_rows - batch + 1
    if row_bound >= budget:
        # Worst-case rows equal the budget: the kernel cannot write past
        # the real packed rows, so the guard stays disarmed harmlessly.
        return None
    if row_bound % _K_BLOCK_M == 0:
        # Keep the replacement off every kBlockM multiple so the launch
        # predicate compiles the guard in.  row_bound < budget and budget
        # is a multiple imply budget >= row_bound + _K_BLOCK_M, so the
        # bumped value stays strictly under the inflated budget while
        # still covering row_bound (and therefore true_max) rows.
        return row_bound + _K_BLOCK_M - 1
    return row_bound


def _install_guard(original: Any) -> Any:
    """Wrap one funnel callable with the launcher-budget shrink guard."""

    signature = inspect.signature(original)

    @functools.wraps(original)
    def flash_attn_varlen_func_vmf_guard(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind(*args, **kwargs)
            arguments = bound.arguments
            _counters["total"] += 1
            budget = arguments.get("max_seqlen_q")
            query = arguments.get("q")
            cu_seqlens_q = arguments.get("cu_seqlens_q")
            if (
                type(budget) is int
                and budget > 0
                and budget % _K_BLOCK_M == 0
                and isinstance(query, torch.Tensor)
                and query.ndim == 3
                and isinstance(cu_seqlens_q, torch.Tensor)
                and cu_seqlens_q.ndim == 1
            ):
                replacement = _shrunken_budget(
                    query.shape[0],
                    cu_seqlens_q.shape[0] - 1,
                    budget,
                )
                if replacement is not None:
                    arguments["max_seqlen_q"] = replacement
                    _counters["shrunk"] += 1
                    if _counters["shrunk"] == 1:
                        logger.warning(
                            "[HCU FA4 VMF GUARD] shrunk launcher max_seqlen_q "
                            "%d -> %d (packed rows cannot reach the budget); "
                            "re-arms the fwd_epilogue row guard",
                            budget,
                            replacement,
                        )
                    bound.arguments = arguments
                    return original(*bound.args, **bound.kwargs)
            _counters["passthrough"] += 1
        except Exception:
            # Probe discipline: this wrapper must never introduce a new
            # crash path; fall through to the original call untouched.
            _counters["errors"] += 1
        return original(*args, **kwargs)

    setattr(flash_attn_varlen_func_vmf_guard, _WRAPPER_MARKER, True)
    return flash_attn_varlen_func_vmf_guard


def apply_to_module(module: ModuleType) -> bool:
    """Patch ``flash_attn_varlen_func`` on the dist funnel module."""

    funnel = load_exact_module(TARGET_MODULE, module)
    original = require_callable(
        funnel,
        "flash_attn_varlen_func",
        f"{TARGET_MODULE}.flash_attn_varlen_func",
    )
    if getattr(funnel, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    try:
        require_exact_signature(
            original,
            f"{TARGET_MODULE}.flash_attn_varlen_func",
            positional=_POSITIONAL_PARAMS,
            keyword_only=(),
            defaults={
                "cu_seqlens_k": None,
                "seqused_k": None,
                "q_v": None,
                "dropout_p": 0.0,
                "softmax_scale": None,
                "causal": False,
                "window_size": None,
                "softcap": 0.0,
                "alibi_slopes": None,
                "deterministic": False,
                "return_attn_probs": False,
                "block_table": None,
                "return_softmax_lse": False,
                "out": None,
                "scheduler_metadata": None,
                "q_descale": None,
                "k_descale": None,
                "v_descale": None,
                "num_splits": 0,
                "output_scale": None,
                "fa_version": getattr(funnel, "DEFAULT_FA_VERSION", 2),
                "s_aux": None,
                "cp_world_size": 1,
                "cp_rank": 0,
                "cp_tot_seqused_k": None,
                "mask_mod": None,
                "block_sparse_tensors": None,
                "aux_tensors": None,
                "aux_tensor_leading_dims": None,
                "dynamic_causal": None,
                "gather_kv_indices": None,
                "gather_kv_valid_length": None,
            },
        )
    except PatchCompatibilityError as exc:
        # Probe discipline: an unexpected dist-wheel funnel degrades to a
        # recorded no-op instead of failing unrelated model startups.
        logger.warning(
            "[HCU FA4 VMF GUARD] funnel signature drifted, guard disabled: %s",
            exc,
        )
        setattr(funnel, _MARKER, True)
        _counters["errors"] += 1
        return False

    setattr(funnel, "flash_attn_varlen_func", _install_guard(original))
    setattr(funnel, _MARKER, True)
    logger.info(
        "[HCU FA4 VMF GUARD] armed on %s.flash_attn_varlen_func (R36)", TARGET_MODULE
    )
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
    "guard_counters",
]
