# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep the DSpark draft-graph all_gather on the captured-role TP comm.

v2 (whole-graph capture). The DSpark draft cudagraph captures the ENTIRE
draft step -- backbone forward, draft-logits all_gather, Markov sampling
-- as one replay unit (the audited upstream design; the pre-v2 adapter
split it at the sharded lm_head write and ran the all_gather + Markov
tail eagerly after every replay). The split kept every communicator in a
single role but broke the replay into graph + host-driven eager tail,
which showed up as 3k+ GPU idle gaps per trace and ~70% idle time in the
steady decode region.

The hazard the split was fixing is addressed at the communicator level
instead: while the draft graph is being captured, the TP coordinator's
``_all_gather_out_place`` is swapped (instance level, the same technique
as ``eager_ar_scope``) to a shim that enqueues the all_gather on the TP
group's pynccl communicator -- the communicator that already carries only
captured-graph collectives (the target and draft graphs' all_reduces).
Replays re-execute the recorded ncclAllGather from that communicator, so
the shared c10d communicator keeps only the per-step eager all_gathers
and every communicator hosts a single role:

  * TP pynccl comm: captured ARs (target + draft graphs) + captured AG
    (draft graph);
  * auxiliary comm: eager ARs (``_role_separation.eager_ar_scope``);
  * c10d device-group comm: eager AGs (target logits + eager draft tails).

Capture is detected with ``torch.cuda.is_current_stream_capturing()``
and NOT from ``cudagraph_runtime_mode`` alone: the capture driver calls
the forward function with ``CUDAGraphMode.NONE`` inside the
``torch.cuda.graph`` block.

Rollback: set ``VLLM_ROLE_SEP=0`` to disable everything (whole-graph
capture still holds; the captured AG then records on the shared c10d
communicator -- the legacy captured-x-eager topology).
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch
from vllm.config.compilation import CUDAGraphMode

from ._common import (
    already_applied,
    load_exact_module,
    require_class,
    require_callable,
    require_exact_signature,
    require_unpatched,
)
from . import _role_separation as rs

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.framework_opt.dspark.draft_role_separation"
TARGETS = (
    f"{TARGET_MODULE}.DSparkSpeculator.__init__",
    f"{TARGET_MODULE}.DSparkSpeculator.load_draft_model",
    f"{TARGET_MODULE}.DSparkSpeculator._generate_draft",
)
_MARKER = "_vllm_hcu_dspark_draft_role_sep_applied"
_WRAPPER = "_vllm_hcu_dspark_draft_role_sep_wrapper"


def _hcu_role_sep_init(self, vllm_config, device):
    _hcu_role_sep_init._hcu_original(self, vllm_config, device)
    # Role separation state: whole-graph capture with the draft-logits AG
    # recorded on the captured-role TP pynccl communicator (see
    # _generate_draft + _role_separation.captured_ag_scope).
    from vllm.distributed import get_tensor_model_parallel_world_size

    self._draft_ag_pynccl = bool(
        rs.role_separation_enabled()
        and get_tensor_model_parallel_world_size() > 1
    )
    from vllm.logger import init_logger

    # The "vllm." prefix is required for serve visibility: vLLM's dictConfig
    # attaches its handler to the vllm logger only (propagate=False), so a
    # plain init_logger(__name__) record is silently dropped and the armed
    # state is unknowable from server logs (patch_kimi_k3_sp_moe.py:334).
    init_logger("vllm.hcu.rolesep").info(
        "[role-sep] DSpark draft AG-comm=pynccl:%s (role_sep=%s tp=%d); "
        "whole-graph capture, no eager tail",
        self._draft_ag_pynccl,
        rs.role_separation_enabled(),
        get_tensor_model_parallel_world_size(),
    )


def _hcu_role_sep_load_draft_model(self, target_model, target_attn_layer_names):
    model = _hcu_role_sep_load_draft_model._hcu_original(
        self, target_model, target_attn_layer_names
    )
    if self._draft_ag_pynccl and rs.tp_pynccl_comm() is None:
        # No enabled TP pynccl communicator: the captured draft AG would
        # fall back to the shared c10d communicator -- the captured-x-eager
        # same-communicator pair that deadlocks RCCL. Degrade loudly; the
        # scope repeats the check at capture time (strict mode: fatal).
        rs.raise_or_warn(
            "DSpark: no enabled TP pynccl communicator for the captured "
            "draft-logits all_gather; it will record on the shared c10d "
            "communicator together with the per-step eager all_gathers "
            "(set VLLM_ROLE_SEP_STRICT=1 to make this fatal)."
        )
        self._draft_ag_pynccl = False
    return model


def _hcu_role_sep_generate_draft(
    self,
    num_reqs: int,
    num_tokens_padded: int,
    attn_metadata,
    slot_mappings,
    num_tokens_across_dp,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
):
    # Thin wrapper around the audited original: activate the captured-AG
    # scope only while the draft graph records (the capture driver calls
    # this method with CUDAGraphMode.NONE inside the torch.cuda.graph
    # block, so is_current_stream_capturing() is the only reliable
    # capture signal). Eager calls (context phase / warmup) keep the
    # stock c10d all_gather path -- every communicator keeps a single
    # role.
    original = _hcu_role_sep_generate_draft._hcu_original
    if self._draft_ag_pynccl and torch.cuda.is_current_stream_capturing():
        with rs.captured_ag_scope():
            return original(
                self,
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
    return original(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
    )


def _install_wrapped(speculator_cls, name, original, replacement) -> None:
    replacement._hcu_original = original

    @functools.wraps(original)
    def hcu_wrapped(self, *args, **kwargs):
        return replacement(self, *args, **kwargs)

    setattr(hcu_wrapped, _WRAPPER, True)
    setattr(speculator_cls, name, hcu_wrapped)


def apply_to_module(module: ModuleType) -> bool:
    spec = load_exact_module(TARGET_MODULE, module)
    speculator_cls = require_class(
        spec, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator"
    )
    wrapped = (
        (speculator_cls, "__init__", TARGETS[0], _WRAPPER),
        (speculator_cls, "load_draft_model", TARGETS[1], _WRAPPER),
        (speculator_cls, "_generate_draft", TARGETS[2], _WRAPPER),
    )
    if already_applied(spec, _MARKER, wrapped):
        return False

    init_original = require_callable(speculator_cls, "__init__", TARGETS[0])
    require_exact_signature(
        init_original, TARGETS[0], positional=("self", "vllm_config", "device")
    )
    load_original = require_unpatched(
        speculator_cls, "load_draft_model", TARGETS[1], _WRAPPER
    )
    require_exact_signature(
        load_original,
        TARGETS[1],
        positional=("self", "target_model", "target_attn_layer_names"),
    )
    draft_original = require_unpatched(
        speculator_cls, "_generate_draft", TARGETS[2], _WRAPPER
    )
    require_exact_signature(
        draft_original,
        TARGETS[2],
        positional=(
            "self",
            "num_reqs",
            "num_tokens_padded",
            "attn_metadata",
            "slot_mappings",
            "num_tokens_across_dp",
            "cudagraph_runtime_mode",
        ),
        defaults={"cudagraph_runtime_mode": CUDAGraphMode.NONE},
    )

    _install_wrapped(speculator_cls, "__init__", init_original, _hcu_role_sep_init)
    _install_wrapped(
        speculator_cls,
        "load_draft_model",
        load_original,
        _hcu_role_sep_load_draft_model,
    )
    _install_wrapped(
        speculator_cls,
        "_generate_draft",
        draft_original,
        _hcu_role_sep_generate_draft,
    )
    setattr(spec, _MARKER, True)
    from vllm.logger import init_logger

    # "vllm." prefix: serve-visible (see patch_kimi_k3_sp_moe.py:334).
    init_logger("vllm.hcu.rolesep").info(
        "[role-sep] armed: DSpark whole-graph capture adapter "
        "(draft AG -> captured-role pynccl comm)"
    )
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
