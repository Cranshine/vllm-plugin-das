# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DSpark draft model: eager-AR scope plus the vocab-shard lm_head API.

Two audited changes on ``K3DSparkForCausalLM``:

  - ``forward`` runs inside the role-separation scope so the draft
    model's eager TP all_reduces (prefill/context phases) never share the
    captured-role TP communicator (RCCL captured-x-eager deadlock).
  - ``compute_draft_logits_shard`` exposes the lm_head projection WITHOUT
    the logits all_gather, letting the draft speculator stop the captured
    draft graph at a sharded logits write and run the all_gather eagerly
    after each replay (see ``patch_dspark_draft_role_sep``). Pairing the
    shard with an explicit ``tensor_model_parallel_all_gather`` reproduces
    ``compute_logits`` bit-for-bit (same kernels, same gather semantics).
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_class,
    require_exact_signature,
    require_unpatched,
)
from ._role_separation import eager_ar_scope

TARGET_MODULE = "vllm.models.kimi_k3.nvidia.dspark_mla"
PATCH_ID = "worker.framework_opt.dspark.eager_ar_scope"
TARGETS = (
    f"{TARGET_MODULE}.K3DSparkForCausalLM.forward",
    f"{TARGET_MODULE}.K3DSparkForCausalLM.compute_draft_logits_shard",
)
_MARKER = "_vllm_hcu_dspark_eager_ar_applied"
_WRAPPER = "_vllm_hcu_dspark_eager_ar_wrapper"


def _compute_draft_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Vocab-shard projection only: the lm_head GEMM without the logits
    all_gather. The speculator aggregates with an explicit
    tensor_model_parallel_all_gather outside the captured graph."""
    assert self.lm_head is not None
    return self.logits_processor._apply_head(self.lm_head, hidden_states, None)


def apply_to_module(module: ModuleType) -> bool:
    mla = load_exact_module(TARGET_MODULE, module)
    causal_cls = require_class(
        mla, "K3DSparkForCausalLM", TARGETS[0].rsplit(".", 1)[0]
    )
    wrapped = (
        (causal_cls, "forward", TARGETS[0], _WRAPPER),
        (causal_cls, "compute_draft_logits_shard", TARGETS[1], _WRAPPER),
    )
    if already_applied(mla, _MARKER, wrapped):
        return False
    original = require_unpatched(causal_cls, "forward", TARGETS[0], _WRAPPER)
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "input_ids", "positions", "inputs_embeds"),
        defaults={"inputs_embeds": None},
    )
    if hasattr(causal_cls, "compute_draft_logits_shard"):
        raise PatchCompatibilityError(
            f"required-new API {TARGETS[1]} already exists on the target "
            "vLLM module; the audited migration is obsolete"
        )

    @functools.wraps(original)
    def hcu_eager_ar_forward(self, input_ids, positions, inputs_embeds=None):
        # Eager draft all_reduces (context/prefill phases; cg_mode == NONE,
        # e.g. profile/dummy) route their TP all_reduces to the dedicated
        # eager comm. The scope is a no-op during draft-graph capture, so
        # captured ARs stay recorded on - and replayed from - the
        # captured-role communicator.
        with eager_ar_scope():
            return original(self, input_ids, positions, inputs_embeds)

    setattr(hcu_eager_ar_forward, _WRAPPER, True)
    shard_fn = _compute_draft_logits_shard
    setattr(shard_fn, _WRAPPER, True)
    causal_cls.forward = hcu_eager_ar_forward
    causal_cls.compute_draft_logits_shard = shard_fn
    setattr(mla, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
