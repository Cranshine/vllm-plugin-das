# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the Kimi-K3.1 Pro lm_head to the K3.1 streaming GEMM.

Routing is the head shape itself, not the model config:

* the target text model's ``LogitsProcessor`` is built from the ``text_config``
  (model_type ``kimi_linear``, same as Kimi-K3), while the DSpark draft's is
  built from the top-level config (model_type ``kimi_k31``), so no config-derived
  marker can see both processors;
* both processors share the same unquantized bf16 head weight (the DSpark draft
  reuses the target ``lm_head``), so the per-rank width of the 163840 vocab plus
  ``K == 7168`` is the single routing key.

``LogitsProcessor._apply_head`` is wrapped to answer that shape from
``vllm_hcu.models.kimi_k31_skinny_lm_head`` when the head is exactly
unquantized, has no embedding bias, keeps the model dtype, and the local shape
passes the gate (M <= 128, one of the tuned per-rank widths).  Everything else,
including every non-Kimi head, keeps the native projection.

Nothing else is shared with Step5: the Step5 adapter and its
``skinny_lm_head`` kernel are untouched by this module.

Disable through the process master switch ``VLLM_HCU_USE_CUSTOM_OPS=0``.
"""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.model_executor.layers.logits_processor"
PATCH_ID = "worker.framework_opt.kimik31_skinny_lm_head"
_APPLY_HEAD = (
    "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head"
)
TARGETS = (_APPLY_HEAD,)
_MARKER = "_vllm_hcu_kimik31_skinny_lm_head_applied"

SIGNATURES = (
    (
        _APPLY_HEAD,
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)

    from vllm_hcu.platforms import envs as henvs

    cls = getattr(target, "LogitsProcessor", None)
    original_apply_head = require_callable(cls, "_apply_head", _APPLY_HEAD)

    @functools.wraps(original_apply_head)
    def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
        if (
            henvs.optional_custom_op_enabled()
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
            and embedding_bias is None
            and type(lm_head.quant_method)
            in (target.UnquantizedEmbeddingMethod, target.UnquantizedLinearMethod)
        ):
            from vllm_hcu.models import kimi_k31_skinny_lm_head

            weight = lm_head.weight
            if kimi_k31_skinny_lm_head.usable(hidden_states, weight):
                target.logger.info_once(
                    "Kimi-K3.1 lm_head: using the K3.1 streaming bf16 GEMM "
                    "for M <= %d (routed by the Kimi head shape)",
                    kimi_k31_skinny_lm_head.MAX_M,
                )
                return kimi_k31_skinny_lm_head.linear(hidden_states, weight)
        return original_apply_head(self, lm_head, hidden_states, embedding_bias)

    cls._apply_head = _apply_head
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
