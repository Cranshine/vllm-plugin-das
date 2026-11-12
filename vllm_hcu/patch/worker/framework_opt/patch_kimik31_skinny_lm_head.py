# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the Kimi-K3.1 Pro lm_head to the K3.1 streaming GEMM.

Scope is the model type plus the local shape, in that order:

* ``LogitsProcessor.__init__`` is wrapped once to record whether the processor
  being built belongs to a Kimi-K3.1 Pro run (``hf_config.model_type ==
  "kimi_k31"``), checked against the target model config and, for a run with a
  DSpark draft, the speculative draft config too.
* ``LogitsProcessor._apply_head`` is wrapped to answer that processor from
  ``vllm_hcu.models.kimi_k31_skinny_lm_head`` when the head is exactly
  unquantized, has no embedding bias, keeps the model dtype, and the local shape
  passes the gate (M <= 64, N >= 16384, contiguous bf16, K % BK == 0).

Nothing else is shared with Step5: the Step5 adapter and its
``skinny_lm_head`` kernel are untouched by this module, and non-K3.1 models keep
their original ``_apply_head`` behavior.

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
_INIT = "vllm.model_executor.layers.logits_processor.LogitsProcessor.__init__"
_APPLY_HEAD = (
    "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head"
)
TARGETS = (_INIT, _APPLY_HEAD)
_MARKER = "_vllm_hcu_kimik31_skinny_lm_head_applied"
_PROCESSOR_ATTR = "_vllm_hcu_kimik31_lm_head"

SIGNATURES = (
    (
        _INIT,
        ("self", "vocab_size", "org_vocab_size", "scale", "logits_as_input", "soft_cap"),
    ),
    (
        _APPLY_HEAD,
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    ),
)
REVIEWED_SOURCES = ()


def _is_kimik31_run() -> bool:
    """True when the processor being constructed belongs to a K3.1 Pro run."""
    try:
        from vllm.config import get_current_vllm_config
        from vllm_hcu.models.kimi_k31 import kimi_k31_is_k31
    except Exception:
        return False
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return False
    speculative_config = getattr(vllm_config, "speculative_config", None)
    candidates = (
        getattr(vllm_config, "model_config", None),
        getattr(speculative_config, "draft_model_config", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if kimi_k31_is_k31(candidate, getattr(candidate, "hf_config", None)):
                return True
        except Exception:
            continue
    return False


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)

    from vllm_hcu.platforms import envs as henvs

    cls = getattr(target, "LogitsProcessor", None)
    original_init = require_callable(cls, "__init__", _INIT)
    original_apply_head = require_callable(cls, "_apply_head", _APPLY_HEAD)

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        setattr(self, _PROCESSOR_ATTR, _is_kimik31_run())

    @functools.wraps(original_apply_head)
    def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
        if (
            getattr(self, _PROCESSOR_ATTR, False)
            and henvs.optional_custom_op_enabled()
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
                    "for M <= %d, N >= %d",
                    kimi_k31_skinny_lm_head.MAX_M,
                    kimi_k31_skinny_lm_head.MIN_N,
                )
                return kimi_k31_skinny_lm_head.linear(hidden_states, weight)
        return original_apply_head(self, lm_head, hidden_states, embedding_bias)

    cls.__init__ = __init__
    cls._apply_head = _apply_head
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
