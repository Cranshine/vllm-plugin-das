# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route eager Kimi-K3 target-model TP all_reduces to the auxiliary comm.

The Kimi-K3 target decode graph captures its TP all_reduces on the TP
communicator. Eager-phase forwards of the SAME model (prefill, profile,
ducible warmup) all_reduce on that same communicator by default, forming
the captured-x-eager same-communicator pair that deadlocks RCCL on DCUs
(16/16 ranks spinning in ``runRing<hip_bfloat16, FuncSum<...>,
ProtoSimple<2, 2, 4, 0, 0>, 4>``). Wrapping ``forward`` with
:eattr:`vllm_hcu.patch.worker.framework_opt._role_separation.\
eager_ar_scope` moves every eager all_reduce of this model onto the
dedicated auxiliary communicator; captures and torch.compile traces
inside the scope are no-ops (the scope defers), so captured/graphed
behavior is unchanged and the auxiliary comm is never built during a
capture.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_class,
    require_callable,
    require_exact_signature,
    require_unpatched,
)
from ._role_separation import eager_ar_scope

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.framework_opt.kimi_k3.eager_ar_scope"
TARGETS = (f"{TARGET_MODULE}.KimiLinearForCausalLM.forward",)
_MARKER = "_vllm_hcu_kimi_eager_ar_applied"
_WRAPPER = "_vllm_hcu_kimi_eager_ar_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    lin = load_exact_module(TARGET_MODULE, module)
    causal_cls = require_class(
        lin, "KimiLinearForCausalLM", TARGETS[0].rsplit(".", 1)[0]
    )
    wrapped = ((causal_cls, "forward", TARGETS[0], _WRAPPER),)
    if already_applied(lin, _MARKER, wrapped):
        return False
    original = require_unpatched(causal_cls, "forward", TARGETS[0], _WRAPPER)
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "self",
            "input_ids",
            "positions",
            "intermediate_tensors",
            "inputs_embeds",
        ),
        defaults={"intermediate_tensors": None, "inputs_embeds": None},
        var_keyword="kwargs",
    )

    @functools.wraps(original)
    def hcu_eager_ar_forward(self, *args, **kwargs):
        # Eager ARs (prefill/profile/warmup) ride the auxiliary comm;
        # captured/graphed ARs stay on the original TP communicator.
        with eager_ar_scope():
            return original(self, *args, **kwargs)

    setattr(hcu_eager_ar_forward, _WRAPPER, True)
    causal_cls.forward = hcu_eager_ar_forward
    setattr(lin, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
