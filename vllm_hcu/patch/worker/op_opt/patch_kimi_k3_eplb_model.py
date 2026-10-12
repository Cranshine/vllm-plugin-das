# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expose Kimi-K3's local routed experts through vLLM's EPLB protocol."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.eplb_model_protocol"
TARGETS = (f"{TARGET_MODULE}.KimiLinearForCausalLM.__init__",)
_MARKER = "_vllm_hcu_kimi_k3_eplb_model_protocol_applied"
_WRAPPER = "_vllm_hcu_kimi_k3_eplb_init_wrapper"


def _init_moe_metadata(self) -> None:
    from vllm.models.kimi_k3.amd.linear import KimiMoE

    layers = [
        layer.mlp.experts
        for layer in self.model.layers[
            self.model.start_layer : self.model.end_layer
        ]
        if isinstance(getattr(layer, "mlp", None), KimiMoE)
    ]
    logical = int(self.config.num_experts or 0)
    counts = {
        (
            int(runner.routed_experts.moe_config.num_experts),
            int(runner.routed_experts.local_num_experts),
        )
        for runner in layers
    }
    if len(counts) > 1:
        raise ValueError("Kimi-K3 local MoE layers disagree on expert counts")
    physical, local = next(iter(counts), (logical, 0))
    if physical < logical or (layers and not 0 < local <= physical):
        raise ValueError("Kimi-K3 has invalid physical/local expert counts")
    self.moe_layers = layers
    self.num_moe_layers = len(layers)
    self.num_expert_groups = int(self.config.num_expert_group or 1)
    self.num_logical_experts = logical
    self.num_routed_experts = logical
    self.num_physical_experts = physical
    self.num_local_physical_experts = local
    self.num_shared_experts = int(self.config.num_shared_experts or 0)
    self.num_redundant_experts = physical - logical
    # The list is populated only after post-load HIPC packing, when EPLB asks
    # for it. Keeping no early snapshot is required for stable ownership.
    self.expert_weights = []


def _set_eplb_state(
    self,
    expert_load_view,
    logical_to_physical_map,
    logical_replica_count,
) -> None:
    self.expert_weights.clear()
    for layer_idx, layer in enumerate(self.moe_layers):
        self.expert_weights.append(list(layer.get_expert_weights()))
        layer.set_eplb_state(
            moe_layer_idx=layer_idx,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )


def _update_physical_experts_metadata(
    self, num_physical_experts: int, num_local_physical_experts: int
) -> None:
    if (num_physical_experts, num_local_physical_experts) != (
        self.num_physical_experts,
        self.num_local_physical_experts,
    ):
        raise ValueError("Kimi-K3 cannot resize packed physical expert storage")
    for layer in self.moe_layers:
        layer.update_expert_map()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        target, "KimiLinearForCausalLM", f"{TARGET_MODULE}.KimiLinearForCausalLM"
    )
    if already_applied(target, _MARKER, ((cls, "__init__", TARGETS[0], _WRAPPER),)):
        return False
    original_init = require_callable(cls, "__init__", TARGETS[0])

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _init_moe_metadata(self)

    setattr(hcu_init, _WRAPPER, True)
    setattr(cls, "__init__", hcu_init)
    cls.set_eplb_state = _set_eplb_state
    cls.update_physical_experts_metadata = _update_physical_experts_metadata
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
