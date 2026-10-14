# SPDX-License-Identifier: Apache-2.0
"""Keep legacy Kimi chunk preparation on scheduler-owned CPU metadata."""
from __future__ import annotations

import os
from types import ModuleType

from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.models.kimi_k3.amd.kda_metadata"
PATCH_ID = "worker.op_opt.kimi_k3.cpu_chunk_metadata"
TARGETS = (f"{TARGET_MODULE}.KimiK3ROCmKDAMetadataBuilder._build_chunk_metadata",)
_MARKER = "_vllm_hcu_kimi_cpu_chunk_metadata_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    builder = require_class(
        target, "KimiK3ROCmKDAMetadataBuilder", TARGETS[0]
    )
    original = builder._build_chunk_metadata

    def build(self, prefill_query_start_loc, prefill_query_start_loc_cpu, device):
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1").lower() in (
            "0", "false"
        ):
            # Upstream's CPU path preserves the FLA fallback payload as well
            # as FlashKDA compatibility, without the three device metadata
            # launches introduced by the Kimi subclass in 028.
            return target.GDNAttentionMetadataBuilder._build_chunk_metadata(
                self, prefill_query_start_loc, prefill_query_start_loc_cpu, device
            )
        return original(
            self, prefill_query_start_loc, prefill_query_start_loc_cpu, device
        )

    builder._build_chunk_metadata = build
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
