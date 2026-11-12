# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU regressions for the shape-routed Kimi lm_head GEMM."""

from types import ModuleType, SimpleNamespace

import pytest
import torch


def _install(monkeypatch):
    """Install the adapter on a stub logits_processor module; return (target, calls)."""
    from vllm_hcu.patch.worker.framework_opt import (
        patch_kimik31_skinny_lm_head as adapter,
    )

    calls = []

    class Unquantized:
        pass

    class Processor:
        def __init__(
            self,
            vocab_size,
            org_vocab_size=None,
            scale=1.0,
            logits_as_input=False,
            soft_cap=None,
        ):
            self.head_dtype = None
            self.vocab_size = vocab_size

        def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
            calls.append("native")
            return "native"

    target = ModuleType(adapter.TARGET_MODULE)
    target.LogitsProcessor = Processor
    target.UnquantizedEmbeddingMethod = Unquantized
    target.UnquantizedLinearMethod = Unquantized
    target.logger = SimpleNamespace(info_once=lambda *args: None)
    assert adapter.apply_to_module(target) is True
    return adapter, target, Processor, Unquantized, calls


def _head(Unquantized, n=32768, k=7168):
    return SimpleNamespace(
        quant_method=Unquantized(), weight=torch.zeros(n, k)
    )


def test_kimi_head_shape_uses_k31_kernel(monkeypatch):
    """Routing is the head shape: both the target text model's processor (built
    from the kimi_linear text_config) and the DSpark draft's (built from the
    top-level kimi_k31 config) reach the kernel through the same weight."""
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    adapter, target, Processor, Unquantized, calls = _install(monkeypatch)
    probed = []
    monkeypatch.setattr(
        k31, "usable", lambda x, w: probed.append((tuple(x.shape), tuple(w.shape))) or True
    )
    monkeypatch.setattr(k31, "linear", lambda x, w: "k31")

    head = _head(Unquantized)
    hidden = torch.zeros(48, 7168, dtype=torch.bfloat16)
    # M=96 is the target verify call (bs*(spec_len+1)); M=80 the draft propose
    # call (bs*spec_len): both route through the same gate.
    assert Processor(163840)._apply_head(head, hidden, None) == "k31"
    hidden80 = torch.zeros(80, 7168, dtype=torch.bfloat16)
    assert Processor(163840)._apply_head(head, hidden80, None) == "k31"
    assert calls == []
    assert probed[0] == ((48, 7168), (32768, 7168))
    assert probed[1] == ((80, 7168), (32768, 7168))


@pytest.mark.parametrize(
    "case",
    ["embedding_bias", "quantized_head", "fp32_head", "master_off"],
)
def test_kimi_head_falls_back(monkeypatch, case):
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    adapter, target, Processor, Unquantized, calls = _install(monkeypatch)
    probed = []
    monkeypatch.setattr(k31, "usable", lambda x, w: probed.append(1) or True)
    monkeypatch.setattr(k31, "linear", lambda x, w: "k31")

    class Quantized(Unquantized):
        pass

    head = _head(Unquantized)
    bias = None
    hidden = torch.zeros(48, 7168, dtype=torch.bfloat16)
    processor = Processor(163840)
    if case == "embedding_bias":
        bias = torch.zeros(32768)
    elif case == "quantized_head":
        head = _head(Quantized)
    elif case == "fp32_head":
        processor.head_dtype = torch.float32
    elif case == "master_off":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")

    assert processor._apply_head(head, hidden, bias) == "native"
    assert calls == ["native"]
    assert probed == []


def test_non_kimi_shape_never_probes_the_k31_kernel(monkeypatch):
    """A different K or a non-Kimi width keeps the native projection: the shape
    is the routing key, so the config model_type does not matter anymore."""
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    adapter, target, Processor, Unquantized, calls = _install(monkeypatch)
    probed = []
    # usable() is the single gate; a foreign K (4096) never matches the Kimi
    # head shape, so the wrapper answers native without calling linear().
    monkeypatch.setattr(
        k31, "usable", lambda x, w: probed.append((tuple(x.shape), tuple(w.shape))) or False
    )
    monkeypatch.setattr(k31, "linear", lambda x, w: "k31")

    hidden = torch.zeros(48, 4096, dtype=torch.bfloat16)
    head = _head(Unquantized, n=32768, k=4096)
    assert Processor(163840)._apply_head(head, hidden, None) == "native"
    assert calls == ["native"]
    assert probed == [((48, 4096), (32768, 4096))]


def test_apply_is_idempotent(monkeypatch):
    adapter, target, _, _, _ = _install(monkeypatch)
    assert adapter.apply_to_module(target) is False


def test_adapter_is_registered_in_the_worker_dispatcher():
    from vllm_hcu.patch.worker import _patch_features, worker_callback_names

    from vllm_hcu.patch.worker.framework_opt import (
        patch_kimik31_skinny_lm_head as adapter,
    )

    callbacks = dict(worker_callback_names())
    assert callbacks[adapter.PATCH_ID] == adapter.TARGET_MODULE
    assert _patch_features()[adapter.PATCH_ID] == "always"


def test_kernel_plan_per_head_width():
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    # Decode at bs<=8 (M<=64): one 64-row tile, K3.1-Pro INT4 head is 163840
    # wide so TP1/TP2/TP4/TP8 give these shards.  Every decode M routes to the
    # kernel now (uniform switch), including M below the old min_m guards.
    assert k31._plan(81920, 48) == (64, 128, 2, 4)  # TP2, the priority target
    assert k31._plan(40960, 56) == (64, 128, 4, 4)  # TP4
    assert k31._plan(20480, 48) == (64, 128, 2, 4)  # TP8
    assert k31._plan(163840, 48) == (64, 128, 4, 4)  # TP1
    assert k31._plan(20480, 1) == (64, 128, 2, 4)  # TP8, uniform switch
    # bs*(spec_len+1) > 64 (--max-num-seqs 16 at 5 speculative tokens): 128-row
    # tile, BK per width; every Kimi width has a wide plan now.
    assert k31._plan(81920, 96) == (128, 128, 4, 8)
    assert k31._plan(40960, 96) == (128, 64, 4, 8)
    assert k31._plan(40960, 128) == (128, 64, 4, 8)
    assert k31._plan(20480, 96) == (128, 64, 2, 8)
    assert k31._plan(163840, 96) == (128, 64, 2, 8)
    assert k31._plan(40960, 129) is None
    # A width that is not a Kimi vocab shard, or does not divide the tile.
    assert k31._plan(65536, 48) is None
    assert k31._plan(8192, 48) is None
    assert k31._plan(81920 + 32, 48) is None
