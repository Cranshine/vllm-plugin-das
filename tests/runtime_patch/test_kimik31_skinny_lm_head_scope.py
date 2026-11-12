# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU regressions for the Kimi-K3.1-only lm_head routing."""

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


def _model_config(model_type, draft_model_type=None):
    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    draft = None
    if draft_model_type is not None:
        draft = SimpleNamespace(
            hf_config=SimpleNamespace(model_type=draft_model_type)
        )
    return SimpleNamespace(
        model_config=model_config, speculative_config=SimpleNamespace(
            draft_model_config=draft
        )
    )


def test_processor_marked_only_for_kimik31(monkeypatch):
    import vllm.config

    adapter, target, Processor, _, _ = _install(monkeypatch)

    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _model_config("kimi_k31")
    )
    assert getattr(Processor(131072), adapter._PROCESSOR_ATTR) is True

    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _model_config("kimi_k3")
    )
    assert getattr(Processor(131072), adapter._PROCESSOR_ATTR) is False

    # A K3.1 run may reach its DSpark draft processor through the draft config.
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: _model_config("kimi_k3", draft_model_type="kimi_k31"),
    )
    assert getattr(Processor(131072), adapter._PROCESSOR_ATTR) is True


def test_kimik31_head_uses_k31_kernel(monkeypatch):
    import vllm.config
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    adapter, target, Processor, Unquantized, calls = _install(monkeypatch)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _model_config("kimi_k31")
    )
    probed = []
    monkeypatch.setattr(
        k31, "usable", lambda x, w: probed.append((tuple(x.shape), tuple(w.shape))) or True
    )
    monkeypatch.setattr(k31, "linear", lambda x, w: "k31")

    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(32768, 7168))
    hidden = torch.zeros(48, 7168, dtype=torch.bfloat16)
    assert Processor(131072)._apply_head(head, hidden, None) == "k31"
    assert calls == []
    assert probed == [((48, 7168), (32768, 7168))]


def test_non_kimik31_never_probes_the_k31_kernel(monkeypatch):
    import vllm.config
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    adapter, target, Processor, Unquantized, calls = _install(monkeypatch)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _model_config("kimi_k3")
    )
    probed = []
    monkeypatch.setattr(k31, "usable", lambda x, w: probed.append(1) or True)
    monkeypatch.setattr(k31, "linear", lambda x, w: "k31")

    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(32768, 7168))
    hidden = torch.zeros(48, 7168, dtype=torch.bfloat16)
    assert Processor(131072)._apply_head(head, hidden, None) == "native"
    assert calls == ["native"]
    assert probed == []


@pytest.mark.parametrize(
    "case",
    ["embedding_bias", "quantized_head", "fp32_head", "master_off"],
)
def test_kimik31_falls_back(monkeypatch, case):
    import vllm.config
    from vllm_hcu.models import kimi_k31_skinny_lm_head as k31

    adapter, target, Processor, Unquantized, calls = _install(monkeypatch)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _model_config("kimi_k31")
    )
    probed = []
    monkeypatch.setattr(k31, "usable", lambda x, w: probed.append(1) or True)
    monkeypatch.setattr(k31, "linear", lambda x, w: "k31")

    class Quantized(Unquantized):
        pass

    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(32768, 7168))
    bias = None
    hidden = torch.zeros(48, 7168, dtype=torch.bfloat16)
    processor = Processor(131072)
    if case == "embedding_bias":
        bias = torch.zeros(32768)
    elif case == "quantized_head":
        head = SimpleNamespace(quant_method=Quantized(), weight=torch.zeros(32768, 7168))
    elif case == "fp32_head":
        processor.head_dtype = torch.float32
    elif case == "master_off":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")

    assert processor._apply_head(head, hidden, bias) == "native"
    assert calls == ["native"]
    assert probed == []


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
    # wide so TP1/TP2/TP4/TP8 give these shards.
    assert k31._plan(81920, 48) == (64, 128, 2, 4)  # TP2, the priority target
    assert k31._plan(40960, 56) == (64, 128, 4, 4)  # TP4
    assert k31._plan(20480, 48) == (64, 128, 2, 4)  # TP8
    assert k31._plan(163840, 48) == (64, 128, 4, 4)  # TP1
    # Narrow shards keep the native projection for small batches.
    assert k31._plan(20480, 8) is None
    assert k31._plan(163840, 8) is None
    # bs*(spec_len+1) > 64 (--max-num-seqs 16 at 5 speculative tokens): 128-row
    # tile, BK per width; TP8 has no wide plan because native is faster there.
    assert k31._plan(81920, 96) == (128, 128, 4, 8)
    assert k31._plan(40960, 96) == (128, 64, 4, 8)
    assert k31._plan(40960, 128) == (128, 64, 4, 8)
    assert k31._plan(20480, 96) is None
    assert k31._plan(163840, 96) is None
    assert k31._plan(40960, 129) is None
    # Other widths keep the conservative (default) small-M plan.
    assert k31._plan(65536, 48) == (64, 128, 2, 4)
    # Below the tuned width, or a width that does not divide the tile: stock.
    assert k31._plan(8192, 48) is None
    assert k31._plan(81920 + 32, 48) is None
