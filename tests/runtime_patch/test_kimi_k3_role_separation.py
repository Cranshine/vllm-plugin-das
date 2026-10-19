# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Audited Kimi-K3/DSpark role-separation runtime patch contracts."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_hcu.patch.worker.framework_opt import (
    patch_dspark_draft_role_sep,
    patch_dspark_eager_ar,
    patch_kimi_k3_eager_ar,
)
from vllm_hcu.patch.worker.framework_opt import _role_separation as rs


@pytest.fixture(autouse=True)
def _reset_role_sep_state(monkeypatch):
    monkeypatch.setattr(rs, "_aux_ar_comm", None)
    monkeypatch.setattr(rs, "_tp_group", None)
    monkeypatch.setattr(rs, "_disabled", False)
    monkeypatch.delenv("VLLM_ROLE_SEP", raising=False)
    monkeypatch.delenv("VLLM_ROLE_SEP_STRICT", raising=False)


def _causal_cls_with_forward(**kwargs):
    class KimiLinearForCausalLM:
        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
            **kwargs,
        ):
            return "hidden"

    return KimiLinearForCausalLM


def test_kimi_eager_ar_scope_wraps_forward_and_is_idempotent(monkeypatch):
    opened = []

    class FakeScope:
        def __enter__(self):
            opened.append("enter")
            return self

        def __exit__(self, *exc):
            opened.append("exit")
            return False

    monkeypatch.setattr(
        patch_kimi_k3_eager_ar, "eager_ar_scope", lambda: FakeScope()
    )
    target = ModuleType(patch_kimi_k3_eager_ar.TARGET_MODULE)
    target.KimiLinearForCausalLM = _causal_cls_with_forward()
    monkeypatch.setitem(
        sys.modules, patch_kimi_k3_eager_ar.TARGET_MODULE, target
    )

    assert patch_kimi_k3_eager_ar.apply_to_module(target)
    instance = target.KimiLinearForCausalLM()
    assert instance.forward(None, None) == "hidden"
    assert opened == ["enter", "exit"]
    wrapper = target.KimiLinearForCausalLM.forward
    assert getattr(wrapper, patch_kimi_k3_eager_ar._WRAPPER)
    assert getattr(target, patch_kimi_k3_eager_ar._MARKER)
    assert patch_kimi_k3_eager_ar.apply_to_module(target) is False
    assert target.KimiLinearForCausalLM.forward is wrapper


def test_dspark_eager_ar_adds_shard_api_and_wraps_forward(monkeypatch):
    opened = []

    class FakeScope:
        def __enter__(self):
            opened.append("enter")
            return self

        def __exit__(self, *exc):
            opened.append("exit")
            return False

    monkeypatch.setattr(
        patch_dspark_eager_ar, "eager_ar_scope", lambda: FakeScope()
    )
    target = ModuleType(patch_dspark_eager_ar.TARGET_MODULE)

    class K3DSparkForCausalLM:
        def forward(self, input_ids, positions, inputs_embeds=None):
            return "hidden"

    target.K3DSparkForCausalLM = K3DSparkForCausalLM
    monkeypatch.setitem(
        sys.modules, patch_dspark_eager_ar.TARGET_MODULE, target
    )

    assert patch_dspark_eager_ar.apply_to_module(target)
    instance = K3DSparkForCausalLM()
    assert instance.forward("ids", "pos") == "hidden"
    assert opened == ["enter", "exit"]
    assert getattr(K3DSparkForCausalLM.forward,
                   patch_dspark_eager_ar._WRAPPER)
    # The new shard API exists and projects through lm_head without a gather.
    instance.lm_head = SimpleNamespace()
    instance.logits_processor = SimpleNamespace(
        _apply_head=lambda lm_head, hidden, bias: ("shard", hidden)
    )
    assert instance.compute_draft_logits_shard("h") == ("shard", "h")
    assert patch_dspark_eager_ar.apply_to_module(target) is False
    with pytest.raises(patch_dspark_eager_ar.PatchCompatibilityError):
        # A second install must refuse because the shard API now exists on a
        # fresh (unpatched) module carrying the same class symbol.
        fresh = ModuleType(patch_dspark_eager_ar.TARGET_MODULE)

        class K3DSparkForCausalLM2:
            def forward(self, input_ids, positions, inputs_embeds=None):
                return "hidden"

            def compute_draft_logits_shard(self, hidden_states):
                raise AssertionError("obsolete migration must not run")

        fresh.K3DSparkForCausalLM = K3DSparkForCausalLM2
        patch_dspark_eager_ar.apply_to_module(fresh)


class _RoleSepFixtures:
    """Fake DSpark speculator chain mirroring the audited target layout."""

    num_speculative_steps = 2
    max_num_tokens = 16
    max_num_reqs = 4
    vocab_size = 64
    dtype = torch.float32
    device = "cpu"

    def __init__(self, vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(draft_vocab_size=None)
        )
        self.query_cudagraph_manager = None
        self.build()

    def init_cudagraph_manager(self, cudagraph_mode):
        self.query_cudagraph_manager = SimpleNamespace(
            run_fullgraph=lambda desc: self.replayed.append(desc)
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.draft_model_stub

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    ):
        raise AssertionError("legacy path must not run under role separation")

    def _run_model(
        self,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
    ):
        return self.head_hidden

    def build(self):
        self.replayed = []
        self.head_hidden = torch.arange(16, dtype=torch.float32)
        self.sample_indices = torch.arange(16, dtype=torch.int64)
        self.draft_model_stub = SimpleNamespace(
            compute_draft_logits_shard=lambda hidden: hidden,
            lm_head=SimpleNamespace(
                num_embeddings_per_partition=32, replicated=False
            ),
        )
        return self


def _make_speculator_cls():
    class DFlashSpeculator(_RoleSepFixtures):
        pass

    class DSparkSpeculator(DFlashSpeculator):
        pass

    return DFlashSpeculator, DSparkSpeculator


def _install_target(monkeypatch, dspark_cls):
    target = ModuleType(patch_dspark_draft_role_sep.TARGET_MODULE)
    target.DSparkSpeculator = dspark_cls
    monkeypatch.setitem(
        sys.modules, patch_dspark_draft_role_sep.TARGET_MODULE, target
    )
    return target


def test_dspark_draft_role_sep_moves_ag_out_of_graph(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )
    dflash_cls, dspark_cls = _make_speculator_cls()
    target = _install_target(monkeypatch, dspark_cls)
    assert patch_dspark_draft_role_sep.apply_to_module(target)
    assert patch_dspark_draft_role_sep.apply_to_module(target) is False

    spec = dspark_cls(SimpleNamespace(), "cpu")
    # __init__ wrapper armed the role-separation state under default env.
    assert spec._ag_out_of_graph is True
    assert spec._logits_shard_width is None

    loaded = spec.load_draft_model(object(), set())
    spec.model = loaded  # the spec-decode worker wires the draft model here
    assert loaded is spec.draft_model_stub
    assert spec._logits_shard_width == 32
    assert spec.draft_logits_shard.shape == (16, 32)

    # init_cudagraph_manager wrapper grows the eager replay tail, DSpark-only.
    spec.init_cudagraph_manager(None)
    mgr = spec.query_cudagraph_manager
    assert getattr(mgr.run_fullgraph,
                   patch_dspark_draft_role_sep._TAIL_WRAPPER)

    # FULL decode step: the captured segment only writes the shard...
    captured_calls = []

    def shard_recorder(hidden):
        captured_calls.append(hidden)
        return torch.full((2, 32), 0.25)

    spec.draft_model_stub.compute_draft_logits_shard = shard_recorder
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )

    spec._generate_draft(
        1,
        4,
        None,
        None,
        None,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )
    assert len(captured_calls) == 1
    assert spec.draft_logits_shard.shape == (16, 32)
    assert torch.all(spec.draft_logits_shard[:2] == 0.25)

    # ...and the eager tail gathers + samples after replay, with the same
    # partition dim semantics as the eager LogitsProcessor path.
    gathered = []

    def fake_all_gather(shard_tensor, dim=-1):
        gathered.append((tuple(shard_tensor.shape), dim))
        return torch.cat([shard_tensor] * 2, dim=dim)

    monkeypatch.setattr(
        "vllm.distributed.tensor_model_parallel_all_gather", fake_all_gather
    )
    markov_inputs = []

    def fake_markov(num_reqs, base_logits):
        markov_inputs.append((num_reqs, tuple(base_logits.shape)))

    spec._markov_sample = fake_markov
    spec.replayed.append("desc")
    # Invoke the tail the way the wrapped manager would: desc carries tokens
    # only, so num_reqs falls back to min(num_tokens, max_num_reqs) = 3.
    desc = SimpleNamespace(num_reqs=None, num_tokens=3)
    mgr.run_fullgraph(desc)
    assert spec.replayed[0] == "desc"
    assert spec.replayed[1] is desc
    assert gathered == [((6, 32), -1)]
    assert markov_inputs == [(3, (3, 2, 64))]


def test_dspark_draft_role_sep_rejects_new_api_and_bad_manager(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )
    dflash_cls, dspark_cls = _make_speculator_cls()
    # The DSpark class already owns init_cudagraph_manager: the audited
    # delegation expects the DFlashSpeculator implementation.
    dspark_cls.init_cudagraph_manager = dflash_cls.init_cudagraph_manager
    target = _install_target(monkeypatch, dspark_cls)
    with pytest.raises(patch_dspark_draft_role_sep.PatchCompatibilityError):
        patch_dspark_draft_role_sep.apply_to_module(target)


def test_gather_guard_uses_strict_env(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setenv("VLLM_ROLE_SEP_STRICT", "1")
    with pytest.raises(RuntimeError, match="captured"):
        patch_dspark_draft_role_sep._guard_captured_draft_all_gather("t")
    monkeypatch.setenv("VLLM_ROLE_SEP_STRICT", "0")
    patch_dspark_draft_role_sep._guard_captured_draft_all_gather("t")


def test_scope_is_noop_when_disabled_or_capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    built = []

    def _fail_build():
        built.append(1)
        raise AssertionError("aux comm must never be built during capture")

    monkeypatch.setattr(rs, "_get_aux_ar_comm", _fail_build)
    with rs.eager_ar_scope():
        pass
    assert built == []

    monkeypatch.setenv("VLLM_ROLE_SEP", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: False)
    with rs.eager_ar_scope():
        pass
    assert built == []


def test_scope_swaps_only_tp_all_reduce_and_restores(monkeypatch):
    class ShimComm:
        available = True
        disabled = False

        def all_reduce(self, tensor):
            return tensor * 3

    class FakeTP:
        cpu_group = object()

        def __init__(self):
            self.calls = []

        def _all_reduce_out_place(self, tensor):
            self.calls.append("original")
            return tensor

    fake_tp = FakeTP()
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator",
        lambda group, device: ShimComm(),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.GroupCoordinator", SimpleNamespace
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    original = FakeTP._all_reduce_out_place
    with rs.eager_ar_scope():
        swapped = fake_tp.__dict__[rs._SWAP_ATTR]
        assert swapped(torch.tensor([1.0])).item() == 3.0
    # Restored: the instance-level swap is gone and the class method resolves.
    assert rs._SWAP_ATTR not in fake_tp.__dict__
    assert FakeTP.__dict__["_all_reduce_out_place"] is original
    fake_tp._all_reduce_out_place(torch.tensor([1.0]))
    assert fake_tp.calls == ["original"]
    # Second scope reuses the cached auxiliary communicator.
    with rs.eager_ar_scope():
        assert fake_tp.__dict__[rs._SWAP_ATTR] is not original


def test_strict_mode_flags_failed_aux_comm(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )

    def failing_comm(group, device):
        raise RuntimeError("pynccl init exploded")

    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator",
        failing_comm,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setenv("VLLM_ROLE_SEP_STRICT", "1")
    with pytest.raises(RuntimeError, match="role separation"):
        with rs.eager_ar_scope():
            pass
