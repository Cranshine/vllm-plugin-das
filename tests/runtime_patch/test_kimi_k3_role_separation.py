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
        self.gathered_logits = None
        self.build()

    def init_cudagraph_manager(self, cudagraph_mode):
        # The audited original: a plain replay entry with NO grown tail.
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
        # Audited original: the whole draft step in one body. The logits
        # all_gather goes through the real chain
        # (tensor_model_parallel_all_gather -> GroupCoordinator.all_gather
        # -> _all_gather_out_place) so the test observes which
        # communicator the captured AG lands on.
        from vllm.distributed import tensor_model_parallel_all_gather

        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        shard = head_hidden[: num_reqs * self.num_speculative_steps]
        self.gathered_logits = tensor_model_parallel_all_gather(shard, dim=-1)

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
        self.head_hidden = torch.arange(16, dtype=torch.float32).view(16, 1)
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


class _FakePynccl:
    """Stand-in for the TP group's captured-role pynccl communicator."""

    disabled = False

    def __init__(self, world_size):
        self.world_size = world_size
        self.ag_calls = []

    def all_gather(self, output, inp):
        self.ag_calls.append((tuple(inp.shape), tuple(output.shape)))
        # ncclAllGather concat semantics: rank shards stacked along dim 0.
        output.copy_(torch.cat([inp] * self.world_size, dim=0))


class _FakeTPGroup:
    """Stand-in for GroupCoordinator with a swappable AG entry point."""

    def __init__(self, world_size=2, with_pynccl=True):
        self.world_size = world_size
        self.calls = []
        if with_pynccl:
            self.pynccl = _FakePynccl(world_size)
            self.device_communicator = SimpleNamespace(pynccl_comm=self.pynccl)
        else:
            self.device_communicator = SimpleNamespace(pynccl_comm=None)

    def _all_gather_out_place(self, input_, dim):
        # Original path: c10d all_gather_into_tensor equivalent.
        self.calls.append((tuple(input_.shape), dim))
        return torch.cat([input_] * self.world_size, dim=dim)


def test_captured_ag_scope_routes_ag_to_pynccl_and_restores(monkeypatch):
    fake_tp = _FakeTPGroup(2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    original = _FakeTPGroup._all_gather_out_place
    with rs.captured_ag_scope():
        shim = fake_tp.__dict__[rs._AG_SWAP_ATTR]
        out = shim(torch.tensor([[1.0, 2.0]]), -1)
        # dim=-1 concat layout, bit-identical to the original path.
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 1.0, 2.0]]))
        out0 = shim(torch.tensor([[1.0, 2.0]]), 0)
        assert torch.equal(out0, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
        assert fake_tp.pynccl.ag_calls == [
            ((1, 2), (2, 2)),
            ((1, 2), (2, 2)),
        ]
        assert fake_tp.calls == []
    # Restored: the instance-level swap is gone, the class method resolves,
    # and later AGs ride the original (c10d) path again.
    assert rs._AG_SWAP_ATTR not in fake_tp.__dict__
    assert _FakeTPGroup.__dict__["_all_gather_out_place"] is original
    fake_tp._all_gather_out_place(torch.tensor([[1.0]]), -1)
    assert fake_tp.calls == [((1, 1), -1)]
    assert fake_tp.pynccl.ag_calls[-1] == ((1, 2), (2, 2))  # unchanged


def test_captured_ag_scope_noop_when_not_capturing_or_disabled(monkeypatch):
    fake_tp = _FakeTPGroup(2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with rs.captured_ag_scope():
        assert rs._AG_SWAP_ATTR not in fake_tp.__dict__
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setenv("VLLM_ROLE_SEP", "0")
    with rs.captured_ag_scope():
        assert rs._AG_SWAP_ATTR not in fake_tp.__dict__
    assert fake_tp.pynccl.ag_calls == []


def test_captured_ag_scope_warns_without_pynccl(monkeypatch):
    fake_tp = _FakeTPGroup(2, with_pynccl=False)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with rs.captured_ag_scope():
        # Degraded: no swap, the captured AG would fall back to the shared
        # c10d communicator.
        assert rs._AG_SWAP_ATTR not in fake_tp.__dict__
    # Strict mode makes the same degradation fatal at capture time.
    monkeypatch.setenv("VLLM_ROLE_SEP_STRICT", "1")
    with pytest.raises(RuntimeError, match="pynccl"):
        with rs.captured_ag_scope():
            pass


def test_dspark_whole_graph_capture_routes_ag_onto_pynccl(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )
    dflash_cls, dspark_cls = _make_speculator_cls()
    target = _install_target(monkeypatch, dspark_cls)
    assert patch_dspark_draft_role_sep.apply_to_module(target)
    assert patch_dspark_draft_role_sep.apply_to_module(target) is False

    spec = dspark_cls(SimpleNamespace(), "cpu")
    # __init__ wrapper armed the role-separation state under default env.
    assert spec._draft_ag_pynccl is True
    # The adapter no longer wraps the cudagraph manager: replay stays the
    # audited original (whole graph, no grown eager tail).
    assert not hasattr(dspark_cls, "init_cudagraph_manager") or not getattr(
        dspark_cls.init_cudagraph_manager, patch_dspark_draft_role_sep._WRAPPER, False
    )

    fake_tp = _FakeTPGroup(2)
    monkeypatch.setattr(rs, "tp_pynccl_comm", lambda: (fake_tp, fake_tp.pynccl))
    loaded = spec.load_draft_model(object(), set())
    spec.model = loaded
    assert spec._draft_ag_pynccl is True

    # Route the fixture's AG through the real coordinator chain.
    monkeypatch.setattr(
        "vllm.distributed.tensor_model_parallel_all_gather",
        lambda t, dim=-1: fake_tp._all_gather_out_place(t, dim),
    )
    spec.init_cudagraph_manager(None)
    mgr = spec.query_cudagraph_manager
    assert getattr(mgr.run_fullgraph, "__name__", "") == "<lambda>"

    # Capture: the whole draft step records with the AG on the pynccl comm.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    spec._generate_draft(
        2, 4, None, None, None, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    assert fake_tp.pynccl.ag_calls == [((4, 1), (8, 1))]
    assert fake_tp.calls == []
    assert spec.gathered_logits.shape == (4, 2)

    # Eager (context phase / warmup): the stock c10d path, untouched.
    spec.gathered_logits = None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    spec._generate_draft(2, 4, None, None, None, cudagraph_runtime_mode=None)
    assert fake_tp.calls == [((4, 1), -1)]
    assert len(fake_tp.pynccl.ag_calls) == 1


def test_dspark_load_draft_model_degrades_without_pynccl(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )
    dflash_cls, dspark_cls = _make_speculator_cls()
    target = _install_target(monkeypatch, dspark_cls)
    assert patch_dspark_draft_role_sep.apply_to_module(target)
    monkeypatch.setattr(rs, "tp_pynccl_comm", lambda: None)

    spec = dspark_cls(SimpleNamespace(), "cpu")
    assert spec._draft_ag_pynccl is True
    spec.load_draft_model(object(), set())
    # Loud degradation: the captured AG would fall back to the shared c10d
    # communicator, so the adapter disarms rather than patching silently.
    assert spec._draft_ag_pynccl is False


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


def test_role_separation_adapters_are_registered_in_worker_callbacks() -> None:
    """The isolation chain only exists if the adapters are actually armed.

    A merge that keeps one side of worker/__init__.py can bring the three
    framework_opt files over while silently dropping their callback
    registrations (observed upstream: the job launched with role separation
    unarmed and deadlocked again). Pin the adapter paths so that regression
    fails a unit test instead of a 32-rank launch.
    """
    from vllm_hcu.patch.worker import _FRAMEWORK_CALLBACKS

    adapters = {spec.adapter for spec in _FRAMEWORK_CALLBACKS}
    for name in (
        "vllm_hcu.patch.worker.framework_opt.patch_kimi_k3_eager_ar",
        "vllm_hcu.patch.worker.framework_opt.patch_dspark_eager_ar",
        "vllm_hcu.patch.worker.framework_opt.patch_dspark_draft_role_sep",
    ):
        assert name in adapters, f"silently unregistered: {name}"
