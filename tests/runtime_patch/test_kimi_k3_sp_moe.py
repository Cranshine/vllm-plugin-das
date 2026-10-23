# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3 SP-MoE patch contracts (CPU only, no CUDA, no distributed init).

Binding contract: for every N (tokens) x TP (shards) x rank combination the
SP forward must produce the rows of the replicated-domain reference
computation. Collectives and the attn_res kernel are replaced by stubs with
exact semantics, so equality is bit-exact (torch.equal).

Review-contract tests (chatgpt-codex-connector P1 round):
- P1-1: the phase-routed communicator selection must survive torch.compile;
  the collectives are registered custom ops (opaque to dynamo, fake impl for
  shapes), and the module must not fall back "while compiling".
- P1-2: the SP forward must be branch-free in the token count (dropped-guard
  hazard), and its window arithmetic must never slice past N.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_sp_moe as sp


# ---------------------------------------------------------------------------
# Stubs with exact collective / kernel semantics.
# ---------------------------------------------------------------------------


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _stub_attn_res(prefix, residual, proj, norm, num_valid_blocks):
    """Row-wise stand-in for the boltops attn_res kernel.

    Mirrors the real wrapper's static gate (num_valid_blocks <= 0 short-
    circuits), then pins the kernel contract under test: padding rows
    (softmax of zeroed prefix) must never be fed in, so every live call
    must carry a positive batch of finite rows whose sizes match.
    """
    if num_valid_blocks <= 0:
        return prefix
    assert prefix.shape[0] > 0, "padding rows reached the attn_res kernel"
    assert residual.shape[0] == prefix.shape[0]
    return prefix + residual.sum(dim=1)


def _make_layer(*, tp_rank: int, tp_size: int, block_write: bool, prev_blocks: int, H: int = 6):
    """Stub KimiDecoderLayer exposing exactly the attributes the SP forward uses."""
    layer = types.SimpleNamespace()
    res_proj = types.SimpleNamespace(weight=torch.eye(H))
    res_norm = types.SimpleNamespace(weight=torch.ones(H), variance_epsilon=1e-6)
    layer.self_attention_res_proj = res_proj
    layer.self_attention_res_norm = res_norm
    layer.mlp_res_proj = res_proj
    layer.mlp_res_norm = res_norm
    layer.is_block_write_layer = block_write
    layer.block_write_idx = 1
    layer.prev_valid_blocks = prev_blocks
    layer.input_layernorm = _Identity()
    layer.post_attention_layernorm = _Identity()
    layer._run_self_attn = lambda positions, h: h * 3.0 + 1.0
    layer.mlp = lambda h: h - 2.0
    setattr(layer, sp._LAYER_STATE, sp._SpState(tp_size, tp_rank, runner=None))
    state = getattr(layer, sp._LAYER_STATE)
    state.checked = True  # skip the real MoERunner introspection
    return layer


def _reference_full_domain(hs, br, *, block_write: bool, prev_blocks: int):
    """Replicated-domain computation of the same layer math (AR == identity).

    Written from the layer equations with the same stubs, never from the SP
    code path. ``br`` is mutated in place exactly like the layer does (block
    write), so ``br.sum(dim=1)`` after the write is the residual stream the
    MLP-side kernel consumes.
    """
    pre = hs if prev_blocks <= 0 else hs + br.sum(dim=1)   # entry kernel; a no-op gate at 0 blocks
    attn = pre * 3.0 + 1.0
    if block_write:
        br[:, 1, :] = hs                        # block write (in place)
        prefix_mid = attn                       # residual stream restarts at attn
    else:
        prefix_mid = hs + attn                  # upstream :  prefix_sum + attn_out
    if prev_blocks + (1 if block_write else 0) > 0:
        hidden_mid = prefix_mid + br.sum(dim=1)  # br reflects a done write
    else:
        hidden_mid = prefix_mid
    return prefix_mid + (hidden_mid - 2.0)


def _run_sp_rank(
    hs,
    br,
    reference,
    *,
    tp_rank: int,
    tp_size: int,
    block_write: bool,
    prev_blocks: int,
    observe: dict,
):
    """Run one rank's SP forward with exact collective stubs.

    RS: rank r receives rows [r*S, (r+1)*S) of the summed padded partials.
    With deterministic stubs every rank's padded partial is identical, so the
    summed tensor is the padded input itself and the window is a slice.
    AG: the gathered stream is the replicated full-domain result, taken from
    the independently derived ``reference`` -- this asserts the window each
    rank contributes sits at its own slot.
    """
    layer = _make_layer(
        tp_rank=tp_rank, tp_size=tp_size, block_write=block_write, prev_blocks=prev_blocks
    )
    state = getattr(layer, sp._LAYER_STATE)
    S = -(-hs.shape[0] // tp_size)
    padded = S * tp_size

    def rs_stub(x, tp):
        # All ranks hold the same padded partial (deterministic stubs), so
        # the reduction is a no-op over identical windows and rank r's
        # received window is the slice of x itself.
        assert x.shape[0] == padded
        observe.setdefault("rs_windows", []).append(
            (tp_rank, x[tp_rank * S : (tp_rank + 1) * S].clone())
        )
        return x[tp_rank * S : (tp_rank + 1) * S].clone()

    def ag_stub(x, tp):
        observe.setdefault("ag_locals", []).append((tp_rank, x.clone()))
        # AG semantics: each rank's window placed at its own slot. Slots of
        # other ranks are reconstructed independently from the reference so
        # the equality below asserts the placement and the final slice, not
        # a passthrough of the reference object. Reference windows past N
        # don't exist -- they stay zero (they get sliced away anyway).
        full = torch.zeros(S * tp_size, x.shape[1])
        for r in range(tp_size):
            g = r * S
            v = max(0, min(S, reference.shape[0] - g))
            if v:
                full[g : g + v] = reference[g : g + v]
        full[tp_rank * S : (tp_rank + 1) * S] = x
        return full

    with pytest.MonkeyPatch.context() as mp:
        # Patch the op namespace attributes the forward calls through. This
        # keeps the numerics contract independent of dispatcher backends:
        # on a CPU-only test box the custom ops may have no backend impl.
        mp.setattr(
            torch.ops.vllm, "hcu_k3_sp_reduce_scatter", rs_stub, raising=False
        )
        mp.setattr(
            torch.ops.vllm, "hcu_k3_sp_all_gather", ag_stub, raising=False
        )
        mp.setattr(sp, "_attn_res", _stub_attn_res)
        prefix, br_out = sp._sp_forward_attn_residual(
            layer, state, torch.zeros(hs.shape[0]), hs, br
        )
    assert br_out is br
    return prefix


# ---------------------------------------------------------------------------
# Numerics contract: SP rows == replicated rows for every N x TP x rank.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("block_write", [False, True])
@pytest.mark.parametrize("prev_blocks", [0, 2])
def test_sp_rows_match_replicated_domain(tp_size, num_tokens, block_write, prev_blocks):
    H, B = 6, 3
    gen = torch.Generator().manual_seed(1234 + num_tokens)
    hs = torch.randn(num_tokens, H, generator=gen)
    br = torch.randn(num_tokens, B, H, generator=gen)
    reference = _reference_full_domain(
        hs.clone(), br.clone(), block_write=block_write, prev_blocks=prev_blocks
    )

    observe: dict = {}
    for r in range(tp_size):
        prefix = _run_sp_rank(
            hs.clone(), br.clone(), reference,
            tp_rank=r, tp_size=tp_size,
            block_write=block_write, prev_blocks=prev_blocks,
            observe=observe,
        )
        assert prefix.shape == (num_tokens, H)
        # Every rank must land on the same replicated-domain stream.
        assert torch.equal(prefix, reference), (
            f"rank {r} mismatch N={num_tokens} TP={tp_size} "
            f"bw={block_write} prev={prev_blocks}"
        )

    # Each rank contributed exactly its own window to the gather. Window
    # rows beyond N are padding rows, which this patch leaves at the mlp
    # stub's bias (-2.0) before the final slice drops them.
    S = -(-num_tokens // tp_size)
    H = hs.shape[1]
    for r, local in observe["ag_locals"]:
        g = r * S
        v = max(0, min(S, num_tokens - g))
        assert torch.equal(local[:v], reference[g : g + v])
        if v < S:
            assert torch.equal(local[v:], torch.full((S - v, H), -2.0))
    # And the reduce-scatter window is the summed padded partial's window.
    padded = S * tp_size
    pre = hs if prev_blocks <= 0 else hs + br.sum(dim=1)
    attn_partial = pre * 3.0 + 1.0
    padded_partial = torch.zeros(padded, H)
    padded_partial[:num_tokens] = attn_partial
    for r, window in observe["rs_windows"]:
        assert torch.equal(window, padded_partial[r * S : (r + 1) * S])


# ---------------------------------------------------------------------------
# Router-side windowing (the REAL router_runtime helper): rank slices,
# short-mask skip, no-window no-op.
# ---------------------------------------------------------------------------

from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    window_sp_padding_mask,
)


def test_router_windows_the_global_suffix_mask():
    N, TP, S = 9, 4, 3  # padded 12; rank 3 owns window rows [9, 12) -> all pad
    global_mask = torch.tensor([i >= N for i in range(N)])
    for rank in range(TP):
        topk_ids = torch.zeros(S, 3, dtype=torch.long)
        out = window_sp_padding_mask(global_mask, topk_ids, (rank, TP))
        assert out.shape == (S,)
        v = max(0, min(S, N - rank * S))
        assert not bool(out[:v].any()), f"rank {rank} real rows must stay unmasked"
        if v < S:
            assert bool(out[v:].all()), f"rank {rank} padding rows must be masked"


def test_router_skips_windowing_for_unrelated_masks():
    """A mask that does not reach this rank's first window row is not a
    global bitmap -- windowing must be skipped (caller keeps generic
    masking) instead of True-padding unknown rows."""
    short_mask = torch.tensor([False, True])  # length 2 < rank*S = 6
    topk_ids = torch.zeros(3, 2, dtype=torch.long)
    out = window_sp_padding_mask(short_mask, topk_ids, (2, 4))
    assert out is short_mask


def test_router_windows_a_full_exactly_n_bitmap():
    """len(mask) == N == rank*S boundary: the fully-padding last window."""
    mask = torch.tensor([False] * 9)
    topk_ids = torch.zeros(3, 2, dtype=torch.long)
    out = window_sp_padding_mask(mask, topk_ids, (3, 4))
    assert out.shape == (3,) and bool(out.all())


def test_router_without_window_attribute_is_noop():
    mask = torch.tensor([False, True])
    topk_ids = torch.zeros(2, 2, dtype=torch.long)
    out = window_sp_padding_mask(mask, topk_ids, None)
    assert out is mask


def test_router_window_major_mask_is_cropped():
    """A mask longer than padded (graph-padded local batches) crops cleanly."""
    N, S, TP = 8, 4, 2
    mask = torch.tensor([i >= 6 for i in range(N + 4)])  # longer than padded
    topk_ids = torch.zeros(S, 2, dtype=torch.long)
    out = window_sp_padding_mask(mask, topk_ids, (1, TP))
    assert out.shape == (S,)
    assert out.tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# Enablement: the window must be attached to the router that masks (P0).
# ---------------------------------------------------------------------------


def test_attach_padding_window_targets_the_router():
    moe = types.SimpleNamespace(
        gate=object(),  # GateLinear: no routing hook (review P0)
        experts=types.SimpleNamespace(
            router=types.SimpleNamespace(_compute_routing=lambda *a, **k: None)
        ),
    )
    sp._attach_padding_window(moe, 1, 4)
    assert moe.experts.router._hcu_sp_padding_window == (1, 4)
    # Arbitrary attributes on the gate must stay untouched.
    assert not hasattr(moe.gate, "_hcu_sp_padding_window")


def test_attach_padding_window_refuses_routerless_moe():
    moe = types.SimpleNamespace(experts=types.SimpleNamespace(router=None))
    with pytest.raises(sp.PatchCompatibilityError, match="_compute_routing"):
        sp._attach_padding_window(moe, 0, 2)


# ---------------------------------------------------------------------------
# P1-2 regression: no branch in the SP body may depend on the token count.
# ---------------------------------------------------------------------------


def _sp_forward_source() -> str:
    src = open(sp.__file__, encoding="utf-8").read()
    return ast.get_source_segment(src, sp._sp_forward_attn_residual) or src


def test_sp_forward_is_branch_free_in_token_count():
    """Dynamo specializes shape branches; vLLM drops the guards (Qwen precedent)."""
    tree = ast.parse(_sp_forward_source())
    banned = {"padded", "valid", "num_tokens", "rows"}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            overlap = banned & names
            assert not overlap, (
                f"SP branch on shape-derived value(s) {sorted(overlap)}: "
                f"{ast.unparse(node.test)[:120]}"
            )


def test_phase_comm_has_no_compiling_fallback():
    """P1-1: compilation must not select the no-role-separation routing."""
    src = open(sp.__file__, encoding="utf-8").read()
    assert "is_compiling" not in src, (
        "_phase_comm must not treat compilation as a no-role-separation "
        "fallback; routing happens at invocation time inside custom ops"
    )


# ---------------------------------------------------------------------------
# P1-1 regression: registered collectives with shape fakes.
# ---------------------------------------------------------------------------


def _ensure_ops():
    sp._ensure_custom_ops_registered()


def test_collective_ops_are_registered_with_shape_fakes():
    _ensure_ops()
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.randn(8, 3)
        rs = torch.ops.vllm.hcu_k3_sp_reduce_scatter(x, 4)
        assert tuple(rs.shape) == (2, 3)
        ag = torch.ops.vllm.hcu_k3_sp_all_gather(x, 4)
        assert tuple(ag.shape) == (32, 3)


def test_registration_is_idempotent():
    _ensure_ops()
    _ensure_ops()  # registry guard: no schema redefinition


def test_reduce_scatter_impl_prefers_role_comm(monkeypatch):
    calls = []

    class FakeComm:
        def reduce_scatter(self, out, x):
            calls.append(("rs", tuple(out.shape)))
            out.copy_(x[: out.shape[0]])

    monkeypatch.setattr(sp, "_phase_comm", lambda: FakeComm())
    x = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)
    out = sp._sp_reduce_scatter(x, 4)
    assert calls == [("rs", (2, 3))]
    assert torch.equal(out, x[:2])


def test_all_gather_impl_prefers_role_comm(monkeypatch):
    calls = []

    class FakeComm:
        def all_gather(self, out, x):
            calls.append(("ag", tuple(out.shape)))
            out.zero_()
            out[3:6] = x

    monkeypatch.setattr(sp, "_phase_comm", lambda: FakeComm())
    x = torch.ones(3, 2)
    out = sp._sp_all_gather(x, 3)
    assert calls == [("ag", (9, 2))]
    assert torch.equal(out[3:6], x)


def test_impls_fall_back_to_stock_group_api(monkeypatch):
    monkeypatch.setattr(sp, "_phase_comm", lambda: None)
    monkeypatch.setattr(sp, "_warned_fallback", True)  # silence warn-once

    import vllm.distributed as vd

    # The real stock APIs (TP > 1) always return freshly allocated tensors,
    # not views -- the alias contract of the custom ops depends on it.
    def fake_rs(x, dim=0):
        return x[: x.shape[0] // 2].clone()

    def fake_ag(x, dim=0):
        return x.repeat(2, 1)
    monkeypatch.setattr(
        vd, "tensor_model_parallel_reduce_scatter", fake_rs
    )
    monkeypatch.setattr(
        vd, "tensor_model_parallel_all_gather", fake_ag
    )
    x = torch.arange(4 * 2, dtype=torch.float32).view(4, 2)
    assert torch.equal(sp._sp_reduce_scatter(x, 2), x[:2])
    assert torch.equal(sp._sp_all_gather(x, 2), x.repeat(2, 1))


def test_phase_comm_selection_table(monkeypatch):
    """capture -> TP pynccl comm; eager + role-sep -> aux comm; off -> None."""
    role_comm = types.SimpleNamespace(disabled=False, available=True)
    aux = types.SimpleNamespace(disabled=False, available=True)

    monkeypatch.setattr(
        sp.torch.cuda, "is_current_stream_capturing", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: types.SimpleNamespace(
            device_communicator=types.SimpleNamespace(pynccl_comm=role_comm)
        ),
        raising=False,
    )
    # Capturing wins over role-sep state: the captured-role comm is selected.
    monkeypatch.setattr(sp, "_role_sep_enabled", lambda: True)
    monkeypatch.setattr(sp, "_role_sep_aux_comm", lambda: object())
    assert sp._phase_comm() is role_comm

    # Eager + role separation: the auxiliary comm.
    monkeypatch.setattr(sp.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(sp, "_role_sep_aux_comm", lambda: aux)
    assert sp._phase_comm() is aux

    # An unusable aux (disabled / absent) degrades to the stock group API.
    monkeypatch.setattr(sp, "_role_sep_aux_comm", lambda: None)
    assert sp._phase_comm() is None

    # Role separation off: None -> stock group API.
    monkeypatch.setattr(sp, "_role_sep_enabled", lambda: False)
    assert sp._phase_comm() is None


def test_op_entries_resolve_the_live_module(monkeypatch):
    """A reload must not strand the registered ops on stale globals (P1-1)."""
    sent_x, sent_tp = object(), 7
    seen = {}

    class FakeModule:
        def _sp_reduce_scatter(self, x, tp):
            seen["args"] = (x, tp)
            return "rs-done"

    monkeypatch.setitem(sys.modules, sp.__name__, FakeModule())
    assert sp._sp_reduce_scatter_op_entry(sent_x, sent_tp) == "rs-done"
    assert seen["args"] == (sent_x, sent_tp)


# ---------------------------------------------------------------------------
# P2 regression: reload resets the role-separation accessors; the first
# post-reload routing call must re-resolve them lazily (never during
# capture), so eager and captured calls cannot split between the stock API
# and pynccl.
# ---------------------------------------------------------------------------


def test_role_sep_accessors_rebind_after_simulated_reload(monkeypatch):
    monkeypatch.setattr(
        sp.torch.cuda, "is_current_stream_capturing", lambda: False, raising=False
    )
    monkeypatch.setattr(sp, "_role_sep_enabled", None, raising=False)
    monkeypatch.setattr(sp, "_role_sep_aux_comm", None, raising=False)
    monkeypatch.setattr(sp, "_role_sep_resolved", False, raising=False)

    aux = types.SimpleNamespace(disabled=False, available=True)
    fake_plugin = types.ModuleType(
        "vllm_hcu.patch.worker.framework_opt._role_separation"
    )
    fake_plugin.role_separation_enabled = lambda: True
    fake_plugin._get_aux_ar_comm = lambda: aux
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.patch.worker.framework_opt._role_separation",
        fake_plugin,
    )

    assert sp._phase_comm() is aux
    assert sp._role_sep_enabled is fake_plugin.role_separation_enabled
    assert sp._role_sep_aux_comm is fake_plugin._get_aux_ar_comm
    assert sp._role_sep_resolved is True


def test_role_sep_unavailable_flips_resolved_flag(monkeypatch):
    monkeypatch.setattr(
        sp.torch.cuda, "is_current_stream_capturing", lambda: False, raising=False
    )
    monkeypatch.setattr(sp, "_role_sep_enabled", None, raising=False)
    monkeypatch.setattr(sp, "_role_sep_aux_comm", None, raising=False)
    monkeypatch.setattr(sp, "_role_sep_resolved", False, raising=False)
    monkeypatch.setattr(sp, "_warned_fallback", True)  # silence warn-once
    # Deterministic: both authority modules pretend to be missing-never-
    # importable (setitem None forces ImportError even if one exists).
    monkeypatch.setitem(
        sys.modules, "vllm_hcu.patch.worker.framework_opt._role_separation", None
    )
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.role_separation", None)

    # Neither module resolves in this tree (the role-separation prerequisite
    # is not part of this PR): the stock-API destination is decided once and
    # must NOT be re-probed on subsequent calls.
    assert sp._phase_comm() is None
    assert sp._role_sep_resolved is True

    resolve_spy = []
    real_resolve = sp._resolve_role_sep

    def spy():
        resolve_spy.append(1)
        return real_resolve()

    monkeypatch.setattr(sp, "_resolve_role_sep", spy)
    assert sp._phase_comm() is None
    assert resolve_spy == []  # latch held: no re-probe on the second call


def test_capture_short_circuits_lazy_resolution(monkeypatch):
    """Capture takes the pynccl branch before any resolution work."""
    monkeypatch.setattr(
        sp.torch.cuda, "is_current_stream_capturing", lambda: True, raising=False
    )
    monkeypatch.setattr(sp, "_role_sep_resolved", False, raising=False)
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: types.SimpleNamespace(
            device_communicator=types.SimpleNamespace(
                pynccl_comm=types.SimpleNamespace(disabled=False, available=True)
            )
        ),
        raising=False,
    )
    comm = sp._phase_comm()
    assert getattr(comm, "available", None) is True
    assert sp._role_sep_resolved is False  # untouched while capturing


# ---------------------------------------------------------------------------
# End-to-end-ish subprocess check: patch applies with the switch on and the
# module compiles clean after the changes.
# ---------------------------------------------------------------------------


def test_module_imports_and_compiles_clean():
    script = textwrap.dedent(
        """
        import ast
        import vllm_hcu.patch.worker.op_opt.patch_kimi_k3_sp_moe as sp
        ast.parse(open(sp.__file__).read())  # file parses
        assert hasattr(sp, "_ensure_custom_ops_registered")
        assert hasattr(sp, "_sp_reduce_scatter_fake")
        assert hasattr(sp, "_sp_all_gather_fake")
        print("IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "IMPORT_OK" in result.stdout
