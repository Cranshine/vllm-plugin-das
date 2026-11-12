# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3.1 Pro only: bandwidth-bound streaming GEMM for the lm_head at decode M.

Kept separate from ``vllm_hcu.model_executor.layers.skinny_lm_head`` (the Step5
kernel) on purpose: the Step5 path stays untouched, and this kernel is
specialised per tensor-parallel head width instead of being a generic skinny-M
GEMM.

Why the shape needs a dedicated kernel: the decode step runs the shared lm_head
twice (DSpark draft logits, then target verify logits) against the bf16 head of
shape ``vocab x 7168`` = ``163840 x 7168`` on the K3.1-Pro INT4 checkpoint, i.e.
a per-rank shard of 81920 (TP2) / 40960 (TP4) / 20480 (TP8) rows. With the
served batch (``--max-num-seqs 16``, 5 speculative tokens, so M = 80/96 as
bs*(spec_len+1) and bs*spec_len) each call is a pure weight stream: the
fp32-accumulating bf16 GEMM needs every weight element once, so the reachable
time is ``bytes / HBM read bandwidth``.

What is specialised here (measured on nmz30 against the real
``language_model.lm_head.weight``, per call; TP2/TP4 bit-exact against the
native projection):

* ``BM`` covers every decode row of one call in a single tile (64 for M <= 64,
  128 for M <= 128), so the weight is streamed exactly once instead of once per
  16/64-row M tile. That is the whole win: the transposed-B BLAS solution picks
  an M tile of 16 or 64 and re-streams the entire head for each tile.
* ``BN=64`` with ``NT`` fused N tiles per program: ``x`` is re-read once per
  program, so fusing tiles cuts the repeated ``x`` traffic; NT of 64x64 fp32
  accumulators is what the VGPR budget allows.
* ``BK=128`` for the small tile and for TP2's wide tile; the wide tile doubles
  the ``x`` staging in LDS so TP4/TP8/TP1 are faster with ``BK=64``.  Other BK
  loses 15-70%.
* ``num_stages=1`` on purpose: a second pipeline stage needs 80-160 KiB of LDS
  against the 64 KiB per-workgroup budget (``OutOfResources``), and the stage
  that does fit is slower because it costs occupancy.
* Per-width plan, because the best ``NT`` depends on the shard width and on how
  many programs remain: TP2 (N=81920) and TP8 (N=20480) want ``NT=2`` (640 and
  160 programs), TP4 (N=40960) and TP1 (N=163840) ``NT=4``.
* Routing is the Kimi head shape itself (2026-11-12): the target text model's
  ``LogitsProcessor`` is built from the ``text_config`` (model_type
  ``kimi_linear``) while the DSpark draft's is built from the top-level config
  (model_type ``kimi_k31``), so no config-derived marker can see both.  Both
  share the same weight, so the per-rank width of the 163840 vocab plus
  ``K == 7168`` is the single routing key.  Every decode M (1..128) routes to
  this kernel: at N=20480 the native projection is faster below M=24 (0.19 vs
  0.24 ms) and even at wide M=96 (0.349 vs 0.372 ms), but the uniform switch
  was taken on purpose; only M > 128 (prefill logits) keeps the native path.

Measured per call (nmz30, native = the projection vLLM runs today):

    M=48    TP2  N= 81920   4.95 ms -> 0.772 ms (6.4x, 1.52 TB/s, bit-exact)
            TP4  N= 40960   4.14 ms -> 0.435 ms (9.5x, 1.35 TB/s, bit-exact)
            TP8  N= 20480  0.363 ms -> 0.243 ms (1.50x, 1.21 TB/s)
    M=96    TP1  N=163840   3.304 ms -> 2.255 ms (1.47x, 1.04 TB/s)
            TP2  N= 81920   4.87 ms -> 1.105 ms (4.4x, 1.06 TB/s, bit-exact)
            TP4  N= 40960   7.96 ms -> 0.598 ms (13.3x, 0.98 TB/s, bit-exact)
            TP8  N= 20480   0.349 ms -> 0.372 ms (0.94x, uniform switch)
    M=128   TP2  N= 81920   7.04 ms -> 1.108 ms (6.4x, bit-exact)
            TP4  N= 40960   9.06 ms -> 0.599 ms (15.1x, bit-exact)
            TP8  N= 20480   0.414 ms -> 0.372 ms (1.11x)

Where the result is not bit-identical it differs from the native projection only
in fp32 accumulation order (max 2e-3 in head-logit units; both are 2.09e-3
relRMS away from an fp64 reference, i.e. the same bf16 GEMM, not a precision
change).

Disable through the process master switch ``VLLM_HCU_USE_CUSTOM_OPS=0``; the
adapter only reaches this module for a head that is exactly unquantized bf16
with no embedding bias, K = 7168 and one of the Kimi vocab shard widths.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.platforms import envs as henvs

# Largest token count that is still a pure weight stream in the K3.1-Pro decode
# path: a serve with --max-num-seqs 16 and 5 speculative tokens runs the head at
# M = 16 * 6 = 96, and bs * (spec_len + 1) stays <= 128 up to bs=21.
MAX_M = 128
# Rows covered by a single tile in the small / wide plan.
_BM_SMALL = 64
_BM_WIDE = 128
# Routing key: the Kimi-K3.1-Pro head itself (hidden 7168, vocab 163840). The
# per-rank widths of that vocab are the only N values with a tuned plan; every
# other shape (a different K, a different vocab, a quantized or biased head)
# keeps the native projection.
_KIMI_K = 7168

_BN = 64
# (num_ntiles, num_warps) for the 64-row tile, per per-rank head width.  The
# K3.1-Pro INT4 checkpoint has vocab 163840, so TP1/TP2/TP4/TP8 give these.
_PLANS_SMALL: dict[int, tuple[int, int]] = {
    163840: (4, 4),  # TP1
    81920: (2, 4),  # TP2
    40960: (4, 4),  # TP4
    20480: (2, 4),  # TP8
}
# (num_ntiles, num_warps, BK) for the 128-row tile (bs*6 > 64).  The wide tile
# doubles the x staging, so TP2 keeps BK=128 while TP4/TP8/TP1 (more programs
# per tile) are faster with BK=64.
_PLANS_WIDE: dict[int, tuple[int, int, int]] = {
    163840: (2, 8, 64),  # TP1: 2.255 ms vs 3.304 native at M=96
    81920: (4, 8, 128),  # TP2
    40960: (4, 8, 64),  # TP4
    20480: (2, 8, 64),  # TP8: 0.372 ms vs 0.349 native at M=96 (0.94x), 1.11x at M=128
}
_BK_SMALL = 128


def _plan(n: int, m: int) -> tuple[int, int, int, int] | None:
    """(BM, BK, NT, num_warps) for a per-rank head width N at M rows, or None."""
    if m <= _BM_SMALL:
        small = _PLANS_SMALL.get(n)
        if small is None:
            return None
        nt, warps = small
        if n % (_BN * nt) != 0:
            return None
        return _BM_SMALL, _BK_SMALL, nt, warps
    if m > _BM_WIDE:
        return None
    wide = _PLANS_WIDE.get(n)
    if wide is None:
        return None
    nt, warps, bk = wide
    if n % (_BN * nt) != 0:
        return None
    return _BM_WIDE, bk, nt, warps


@triton.jit
def _k31_lm_head_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
):
    """o[M, N] = x[M, K] @ w[N, K].T: all M rows in one tile, NT N tiles per program."""
    pid = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pid * BN * NT + tl.arange(0, BN)
    mmask = rm < M
    acc0 = tl.zeros((BM, BN), tl.float32)
    acc1 = tl.zeros((BM, BN), tl.float32)
    acc2 = tl.zeros((BM, BN), tl.float32)
    acc3 = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0
        )
        acc0 += tl.dot(x, tl.load(w_ptr + rn[None, :] * K + rk[:, None]))
        if NT >= 2:
            acc1 += tl.dot(x, tl.load(w_ptr + (rn + BN)[None, :] * K + rk[:, None]))
        if NT >= 4:
            acc2 += tl.dot(x, tl.load(w_ptr + (rn + 2 * BN)[None, :] * K + rk[:, None]))
            acc3 += tl.dot(x, tl.load(w_ptr + (rn + 3 * BN)[None, :] * K + rk[:, None]))
    op = o_ptr + rm[:, None] * N + rn[None, :]
    tl.store(op, acc0.to(tl.bfloat16), mask=mmask[:, None])
    if NT >= 2:
        tl.store(op + BN, acc1.to(tl.bfloat16), mask=mmask[:, None])
    if NT >= 4:
        tl.store(op + 2 * BN, acc2.to(tl.bfloat16), mask=mmask[:, None])
        tl.store(op + 3 * BN, acc3.to(tl.bfloat16), mask=mmask[:, None])


def enabled() -> bool:
    return henvs.optional_custom_op_enabled()


def usable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Shape/layout gate: the Kimi-K3.1-Pro head at decode M."""
    if not enabled() or not x.is_cuda or x.dim() < 2:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if weight.dim() != 2 or not weight.is_contiguous():
        return False
    n, k = weight.shape
    if k != _KIMI_K or k != x.shape[-1]:
        return False
    m = x.numel() // k
    return 0 < m <= MAX_M and _plan(n, m) is not None


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x[..., K] @ weight[N, K].T in bf16 with fp32 accumulation."""
    n, k = weight.shape
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]
    plan = _plan(n, m)
    if plan is None:
        raise ValueError(f"K3.1 lm_head GEMM is not specialised for N={n}, M={m}")
    bm, bk, nt, num_warps = plan
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _k31_lm_head_kernel[(n // (_BN * nt),)](
        x2,
        weight,
        out,
        m,
        n,
        k,
        BM=bm,
        BN=_BN,
        BK=bk,
        NT=nt,
        num_warps=num_warps,
        num_stages=1,
    )
    return out.reshape(*x.shape[:-1], n)


__all__ = ["MAX_M", "enabled", "linear", "usable"]
