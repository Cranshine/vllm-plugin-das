# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Sequence-parallel MoE (SP-MoE) for the Kimi-K3 AMD attn-residual layer.

Opt-in with ``VLLM_HCU_K3_SP_MOE=1``. Unset or ``0`` (the default) leaves the
target module untouched: the patch only sets its marker and installs nothing.

Problem
=======
Deployment: Kimi-K3 W4A8, DP8 x TP4 x EP32, DeepEP low-latency, RCCL, decode
captured with ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``. ``o_proj`` is a
``RowParallelLinear(reduce_results=True)``, so after attention all four TP
ranks hold the same ``N`` rows. Each rank then routes and dispatches all
``N`` rows through DeepEP. That means 4x the a2a traffic, 4x the masked
GEMM ``M`` and 4x the latent down/up-proj and gate work, which is the
duplication the upstream ``batched_deep_gemm_moe.py`` FIXME describes.

Design (layer-internal SP, applied to MoE layers only)
======================================================
The attn->MoE boundary moves from the replicated row domain to the
row-sharded domain, using ``AR(x) == AG(RS(x))``. ``N_pad = TP*ceil(N/TP)``,
``S = N_pad/TP``, rank ``r`` owns global rows ``[r*S, r*S+S)``, and ``v_r``
is how many of those rows are real (``< N``)::

    full   prefix_sum[N]  (layer input, replicated, bit-identical on all ranks)
    full   pre-attn attn_res + input_layernorm + block write   (unchanged)
    full   attention (TP head-sharded) -> o_proj PARTIAL sums  (no AR)
    pad    partial -> [N_pad]                     (only when N % TP != 0)
    RS     -> attn_local[S]                       (collective 1)
    shard  prefix_local = prefix_sum[rows_r] + attn_local
    shard  mlp attn_res(prefix_local, block_residual[rows_r]) + post-norm
    shard  KimiMoE: gate, latent down-proj, DeepEP dispatch of S rows,
           masked GEMM, combine, latent up-proj
           shared expert: AG(x_local)->TP-sharded MLP->RS  (collectives 2,3)
    shard  prefix_local += moe_local
    AG     -> prefix_sum[N_pad] -> [:N]           (collective 4)

Per MoE layer this is RS+AG+RS+AG. Today it is AR(o_proj) + AR(shared
expert), and an AR moves the same bytes as RS+AG, so TP communication
volume stays the same. Each rank's DeepEP dispatch drops from ``N`` to
``ceil(N/TP)`` rows. The trade is 4 collective launches instead of 2, which
matters when decode is latency-bound (see risk R1).

Dense layers (``first_k_dense_replace``) keep the original forward. Their
TP-sharded ``KimiMLP`` would need the same AG/RS pair and gains nothing from
SP, since no dispatch happens there. Layers without attn residuals, such as
the MTP block (``KimiDecoderLayer`` built with attn-res disabled, and the
DSpark draft which reuses it), are also left alone: their eager warmup
collectives never bypass the role-comm pairing that way. Every layer still
returns a full-row ``prefix_sum``, so SP and non-SP layers mix freely.

``KimiDecoderLayer.forward`` is not replaced: when ``use_attn_residuals`` is
set it only dispatches to ``self.forward_attn_residual`` (the patched
method). Its non-attn-res branch is never taken by an SP layer, because SP is
enabled only for layers with ``use_attn_residuals``.

Hidden prerequisite: the shared expert
--------------------------------------
``KimiMoE.shared_experts`` is a TP-sharded ``KimiMLP`` with
``down_proj.reduce_results=False``. The HCU ``MoERunner`` all-reduces its
partial output over TP in ``_maybe_reduce_shared_expert_output``. Once TP
ranks hold different rows, an all-reduce over that sharded domain would sum
index-aligned partials of DIFFERENT global rows (rank r's index u is global
row r*S+u), so the result would be wrong with no error raised. For SP
layers this patch therefore, at the instance level only:

* wraps the shared expert ``forward`` as ``AG(x_local) -> MLP -> RS``, which
  gives exact rows for this rank's shard
  (``RS(MLP_r(AG(x)))[rows_r] == AR(MLP_r(x))[rows_r]``);
* makes that runner's ``_maybe_reduce_shared_expert_output`` a passthrough;
* checks on layer construction (Python only, before any capture) that the
  runner has no other TP reduction left: ``moe_config.tp_size == 1`` (EP),
  the combine output is already reduced (DeepEP LL/HT), there is no EP fused
  combine, the runner is not in upstream SP mode, and DBO is off. If any
  check fails it raises instead of producing wrong numerics.

Upstream solves this with ``ReplicatedLinear`` shared experts when
``is_sequence_parallel`` is set (deepseek_v2). That is not possible here
without re-sharding the loaded W4A8 shared-expert weights.

Trap 1: the attn-res residual domain
====================================
``block_residual`` is ``[N, B, H]``, allocated by ``KimiLinearModel.forward``
with full rows and carried across layers. Ruling:

* Writes (``block_residual[:, idx, :].copy_(prefix_sum)``) happen at layer
  entry, before attention, with the full, replicated ``prefix_sum``. They stay
  in the full domain and are unchanged.
* The pre-attention ``attn_res`` reads all ``N`` rows in the full domain and
  is unchanged.
* The MLP-side ``attn_res`` runs in the shard domain. Its ``prefix_local`` row
  ``i`` is global row ``r*S+i``, so it is called with the aligned view
  ``block_residual[r*S : r*S+v_r]``. A dim-0 slice of a contiguous tensor is
  still contiguous (only the storage offset changes), and attn_res is
  row-independent (a per-row softmax over blocks plus a weighted sum), so
  per-row indexing stays valid. Padding rows (``i >= v_r``) have no
  ``block_residual`` row. They are never passed to the kernel; their hidden
  state is zero, which ``RMSNorm`` maps to zero.
* The adds ``prefix + attn`` and ``prefix + moe`` are elementwise. They run on
  the shard and are bit-identical to the same rows of the full computation.
* Returned to the next layer: ``prefix_sum`` is full ``[N, H]``, restored by
  this layer's final AG (a view of the ``[N_pad, H]`` gather buffer).
  ``block_residual`` is the same full object, never sharded, never
  copied. Rows produced by AG are bit-identical across TP ranks, so the
  replicated-domain invariant the next layer relies on still holds.
  ``AR`` gave the same guarantee.
* The model-level final ``attn_res`` and the aux hidden-state taps
  (EAGLE/DSpark) see full rows exactly as before.

Trap 2: graph capture and padding
=================================
* ``gpu_model_runner._pad_for_sequence_parallelism`` keys on
  ``compilation_config.pass_config.enable_sp``, not on
  ``parallel_config.use_sequence_parallel_moe``. The latter is already True in
  this deployment (DeepEP LL + EP + TP>1 + DP>1), but it never triggers row
  padding. The HCU model runner inherits that method unchanged, and
  ``enable_sp`` is off for this model, so nothing upstream guarantees
  ``N % TP == 0``. The HCU ``enable_custom_sp`` feature
  (``platform/core_fix/patch_compilation_config.py``) only rounds the
  *capture sizes* through upstream's ``enable_sp`` branch of
  ``adjust_cudagraph_sizes_for_sp``. It does not pad eager batches and is not
  relied upon here.
* So padding is done inside this layer: zero rows go onto the un-reduced
  ``o_proj`` output (zero partial sums reduce to zero), and the AG output is
  sliced back to ``N``. ``N``, ``S``, ``N_pad`` and ``v_r`` are Python ints
  derived from ``hidden_states.shape[0]`` and the static TP rank. Under capture
  ``N`` is the captured batch size, so every shape and branch is baked once per
  graph and identical on every replay. No data-dependent shape, no host sync.
* Pure decode with ``num_spec_tokens=7`` gives ``N = 8 * num_reqs`` (padded to
  the capture size), which is always divisible by 4, so padding costs nothing.
  Admission steps that mix spec and non-spec requests, DP dummy batches
  (``N=1``) and arbitrary prefill chunks do not satisfy the assumption. They
  take the pad-and-slice path, which is eager, so the extra pad copy and slice
  are acceptable. When ``N < TP``, trailing ranks own only padding rows
  (``v_r == 0``) and still join every collective and DeepEP dispatch,
  sending zero rows.
* Padding rows are routed and dispatched (at most TP-1 rows per DP rank).
  Their latent and shared outputs are about zero and get sliced away.

Trap 3: expert-side expected_m
==============================
``BatchedDeepGemmExperts.estimate_expected_m`` uses either
(a) ``sum(num_tokens_across_dp)`` when ``DPMetadata`` exists, or
(b) ``topk_ids.size(0) * num_dispatchers``. ``topk_ids`` is the pre-dispatch
local routing of this rank.

* (a) ``num_tokens_across_dp`` counts each DP rank's ``N`` once. Before SP the
  real dispatch was ``TP * sum(N)`` rows, so the estimate was already 4x too
  LOW. After SP the real dispatch is ``sum(N_pad) ~= sum(N)``, so the
  estimate becomes accurate. Nothing is overestimated.
* (b) ``local`` changes from ``N`` to ``S`` together with the real per-rank
  dispatch, so ``S * 32`` is exactly the dispatched total. It stays accurate
  and it is still a Python int, which is cudagraph-safe.

``expected_m`` is only a tile/wave hint for the masked GEMM. Correctness
comes from the dispatch counts (``masked_m``), and the per-rank buffer
capacity ``max_num_tokens`` only needs ``S <= N``. Ruling: no change. The
minimal-change rule forbids touching ``batched_deep_gemm_moe.py`` without a
demonstrated overestimate, and there is none.

Collectives and role separation (``VLLM_ROLE_SEP``)
===================================================
``role_separation`` reroutes only TP ``all_reduce`` and splits by phase:
captured ARs go to the TP pynccl comm, eager ARs go to the auxiliary comm.
The stock group APIs would put the new collectives on new mixed-role pairs.
On ROCm, an eager prefill ReduceScatter through the stock dispatcher would
land on the TP pynccl comm that carries captured collectives, and a captured
AllGather would land on the c10d comm that also serves eager collectives.
This patch therefore picks the comm explicitly, per call, by phase -- the
same predicate the role-sep shim uses:

* capturing -> ``get_tp_group().device_communicator.pynccl_comm`` (the
  captured-role comm, which already hosts the captured ARs);
* eager with role-sep enabled -> the role-sep auxiliary comm (eager role),
  obtained through ``role_separation._get_aux_ar_comm``; availability is
  verified at layer enablement so the eager phase can never silently fall
  back into the captured comm;
* role-sep off/unavailable, or torch.compile tracing -> the stock
  ``tensor_model_parallel_reduce_scatter`` / ``_all_gather`` (the same legacy
  single-comm topology role-sep itself falls back to), with a warn-once so
  the degraded routing is observable in the log.

Breakable cudagraph: the MoE body (including the shared expert's AG/RS) is
captured by the HCU MoE replacement, so every SP collective inside a decode
step is captured alongside it. The per-call predicate below re-evaluates at
capture time and selects the TP pynccl comm for captured segments, so capture
and replay use the same communicator. Eager steps (prefill, DP dummies,
warmup) select the aux comm; capturing and eager steps can interleave across
steps without changing which comm each collective uses.

Composition and provenance
==========================
This wrapper must apply after ``worker.op_opt.kimi_k3.reference_paths`` (it
captures that patch's ``hcu_residual_forward`` as the non-SP chain, and its
registration sits right below it in ``_OP_CALLBACKS``). ``_maybe_enable_layer``
verifies per layer at construction that the class method is still this
wrapper, and raises otherwise, because a later overwrite would run
un-reduced ``o_proj`` through the non-SP path.

Dual-implementation debate provenance: this file merges two independent
implementations (GLM main session, claude code review channel). The shared
expert (Trap-S) diagnosis, the phase-routed comm selection, init-time
enablement with runner validation, and the A1-A4 acceptance criteria are the
claude design, adopted after its AR index-alignment proof refuted the GLM
draft's runner-reduction stance. The custom-ops discipline (SP enablement
requires ``henvs.custom_ops_enabled()``, matching the reference patch's
runtime gate), the aux-comm availability preflight, and the warn-once on
degraded comm routing are the GLM corrections, adopted after review.

Known risks
===========
R1 latency: decode does 4 RCCL collectives per MoE layer instead of 2 ARs,
   and the displaced ARs were most likely custom quick-reduce/aiter IPC
   kernels (``CudaCommunicator.all_reduce`` prefers them over pynccl on
   ROCm), while the RS/AG here go to raw pynccl. At ``N <= 128`` both the
   a2a and the GEMM are latency-bound, so the net gain must be measured.
   Follow-up: use ``ca_comm.custom_reduce_scatter`` / ``custom_all_gather``
   for the captured phase (same approach as ``patch_qwen4_exp_sp``, which
   tries custom kernels first), or cross-layer SP that keeps ``prefix_sum``
   and ``block_residual`` sharded and AGs only the normed attention input.
   That needs ``KimiLinearModel.forward`` changes and is out of scope here.
R2 numerics: RS+AG versus AR can sum in a different order, so results match
   within fp tolerance, not bit-for-bit. Row-wise kernels (attn_res, RMSNorm,
   router GEMM) may choose different tiles for ``S`` rows than for ``N`` rows.
R3 shared-expert collectives inside the DeepEP MK window: ``reference_paths``
   pins the shared expert to the main stream, but DeepEP LL supports async
   finalize, so the shared expert (now including its AG and RS) runs in the
   MK_INTERNAL_OVERLAPPED window between ``finalize_async`` and the receive
   hook, where previously no TP collective ran. Every rank runs the same
   order and DeepEP LL does not use RCCL, so this is stream-safe, but two
   RCCL kernels now sit between the captured DeepEP send and recv kernels:
   treat long prefill/decode interleave soak as a mandatory validation
   (see A4). Correcting the earlier claim that the window could not host
   these collectives.
R4 coupling: depends on the HCU ``MoERunner`` reduction layout. The
   construction-time check fails hard if that layout changes. Also depends on
   ``o_proj`` being the last op of ``self_attn`` and on ``RowParallelLinear``
   reading ``reduce_results`` at call time.
R5 ordering: see "Composition and provenance" above. Note also that the
   enablement preflight calls ``_get_aux_ar_comm`` at model construction,
   which builds the auxiliary NCCL communicator before memory profiling and
   DeepEP buffer init instead of on the first forward (role_separation's
   documented construction point). This is rank-symmetric and harmless for
   correctness, but it moves RCCL buffer allocation earlier; keep it in mind
   when interpreting memory-profile deltas.

Rollback
========
Leave ``VLLM_HCU_K3_SP_MOE`` unset (it defaults to off) or set it to ``0``
and restart the workers.
Nothing is installed in that case, and model state, weights and graphs are
the same as before this patch.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.sp_moe"
TARGETS = (
    f"{TARGET_MODULE}.KimiDecoderLayer.__init__",
    f"{TARGET_MODULE}.KimiDecoderLayer.forward_attn_residual",
)
ENV_SWITCH = "VLLM_HCU_K3_SP_MOE"

# User-visible SP messages are routed through the ``vllm`` root logger.
# ``init_logger(__name__)`` hands back a child logger with NO handler of
# its own: the vLLM dictConfig attaches the StreamHandler to the ``vllm``
# logger only (its own ``propagate`` is False), so INFO-level records from
# the ``vllm_hcu.*`` child never reach any handler and are silently
# dropped -- the enablement banner, the whitelist admission line and the
# clamp line all vanished from server logs even while the patch was fully
# live. ``vllm.logger.logger`` is that configured root instance (and
# carries the same monkey-patched ``info_once``). Lazy import keeps the
# module import free of body-vllm side effects, matching this file's
# style; module-level names stay monkeypatch-able for the test suite.
_INFO_ONCE = None
_WARN = None


def _vllm_info_once(msg: str, *args) -> None:
    if _INFO_ONCE is not None:
        _INFO_ONCE(msg, *args)
        return
    from vllm.logger import logger as _vllogger

    _vllogger.info_once(msg, *args)


def _vllm_warn(msg: str, *args) -> None:
    if _WARN is not None:
        _WARN(msg, *args)
        return
    from vllm.logger import logger as _vllogger

    _vllogger.warning(msg, *args)

_ROLE_SEP_MODULE = "vllm.models.kimi_k3.role_separation"
_MARKER = "_vllm_hcu_kimi_k3_sp_moe_applied"
_INIT_WRAPPER = "_vllm_hcu_kimi_k3_sp_moe_init_wrapper"
_FORWARD_WRAPPER = "_vllm_hcu_kimi_k3_sp_moe_forward_wrapper"
_LAYER_STATE = "_vllm_hcu_kimi_k3_sp_moe_state"
# Tag written onto the layer's MoERunner (layer.mlp.experts) once SP
# enablement succeeds. It marks the only runners whose DeepEP-LL
# dispatch settings this patch is allowed to touch: the same enablement
# clamps THAT runner's ``FusedMoEConfig.max_num_tokens`` to the SP row
# window (see _clamp_ll_dispatch_slots, called by _maybe_enable_layer).
# Runners without the tag keep stock binary config: never clamped, the
# SP code never reads or writes their settings. This is what scopes the
# clamp to Kimi-K3 SP layers -- Kimi-K3 only runs these MoE layers, and
# any other model in-process (or the Kimi MTP / DSpark draft blocks,
# which sp_moe skips by design) never reaches this code.
SP_LL_TAG = "_vllm_hcu_kimi_k3_sp_moe_ll_ok"

# Lower bound for the shrunken LL dispatch budget. The clamp is a
# resource optimization, never an accuracy constraint: if a tiny row
# window would ask for fewer slots than the kernels' smallest sensible
# tile batch, keep enough slots for the hardware instead of chasing the
# exact ratio. TP8 main targets (decode window 8) compute to 16 here and
# are unaffected; the floor only bites hypothetically small windows.
_LL_SLOTS_FLOOR = 16

# Resolved once at apply time when the switch is on.
_role_sep_enabled = None
_role_sep_aux_comm = None

# Latent-width admission for the DeepEP-LL path (Kimi-K3 latent MoE).
#
# Kimi-K3 routes experts in a 3584-wide latent (routed_expert_hidden_size;
# FusedMoEFactory is built with hidden_size=3584 and the checkpoint's
# routed weights are natively 3584-wide -- verified by safetensors shape
# census, no 4096 dimension exists in any routed-expert tensor). The body
# resolves the routed width through a hardcoded SUPPORTED_HIDDEN_SIZES
# whitelist at RoutedExperts construction (maybe_roundup_sizes ->
# maybe_roundup_layer_hidden_size): the stock list stops at 4096, so the
# expert weights AND the forward activation pad are both sized to 4096
# and every routed byte pays a 3584->4096 growth.
#
# With SP enabled we admit 3584 at patch-apply time (before any layer
# constructs its RoutedExperts, so weights width and pad width both
# follow), gated by a byte probe of the deep_ep extension: every
# function family that instantiates a legacy width must also
# instantiate 3584, else the deployment keeps its stock 4096-rounded
# path (verified 3584 kernels exist in this wheel; guards against forks
# whose C++ SWITCH_HIDDEN would host-assert on the width). A per-layer
# verifier re-reads moe_config.hidden_dim after enablement and warns
# when admission landed after expert construction -- that layer keeps
# the padded path and loses the width gain, which must not silently
# contaminate a TPOT A/B run.
_LL_ADMIT_HIDDEN_SIZE = 3584
_LL_ADMIT_MODULES = (
    ("vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll",
     "DeepEPLLPrepareAndFinalize"),
    ("vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep",
     "NixlEPPrepareAndFinalize"),
)


# (family probes at a legacy width, 3584 instantiations any-of) --
# a requirement is ACTIVE only when its function family exists in the
# wheel at a legacy width, so a wheel that predates a function entirely
# is not failed for lacking its 3584 case, while a fork that adds 3584
# to some switch sites but not others is caught. Census on the deployed
# das wheel: dispatch 265 hits, combine 26+26 (LogFMT on/off), sbo 26,
# dispatch_ll_layered family absent (auto-skipped).
_LL_NEEDLE_REQUIREMENTS: tuple[tuple[tuple[bytes, ...], tuple[bytes, ...]], ...] = (
    (
        (b"internode_ll8dispatchILi2048E",),
        (b"internode_ll8dispatchILi3584E",),
    ),
    (
        (b"internode_ll7combineILb0ELi2048E",
         b"internode_ll7combineILb1ELi2048E"),
        (b"internode_ll7combineILb0ELi3584E",
         b"internode_ll7combineILb1ELi3584E"),
    ),
    (
        (b"internode_ll11combine_sboILi2048E",),
        (b"internode_ll11combine_sboILi3584E",),
    ),
    (
        (b"internode_ll19dispatch_ll_layeredILi2048E",),
        (b"internode_ll19dispatch_ll_layeredILi3584E",),
    ),
)


def _scan_ll_needles(data: bytes) -> list[bytes] | None:
    """Analyze extension bytes: which required 3584 needles are missing.

    Returns the missing needles for every ACTIVE requirement family
    ([] = fully provisioned), or ``None`` when the bytes match no legacy
    probe at all -- meaning the binary is unverifiable (stripped symbols,
    compressed device code, different mangling) and no claim about 3584
    support can be made from it either way.
    """
    missing: list[bytes] = []
    active = False
    for family_probes, targets in _LL_NEEDLE_REQUIREMENTS:
        if not any(probe in data for probe in family_probes):
            continue
        active = True
        if not any(target in data for target in targets):
            missing.extend(targets[:1])
    if not active:
        return None
    return missing


def _deep_ep_kernel_needles() -> list[bytes] | None:
    """Missing 3584 LL template instantiations in the deep_ep extension.

    Reads each extension binary once and checks the mangled-name needles
    for every active requirement (see ``_scan_ll_needles``). Returns the
    missing needles, or ``None`` when the extension could not be located,
    read, or recognized at all (the caller then abstains from admission).
    """
    try:
        import deep_ep

        pkg_dir = os.path.dirname(deep_ep.__file__)
        so_paths = [
            os.path.join(pkg_dir, name)
            for name in os.listdir(pkg_dir)
            if name.endswith(".so")
        ]
    except Exception:
        return None
    data = b""
    for so_path in so_paths:
        try:
            with open(so_path, "rb") as handle:
                data += handle.read()
        except OSError:
            continue
    if not data:
        return None
    return _scan_ll_needles(data)


def _admit_latent_hidden_size(logger) -> tuple[int, int]:
    """Add 3584 to the body's DeepEP-LL hidden-size whitelists, in place.

    The lists are mutated (append), never rebound: modules capture the
    same list object in module-level aliases (``_supported_hs``), and an
    in-place append updates every alias at once. Returns
    ``(admitted, modules_searched)``.

    Guard: the body whitelist is not the only width gate. The deep_ep
    C++ dispatches through a hardcoded ``SWITCH_HIDDEN`` case table whose
    ``default`` is a host assert ("Unsupported hidden") -- a wheel built
    from sources without the 3584 case would accept the python admission
    and then abort at the first token. The installed extension is
    therefore probed for the 3584 template instantiations first and the
    admission is refused when any active requirement is missing: the
    deployment keeps its stock gated path (which rounds 3584 up to 4096)
    instead of trading a ValueError for a C++ assert.
    """
    admitted = 0
    missing = _deep_ep_kernel_needles()
    if missing is None:
        logger.warning_once(
            "SP-MoE: the deep_ep extension could not be probed for native "
            "%d-width LL kernels; keeping the stock hidden-size gate.",
            _LL_ADMIT_HIDDEN_SIZE,
        )
        return 0, 0
    if missing:
        logger.warning_once(
            "SP-MoE: deep_ep lacks native 3584 LL kernels (missing %s); "
            "keeping the stock hidden-size gate so the C++ switch cannot "
            "hit its unsupported-hidden assert.",
            ", ".join(item.decode("ascii", "replace") for item in missing),
        )
        return 0, len(_LL_ADMIT_MODULES)
    searched = 0
    for module_name, class_name in _LL_ADMIT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.debug(
                "SP-MoE: %s unavailable; skipping 3584 admission",
                module_name,
                exc_info=True,
            )
            continue
        searched += 1
        cls = getattr(module, class_name, None)
        sizes = getattr(cls, "SUPPORTED_HIDDEN_SIZES", None)
        if not isinstance(sizes, list):
            logger.debug(
                "SP-MoE: %s.%sSUPPORTED_HIDDEN_SIZES missing or not a "
                "list; skipping 3584 admission",
                module_name,
                class_name,
            )
            continue
        if _LL_ADMIT_HIDDEN_SIZE in sizes:
            admitted += 1
            continue
        sizes.append(_LL_ADMIT_HIDDEN_SIZE)
        sizes.sort()
        admitted += 1
        _vllm_info_once(
            "SP-MoE: %s now admits hidden size %d (Kimi-K3 latent width).",
            class_name,
            _LL_ADMIT_HIDDEN_SIZE,
        )
    return admitted, searched


# Whether the whitelist admission took effect at patch-apply time. The
# per-layer verifier only rates the width when this is true; when false
# the deployment is intentionally on its stock rounded path.
_admitted_wide = False


def _verify_latent_width(runner, logger) -> None:
    """Confirm the runner was sized for the native latent width.

    The body resolves the routed width through the LL whitelist at
    RoutedExperts construction time (``maybe_roundup_sizes`` ->
    ``moe_config.hidden_dim``), which sizes expert weights and the
    forward activation pad together. Admission runs at patch-apply time,
    so a still-rounded ``hidden_dim`` means the whitelist change landed
    after (or was shadowed at) expert construction: the layer silently
    keeps the stock padded path. Numerically safe, so rate it once
    instead of raising.
    """
    if not _admitted_wide:
        return
    hidden_dim = getattr(getattr(runner, "moe_config", None), "hidden_dim",
                         None)
    if hidden_dim == _LL_ADMIT_HIDDEN_SIZE:
        logger.debug(
            "SP-MoE: %s sized at native latent width %d.",
            type(runner).__name__,
            _LL_ADMIT_HIDDEN_SIZE,
        )
        return
    logger.warning_once(
        "SP-MoE: %s.moe_config.hidden_dim=%r still rounds past the native "
        "%d latent width; the whitelist change landed after expert "
        "construction, so this layer keeps the stock padded path and "
        "loses the width gain -- exclude it from TPOT comparisons.",
        type(runner).__name__,
        hidden_dim,
        _LL_ADMIT_HIDDEN_SIZE,
    )

_warned_fallback = False


def _switch_enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "0").strip().lower() in ("1", "true")


def _custom_ops_enabled() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.custom_ops_enabled())
    except Exception:
        # Fail closed: without the envs module this is not the HCU plugin
        # runtime that deploys the boltops attn_res kernel.
        return False


class _SpState:
    """Per-layer static SP facts; plain Python, baked into captured graphs."""

    __slots__ = ("tp_size", "tp_rank", "runner", "checked")

    def __init__(self, tp_size: int, tp_rank: int, runner) -> None:
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.runner = runner
        self.checked = False


# ---------------------------------------------------------------------------
# Collectives: phase-routed so no communicator gains a second role.
# ---------------------------------------------------------------------------


def _usable(comm) -> bool:
    return (
        comm is not None
        and not getattr(comm, "disabled", True)
        and getattr(comm, "available", True)
    )


def _phase_comm():
    """PyNccl comm owning this call's role, or None for the group API."""

    if torch.compiler.is_compiling():
        return None
    if torch.cuda.is_current_stream_capturing():
        from vllm.distributed import get_tp_group

        comm = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
        return comm if _usable(comm) else None
    if _role_sep_enabled is None or not _role_sep_enabled():
        return None
    comm = _role_sep_aux_comm()
    return comm if _usable(comm) else None


def _warn_fallback(phase: str) -> None:
    global _warned_fallback
    if not _warned_fallback:
        _warned_fallback = True
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "SP-MoE %s routed through the stock group API (no role comm "
            "available: tracing or role-sep disabled). If VLLM_ROLE_SEP=1 "
            "and this appears outside tracing, RCCL mixed-role hazards are "
            "possible -- pause traffic and check the aux comm.",
            phase,
        )


def _sp_reduce_scatter(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    x = x.contiguous()
    comm = _phase_comm()
    if comm is None:
        _warn_fallback("reduce_scatter")
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        return tensor_model_parallel_reduce_scatter(x, dim=0)
    out = x.new_empty((x.shape[0] // tp_size, *x.shape[1:]))
    comm.reduce_scatter(out, x)
    return out


def _sp_all_gather(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    x = x.contiguous()
    comm = _phase_comm()
    if comm is None:
        _warn_fallback("all_gather")
        from vllm.distributed import tensor_model_parallel_all_gather

        return tensor_model_parallel_all_gather(x, dim=0)
    out = x.new_empty((x.shape[0] * tp_size, *x.shape[1:]))
    comm.all_gather(out, x)
    return out


# ---------------------------------------------------------------------------
# Layer forward.
# ---------------------------------------------------------------------------


def _attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj,
    norm,
    num_valid_blocks: int,
) -> torch.Tensor:
    # Same kernel as op_opt.patch_kimi_k3_reference_paths (the deployed path).
    if num_valid_blocks <= 0:
        return prefix_sum

    from boltops.generic import attn_res

    return attn_res(
        prefix_sum,
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        num_valid_blocks,
        norm.variance_epsilon,
    )


def _check_runner(runner) -> None:
    """Prove the runner performs no TP reduction besides the shared AR."""

    moe_config = runner.moe_config
    problems = []
    if moe_config.tp_size != 1:
        problems.append(f"moe tp_size={moe_config.tp_size} (EP required)")
    if moe_config.is_sequence_parallel:
        problems.append("runner already in upstream sequence-parallel mode")
    if getattr(moe_config, "skip_final_all_reduce", False):
        problems.append("skip_final_all_reduce is set")
    if not runner._fused_output_is_reduced:
        problems.append("combine output is not reduced (DeepEP LL/HT required)")
    # The runner can additionally fuse the shared-partials AR and the routed
    # output into one TP collective (_fuse_kimi_k3_latent_all_reduce), but its
    # own preconditions require tp_size > 1 AND a non-reduced combine here --
    # both of which the checks above already forbid. The attribute on the
    # routed transform is set True by reference_paths on every layer, so it
    # must not be treated as a problem on its own.
    if getattr(runner, "_ep_fused_combine", False):
        problems.append("EP fused combine is enabled")
    if getattr(runner, "enable_dbo", False):
        problems.append("DBO is enabled")
    if problems:
        raise RuntimeError(
            f"{ENV_SWITCH}=1 is incompatible with this MoE runner: "
            + "; ".join(problems)
            + f". Unset {ENV_SWITCH} to restore the replicated MoE path."
        )


def _sp_forward_attn_residual(
    self,
    state: _SpState,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_size = state.tp_size
    num_tokens = hidden_states.shape[0]
    shard = -(-num_tokens // tp_size)
    padded = shard * tp_size
    start = state.tp_rank * shard
    valid = max(0, min(shard, num_tokens - start))

    # Full (replicated) domain: identical to the reference path.
    prefix_sum = hidden_states
    hidden_states = _attn_res(
        prefix_sum,
        block_residual,
        self.self_attention_res_proj,
        self.self_attention_res_norm,
        self.prev_valid_blocks,
    )

    if self.is_block_write_layer:
        block_residual[:, self.block_write_idx, :].copy_(prefix_sum)
        prefix_sum = None

    hidden_states = self.input_layernorm(hidden_states)
    # o_proj.reduce_results is False on SP layers: TP partial sums.
    attn_partial = self._run_self_attn(positions, hidden_states)
    if padded != num_tokens:
        attn_partial = torch.nn.functional.pad(
            attn_partial, (0, 0, 0, padded - num_tokens)
        )
    attn_local = _sp_reduce_scatter(attn_partial, tp_size)

    # Row-sharded domain: local row i is global row start + i.
    if prefix_sum is None:
        prefix_local = attn_local
    elif valid == shard:
        prefix_local = prefix_sum[start : start + shard] + attn_local
    else:
        prefix_local = attn_local.clone()
        if valid:
            prefix_local[:valid].add_(prefix_sum[start : start + valid])

    mlp_valid_blocks = self.prev_valid_blocks + (
        1 if self.is_block_write_layer else 0
    )
    if mlp_valid_blocks <= 0:
        hidden_local = prefix_local
    elif valid == shard:
        hidden_local = _attn_res(
            prefix_local,
            block_residual[start : start + shard],
            self.mlp_res_proj,
            self.mlp_res_norm,
            mlp_valid_blocks,
        )
    else:
        # Padding rows have no block_residual row; keep them zero.
        # Invariant: start + valid <= N because valid = min(shard, N - start),
        # so every block_residual[start:start + valid] below stays in range.
        hidden_local = torch.zeros_like(prefix_local)
        if valid:
            hidden_local[:valid] = _attn_res(
                prefix_local[:valid],
                block_residual[start : start + valid],
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
            )

    hidden_local = self.post_attention_layernorm(hidden_local)
    if not state.checked:
        _check_runner(state.runner)
        state.checked = True
    # The router's VLLM_MOE_SKIP_PADDING mask is a global-row suffix bitmap
    # (model_runner builds it before the layers). topk_ids rows are now local
    # window rows, so republish the mask in window coordinates: pad to full
    # rows with True, then slice this rank's window. Restore right after so
    # the next (non-SP or SP) layer sees the global mask again.
    forward_context = None
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            forward_context = get_forward_context()
            global_is_padding = getattr(forward_context, "is_padding", None)
            if global_is_padding is not None:
                padded_mask = torch.nn.functional.pad(
                    global_is_padding, (0, padded - num_tokens), value=True
                )
                forward_context.is_padding = padded_mask[start : start + shard]
            else:
                forward_context = None
    except ImportError:
        forward_context = None
    try:
        hidden_local = self.mlp(hidden_local)
    finally:
        if forward_context is not None:
            forward_context.is_padding = global_is_padding
    prefix_local = prefix_local + hidden_local

    # Back to the full domain for the next layer.
    prefix_sum = _sp_all_gather(prefix_local, tp_size)
    if padded != num_tokens:
        prefix_sum = prefix_sum[:num_tokens]
    return prefix_sum, block_residual


# ---------------------------------------------------------------------------
# Per-layer enablement (instance level, at model construction).
# ---------------------------------------------------------------------------


def _find_o_proj(self_attn: torch.nn.Module, tp_size: int):
    found = {
        id(module): module
        for name, module in self_attn.named_modules()
        if name.rsplit(".", 1)[-1] == "o_proj" and hasattr(module, "reduce_results")
    }
    if len(found) != 1:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: expected exactly one o_proj RowParallelLinear in "
            f"{type(self_attn).__name__}, found {len(found)}"
        )
    (o_proj,) = found.values()
    if "RowParallelLinear" not in {cls.__name__ for cls in type(o_proj).__mro__}:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: o_proj is {type(o_proj).__name__}, not RowParallelLinear"
        )
    if o_proj.reduce_results is not True or getattr(o_proj, "tp_size", tp_size) != tp_size:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: o_proj must be a TP-{tp_size} reducing RowParallelLinear"
        )
    return o_proj


def _wrap_shared_experts(shared: torch.nn.Module, tp_size: int) -> None:
    down_proj = getattr(shared, "down_proj", None)
    if getattr(down_proj, "reduce_results", None) is not False:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: shared expert down_proj must emit TP partial sums"
        )
    # KimiMLP.forward has no x_and_scale_quanted parameter, so the HCU
    # SharedExperts fast path (_run_layer with a pre-quantized tuple) is
    # structurally unreachable for this layer: it always calls
    # self._layer(x), i.e. exactly the instance forward wrapped below.
    original_forward = shared.forward

    def sp_shared_forward(x: torch.Tensor) -> torch.Tensor:
        # RS(MLP_r(AG(x_local))) == rows of AR(MLP_r(x_full)) for this rank.
        partial = original_forward(_sp_all_gather(x, tp_size))
        return _sp_reduce_scatter(partial, tp_size)

    shared.forward = sp_shared_forward


def _clamp_ll_dispatch_slots(runner, tp_size: int, logger) -> None:
    """Shrink the latched layer's LL dispatch budget to the SP row window.

    Called from ``_maybe_enable_layer`` immediately AFTER ``SP_LL_TAG``
    is set, so no code path can observe a clamped-but-untagged or
    tagged-but-unclamped runner: the two flips are one enablement
    transaction on this single-threaded construction path.

    ``FusedMoEConfig.max_num_tokens`` is the single seed field of the
    DeepEP-LL chain: the all2all args derive
    ``max_num_tokens_per_dp_rank`` / ``max_tokens_per_rank`` from it,
    and the masked-GEMM ``estimate_expected_m`` clamps through the same
    value. SP dispatches ``ceil(N/tp_size)`` rows per rank, so a buffer
    sized for ``N`` wastes ``1 - 1/tp_size`` of its slots. The clamp
    runs during layer enablement, before the MoE factory builds the LL
    handle and workspace from this config, so every derived buffer is
    sized once, correctly.

    Scope: this runner was located from a ``use_attn_residuals`` Kimi
    MoE decoder layer and has just been latched, so the clamp applies
    only to Kimi-K3 SP layers. Any other model's MoE layers in-process,
    and the Kimi MTP / DSpark draft blocks (sp_moe deliberately skips
    them), never reach this function and keep the stock config.

    The function shares the ``FusedMoEConfig`` instance with the MoE
    factory: mutating the attribute in place (never rebinding) is what
    propagates the new slot count to the LL handle and the expected_m
    clamp on the same object.
    """
    cfg = getattr(runner, "moe_config", None)
    stock = getattr(cfg, "max_num_tokens", None)
    if not isinstance(stock, int) or isinstance(stock, bool) or stock <= 1:
        logger.warning_once(
            "Kimi-K3 SP-MoE: runner %r exposes no usable max_num_tokens "
            "(%r); LL dispatch slots left at stock.",
            type(runner).__name__,
            stock,
        )
        return
    shrunk = max(_LL_SLOTS_FLOOR, -(-stock // tp_size))
    if shrunk >= stock:
        return
    cfg.max_num_tokens = shrunk
    _vllm_info_once(
        "Kimi-K3 SP-MoE: LL dispatch slots clamped %d -> %d "
        "(ceil(max_num_tokens/%d), floor %d).",
        stock,
        shrunk,
        tp_size,
        _LL_SLOTS_FLOOR,
    )


def _maybe_enable_layer(layer, decoder_cls, logger) -> _SpState | None:
    if not getattr(layer, "use_attn_residuals", False):
        return None
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is None or layer.mlp is not moe:
        return None

    from vllm.config import get_current_vllm_config
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None

    parallel_config = get_current_vllm_config().parallel_config
    problems = []
    if not _custom_ops_enabled():
        problems.append(
            "HCU custom ops are disabled; the SP layer body needs the "
            "boltops attn_res kernel deployed by worker.op_opt."
            "kimi_k3.reference_paths"
        )
    if not parallel_config.use_sequence_parallel_moe:
        problems.append("requires EP + all2all backend + TP>1 + DP>1")
    if getattr(parallel_config, "use_ubatching", False):
        problems.append("DBO/ubatching is unsupported")
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        problems.append("prefill context parallel is unsupported")
    if getattr(parallel_config, "decode_context_parallel_size", 1) != 1:
        problems.append("decode context parallel is unsupported")
    if (
        _role_sep_enabled is not None
        and _role_sep_enabled()
        and not _usable(_role_sep_aux_comm())
    ):
        problems.append(
            "VLLM_ROLE_SEP=1 but the auxiliary eager comm is unavailable; "
            "eager-phase SP collectives would fall back into the captured "
            "comm (RCCL mixed-role hazard)"
        )
    if not getattr(vars(decoder_cls).get("forward_attn_residual"), _FORWARD_WRAPPER, False):
        problems.append("KimiDecoderLayer.forward_attn_residual was re-patched after SP-MoE")
    if problems:
        raise RuntimeError(
            f"{ENV_SWITCH}=1 cannot be applied: " + "; ".join(problems)
        )

    from vllm_hcu.model_executor.layers.fused_moe.moe_runner import MoERunner

    runner = moe.experts
    if not isinstance(runner, MoERunner):
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: KimiMoE.experts is {type(runner).__name__}, "
            "not the HCU MoERunner"
        )
    require_callable(
        runner,
        "_maybe_reduce_shared_expert_output",
        "MoERunner._maybe_reduce_shared_expert_output",
    )
    shared = getattr(moe, "shared_experts", None)
    runner_shared = getattr(runner, "_shared_experts", None)
    if (shared is None) != (runner_shared is None) or (
        shared is not None and runner_shared._layer is not shared
    ):
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: runner shared expert is not KimiMoE.shared_experts"
        )

    o_proj = _find_o_proj(layer.self_attn, tp_size)
    o_proj.reduce_results = False
    if shared is not None:
        _wrap_shared_experts(shared, tp_size)

        def shared_output_already_sharded(shared_output, fused_output_is_reduced=None):
            return shared_output

        runner._maybe_reduce_shared_expert_output = shared_output_already_sharded

    _vllm_info_once(
        "Kimi-K3 SP-MoE enabled (%s=1): o_proj reduce-scatter, MoE dispatch "
        "of ceil(N/%d) rows per TP rank, all-gather after MoE.",
        ENV_SWITCH,
        tp_size,
    )
    # The enablement outcome is now final and irreversible (this is
    # server construction, single-shot, no disable path): latch the tag
    # first so the runner is identifiable, then shrink ITS OWN config to
    # the SP window. One transaction: no observer can see the runner
    # tagged-but-unclamped; nothing else can ever clamp an untagged one.
    setattr(moe.experts, SP_LL_TAG, True)
    _clamp_ll_dispatch_slots(runner, tp_size, logger)
    _verify_latent_width(runner, logger)
    return _SpState(tp_size, get_tensor_model_parallel_rank(), runner)


# ---------------------------------------------------------------------------
# Patch entry points.
# ---------------------------------------------------------------------------


def _resolve_role_sep() -> None:
    """Bind the role-separation accessors of this deployment's authority.

    Branch-dependent: the d-lwc line ships role separation inside the plugin
    (``vllm_hcu.patch.worker.framework_opt._role_separation``, ported from
    the body-side fix, public ``role_separation_enabled``), while other
    trees carry it in the vLLM body
    (``vllm.models.kimi_k3.role_separation``, ``_role_separation_enabled``).
    The plugin module is preferred: binding the body module when the plugin
    also builds its auxiliary comm would create a SECOND eager NCCL
    communicator for the SP collectives to ride. When neither module exists
    the deployment has no role separation at all, so the stock single-comm
    group API is a consistent destination and we degrade with a warning
    instead of failing worker patch loading.
    """
    global _role_sep_enabled, _role_sep_aux_comm

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    try:
        plugin = importlib.import_module(
            "vllm_hcu.patch.worker.framework_opt._role_separation"
        )
        enabled = getattr(plugin, "role_separation_enabled", None)
        aux = getattr(plugin, "_get_aux_ar_comm", None)
        if callable(enabled) and callable(aux):
            _role_sep_enabled = enabled
            _role_sep_aux_comm = aux
            return
    except Exception:
        logger.debug(
            "SP-MoE: plugin role-separation module unavailable", exc_info=True
        )

    try:
        body = sys.modules.get(_ROLE_SEP_MODULE)
        if body is None:
            body = importlib.import_module(_ROLE_SEP_MODULE)
        enabled = getattr(body, "_role_separation_enabled", None)
        aux = getattr(body, "_get_aux_ar_comm", None)
        if callable(enabled) and callable(aux):
            _role_sep_enabled = enabled
            _role_sep_aux_comm = aux
            return
    except Exception:
        logger.debug(
            "SP-MoE: body role-separation module unavailable", exc_info=True
        )

    logger.warning_once(
        "SP-MoE: no role-separation module found (plugin or vLLM body); "
        "SP collectives will use the stock group API on the shared TP "
        "communicator, matching this deployment's no-role-sep topology."
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    decoder = require_class(target, "KimiDecoderLayer", f"{TARGET_MODULE}.KimiDecoderLayer")
    state = getattr(target, _MARKER, None)
    if state is not None:
        if state == "enabled" and not (
            getattr(vars(decoder).get("__init__"), _INIT_WRAPPER, False)
            and getattr(vars(decoder).get("forward_attn_residual"), _FORWARD_WRAPPER, False)
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[1]} is stale; restart the process"
            )
        return False
    if not _switch_enabled():
        setattr(target, _MARKER, "disabled")
        return True

    original_init = require_callable(decoder, "__init__", TARGETS[0])
    original_forward = require_callable(decoder, "forward_attn_residual", TARGETS[1])
    # inspect.signature follows __wrapped__, so against the
    # reference_paths wrapper this resolves to the upstream True signature.
    require_exact_signature(
        original_forward,
        TARGETS[1],
        positional=("self", "positions", "hidden_states", "block_residual"),
    )
    require_callable(decoder, "_run_self_attn", f"{TARGET_MODULE}.KimiDecoderLayer._run_self_attn")
    _resolve_role_sep()

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    # Admit the Kimi latent width (3584) into the DeepEP-LL whitelists so
    # the LL path can run the native width under SP. Done once per
    # process, at patch-apply time -- this MUST land before any layer
    # constructs its RoutedExperts, because the body resolves the routed
    # width through the whitelist then (maybe_roundup_sizes ->
    # moe_config.hidden_dim weights + forward pad).
    global _admitted_wide
    admitted, searched = _admit_latent_hidden_size(logger)
    _admitted_wide = admitted > 0
    if not _admitted_wide and searched == 0:
        _vllm_warn(
            "SP-MoE: no DeepEP-LL prepare/finalize whitelist module was "
            "found; the %d-wide latent path keeps whatever gate this "
            "deployment ships.",
            _LL_ADMIT_HIDDEN_SIZE,
        )

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        setattr(self, _LAYER_STATE, _maybe_enable_layer(self, decoder, logger))

    @functools.wraps(original_forward)
    def hcu_forward_attn_residual(self, positions, hidden_states, block_residual):
        sp_state = getattr(self, _LAYER_STATE, None)
        if sp_state is None:
            return original_forward(self, positions, hidden_states, block_residual)
        return _sp_forward_attn_residual(
            self, sp_state, positions, hidden_states, block_residual
        )

    setattr(hcu_init, _INIT_WRAPPER, True)
    setattr(hcu_forward_attn_residual, _FORWARD_WRAPPER, True)
    setattr(decoder, "_vllm_hcu_sp_moe_original_init", original_init)
    setattr(decoder, "_vllm_hcu_sp_moe_original_forward_attn_residual", original_forward)
    decoder.__init__ = hcu_init
    decoder.forward_attn_residual = hcu_forward_attn_residual
    setattr(target, _MARKER, "enabled")
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["ENV_SWITCH", "PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]


# ---------------------------------------------------------------------------
# Acceptance criteria (not run here; for the validation deployment).
#
# A1 dispatch rows: with VLLM_HCU_K3_SP_MOE=1, DeepEP LL dispatch per rank goes
#    from N to ceil(pad(N)/4) rows, where pad(N) = 4*ceil(N/4). The
#    is_padding republication below is REQUIRED for imbalanced DP steps:
#    without it, every rank dispatches its padding rows and an 8-real-token
#    step in a 128-row graph would dispatch 104 rows instead of 32 (and 8
#    after the mask fix). Measure topk_ids.size(0) / DeepEP send counts on
#    both balanced and imbalanced DP steps.
# A2 numerics: same prompts with greedy sampling, switch 0 vs 1. Per-layer
#    prefix_sum and final logits must match within bf16 tolerance (AR vs
#    RS+AG ordering only). Greedy tokens should match, and spec-decode
#    acceptance rate should not change. Include a prefill with N % 4 != 0
#    and a DP dummy batch with N=1.
# A3 graphs: with VLLM_USE_BREAKABLE_CUDAGRAPH=1, capture every decode size
#    with the switch on, then replay mixed batch sizes. No shape mismatch, no
#    capture error, and the output for each size equals an eager run of that
#    size. All SP shapes are Python ints of N and the static TP rank.
# A4 role separation and soak: with VLLM_ROLE_SEP=1 and
#    VLLM_ROLE_SEP_STRICT=1, run NCCL_DEBUG=INFO with collective tracing.
#    Every eager ReduceScatter and AllGather issued by this layer must be on
#    the aux comm, every captured one on the TP pynccl comm, and the c10d TP
#    comm must carry no new collectives. Then soak a long prefill/decode
#    interleave (hours) with no RCCL hang: the shared-expert AG/RS now run
#    between the captured DeepEP send and recv kernels, which is a new
#    position for RCCL work on these comms.
# ---------------------------------------------------------------------------
