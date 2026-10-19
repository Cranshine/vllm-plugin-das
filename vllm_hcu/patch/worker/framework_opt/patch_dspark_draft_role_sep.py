# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep the DSpark draft-logits all_gather out of the captured draft graph.

The only collective inside the DSpark draft graph is the draft-logits c10d
all_gather, which shares the TP communicator with the eager-phase
all_gathers outside the graphs (target logits + draft tail). That
captured-AG x eager-AG same-communicator pair deadlocks RCCL on DCUs (all
16 ranks spinning in ``runRing<hip_bfloat16, ...>`` until watchdog kill).

With role separation the captured draft graph stops at the sharded lm_head
GEMM (a persistent-buffer write, collective-free); the all_gather plus the
Markov sampling loop run eagerly right after each FULL replay via a
DSpark-owned instance patch on the draft cudagraph manager's
``run_fullgraph``. Capture is detected with
``torch.cuda.is_current_stream_capturing()`` and NOT from
``cudagraph_runtime_mode`` alone: the capture driver calls
``forward_fn(CUDAGraphMode.NONE)`` inside the ``torch.cuda.graph`` block,
so keying off the mode left the branch dead during capture.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch
from vllm.config.compilation import CUDAGraphMode

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_class,
    require_callable,
    require_exact_signature,
    require_unpatched,
)
from ._role_separation import raise_or_warn, role_separation_enabled

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.framework_opt.dspark.draft_role_separation"
TARGETS = (
    f"{TARGET_MODULE}.DSparkSpeculator.__init__",
    f"{TARGET_MODULE}.DSparkSpeculator.load_draft_model",
    f"{TARGET_MODULE}.DSparkSpeculator.init_cudagraph_manager",
    f"{TARGET_MODULE}.DSparkSpeculator._generate_draft",
)
_MARKER = "_vllm_hcu_dspark_draft_role_sep_applied"
_WRAPPER = "_vllm_hcu_dspark_draft_role_sep_wrapper"
_TAIL_WRAPPER = "_vllm_hcu_dspark_replay_tail_wrapper"


def _guard_captured_draft_all_gather(where: str) -> None:
    msg = (
        "DSpark: the draft-logits all_gather would stay inside the captured "
        "draft graph (%s). c10d already carries the per-step eager "
        "all_gathers, so this re-creates the captured-x-eager same-"
        "communicator pair that deadlocks RCCL. Fix the cause "
        "(VLLM_ROLE_SEP=1 + a sharded, non-replicated lm_head with "
        "compute_draft_logits_shard), or set VLLM_ROLE_SEP_STRICT=0 to "
        "downgrade this to a warning." % where
    )
    raise_or_warn(msg)


def _mro_original(cls: type, name: str):
    for klass in cls.__mro__:
        function = klass.__dict__.get(name)
        if function is not None:
            if getattr(function, _WRAPPER, False):
                raise PatchCompatibilityError(
                    f"target {name} on the {TARGET_MODULE} chain is already "
                    "wrapped; refusing to double-wrap"
                )
            return function
    raise PatchCompatibilityError(
        f"class chain of the {TARGET_MODULE} target has no {name}"
    )


def _hcu_role_sep_init(self, vllm_config, device):
    _hcu_role_sep_init._hcu_original(self, vllm_config, device)
    # Role separation state. Keep the draft-logits all_gather out of the
    # captured draft graph: the graph stops at the sharded lm_head GEMM
    # (a persistent-buffer write); the AG + Markov sampling run eagerly
    # after each replay (see init_cudagraph_manager).
    from vllm.distributed import get_tensor_model_parallel_world_size

    self._ag_out_of_graph = bool(
        role_separation_enabled()
        and get_tensor_model_parallel_world_size() > 1
    )
    # Width of the per-rank logits shard. Exact value (the lm_head's
    # num_embeddings_per_partition) is read once the draft model exists.
    self._draft_vocab_size = (
        getattr(self.draft_model_config.hf_config, "draft_vocab_size", None)
        or self.vocab_size
    )
    self._logits_shard_width = None
    self.draft_logits_shard = None


def _hcu_role_sep_load_draft_model(self, target_model, target_attn_layer_names):
    model = _hcu_role_sep_load_draft_model._hcu_original(
        self, target_model, target_attn_layer_names
    )
    if self._ag_out_of_graph:
        shard_fn = getattr(model, "compute_draft_logits_shard", None)
        lm_head = getattr(model, "lm_head", None)
        shard_width = getattr(lm_head, "num_embeddings_per_partition", None)
        if (
            shard_fn is None
            or shard_width is None
            or getattr(lm_head, "replicated", False)
        ):
            # Shard API missing or the head is replicated (no AG to split
            # out): the legacy in-graph all_gather would come back -- the
            # captured-x-eager same-communicator pair with it.
            _guard_captured_draft_all_gather(
                "draft model has no compute_draft_logits_shard with a "
                "non-replicated lm_head"
            )
            self._ag_out_of_graph = False
        else:
            self._logits_shard_width = shard_width
            self.draft_logits_shard = torch.zeros(
                self.max_num_tokens,
                self._logits_shard_width,
                dtype=self.dtype,
                device=self.device,
            )
    return model


def _hcu_role_sep_init_cudagraph_manager(self, cudagraph_mode):
    _hcu_role_sep_init_cudagraph_manager._hcu_original(self, cudagraph_mode)
    if not self._ag_out_of_graph:
        return
    mgr = self.query_cudagraph_manager
    assert mgr is not None
    original_replay = mgr.run_fullgraph
    if getattr(original_replay, _TAIL_WRAPPER, False):
        return

    def run_fullgraph_with_tail(desc, _orig=original_replay):
        _orig(desc)
        # The captured draft graph ends at the sharded-logits write
        # (collective-free capture); finish the draft step eagerly, still
        # before propose() returns draft_tokens.
        num_reqs = desc.num_reqs
        if num_reqs is None:
            # Same padding rule the capture driver uses (gpu/cudagraph_utils
            # create_forward_fn), so the eager tail reads exactly the rows
            # the graph wrote.
            num_reqs = min(desc.num_tokens, self.max_num_reqs)
        self._gather_and_markov(num_reqs)

    setattr(run_fullgraph_with_tail, _TAIL_WRAPPER, True)
    # DSpark-owned instance patch: only THIS draft graph's replay grows the
    # eager tail. DFlash and every other speculator keep their original
    # replay path untouched.
    mgr.run_fullgraph = run_fullgraph_with_tail


def _hcu_markov_sample(self, num_reqs, base_logits) -> None:
    # Sequential Markov sampling loop shared by the sequential paths. No
    # collectives (the Markov head is replicated); safe to run eagerly or
    # inside a graph.
    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec

    idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
    confidence_markov_embeds = []

    # Anchor (bonus) token per request = the input id at query offset 0,
    # read via the precomputed persistent index (fixed buffer for capture).
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

    for i in range(n_spec):
        # Sequential stage: Markov bias from the previously sampled token.
        markov_embed = self.model.markov_embed(prev)
        if self.enable_adaptive_verification:
            confidence_markov_embeds.append(markov_embed)
        bias = self.model.markov_bias(markov_embed)
        logits_i = base_logits[:, i] + bias
        draft_sampled_i = self._sample_logits(
            logits_i, idx_map[:, i], sample_pos[:, i], i
        )
        self.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i


def _hcu_shard_logits_in_graph(self, num_reqs, head_hidden) -> None:
    """Graph-captured segment, collective-free: write the per-rank draft
    logits shard into a persistent buffer. Replaces the captured
    all_gather inside compute_draft_logits. Runs during capture AND on
    every FULL replay."""
    assert self.draft_logits_shard is not None
    num_sample = num_reqs * self.num_speculative_steps
    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
    shard = self.model.compute_draft_logits_shard(sample_hidden)
    assert shard.shape[-1] == self._logits_shard_width, (
        f"lm_head shard width {shard.shape[-1]} != expected "
        f"{self._logits_shard_width}"
    )
    self.draft_logits_shard[: shard.shape[0]].copy_(shard)


def _hcu_gather_and_markov(self, num_reqs: int) -> None:
    """Eager companion of _shard_logits_in_graph: all_gather the sharded
    draft logits (identical call to the eager LogitsProcessor path) and
    run the Markov sampling loop. Executed after each FULL replay."""
    if torch.cuda.is_current_stream_capturing():
        # The whole point of this method is to run OUTSIDE the graph; if
        # it ever runs during a capture, the all_gather is baked in again.
        _guard_captured_draft_all_gather(
            "_gather_and_markov executed while capturing"
        )
    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec
    assert self.draft_logits_shard is not None
    shard = self.draft_logits_shard[:num_sample]
    # Same all_gather, same shapes, same dim semantics as the eager
    # LogitsProcessor gather path -> bit-identical logits.
    from vllm.distributed import tensor_model_parallel_all_gather

    full = tensor_model_parallel_all_gather(shard, dim=-1)
    vocab_size = self._draft_vocab_size
    base_logits = full[..., :vocab_size].view(num_reqs, n_spec, vocab_size)
    self._markov_sample(num_reqs, base_logits)


def _hcu_role_sep_generate_draft(
    self,
    num_reqs: int,
    num_tokens_padded: int,
    attn_metadata,
    slot_mappings,
    num_tokens_across_dp,
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
):
    # Full method rewrite (the audited original samples internally and
    # returns None); the inherited _run_model keeps backbone semantics.
    head_hidden = self._run_model(
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
    )
    capturing = torch.cuda.is_current_stream_capturing()
    in_full_graph = capturing or (
        cudagraph_runtime_mode == CUDAGraphMode.FULL
    )
    if self._ag_out_of_graph and in_full_graph:
        # Collective-free capture: stop the graph at the sharded logits
        # write; the all_gather + Markov sampling execute eagerly right
        # after each replay (run_fullgraph_with_tail in
        # init_cudagraph_manager).
        self._shard_logits_in_graph(num_reqs, head_hidden)
    else:
        if in_full_graph:
            # Reached when the shard path is unavailable: this enqueue
            # happens *inside* the capture, so the all_gather would be
            # baked into the graph.
            _guard_captured_draft_all_gather(
                "_sample_sequential while capturing the draft FULL graph"
            )
        self._sample_sequential(num_reqs, head_hidden)


def _install_wrapped(speculator_cls, name, original, replacement) -> None:
    replacement._hcu_original = original

    @functools.wraps(original)
    def hcu_wrapped(self, *args, **kwargs):
        return replacement(self, *args, **kwargs)

    setattr(hcu_wrapped, _WRAPPER, True)
    setattr(speculator_cls, name, hcu_wrapped)


def apply_to_module(module: ModuleType) -> bool:
    spec = load_exact_module(TARGET_MODULE, module)
    speculator_cls = require_class(
        spec, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator"
    )
    wrapped = (
        (speculator_cls, "__init__", TARGETS[0], _WRAPPER),
        (speculator_cls, "load_draft_model", TARGETS[1], _WRAPPER),
        (speculator_cls, "init_cudagraph_manager", TARGETS[2], _WRAPPER),
        (speculator_cls, "_generate_draft", TARGETS[3], _WRAPPER),
    )
    if already_applied(spec, _MARKER, wrapped):
        return False

    init_original = require_callable(speculator_cls, "__init__", TARGETS[0])
    require_exact_signature(
        init_original, TARGETS[0], positional=("self", "vllm_config", "device")
    )
    load_original = require_unpatched(
        speculator_cls, "load_draft_model", TARGETS[1], _WRAPPER
    )
    require_exact_signature(
        load_original,
        TARGETS[1],
        positional=("self", "target_model", "target_attn_layer_names"),
    )
    # The DSpark class must NOT own init_cudagraph_manager yet; the audited
    # original lives on a superclass (DFlashSpeculator).
    if "init_cudagraph_manager" in speculator_cls.__dict__:
        raise PatchCompatibilityError(
            f"{TARGETS[2]} is defined on the DSpark target itself; the "
            "audited delegation expects the DFlashSpeculator implementation"
        )
    mgr_original = _mro_original(speculator_cls, "init_cudagraph_manager")
    require_exact_signature(
        mgr_original,
        TARGETS[2],
        positional=("self", "cudagraph_mode"),
    )
    draft_original = require_unpatched(
        speculator_cls, "_generate_draft", TARGETS[3], _WRAPPER
    )
    require_exact_signature(
        draft_original,
        TARGETS[3],
        positional=(
            "self",
            "num_reqs",
            "num_tokens_padded",
            "attn_metadata",
            "slot_mappings",
            "num_tokens_across_dp",
            "cudagraph_runtime_mode",
        ),
        defaults={"cudagraph_runtime_mode": CUDAGraphMode.NONE},
    )
    for name, new_api in (
        ("_markov_sample", f"{TARGET_MODULE}.DSparkSpeculator._markov_sample"),
        (
            "_shard_logits_in_graph",
            f"{TARGET_MODULE}.DSparkSpeculator._shard_logits_in_graph",
        ),
        (
            "_gather_and_markov",
            f"{TARGET_MODULE}.DSparkSpeculator._gather_and_markov",
        ),
    ):
        if hasattr(speculator_cls, name):
            raise PatchCompatibilityError(
                f"required-new API {new_api} already exists on the target "
                "vLLM module; the audited migration is obsolete"
            )

    _install_wrapped(speculator_cls, "__init__", init_original, _hcu_role_sep_init)
    _install_wrapped(
        speculator_cls,
        "load_draft_model",
        load_original,
        _hcu_role_sep_load_draft_model,
    )
    _install_wrapped(
        speculator_cls,
        "init_cudagraph_manager",
        mgr_original,
        _hcu_role_sep_init_cudagraph_manager,
    )
    _install_wrapped(
        speculator_cls,
        "_generate_draft",
        draft_original,
        _hcu_role_sep_generate_draft,
    )
    # New stateless helpers: plain functions so normal bound-method lookup
    # applies on every instance.
    speculator_cls._markov_sample = _hcu_markov_sample
    speculator_cls._shard_logits_in_graph = _hcu_shard_logits_in_graph
    speculator_cls._gather_and_markov = _hcu_gather_and_markov
    setattr(spec, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
