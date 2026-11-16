# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fast, batch-capable encode for the Kimi tiktoken slow wrapper.

Why (nmz22 K3.1-Pro 12L serve, 2026-11-16): TTFT of a 16x128k fake-prefill
bench is dominated by server-side tokenization of ~500 KB prompt texts.
The K3.1 repo ships no ``tokenizer.json``; ``TikTokenTokenizer`` is a
``PreTrainedTokenizer`` whose ``__call__`` burns ~0.41 s per 128k text in
HF slow-tokenizer ceremony (``tokenize()`` regex split 0.31 s +
``convert_tokens_to_ids`` dict walk 0.16 s) while the underlying tiktoken
Rust core needs only ~0.06 s (live A/B: outputs token-identical through
every intercepted call shape).

vLLM's own parallelism knob cannot help: ``maybe_make_thread_pool``
deep-copies *fast* tokenizers only and is a no-op for this wrapper, and
``TOKENIZERS_PARALLELISM`` only reaches the HF fast (Rust ``tokenizers``)
batch path which this model never uses.

Intercepted call shapes (token-exact to stock, verified by self-check):

* plain ``str`` and ``list[str]`` batches whose kwargs stay inside the
  allowlist (``add_special_tokens``/``truncation``/``max_length``) --
  exactly what the OpenAI server's async microbatch tokenizer issues.
* Truncation follows stock semantics incl. the HF compatibility rule
  that ``max_length`` without explicit ``truncation`` implies
  ``"longest_first"``.  Manual ``ids[:max_length]`` is only faithful on
  ``truncation_side="right"``; anything else delegates to stock.
* Every other shape (``padding``, ``return_tensors``, text pairs,
  non-str items) goes to the original implementation unchanged.

Per-instance state is read through ``self`` inside the patched methods
(``self.model``, the instance splitter, ``self.truncation_side``), so
deep-copies and reloads never inherit a stale encoder.

Parallelism is flat, never nested: all (text, chunk) pieces of one call
are one level of sibling tasks on a module-level pool over the shared
Rust core (``Encoding.encode`` is ``&self`` thread-safe).  No pool task
submits further tasks, so a saturated pool cannot self-deadlock.
``submit`` failure (shutdown / fork race) degrades to serial encode;
the pool is rebuilt in forked children with fresh locks.

Arming is gated by a self-check (probe strings with structural markers,
reserved tokens, mixed scripts; both ``allow_special_tokens`` modes)
and an atomic class marker.  Kill switch: ``VLLM_HCU_TIKTOKEN_FAST=0``.

Hook point: ``vllm.tokenizers.registry``'s ``load_tokenizer`` wraps
every tokenizer construction on serve startup.
"""
from __future__ import annotations

import functools
import logging
import threading
from types import ModuleType
from typing import Any

PATCH_ID = "platform.core.kimi_tiktoken_fast_encode"
TARGET_MODULE = "vllm.tokenizers.registry"
TARGETS = (f"{TARGET_MODULE}.TokenizerRegistry",
           "vllm.tokenizers.hf.get_cached_tokenizer")

_MARKER = "_k3_tiktoken_fast_encode_armed"
_DISABLE_ENV = "VLLM_HCU_TIKTOKEN_FAST"
_POOL_MAX = 4
_CLASS_NAME = "TikTokenTokenizer"
_MODULE_HINT = "tokenization_kimi"

# single-sequence strategies whose effect is exactly ids[:max_length]
_TRUNCATION_STRATEGIES = (True, "longest_first", "only_first")

_arm_lock = threading.Lock()
_pool_lock = threading.Lock()
_pool_executor = None
_fork_registered = False


def _enabled() -> bool:
    import os

    return os.environ.get(_DISABLE_ENV, "").strip().lower() not in (
        "0", "false", "off", "no")


def _debug() -> bool:
    import os

    return os.environ.get("VLLM_HCU_TIKTOKEN_FAST_DEBUG", "") == "1"


def _reset_pool() -> None:
    """Post-fork child reset: fresh pool and fresh locks (a lock held by
    another thread at fork time would stay locked forever in the child)."""

    global _pool_executor, _pool_lock
    _pool_executor = None
    _pool_lock = threading.Lock()


def _register_fork_reset() -> None:
    global _fork_registered
    if _fork_registered:
        return
    import os

    try:
        os.register_at_fork(after_in_child=_reset_pool)
        _fork_registered = True
    except (AttributeError, OSError):  # pragma: no cover
        pass


def _get_pool():
    with _pool_lock:
        global _pool_executor
        if _pool_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            _pool_executor = ThreadPoolExecutor(
                max_workers=_POOL_MAX,
                thread_name_prefix="k3-tiktoken",
            )
            _register_fork_reset()
        return _pool_executor


def apply_to_module(module: ModuleType) -> None:
    log = logging.getLogger("vllm.hcu.tiktoken_fast_encode")

    import inspect

    registry = getattr(module, "TokenizerRegistry", None)
    original = getattr(registry, "load_tokenizer", None)
    if registry is None or original is None:
        log.warning("[tiktoken-fast] registry shape unexpected; skip")
        return
    if getattr(original, _MARKER, False):
        return

    holder = registry if isinstance(registry, type) else type(registry)

    # unwrap staticmethod/classmethod/bound layers to the raw function so
    # the wrapper (stored as a plain class attribute) binds args exactly
    # like the original did — no double self, no unhashable tokenizer_mode
    raw_descriptor = inspect.getattr_static(registry, "load_tokenizer")
    original = raw_descriptor
    while isinstance(original, (classmethod, staticmethod)):
        original = original.__func__
    original = getattr(original, "__func__", original)  # bound -> raw

    @functools.wraps(original)
    def load_and_arm(*args, **kwargs):
        # stored as a plain class-attribute function: python binds the
        # instance naturally, so args flow to the raw function unchanged
        tokenizer = original(*args, **kwargs)
        try:
            if not _enabled():
                log.info("[tiktoken-fast] kill switch %s=0 set; hook idle",
                         _DISABLE_ENV)
            elif _arm_tokenizer(tokenizer, log):
                log.info(
                    "[tiktoken-fast] armed for %s "
                    "(kill switch: %s=0, pool<=%d threads)",
                    args[-1] if args else kwargs.get("tokenizer_mode", "?"),
                    _DISABLE_ENV, _POOL_MAX)
            else:
                log.info(
                    "[tiktoken-fast] no armable TikTokenTokenizer "
                    "(mro=%s)", [k.__name__ for k in type(tokenizer).__mro__])
        except Exception:  # pragma: no cover - fail open to stock encoder
            log.exception("[tiktoken-fast] arming failed; stock path kept")
        return tokenizer

    setattr(load_and_arm, _MARKER, True)
    setattr(holder, "load_tokenizer", load_and_arm)
    log.info("[tiktoken-fast] hook installed on %s.load_tokenizer",
             holder.__name__)

    # ---- second hook: the path the OpenAI server actually uses ----
    # vllm 0.28 get_tokenizer only consults load_tokenizer_CLS and then
    # calls cls.from_pretrained itself; api/server tokenizers come through
    # vllm.tokenizers.hf.get_cached_tokenizer.  Wrap the cached-wrapper
    # factory: it receives a live tokenizer instance before caching, which
    # is the ideal arming point (instance-level, error-safe, idempotent).
    from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR

    hf_mod_name = "vllm.tokenizers.hf"

    def _patch_hf_cached_factory(hf_module: ModuleType) -> None:
        factory = getattr(hf_module, "get_cached_tokenizer", None)
        if factory is None or getattr(factory, _MARKER, False):
            return
        hf_raw = factory
        while isinstance(hf_raw, (classmethod, staticmethod)):
            hf_raw = hf_raw.__func__
        hf_raw = getattr(hf_raw, "__func__", hf_raw)  # bound -> raw

        @functools.wraps(hf_raw)
        def cached_factory_wrapper(*args, **kwargs):
            tokenizer = hf_raw(*args, **kwargs)
            try:
                if _enabled():
                    _arm_tokenizer(tokenizer, log)
            except Exception:  # never break tokenizer construction
                log.exception("[tiktoken-fast] arming failed; stock kept")
            return tokenizer

        setattr(cached_factory_wrapper, _MARKER, True)
        # vllm.tokenizers.hf may have been whole-module-replaced by the
        # coordinator into an immutable module object: republish through
        # sys.modules instead of setattr on the module itself
        import sys

        hf_module.get_cached_tokenizer = cached_factory_wrapper
        current = sys.modules.get("vllm.tokenizers.hf")
        if current is not None:
            try:
                current.get_cached_tokenizer = cached_factory_wrapper
            except TypeError:  # immutable module
                import types

                clone = types.ModuleType("vllm.tokenizers.hf")
                clone.__dict__.update(vars(current))
                clone.get_cached_tokenizer = cached_factory_wrapper
                clone.__class__ = type(current)
                sys.modules["vllm.tokenizers.hf"] = clone
        log.info("[tiktoken-fast] hook installed on "
                 "vllm.tokenizers.hf.get_cached_tokenizer")

    if getattr(apply_to_module, "_hf_hook_registered", False):
        pass  # callback was registered on a previous call; just re-patch
    else:
        setattr(apply_to_module, "_hf_hook_registered", True)
        IMPORT_COORDINATOR.register_callback(
            PATCH_ID + ".hf_cached_factory",
            hf_mod_name,
            _patch_hf_cached_factory,
            targets=("vllm.tokenizers.hf.get_cached_tokenizer",),
        )
    # already-imported case: coordinator fires this on registration too,
    # but test harnesses may import hf before the registry module
    loaded_hf = _sys_modules_get(hf_mod_name)
    if loaded_hf is not None:
        _patch_hf_cached_factory(loaded_hf)


def _find_target_class(tokenizer: Any):
    """Locate the real ``TikTokenTokenizer`` base through the Cached* mro."""

    cls = type(tokenizer)
    for klass in cls.__mro__:
        if klass.__name__ == _CLASS_NAME and _MODULE_HINT in (
                klass.__module__ or ""):
            return klass
    return None


def _chunk_limits(cls: type) -> tuple[int, int]:
    """Chunk constants from the wrapper module, with faithful fallbacks."""

    import sys

    mod = sys.modules.get(cls.__module__)
    outer = getattr(mod, "TIKTOKEN_MAX_ENCODE_CHARS", 400_000) \
        if mod else 400_000
    inner = getattr(mod, "MAX_NO_WHITESPACES_CHARS", 25_000) \
        if mod else 25_000
    return int(outer), int(inner)


def _split_chunks(text: str, splitter, outer: int, inner: int) -> list[str]:
    """Exactly the wrapper's own chunking, materialised."""

    pieces: list[str] = []
    for i in range(0, len(text), outer):
        pieces.extend(splitter(text[i:i + outer], inner))
    return pieces or [""]


def _sys_modules_get(name: str):
    import sys

    return sys.modules.get(name)


def _arm_tokenizer(tokenizer: Any, log: logging.Logger) -> bool:
    """Upgrade the TikTokenTokenizer base class behind this instance."""

    cls = _find_target_class(tokenizer)
    if cls is None:
        return False
    with _arm_lock:
        if getattr(cls, _MARKER, False):
            return True
        core = getattr(tokenizer, "model", None)
        split_ws = getattr(tokenizer,
                           "_split_whitespaces_or_nonwhitespaces", None)
        if core is None or split_ws is None or not hasattr(core, "encode"):
            return False
        if not _selfcheck(tokenizer, cls, split_ws):
            log.info("[tiktoken-fast] self-check mismatch on %s; stock kept",
                     cls.__module__)
            return False

        outer_chars, inner_chars = _chunk_limits(cls)

        def fast_rows(inst, texts: list[str], allow: bool) -> list:
            """Token-exact ids for every text, pool-parallel, in order.

            All (text, chunk) pieces of the call form ONE level of pool
            tasks; each task returns one chunk's ids and results are
            re-grouped per owning text.  Tasks never submit tasks.
            """

            splitter = inst._split_whitespaces_or_nonwhitespaces
            chunk_lists = [
                _split_chunks(t, splitter, outer_chars, inner_chars)
                for t in texts
            ]
            owners: list[int] = []
            flat: list[str] = []
            for text_index, chunks in enumerate(chunk_lists):
                for chunk in chunks:
                    owners.append(text_index)
                    flat.append(chunk)
            if len(flat) == 1:
                chunk_ids = [_encode_chunk_core(inst.model, flat[0], allow)]
            else:
                try:
                    futures = [_get_pool().submit(
                        _encode_chunk_core, inst.model, chunk, allow)
                        for chunk in flat]
                except RuntimeError:  # shutdown / post-fork race
                    futures = None
                if futures is None:
                    chunk_ids = [_encode_chunk_core(inst.model, chunk,
                                                    allow) for chunk in flat]
                else:
                    chunk_ids = [f.result() for f in futures]
            rows: list[list[int]] = [[] for _ in texts]
            for owner, ids in zip(owners, chunk_ids):
                rows[owner].extend(ids)
            return rows

        def truncation_goal(kwargs: dict) -> int | None:
            """Wanted cut length, ``None`` for none; raises = delegate."""

            truncation = kwargs.get("truncation")
            max_length = kwargs.get("max_length")
            if truncation is None:
                if max_length is None:
                    return None  # no truncation at all
                truncation = "longest_first"  # HF inferred default
            elif truncation is False:
                return None  # explicit off: stock ignores max_length too
            if truncation not in _TRUNCATION_STRATEGIES:
                raise ValueError("strategy not single-sequence-exact")
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("stock would use model_max_length")
            return max_length

        def truncation_ok(inst, kwargs: dict) -> bool:
            try:
                cut = truncation_goal(kwargs)
            except ValueError:
                return False
            if cut is None:
                # no truncation will be applied: side is irrelevant
                return True
            # single-sequence left/right truncation is exactly a head or
            # tail slice; anything else is not modelled here
            return getattr(inst, "truncation_side", "right") in (
                "left", "right")

        def apply_cut(inst, ids: list[int], cut: int | None) -> list[int]:
            if cut is None:
                return ids
            if getattr(inst, "truncation_side", "right") == "left":
                return ids[-cut:]
            return ids[:cut]

        def make_call(orig_call):
            @functools.wraps(orig_call)
            def patched_call(self, text=None, *args, **kwargs):
                if args:
                    return orig_call(self, text, *args, **kwargs)
                if _kwarg_ok(kwargs) and truncation_ok(self, kwargs):
                    try:
                        cut = truncation_goal(kwargs)
                        if isinstance(text, str):
                            ids = fast_rows(self, [text], True)[0]
                            return _batch_encoding(
                                apply_cut(self, ids, cut), single=True)
                        if isinstance(text, list) and text and all(
                                isinstance(t, str) for t in text):
                            rows = fast_rows(self, text, True)
                            return _batch_encoding(
                                [apply_cut(self, row, cut)
                                 for row in rows],
                                single=False)
                    except Exception:  # never break the hot path
                        if _debug():
                            log.exception(
                                "[tiktoken-fast] fast call failed; "
                                "stock fallback")
                return orig_call(self, text, *args, **kwargs)

            return patched_call

        def make_encode(orig_encode):
            @functools.wraps(orig_encode)
            def patched_encode(self, text, allow_special_tokens=True,
                               **kwargs):
                if not kwargs and isinstance(text, str):
                    try:
                        return fast_rows(
                            self, [text], allow_special_tokens)[0]
                    except Exception:
                        pass
                return orig_encode(self, text, allow_special_tokens,
                                   **kwargs)

            return patched_encode

        if not hasattr(cls, "__call__") or not hasattr(cls, "encode"):
            return False
        cls.__call__ = make_call(cls.__call__)
        cls.encode = make_encode(cls.encode)
        setattr(cls, _MARKER, True)
        log.info(
            "[tiktoken-fast] class upgraded: %s from %s (flat pool, "
            "<=%d threads, no nested submits)",
            cls.__name__, cls.__module__, _POOL_MAX)
        return True


def _kwarg_ok(kwargs: dict) -> bool:
    allowed = {"add_special_tokens", "truncation", "max_length"}
    return set(kwargs) <= allowed


def _batch_encoding(ids, single: bool):
    from transformers.tokenization_utils_base import BatchEncoding

    if single:
        return BatchEncoding({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        })
    return BatchEncoding({
        "input_ids": ids,
        "attention_mask": [[1] * len(i) for i in ids],
    })


def _encode_chunk_core(core: Any, chunk: str, allow: bool) -> list[int]:
    if allow:
        return core.encode(chunk, allowed_special="all")
    return core.encode(chunk, disallowed_special=())


_PROBE_SPECIAL = ("[BOS] hello <|im_end|> 汉字 installs "
                  "<|reserved_token_9|> tab\t newline\n\n café "
                  "1234567890 [EOS]")


def _fast_ids_now(tokenizer: Any, split_ws, outer: int, inner: int,
                  text: str, allow: bool) -> list[int]:
    out: list[int] = []
    for chunk in _split_chunks(text, split_ws, outer, inner):
        out.extend(_encode_chunk_core(tokenizer.model, chunk, allow))
    return out


def _selfcheck(tokenizer: Any, cls: type, split_ws) -> bool:
    """Fast ids must equal stock ids in the modes the wrapper actually uses.

    The wrapper's ``encode(text)`` keeps structural markers special
    (``allow_special_tokens=True``), and so does its ``__call__``; the
    ``allow=False`` fast branch must reproduce the wrapper's
    ``encode(text, allow_special_tokens=False)`` semantics.
    """

    stock_call = cls.__dict__.get("__call__") or cls.__call__
    stock_encode = cls.__dict__.get("encode") or cls.encode
    outer, inner = _chunk_limits(cls)
    try:
        stock_ids = stock_call(tokenizer, _PROBE_SPECIAL)["input_ids"]
        fast_ids = _fast_ids_now(tokenizer, split_ws, outer, inner,
                                 _PROBE_SPECIAL, True)
        if fast_ids != list(stock_ids):
            return False
        stock_plain = stock_encode(tokenizer, _PROBE_SPECIAL,
                                   allow_special_tokens=False)
        fast_plain = _fast_ids_now(tokenizer, split_ws, outer, inner,
                                   _PROBE_SPECIAL, False)
        return fast_plain == list(stock_plain)
    except Exception:
        return False
