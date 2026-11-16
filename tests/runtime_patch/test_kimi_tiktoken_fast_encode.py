# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the kimi tiktoken fast/parallel encode patch (CPU only).

No torch, no GPU: the patch only touches python objects, so a fake
tokenizer class with a tiny fake tiktoken core is enough.
"""

import importlib
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

patch_mod = importlib.import_module(
    "vllm_hcu.patch.platform.core_fix.patch_kimi_tiktoken_encode")

_FAKE_LOG = None


def _fake_log():
    global _FAKE_LOG
    if _FAKE_LOG is None:
        import logging
        _FAKE_LOG = logging.getLogger("test.tiktoken_fast")
        _FAKE_LOG.addHandler(logging.NullHandler())
    return _FAKE_LOG


def make_fake_class(name="TikTokenTokenizer", sleep=0.0):
    """Fresh slow-wrapper-shaped class with a sleeping fake core."""

    class FakeCore:
        """Deterministic, chunk-order-independent, thread-name recording."""

        def __init__(self, sleep=0.0):
            self.sleep = sleep
            self.threads = set()

        def encode(self, text, allowed_special=(), disallowed_special=None,
                   **kw):
            self.threads.add(threading.current_thread().name)
            if self.sleep:
                time.sleep(self.sleep)
            return [(ord(ch) * 7919) % 100003 for ch in text]

    class TikTokenTokenizer:
        model_input_names = ["input_ids", "attention_mask"]
        _default_sleep = sleep

        def __init__(self, sleep=None):
            self.model = FakeCore(
                sleep if sleep is not None else self._default_sleep)

        @staticmethod
        def _split_whitespaces_or_nonwhitespaces(text, max_consecutive):
            pieces = []
            cur = []
            cur_is_ws = None
            run = 0
            for ch in text:
                is_ws = ch.isspace()
                if cur_is_ws is None or is_ws == cur_is_ws:
                    if run == max_consecutive and is_ws == cur_is_ws:
                        pieces.append("".join(cur))
                        cur = []
                        run = 0
                    cur.append(ch)
                    run += 1
                    cur_is_ws = is_ws
                else:
                    pieces.append("".join(cur))
                    cur = [ch]
                    cur_is_ws = is_ws
                    run = 1
            if cur:
                pieces.append("".join(cur))
            return pieces or [""]

        def _encode_text_piece(self, text, allow_special_tokens=True):
            out = []
            for i in range(0, len(text), 8):
                for sub in self._split_whitespaces_or_nonwhitespaces(
                        text[i:i + 8], 3):
                    out.extend(self.model.encode(sub))
            return out

        def encode(self, text, allow_special_tokens=True, **kwargs):
            assert not kwargs and isinstance(text, str), \
                "stock path expected here"
            return self._encode_text_piece(text)

        def __call__(self, text, **kwargs):
            assert not kwargs or not hasattr(
                self, patch_mod._MARKER), "stock call expected here"
            if isinstance(text, list):
                return [self.__call__(t, **{}) for t in text]
            assert isinstance(text, str), "stock batching path"
            ids = self.encode(text)
            return {"input_ids": ids,
                    "attention_mask": [1] * len(ids)}

    TikTokenTokenizer.__name__ = name
    TikTokenTokenizer.__qualname__ = name
    # the real wrappers load from transformers_modules/...tokenization_kimi
    if name == "TikTokenTokenizer":
        TikTokenTokenizer.__module__ = (
            "transformers_modules.Kimi.tokenization_kimi")
    return TikTokenTokenizer


def _arm(tok):
    return patch_mod._arm_tokenizer(tok, _fake_log())


class TiktokenFastEncodeTest(unittest.TestCase):
    def test_arm_only_targets_tiktoken_class(self):
        other = make_fake_class("SomeOtherTokenizer")()
        self.assertFalse(_arm(other))
        self.assertNotIn(patch_mod._MARKER,
                         [getattr(type(other), patch_mod._MARKER, None)])
        self.assertEqual(
            getattr(type(other).encode, "__wrapped__",
                    type(other).encode),
            type(other).encode)

    def test_arm_sets_marker_once(self):
        cls = make_fake_class()
        tok = cls()
        self.assertTrue(_arm(tok))
        self.assertTrue(getattr(cls, patch_mod._MARKER, False))
        # second arm on the same class: still true, no crash
        self.assertTrue(_arm(tok))

    def test_equivalence_against_stock_short_inputs(self):
        ref_cls = make_fake_class()
        patched_cls = make_fake_class()
        ref, tok = ref_cls(), patched_cls()
        _arm(tok)
        texts = ["hello world", "x" + " " * 50 + "y", "", "  \n\t ",
                 "汉字" * 20, "ab " * 37]
        for t in texts:
            self.assertEqual(ref.encode(t), tok.encode(t), repr(t))
            self.assertEqual(ref(t), tok(t), repr(t))

    def test_single_chunk_hits_core_exactly_once(self):
        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        tok.encode("short")          # small: direct core call, no pool
        self.assertEqual(tok.model.threads,
                         {threading.current_thread().name})

    def test_multichunk_fans_out_to_threads(self):
        cls = make_fake_class(sleep=0.002)
        tok = cls()
        _arm(tok)
        text = "ab c " * 200         # many whitespace boundary chunks
        tok.encode(text)
        self.assertGreater(len(tok.model.threads), 1)

    def test_kwargs_delegate_to_stock(self):
        """Exotic kwargs must reach the ORIGINAL stock encode, not crash."""

        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        # the fake stock encode asserts on kwargs; reaching it proves
        # the patched wrapper delegated instead of intercepting
        with self.assertRaises(AssertionError):
            tok.encode("x", padding=True)
        # and fast path works without kwargs
        self.assertEqual(tok.encode("x"), tok.model.encode("x"))

    def test_left_truncation_side_falls_back(self):
        """Review P1-B, round-5 refinement: single-sequence LEFT side
        truncation is exactly a tail slice, so the fast path serves it
        (ids[-cut:]) instead of delegating."""

        cls = make_fake_class()
        tok = cls()
        tok.truncation_side = "left"
        _arm(tok)
        ids = tok.model.encode("hello world")
        out = tok("hello world", truncation=True, max_length=4)
        self.assertEqual(list(out["input_ids"]), ids[-4:])
        # plain call still fast
        self.assertEqual(tok("hello world")["input_ids"],
                         list(ids))

    def test_truncation_no_max_length_delegates(self):
        """truncation=True without max_length: stock uses
        model_max_length semantics -> must delegate, not skip cutting."""

        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        with self.assertRaises(AssertionError):
            tok("hello world", truncation=True)

    def test_do_not_truncate_delegates(self):
        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        with self.assertRaises(AssertionError):
            tok("hello world", truncation="do_not_truncate",
                max_length=4)

    def test_batch_of_n_strings_parallel_fast(self):
        """Review P1-C: N-item batches with whitelist kwargs are cut
        through the fast path fan-out rather than delegated."""

        cls = make_fake_class(sleep=0.002)
        tok = cls()
        _arm(tok)
        tok.model.threads.clear()
        out = tok(["hello world", "second prompt here", "third one 3.14"],
                  add_special_tokens=True, truncation=True, max_length=7)
        ref_single = tok.model.encode("hello world")[:7]
        self.assertEqual(len(out["input_ids"]), 3)
        self.assertEqual(list(out["input_ids"][0]), ref_single)
        self.assertGreater(len(tok.model.threads), 1)

    def test_batch_wide_multichunk_no_deadlock(self):
        """Review round-3 P0: >=POOL_MAX items each multi-chunk used to
        nest submissions on one pool and wait on itself forever.  The
        flat design must complete; bound it with a timeout."""

        cls = make_fake_class(sleep=0.001)
        tok = cls()
        _arm(tok)
        texts = [" ".join("tok%d" % i for i in range(12))] * 6
        target = tok.encode  # plain callable for clarity
        result = {}

        def run():
            result["out"] = tok(texts)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=20)
        self.assertFalse(worker.is_alive(),
                         "wide multi-chunk batch deadlocked")
        rows = result["out"]["input_ids"]
        self.assertEqual(len(rows), 6)
        ref = [tok.model.encode(t) for t in texts]
        self.assertEqual([list(r) for r in rows], ref)

    def test_max_length_without_truncation_key_cuts(self):
        """HF infers longest_first when max_length is given without an
        explicit truncation kwarg: fast path must cut identically."""

        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        ids = tok.model.encode("hello world")
        out = tok("hello world", max_length=5)
        self.assertEqual(list(out["input_ids"]), ids[:5])

    def test_truncation_false_is_explicit_off(self):
        """truncation=False means stock ignores max_length: fast path
        must also return the full ids."""

        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        ids = tok.model.encode("hello world")
        out = tok("hello world", truncation=False, max_length=3)
        self.assertEqual(list(out["input_ids"]), ids)

    def test_call_list_batch_single(self):
        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        out = tok(["hello world"])
        self.assertEqual(out["input_ids"],
                         [tok.model.encode("hello world")])

    def test_loader_arms_and_kill_switch(self):
        registry_mod = types.ModuleType("vllm.tokenizers.registry")

        class TokenizerRegistry:
            pass

        @staticmethod
        def load_tokenizer(tokenizer_mode, *a, **kw):
            return make_fake_class()()

        TokenizerRegistry.load_tokenizer = staticmethod(load_tokenizer)
        registry_mod.TokenizerRegistry = TokenizerRegistry
        patch_mod.apply_to_module(registry_mod)
        tok = TokenizerRegistry().load_tokenizer("auto")
        self.assertTrue(getattr(type(tok), patch_mod._MARKER, False))

        # kill switch: fresh module -> loader patched but must NOT arm
        registry_mod2 = types.ModuleType("vllm.tokenizers.registry")

        class TokenizerRegistry2:
            pass

        @staticmethod
        def load_tokenizer2(tokenizer_mode, *a, **kw):
            return make_fake_class()()

        TokenizerRegistry2.load_tokenizer = staticmethod(load_tokenizer2)
        registry_mod2.TokenizerRegistry = TokenizerRegistry2
        with mock.patch.dict(os.environ,
                             {patch_mod._DISABLE_ENV: "0"}):
            patch_mod.apply_to_module(registry_mod2)
            tok2 = TokenizerRegistry2().load_tokenizer("auto")
            self.assertFalse(getattr(type(tok2), patch_mod._MARKER, False))

    def test_registry_rearm_is_noop(self):
        registry_mod = types.ModuleType("vllm.tokenizers.registry")

        class TokenizerRegistry:
            pass

        @staticmethod
        def load_tokenizer(tokenizer_mode, *a, **kw):
            return make_fake_class()()

        TokenizerRegistry.load_tokenizer = staticmethod(load_tokenizer)
        registry_mod.TokenizerRegistry = TokenizerRegistry
        patch_mod.apply_to_module(registry_mod)
        armed = TokenizerRegistry.load_tokenizer
        patch_mod.apply_to_module(registry_mod)
        self.assertIs(TokenizerRegistry.load_tokenizer, armed)

    def test_non_str_delegates(self):
        cls = make_fake_class()
        tok = cls()
        _arm(tok)
        with self.assertRaises(AssertionError):
            tok.encode(12345)

    def test_cached_subclass_bases_are_patched(self):
        """Review P0: the serve wraps in ``Cached*`` subclasses; the MRO

        walk must find and patch the original TikTokenTokenizer base.
        """

        base = make_fake_class()
        tok = base()

        class CachedTikTokenTokenizer(base):
            pass

        wrapped = CachedTikTokenTokenizer()
        armed = patch_mod._arm_tokenizer(wrapped, _fake_log())
        self.assertTrue(armed)
        self.assertTrue(getattr(base, patch_mod._MARKER, False))
        # the cached subclass calls now hit the patched base methods
        out = wrapped("hello world")
        ref = base()("hello world")
        self.assertEqual(out["input_ids"], ref["input_ids"])

    def test_foreign_same_name_class_rejected(self):
        """Review P1-5: a same-name class from another module is ignored."""

        other = make_fake_class("TikTokenTokenizer")
        # pretend it lives in an unrelated dynamic module
        other.__module__ = "other_repo.other_tokenization"
        tok = other()
        self.assertFalse(patch_mod._arm_tokenizer(tok, _fake_log()))
        self.assertNotIn(patch_mod._MARKER, vars(other))
    def test_selfcheck_failure_keeps_stock(self):
        """When fast ids diverge from stock ids the patch must abstain."""

        base = make_fake_class()
        tok = base()
        real_split = base._split_whitespaces_or_nonwhitespaces

        def broken_split(text, max_consecutive):
            # corrupt output so the selfcheck probe mismatches
            return ["".join(reversed(real_split(text, max_consecutive)[0]))]

        base._split_whitespaces_or_nonwhitespaces = staticmethod(broken_split)
        armed = patch_mod._arm_tokenizer(tok, _fake_log())
        self.assertFalse(armed)
        self.assertNotIn(patch_mod._MARKER, vars(base))

    def test_call_returns_batch_encoding(self):
        base = make_fake_class()
        tok = base()
        _arm(tok)
        out = tok("hello world")
        from transformers.tokenization_utils_base import BatchEncoding

        self.assertIsInstance(out, BatchEncoding)
        self.assertEqual(list(out.input_ids),
                         list(tok.model.encode("hello world")))

    def test_simple_kwargs_intercepted(self):
        base = make_fake_class()
        tok = base()
        _arm(tok)
        ref = tok.model.encode("hello world")
        out = tok("hello world", add_special_tokens=True)
        self.assertEqual(list(out.input_ids), ref)
        out = tok("hello world", truncation=True, max_length=5)
        self.assertEqual(list(out.input_ids), ref[:5])

    def test_exotic_kwargs_delegate_to_stock_call(self):
        """A kwarg outside the allowlist must reach the stock __call__."""

        fresh = make_fake_class()
        fresh_tok = fresh()
        self.assertTrue(_arm(fresh_tok))
        patched = type(fresh_tok).__dict__["__call__"]
        self.assertTrue(hasattr(patched, "__wrapped__"))
        # 2-item batch + exotic kwarg: fast branches all refuse, the
        # wrapper must hand off to the stock implementation, which for
        # the fake class asserts on list input ("stock batching path").
        with self.assertRaises(AssertionError):
            patched(fresh_tok, ["a", "b"], padding=True)

    def test_pool_shutdown_falls_back_serial(self):
        """A shutdown pool must degrade to serial encode, never raise."""

        from concurrent.futures import ThreadPoolExecutor

        base = make_fake_class(sleep=0.001)
        tok = base()
        _arm(tok)
        old_pool = patch_mod._pool_executor
        try:
            dead = ThreadPoolExecutor(max_workers=1)
            dead.shutdown(wait=False)
            patch_mod._pool_executor = dead  # inject the dead pool
            ids = tok.encode("hello world")
            self.assertTrue(ids)
        finally:
            patch_mod._pool_executor = old_pool

    def test_stale_pool_none_rebuilds(self):
        base = make_fake_class()
        tok = base()
        _arm(tok)
        old_pool = patch_mod._pool_executor
        try:
            patch_mod._pool_executor = None
            ids = tok.encode("hello world")
            self.assertTrue(ids)
            self.assertIsNotNone(patch_mod._pool_executor)
        finally:
            patch_mod._pool_executor = old_pool


if __name__ == "__main__":
    unittest.main(verbosity=2)
