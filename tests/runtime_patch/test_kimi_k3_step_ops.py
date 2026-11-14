# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the K3.1 TPOT step-ops optimization switches."""

import pytest

from vllm_hcu.platforms import envs as henvs


def test_decode_clean_logits_flag_lazy_resolution(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS", raising=False)
    assert henvs.VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS is False
    monkeypatch.setenv("VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS", "1")
    assert henvs.VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS is True
    monkeypatch.setenv("VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS", "0")
    assert henvs.VLLM_HCU_INDEXER_DECODE_CLEAN_LOGITS is False
