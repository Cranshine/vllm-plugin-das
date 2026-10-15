from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.v1.hcu_model_runner_v2 import (
    HcuGPUModelRunnerV2,
    suppress_sampler_warmup_logprobs,
)


def test_runner_v2_overrides_match_installed_vllm_parameter_contract() -> None:
    # Runner V2 owns different call contracts from the legacy HCU runner.
    # Keep every override that crosses that boundary aligned with the actual
    # installed vLLM API, including HCU's allocation-context arguments.
    methods = (
        "initialize_kv_cache",
        "prepare_inputs",
        "prepare_attn",
        "prepare_dummy_attn",
        "execute_model",
        "sample_tokens",
    )
    for name in methods:
        upstream = signature(getattr(GPUModelRunner, name)).parameters
        hcu = signature(getattr(HcuGPUModelRunnerV2, name)).parameters
        assert tuple(hcu) == tuple(upstream), name

    dummy_attn = signature(HcuGPUModelRunnerV2.prepare_dummy_attn).parameters
    assert dummy_attn["valid_state_slots"].default is False


def test_sampler_warmup_logprobs_are_suppressed_and_restored() -> None:
    num_logprobs = np.array([5, 3, -1, 7], dtype=np.int32)
    model_runner = SimpleNamespace(
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(num_logprobs=num_logprobs)
        )
    )
    warmup_batch = SimpleNamespace(
        req_ids=["_warmup_0_", "_warmup_1_"],
        num_reqs=2,
        idx_mapping_np=np.array([3, 1], dtype=np.intp),
    )

    with pytest.raises(RuntimeError, match="synthetic warmup"):
        with suppress_sampler_warmup_logprobs(model_runner, warmup_batch):
            assert num_logprobs.tolist() == [5, -1, -1, -1]
            raise RuntimeError("synthetic warmup")

    assert num_logprobs.tolist() == [5, 3, -1, 7]


def test_sampler_logprobs_are_unchanged_for_real_requests() -> None:
    num_logprobs = np.array([5, 3], dtype=np.int32)
    model_runner = SimpleNamespace(
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(num_logprobs=num_logprobs)
        )
    )
    real_batch = SimpleNamespace(
        req_ids=["request-0"],
        num_reqs=1,
        idx_mapping_np=np.array([1], dtype=np.intp),
    )

    with suppress_sampler_warmup_logprobs(model_runner, real_batch):
        assert num_logprobs.tolist() == [5, 3]

    assert num_logprobs.tolist() == [5, 3]
