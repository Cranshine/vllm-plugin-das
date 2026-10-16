# PR #239: Kimi-K3 redundant expert checkpoint loading

## Scope and root cause

Base: `7d52be2dde99519d103afed9de94eeed74eaba1d`.

With EPLB enabled and redundant experts allocated, the Kimi main model's
inline checkpoint loader only constructs mappings for logical expert IDs.
It also breaks after the first mapping match. Consequently, the physical
replicas can enter routing with uninitialized weights and quantization scales.
Passing the redundant count alone would still leave the first-match bug.

## Fix

The plugin's existing Kimi EPLB adapter wraps `KimiLinearModel.load_weights`.
For layers with redundant storage, it delegates expert tensors to the owning
`RoutedExperts.load_weights` implementation. That loader visits every physical
mapping and uses each parameter's loader for rank-local placement. Bound static
EPLB loaders retain their logical-to-physical fanout.

The adapter translates Kimi checkpoint `w1/w2/w3` names to the owner's configured
projection names and preserves the existing `weight_packed` to `weight` alias
when storage is unpacked. Other tensors and partial-layer filtering continue
through the Kimi loader. Layers without redundant storage use the original
loading path. The patch remains in the plugin; no installed package is changed.

## CPU regression coverage

`tests/runtime_patch/test_kimi_k3_eplb_model.py` includes:

- A pre-fix probe using the installed Kimi loader: redundant `w13_weight`,
  `w2_weight`, and both scales retain their sentinel values.
- EP1, EP2, and EP16 placement, iterating every simulated rank, with default
  initial placement and an alternate static placement.
- FP32 weights and synthetic INT8 checkpoint bytes with `weight_packed` names,
  plus per-channel weight scales for all w1/w2/w3 projections.
- Exact comparison of every local physical slot against its logical checkpoint
  weights/scales; a CPU SiLU MLP calculation through every FP32 physical slot.
- No-redundancy behavior, non-expert weights, partial-layer filtering, adapter
  registration and idempotence.

Mapping construction, expert loading, parameter weight loaders and Kimi inline
loading use the installed vLLM implementations. Fixtures bypass accelerator
allocation and supply CPU placement/configuration. These are small synthetic
weights: they do not validate HIPC packing, device MoE kernels, distributed
communication, or full-model HumanEval.

## Reproduce

Run in `xiesl_kimi3_plugin_028`, from
`/home/xiesl/inference/kimi3_plugin_vllm/plugin_dir/plugin-v0.28.1-dev-kimi-k3-xiesl`:

```bash
export VLLM_PLUGINS=__disabled__
export CUDA_VISIBLE_DEVICES=
export HIP_VISIBLE_DEVICES=
python -m pytest -q --disable-warnings \
  tests/runtime_patch/test_kimi_k3_eplb_model.py \
  tests/runtime_patch/test_kimi_k3_reference_adapters.py \
  tests/runtime_patch/test_model_loader_static_eplb.py \
  tests/model_executor/layers/fused_moe/test_static_eplb.py \
  tests/runtime_patch/test_moe_deepep.py
python tools/check_patch_test_coverage.py
python tools/check_production_boundary.py
```

No service is started and no accelerator is exposed to these commands.

## Results (2026-10-16)

- Final combined regression: **232 passed**, 14 warnings, 64.07 seconds.
- Focused redundant-expert regression: **17 passed**.
- Patch coverage: 171 files, 170 adapters, 0 untested, 0 invalid contracts.
- Production boundary: 401 Python files clean.
- `git diff --check`: clean.

Host artifacts: `task_dir/debug_028/eplb_redundant_fix_tests.log` and
`task_dir/debug_028/eplb_redundant_fix_regression.log`.
No new full-model HumanEval result is claimed for this fix.
