"""Shrink the DeepEP-LL dispatch buffer to SP-MoE row windows.

Why: the low-latency all2all path sizes every dispatch buffer, slot table and
the masked-GEMM ``expected_m`` clamp from ``FusedMoEConfig.max_num_tokens``
(the scheduler's ``max_num_batched_tokens``). With SP-MoE row sharding a TP
rank only dispatches ``ceil(N / TP)`` rows per layer, so the stock slots are
TP-x oversized (4x on TP4) with no consumer behind them.

What: shrink ``max_num_tokens`` to ``ceil(stock / TP)`` inside
``FusedMoEConfig.__post_init__`` while the SP-MoE switch is armed. Single
choke point: the config object is created once per MoE inside the factory and
is referenced unchanged by the MoERunner, the DeepEP-LL handle
(``max_num_tokens_per_dp_rank`` + ``max_tokens_per_rank``) and the masked-GEMM
expected_m clamp, so one field write before runner construction moves the
whole chain. Shrinkage touches the metadata/buffer dimension only, never a
tensor copy. This tree's ``layer.py`` has no ``FusedMoE`` class -- the
factory returns a ``MoERunner`` directly -- so the config class is the only
durable anchor for the clamp, and the runner forward guards the invariant.

Safety: a shrunken buffer is only valid for an SP-managed runner (its
dispatch never exceeds ``ceil(N / TP)`` rows). ``patch_kimi_k3_sp_moe`` tags
the runner it enables with ``SP_LL_TAG``. A forward guard on both runner
classes (the body original and the HCU replacement that this deployment
actually runs) raises when an untagged runner feeds more rows than the
shrunken slot count into the dispatch path -- the signature of a layer that
silently stayed upstream and would otherwise corrupt its deep_ep buffer.
Tagged runners with an unclamped config only log once (wasteful, not
unsafe). The guard is a no-op under the stock config, and tp<=1
(single-process / CPU tests) never clamps. EPLB-style ``_set_moe_config``
replacement of the config object is not re-clamped and unsupported here
(EPLB is off in this deployment).
"""

from __future__ import annotations

import functools
import importlib
import os
from types import ModuleType

from ._common import require_callable, require_class
from .patch_kimi_k3_sp_moe import ENV_SWITCH as _SP_ENV
from .patch_kimi_k3_sp_moe import SP_LL_TAG

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.config"
BODY_RUNNER_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
HCU_RUNNER_MODULE = "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
PATCH_ID = "worker.op_opt.kimi_k3.ll_shrink"
TARGETS = (
    f"{TARGET_MODULE}.FusedMoEConfig.__post_init__",
    f"{BODY_RUNNER_MODULE}.MoERunner.forward",
    f"{HCU_RUNNER_MODULE}.MoERunner.forward",
)

_MARKER = "_vllm_hcu_kimi_k3_ll_shrink_applied"
_CONFIG_MARKER = "_vllm_hcu_kimi_k3_ll_shrink_clamped"
_POST_INIT_WRAPPER = "_vllm_hcu_kimi_k3_ll_shrink_post_init"
_FORWARD_WRAPPER = "_vllm_hcu_kimi_k3_ll_shrink_forward_wrapper"

# Stock slot count of the baseline scheduler config, captured at the first
# clamp. Guards compare a runner's current config against it.
_stock_slots: int | None = None


def _sp_armed() -> bool:
    return os.environ.get(_SP_ENV, "1").strip().lower() in ("1", "true")


def _shrink_slots(max_num_tokens: int, tp_size: int) -> int:
    return -(-max_num_tokens // tp_size)


def _tp_world_size() -> int:
    from vllm.distributed import get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size()


def _patch_config_class(module: ModuleType, logger) -> bool:
    """Wrap FusedMoEConfig.__post_init__ with the row-window clamp."""
    global _stock_slots

    config_cls = require_class(
        module, "FusedMoEConfig", f"{TARGET_MODULE}.FusedMoEConfig"
    )
    post_init = require_callable(
        config_cls, "__post_init__", f"{TARGET_MODULE}.FusedMoEConfig"
    )
    if getattr(vars(config_cls)["__post_init__"], _POST_INIT_WRAPPER, False):
        return False

    @functools.wraps(post_init)
    def wrapped_post_init(self):
        post_init(self)
        global _stock_slots
        if not _sp_armed() or getattr(self, _CONFIG_MARKER, False):
            return
        tp_size = _tp_world_size()
        if tp_size <= 1:
            return
        stock = self.max_num_tokens
        slots = _shrink_slots(stock, tp_size)
        if slots >= stock:
            return
        if _stock_slots is None:
            _stock_slots = stock
        elif _stock_slots != stock:
            # A config with a different scheduler setting; leave it stock.
            return
        self.max_num_tokens = slots
        setattr(self, _CONFIG_MARKER, True)
        logger.info_once(
            "Kimi-K3 LL dispatch slots clamped %d -> %d by SP-MoE row "
            "sharding (%s=1): the deep_ep buffer, slot metadata and the "
            "masked-GEMM expected_m clamp all follow this field.",
            stock,
            slots,
            _SP_ENV,
        )

    setattr(wrapped_post_init, _POST_INIT_WRAPPER, True)
    config_cls.__post_init__ = wrapped_post_init
    return True


def _wrap_runner_forward(cls, logger) -> bool:
    """Guard the row-window invariant on one runner class.

    Raises only on the dangerous direction: a runner whose config is shrunken
    but which SP-MoE never enabled, dispatching more rows than slots. A
    tagged runner left with a stock config is wasteful, not unsafe, and only
    logs.
    """
    original_forward = vars(cls).get("forward")
    if not callable(original_forward):
        logger.debug(
            "SP-MoE ll_shrink: %s has no class-level forward; guard skipped",
            cls.__qualname__,
        )
        return False
    if getattr(original_forward, _FORWARD_WRAPPER, False):
        return True

    @functools.wraps(original_forward)
    def wrapped_forward(self, hidden_states, *args, **kwargs):
        stock = _stock_slots
        config = getattr(self, "moe_config", None)
        if stock is not None and config is not None:
            slots = config.max_num_tokens
            tagged = getattr(self, SP_LL_TAG, False)
            if slots < stock and not tagged:
                rows = hidden_states.shape[0]
                if rows > slots:
                    raise RuntimeError(
                        f"{PATCH_ID}: {rows} rows exceed the {slots}-slot LL "
                        "buffer on a runner that never enabled SP-MoE; an "
                        "upstream-path layer is running on a shrunken "
                        "deep_ep buffer. Refusing to dispatch."
                    )
            elif slots >= stock and tagged:
                logger.info_once(
                    "Kimi-K3 SP-MoE runner dispatch buffer is stock-sized "
                    "(%d slots); row sharding works but the LL buffer stays "
                    "full-width.", slots,
                )
        return original_forward(self, hidden_states, *args, **kwargs)

    setattr(wrapped_forward, _FORWARD_WRAPPER, True)
    cls.forward = wrapped_forward
    return True


def apply_to_module(module: ModuleType) -> bool:
    """Registry entry: ``module`` is the FusedMoE config module."""
    if getattr(module, _MARKER, None) == "applied":
        return False

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    _patch_config_class(module, logger)

    for runner_path in (BODY_RUNNER_MODULE, HCU_RUNNER_MODULE):
        try:
            runner_mod = importlib.import_module(runner_path)
            runner_cls = require_class(
                runner_mod, "MoERunner", f"{runner_path}.MoERunner"
            )
        except Exception:
            logger.debug(
                "SP-MoE ll_shrink: runner module %s unavailable",
                runner_path,
                exc_info=True,
            )
            continue
        _wrap_runner_forward(runner_cls, logger)

    setattr(module, _MARKER, "applied")
    return True


__all__ = [
    "PATCH_ID",
    "TARGETS",
    "TARGET_MODULE",
    "apply_to_module",
]
