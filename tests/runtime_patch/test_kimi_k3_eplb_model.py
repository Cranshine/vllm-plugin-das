from types import ModuleType, SimpleNamespace

import torch

from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_eplb_model as patch


def test_kimi_model_exposes_moe_protocol_and_current_weight_views(monkeypatch):
    class Runner:
        def __init__(self):
            self.routed_experts = SimpleNamespace(
                moe_config=SimpleNamespace(num_experts=10),
                local_num_experts=5,
            )
            self.weights = [torch.arange(6).view(2, 3)]
            self.state = None
            self.updated = 0

        def get_expert_weights(self):
            return self.weights

        def set_eplb_state(self, **kwargs):
            self.state = kwargs

        def update_expert_map(self):
            self.updated += 1

    class KimiMoE:
        def __init__(self, config):
            self.config = config

    class Model:
        def __init__(self):
            self.start_layer = 0
            self.end_layer = 2
            self.layers = [
                SimpleNamespace(mlp=SimpleNamespace(experts=Runner())),
                SimpleNamespace(mlp=KimiMoE(SimpleNamespace())),
            ]
            self.layers[1].mlp.experts = Runner()

    class KimiLinearForCausalLM:
        def __init__(self):
            self.model = Model()
            self.config = SimpleNamespace(
                num_experts=10,
                num_expert_group=2,
                num_shared_experts=1,
            )

    target = ModuleType(patch.TARGET_MODULE)
    target.KimiLinearForCausalLM = KimiLinearForCausalLM
    target.KimiMoE = KimiMoE
    monkeypatch.setitem(__import__("sys").modules, patch.TARGET_MODULE, target)

    assert patch.apply_to_module(target)
    model = KimiLinearForCausalLM()
    assert patch.apply_to_module(target) is False
    assert model.num_moe_layers == 1
    assert model.num_logical_experts == 10
    assert model.num_physical_experts == 10
    assert model.num_local_physical_experts == 5
    assert model.num_redundant_experts == 0
    assert model.expert_weights == []
    from vllm.model_executor.models.interfaces import MixtureOfExperts

    assert isinstance(model, MixtureOfExperts)

    load = torch.zeros((1, 1, 1))
    mapping = torch.zeros((1, 1), dtype=torch.int32)
    replicas = torch.ones((1, 1), dtype=torch.int32)
    model.set_eplb_state(load, mapping, replicas)
    runner = model.moe_layers[0]
    assert model.expert_weights == [runner.weights]
    assert runner.state["moe_layer_idx"] == 0
    assert runner.state["expert_load_view"] is load

    model.update_physical_experts_metadata(10, 5)
    assert runner.updated == 1


def test_kimi_eplb_protocol_patch_is_in_worker_callback_inventory():
    from vllm_hcu.patch.worker import worker_callback_names

    callbacks = dict(worker_callback_names())
    assert callbacks[patch.PATCH_ID] == patch.TARGET_MODULE
