"""Config-time resolution for the Nemotron-H family (Nemotron-3.5 Lightning shape).

Lightning is a hybrid Mamba-2 / full-attention / NVFP4-MoE checkpoint whose routed
experts are *ungated* ReLU^2 (up+down only). Two things have to hold before any weight
is resident:

- the Marlin NVFP4 MoE kernel, which hard-codes a gated [2I, H] gate_up bank, must be
  refused at config time rather than mis-shaping banks after the load;
- dropping ``single_stream_only`` (Super forced it; Lightning's 47 MiB SSM state does
  not) must leave the multi-request knobs alone -- 16 running requests and the elastic
  decode-graph set sized to the *initial* capacity, not the ceiling.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.models.config import KVCacheGroupSpec

MODEL_PATH = "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _spec():
    # The 6 full-attention layers are the only paged-KV group; the 23 Mamba-2 layers
    # hold recurrent state instead and never reach the KV pool.
    return KVCacheGroupSpec(
        name="full",
        layer_ids=(5, 12, 19, 26, 33, 42),
        num_kv_heads=2,
        head_dim=128,
        sliding_window=None,
        attn_type=AttnType.FULL,
    )


def _model_config(*, expert_gated: bool = False, single_stream_only: bool = False):
    mc = SimpleNamespace(
        model_type="nemotron_h",
        single_stream_only=single_stream_only,
        is_moe=True,
        expert_quant="nvfp4",
        expert_gated=expert_gated,
        hidden_act="relu2",
        moe_weight_format=None,
        has_swa_attention=False,
        has_linear_attention=True,
        num_layers=52,
        num_moe_layers=23,
        num_experts=128,
        hidden_size=2688,
        expert_hidden_size=None,
        moe_intermediate_size=1856,
        rotary_config=SimpleNamespace(max_position=1048576),
    )
    specs = (_spec(),)
    mc.kv_cache_group_specs = lambda: specs
    return mc


def _config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.scheduler.config import SchedulerConfig

    model_config = overrides.pop("model_config", None) or _model_config()
    defaults = dict(
        attention_backend="triton",
        moe_backend="offload",
        moe_cache_auto=True,
    )
    defaults.update(overrides)
    config = SchedulerConfig(
        model_path=MODEL_PATH,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **defaults,
    )
    object.__setattr__(config, "model_config", model_config)
    return config


# --- ungated experts x NVFP4 backend -----------------------------------------------


def test_marlin_is_rejected_for_ungated_experts():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="requires gated"):
        _adjust_config(_config(nvfp4_backend="marlin"))


def test_marlin_stays_available_for_gated_nvfp4_experts():
    from freetoken.engine.engine import _adjust_config

    config = _config(
        nvfp4_backend="marlin", model_config=_model_config(expert_gated=True)
    )
    _adjust_config(config)
    assert config.nvfp4_backend == "marlin"


def test_marlin_is_not_rejected_when_the_model_declares_no_gating():
    """Every other MoE family omits ``expert_gated`` entirely and is gated."""
    from freetoken.engine.engine import _adjust_config

    model_config = _model_config()
    del model_config.expert_gated
    config = _config(nvfp4_backend="marlin", model_config=model_config)
    _adjust_config(config)
    assert config.nvfp4_backend == "marlin"


@pytest.mark.parametrize("backend", ["triton", "auto", "flashinfer"])
def test_other_nvfp4_backends_pass_config_time(backend):
    """triton is the Phase-1 lane; flashinfer/b12x is enabled for relu2 in Phase 2, so
    config time must not be the thing that blocks it."""
    from freetoken.engine.engine import _adjust_config

    config = _config(nvfp4_backend=backend)
    _adjust_config(config)
    assert config.nvfp4_backend == backend


# --- multi-request / elastic knobs -------------------------------------------------


def test_multi_stream_model_keeps_its_running_request_ceiling():
    from freetoken.engine.engine import _adjust_config

    config = _config(max_running_req=16)
    _adjust_config(config)
    assert config.max_running_req == 16
    assert config.cuda_graph_max_bs == 16


def test_single_stream_only_still_collapses_to_one(caplog):
    """Nemotron-3 Super's regime is unchanged: one sequence, one captured graph."""
    from freetoken.engine.engine import _adjust_config

    config = _config(
        max_running_req=16, model_config=_model_config(single_stream_only=True)
    )
    _adjust_config(config)
    assert config.max_running_req == 1
    assert config.cuda_graph_bs == [1]
    assert config.cuda_graph_max_bs == 1


def test_elastic_start_sizes_graphs_to_the_initial_capacity():
    from freetoken.engine.engine import _adjust_config

    config = _config(
        max_running_req=16,
        elastic_initial_requests=4,
        kv_grow_step_tokens=65536,
        num_token_override=262144,
    )
    _adjust_config(config)
    # Admission still accepts 16; only the recurrent-state/graph working set starts at 4.
    assert config.max_running_req == 16
    assert config.cuda_graph_bs == [1, 2, 3, 4]
    assert config.cuda_graph_max_bs == 4
    assert config.cache_type == "hybrid_radix"


def test_elastic_initial_must_be_below_the_ceiling():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="must be smaller than"):
        _adjust_config(
            _config(
                max_running_req=4,
                elastic_initial_requests=4,
                kv_grow_step_tokens=65536,
                num_token_override=262144,
            )
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
