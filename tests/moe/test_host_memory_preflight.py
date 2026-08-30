from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.moe.expert_banks import bank_bytes_estimate, validate_host_bank_memory


def _gguf_config(qtype: int):
    return SimpleNamespace(
        expert_quant="gguf",
        moe_weight_format="gguf",
        num_layers=2,
        num_moe_layers=2,
        num_experts=4,
        hidden_size=256,
        expert_hidden_size=256,
        moe_intermediate_size=256,
        gguf_expert_types=((qtype, qtype), (qtype, qtype)),
    )


def test_gguf_bank_estimator_uses_exact_quantized_row_geometry():
    from freetoken.models.gguf.dequant import row_bytes

    config = _gguf_config(14)
    def align(n: int) -> int:
        return (n + 63) // 64 * 64
    per_layer = config.num_experts * (
        align(2 * config.moe_intermediate_size * row_bytes(config.hidden_size, 14))
        + align(config.hidden_size * row_bytes(config.moe_intermediate_size, 14))
    )
    assert bank_bytes_estimate(config) == 2 * per_layer


def test_host_memory_gate_rejects_before_an_unsafe_bank_allocation():
    config = _gguf_config(14)
    banks = bank_bytes_estimate(config)
    assert banks is not None
    with pytest.raises(MemoryError, match="Refusing before allocation"):
        validate_host_bank_memory(
            config,
            reserve_bytes=4 << 30,
            meminfo={"MemAvailable": banks + (3 << 30)},
        )


def test_host_memory_gate_accepts_exact_bank_plus_reserve_floor():
    config = _gguf_config(14)
    banks = bank_bytes_estimate(config)
    assert banks is not None
    validate_host_bank_memory(
        config,
        reserve_bytes=4 << 30,
        meminfo={"MemAvailable": banks + (4 << 30)},
    )


def test_pageable_layers_are_not_charged_to_resident_ram():
    from freetoken.moe.expert_banks import bank_layer_bytes_estimate
    from freetoken.moe.host_banks import HostResidency

    config = _gguf_config(14)
    layers = bank_layer_bytes_estimate(config)
    assert layers is not None
    validate_host_bank_memory(
        config,
        reserve_bytes=4 << 30,
        meminfo={"MemAvailable": layers[0] + (4 << 30)},
        layer_residency=[HostResidency.PINNED.value, HostResidency.PAGEABLE.value],
        disk_free_bytes=layers[1] + (2 << 30),
    )


def test_pageable_layers_require_backing_disk_headroom():
    from freetoken.moe.expert_banks import bank_layer_bytes_estimate
    from freetoken.moe.host_banks import HostResidency

    config = _gguf_config(14)
    layers = bank_layer_bytes_estimate(config)
    assert layers is not None
    with pytest.raises(MemoryError, match="temporary storage"):
        validate_host_bank_memory(
            config,
            reserve_bytes=0,
            meminfo={"MemAvailable": sum(layers)},
            layer_residency=[HostResidency.PINNED.value, HostResidency.PAGEABLE.value],
            disk_free_bytes=layers[1],
        )


def test_pageable_split_uses_post_weight_memavailable(monkeypatch):
    import freetoken.engine.engine as engine
    import freetoken.moe.expert_banks as expert_banks

    model_config = _gguf_config(14)
    layers = expert_banks.bank_layer_bytes_estimate(model_config)
    assert layers is not None and layers[0] == layers[1]
    reserve = 4 << 30
    config = SimpleNamespace(
        model_path="/not-ftw",
        model_config=model_config,
        host_ram_reserve_gb=4.0,
    )
    monkeypatch.setattr(engine, "_pin_budget_bytes", lambda: sum(layers))
    monkeypatch.setattr(
        expert_banks,
        "_host_meminfo_bytes",
        lambda: {"MemAvailable": layers[0] + reserve},
    )

    assert engine._auto_pageable_gpu_layers(config, 2) == frozenset({0})


def test_pageable_split_charges_staging_to_host_reserve(monkeypatch):
    import freetoken.engine.engine as engine
    import freetoken.moe.expert_banks as expert_banks

    model_config = _gguf_config(14)
    model_config.num_experts_per_tok = 1
    banks = expert_banks.bank_bytes_estimate(model_config)
    assert banks is not None
    raw_budget = banks // 2 + banks // 16
    config = SimpleNamespace(
        model_path="/not-ftw",
        model_config=model_config,
        host_ram_reserve_gb=4.0,
        max_running_req=1,
    )
    monkeypatch.setattr(engine, "_pin_budget_bytes", lambda: raw_budget)
    monkeypatch.setattr(
        expert_banks,
        "_host_meminfo_bytes",
        lambda: {"MemAvailable": raw_budget + (4 << 30)},
    )

    # One staged expert row is 1/8 of the two-layer bank. Without charging
    # staging, one layer would appear to fit; with the 4 GiB reserve honored,
    # both tiny test layers must remain pageable.
    assert engine._auto_pageable_gpu_layers(config, 2) == frozenset({0, 1})


def test_ornith_q6_pageable_split_uses_profiled_low_miss_layers(monkeypatch):
    import freetoken.engine.engine as engine
    import freetoken.moe.expert_banks as expert_banks

    config = SimpleNamespace(
        model_path="/not-ftw",
        model_config=SimpleNamespace(
            expert_quant="gguf",
            moe_weight_format="gguf",
            num_layers=40,
            num_moe_layers=40,
            num_experts=256,
            hidden_size=2048,
            expert_hidden_size=2048,
            moe_intermediate_size=512,
            gguf_expert_types=((14, 14),) * 40,
        ),
        host_ram_reserve_gb=4.0,
    )
    layers = expert_banks.bank_layer_bytes_estimate(config.model_config)
    assert layers is not None
    # Force exactly 11 pageable layers.
    resident_budget = sum(layers[:29]) + layers[29] // 2
    monkeypatch.setattr(engine, "_pin_budget_bytes", lambda: resident_budget)
    monkeypatch.setattr(
        expert_banks,
        "_host_meminfo_bytes",
        lambda: {"MemAvailable": resident_budget + (4 << 30)},
    )

    assert engine._auto_pageable_gpu_layers(config, 40) == frozenset(
        {39, 8, 20, 7, 15, 18, 30, 14, 19, 17, 12}
    )


def test_ornith_q6_pageable_profile_is_explicit(monkeypatch):
    import freetoken.engine.engine as engine
    import freetoken.moe.expert_banks as expert_banks
    import freetoken.moe.placement as placement

    config = SimpleNamespace(
        model_path="/not-ftw",
        model_config=SimpleNamespace(
            expert_quant="gguf",
            moe_weight_format="gguf",
            num_layers=40,
            num_moe_layers=40,
            num_experts=256,
            num_experts_per_tok=8,
            hidden_size=2048,
            expert_hidden_size=2048,
            moe_intermediate_size=512,
            gguf_expert_types=((14, 14),) * 40,
        ),
        host_ram_reserve_gb=4.0,
        max_running_req=1,
        moe_pageable_profile="read",
    )
    layers = expert_banks.bank_layer_bytes_estimate(config.model_config)
    assert layers is not None
    resident_budget = sum(layers[:29]) + layers[29] // 2
    monkeypatch.setattr(engine, "_pin_budget_bytes", lambda: resident_budget)
    monkeypatch.setattr(
        expert_banks,
        "_host_meminfo_bytes",
        lambda: {"MemAvailable": resident_budget + (4 << 30)},
    )
    ranking = tuple(range(29, -1, -1)) + tuple(range(30, 40))
    monkeypatch.setattr(placement, "load_pageable_ranking", lambda *_: ranking)

    assert engine._auto_pageable_gpu_layers(config, 40) == frozenset(ranking[:11])
