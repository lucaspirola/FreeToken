"""The S0 merge of upstream #418 dropped the "nvfp4" and "none" rows of the provider table
with no conflict (upstream deleted them, the fork added rows next to them), and the first real
Nemotron boot refused its experts. CPU-only guard: every format a method-less model can carry
must still reach a provider."""

import pytest

from freetoken.moe import expert_banks


@pytest.mark.parametrize("quant", ["none", "nvfp4", "q4_0", "gguf", "laguna_int4"])
def test_method_less_expert_formats_keep_their_provider(quant):
    assert quant in expert_banks._PROVIDERS


def test_nemotron_nvfp4_experts_are_not_refused(monkeypatch):
    seen = {}

    def fake(*args, **kwargs):
        seen["called"] = True
        return "banks"

    monkeypatch.setitem(expert_banks._PROVIDERS, "nvfp4", fake)

    class Cfg:
        expert_quant = "nvfp4"

    out = expert_banks._legacy_expert_banks(
        "/nonexistent", Cfg(), "cpu", None, True, False, 1, 1 << 20
    )
    assert out == "banks" and seen["called"]
