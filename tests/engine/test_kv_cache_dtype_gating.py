"""--kv-cache-dtype gating: the combinations the quantized path does not implement
must be refused at config time, not at the first kernel launch."""

from __future__ import annotations

import pytest

from freetoken.engine.engine import _validate_kv_cache_dtype
from freetoken.kvcache.quant import FP8_E4M3, INT4, NONE, Q5_0, Q6_0, Q8_0
from freetoken.models.config import KVCacheGroupSpec


class _Cfg:
    def __init__(self, quant=Q8_0, backend="triton", quant_k=None, quant_v=None):
        self.kv_quant = quant
        if quant_k is not None:
            self.kv_quant_k = quant_k
        if quant_v is not None:
            self.kv_quant_v = quant_v
        self.attention_backend = backend


class _Model:
    def __init__(self, *specs):
        self._specs = specs
        self.has_swa_attention = False

    def kv_cache_group_specs(self):
        return self._specs


def _spec(name="full", head_dim=256, mla=False, index_head_dim=0):
    return KVCacheGroupSpec(
        name=name,
        layer_ids=(0, 1),
        num_kv_heads=2,
        head_dim=head_dim,
        sliding_window=None,
        mla=mla,
        index_head_dim=index_head_dim,
    )


@pytest.mark.parametrize("quant", [Q8_0, FP8_E4M3, INT4], ids=lambda spec: spec.name)
def test_triton_backend_with_aligned_head_dim_is_accepted(quant):
    _validate_kv_cache_dtype(_Cfg(quant=quant), _Model(_spec(head_dim=256), _spec("swa", 512)))


def test_auto_dtype_skips_every_check():
    # Unquantized configs must pass even on backends/pools the quantized path rejects.
    _validate_kv_cache_dtype(_Cfg(quant=NONE, backend="fi"), _Model(_spec(mla=True)))


@pytest.mark.parametrize("backend", ["fi", "fa", "trtllm", "fa,fi", "triton,fi"])
def test_non_triton_backends_are_rejected(backend):
    with pytest.raises(ValueError, match="needs the triton attention backend"):
        _validate_kv_cache_dtype(_Cfg(backend=backend), _Model(_spec()))


def test_mla_pool_is_rejected():
    with pytest.raises(ValueError, match="MLA/DSA"):
        _validate_kv_cache_dtype(_Cfg(), _Model(_spec(mla=True)))


def test_dsa_index_tier_is_rejected():
    with pytest.raises(ValueError, match="MLA/DSA"):
        _validate_kv_cache_dtype(_Cfg(), _Model(_spec(index_head_dim=128)))


def test_head_dim_not_a_multiple_of_the_block_is_rejected():
    with pytest.raises(ValueError, match="multiple of 32"):
        _validate_kv_cache_dtype(_Cfg(), _Model(_spec(head_dim=80)))


def test_the_error_names_the_offending_group():
    with pytest.raises(ValueError, match=r"swa \(head_dim 100\)"):
        _validate_kv_cache_dtype(_Cfg(), _Model(_spec(head_dim=256), _spec("swa", 100)))


def test_q8_k_q6_v_pair_is_accepted():
    _validate_kv_cache_dtype(
        _Cfg(quant_k=Q8_0, quant_v=Q6_0), _Model(_spec(head_dim=256))
    )


def test_q6_k_q5_v_pair_is_accepted():
    _validate_kv_cache_dtype(
        _Cfg(quant_k=Q6_0, quant_v=Q5_0), _Model(_spec(head_dim=256))
    )


def test_other_asymmetric_pair_is_rejected():
    with pytest.raises(ValueError, match="not a validated kernel pair"):
        _validate_kv_cache_dtype(
            _Cfg(quant_k=Q8_0, quant_v=INT4), _Model(_spec(head_dim=256))
        )


def test_q6_is_not_exposed_as_a_symmetric_format_yet():
    with pytest.raises(ValueError, match="validated asymmetric"):
        _validate_kv_cache_dtype(_Cfg(quant=Q6_0), _Model(_spec(head_dim=256)))


def test_q5_is_not_exposed_as_a_symmetric_format_yet():
    with pytest.raises(ValueError, match="validated asymmetric"):
        _validate_kv_cache_dtype(_Cfg(quant=Q5_0), _Model(_spec(head_dim=256)))
