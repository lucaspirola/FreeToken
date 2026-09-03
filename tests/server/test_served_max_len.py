"""Served context window = min(positional geometry, tokenizer window).

Nemotron-3.5 Lightning ships ``max_position_embeddings`` 1,048,576 against a
tokenizer ``model_max_length`` of 262,144. Advertising the geometry makes the
server accept prompts the publisher never validated, and -- for a Switchyard
upstream -- pushes the 400 ``context_length_exceeded`` fallthrough past the
window the router was told about. The served default therefore clamps to the
tokenizer window whenever the checkpoint states a finite one.

No weights are needed: ``EngineConfig.max_seq_len`` reads only ``config.json``
(through the model registry) and ``tokenizer_config.json``.
"""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig

# A minimal Llama config: real enough for the registry's parse_config, tiny
# enough to write in a tmp dir. Only rotary_config.max_position is read here.
_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16",
}


def _checkpoint(tmp_path, *, tokenizer_config: dict | None) -> str:
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    if tokenizer_config is not None:
        (path / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config), encoding="utf-8"
        )
    return str(path)


def _engine(model_path: str, **overrides) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )


def test_tokenizer_window_clamps_served_length(tmp_path):
    # Lightning's exact pair: 262144 tokenizer window vs 1M positions.
    path = _checkpoint(tmp_path, tokenizer_config={"model_max_length": 262144})
    config = _engine(path)
    assert config.model_config.rotary_config.max_position == 1048576
    assert config.tokenizer_model_max_length == 262144
    assert config.max_seq_len == 262144
    assert config.max_forward_len == 262144


def test_override_wins_over_tokenizer_window(tmp_path):
    path = _checkpoint(tmp_path, tokenizer_config={"model_max_length": 262144})
    # The P2 launch profile pins 131072; an override above the tokenizer window
    # is still the operator's call and must not be clamped.
    assert _engine(path, max_seq_len_override=131072).max_seq_len == 131072
    assert _engine(path, max_seq_len_override=524288).max_seq_len == 524288


def test_missing_tokenizer_config_keeps_geometry(tmp_path):
    path = _checkpoint(tmp_path, tokenizer_config=None)
    config = _engine(path)
    assert config.tokenizer_model_max_length is None
    assert config.max_seq_len == 1048576


@pytest.mark.parametrize(
    "value",
    [
        int(1e30),  # transformers VERY_LARGE_INTEGER
        1000000000000000019884624838656,  # int(1e30) as written by tokenizers
        1e30,  # float form
        None,
        "262144",  # string: not a usable bound
        0,
        -1,
        True,  # bool is not a length
    ],
    ids=[
        "sentinel_int",
        "sentinel_bigint",
        "sentinel_float",
        "null",
        "string",
        "zero",
        "negative",
        "bool",
    ],
)
def test_sentinel_and_malformed_values_keep_geometry(tmp_path, value):
    path = _checkpoint(tmp_path, tokenizer_config={"model_max_length": value})
    config = _engine(path)
    assert config.tokenizer_model_max_length is None
    assert config.max_seq_len == 1048576


def test_tokenizer_window_above_geometry_is_not_raised(tmp_path):
    # Some checkpoints state a tokenizer window larger than the rope geometry
    # (or equal to it). The served window must never exceed max_position.
    path = _checkpoint(tmp_path, tokenizer_config={"model_max_length": 4194304})
    assert _engine(path).max_seq_len == 1048576


def test_unreadable_tokenizer_config_keeps_geometry(tmp_path):
    path = _checkpoint(tmp_path, tokenizer_config=None)
    (tmp_path / "ckpt" / "tokenizer_config.json").write_text("{not json", "utf-8")
    config = _engine(path)
    assert config.tokenizer_model_max_length is None
    assert config.max_seq_len == 1048576
