"""qwen3_5_moe compressed-tensors fp8 detection: which weight geometries route to the
per-tensor fp8 layer.

``_attn_quant`` is a routing key, not just a label: ``iter_weights`` dispatches a
compressed-tensors MoE checkpoint to ``_iter_weights_attn_fp8`` only when it returns
``"fp8_pertensor"``. A checkpoint that is not detected does not merely lose an optimisation --
it reaches a branch that cannot read its tensors.

Runs off trimmed copies of real ``quantization_config`` blocks.
"""

from __future__ import annotations

import pytest

from freetoken.models.qwen3_5_moe.config import _attn_quant


class _Cfg:
    def __init__(self, quantization_config: dict):
        self.quantization_config = quantization_config


def _ct_config(strategy: str | None, *, group_size=None, num_bits: int = 8,
               wtype: str = "float", targets=(".self_attn.q_proj",)) -> _Cfg:
    return _Cfg({
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "targets": list(targets),
                "weights": {"num_bits": num_bits, "type": wtype,
                            "strategy": strategy, "group_size": group_size},
            },
        },
    })


@pytest.mark.parametrize("strategy", ["tensor", "channel"])
def test_per_tensor_and_per_channel_both_route_to_the_fp8_layer(strategy):
    """Fp8PerTensorLinear's weight_scale is per output row. A per-tensor weight stores the
    same scalar in every row, so neither strategy needs a different layer."""
    assert _attn_quant(_ct_config(strategy)) == "fp8_pertensor"


@pytest.mark.parametrize("strategy", ["group", "block", "tensor_group", None])
def test_block_and_group_scales_do_not_route_to_the_fp8_layer(strategy):
    """A block or group scale is a different layout; routing it here would mis-read it."""
    assert _attn_quant(_ct_config(strategy, group_size=128)) == "none"


def test_a_group_size_disqualifies_even_a_named_per_channel_strategy():
    assert _attn_quant(_ct_config("channel", group_size=16)) == "none"


@pytest.mark.parametrize("num_bits,wtype", [(4, "float"), (8, "int"), (4, "int")])
def test_only_8_bit_float_weights_route_to_the_fp8_layer(num_bits, wtype):
    assert _attn_quant(_ct_config("channel", num_bits=num_bits, wtype=wtype)) == "none"


@pytest.mark.parametrize("targets", [
    (".linear_attn.in_proj_qkv",),
    ("re:.*\\.self_attn\\..*",),
])
def test_gdn_and_regex_targets_are_recognised(targets):
    """Targets are regex strings in the checkpoint, so the match is on the leaf path."""
    assert _attn_quant(_ct_config("channel", targets=targets)) == "fp8_pertensor"


def test_a_group_that_targets_neither_attention_nor_gdn_is_ignored():
    """An fp8 group covering only the experts must not claim the dense projections."""
    assert _attn_quant(_ct_config("channel", targets=("re:.*mlp.experts.*",))) == "none"


def test_no_quantization_config_is_not_fp8():
    assert _attn_quant(_Cfg(None)) == "none"
