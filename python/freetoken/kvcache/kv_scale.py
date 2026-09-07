from __future__ import annotations

import math
from collections.abc import Sequence

import torch

# e4m3's largest finite magnitude. flashinfer's fp8 KV path assumes this saturation bound.
FP8_E4M3_MAX = 448.0
# Floor for a derived scale, so an all-zero activation cannot produce a divide-by-zero.
SCALE_FLOOR = 1e-12


class KVScaleTable:
    """One FROZEN (k_scale, v_scale) pair per KV layer, for a per-tensor fp8 KV cache.

    flashinfer's fa2 path takes SCALAR k_scale/v_scale and folds them algebraically
    (``sm_scale *= k_scale``, ``out *= v_scale``) rather than dequantizing in-kernel, so a
    per-tensor scale is exact there -- and per-tensor is the only granularity that ABI can
    express.

    The freeze is the load-bearing property. A stored fp8 value has already been divided by
    the scale, so changing the scale later would silently corrupt every token already in the
    pool. A running amax is therefore not an option: once ``ensure`` hands a pair out for a
    layer, that pair is fixed for the life of the process.

    Keyed by GLOBAL layer id, and ``layer_ids`` enumerates exactly which ids are legal --
    on this model the 10 full-attention layers out of 40, the rest being Gated DeltaNet with
    no paged KV. A bare count would not do: the pool, the checkpoint reader and the store
    path all speak global ids, and validating against a count would let a wrong-key-space bug
    (dense 0..9 vs global 3..39) pad ``all_frozen()`` while a real layer stayed uncalibrated.
    ``all_frozen()`` is the pre-graph-capture gate, so it must not be satisfiable by accident.
    """

    def __init__(self, layer_ids: Sequence[int], margin: float = 2.0) -> None:
        valid = frozenset(int(i) for i in layer_ids)
        if not valid:
            raise ValueError("layer_ids must name at least one KV layer")
        if margin < 1.0:
            raise ValueError(f"margin must be >= 1.0, got {margin}")
        self._valid = valid
        self._margin = float(margin)
        self._scales: dict[int, tuple[float, float]] = {}
        self._from_checkpoint: set[int] = set()
        self._probed: set[int] = set()
        self._clamp_probe: dict[int, tuple[float, float]] = {}

    @property
    def layer_ids(self) -> frozenset[int]:
        """The global ids this table accepts. Iterate these, never ``range(len(...))``."""
        return self._valid

    def _check_layer(self, layer_id: int) -> None:
        if layer_id not in self._valid:
            raise KeyError(
                f"layer {layer_id} is not a KV layer of this pool; valid ids are "
                f"{sorted(self._valid)}"
            )

    def set_checkpoint(self, layer_id: int, k_scale: float, v_scale: float) -> None:
        """Install a calibrated pair shipped by the checkpoint. Must precede any ``ensure``."""
        self._check_layer(layer_id)
        if layer_id in self._scales:
            raise RuntimeError(
                f"layer {layer_id} KV scales are already frozen; a checkpoint scale must be "
                "installed before the first store"
            )
        for name, value in (("k_scale", k_scale), ("v_scale", v_scale)):
            fv = float(value)
            if not math.isfinite(fv) or fv <= 0.0:
                raise ValueError(
                    f"checkpoint {name} for layer {layer_id} must be finite and positive, "
                    f"got {value!r}"
                )
        self._scales[layer_id] = (float(k_scale), float(v_scale))
        self._from_checkpoint.add(layer_id)

    def ensure(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> tuple[float, float]:
        """Return this layer's frozen pair, deriving and freezing it from k/v on first call."""
        self._check_layer(layer_id)
        cached = self._scales.get(layer_id)
        if cached is not None:
            # A checkpoint scale was calibrated by a different quantizer on different data.
            # Probe it ONCE against real activations so a mismatched checkpoint is visible
            # instead of silently saturating. One sync per checkpoint layer, during
            # calibration only -- never on the steady-state path.
            if layer_id in self._from_checkpoint and layer_id not in self._probed:
                self._probed.add(layer_id)
                self._clamp_probe[layer_id] = self.clamp_ratio(layer_id, k, v)
            return cached
        pair = (self._observe(k, layer_id, "k"), self._observe(v, layer_id, "v"))
        self._scales[layer_id] = pair
        return pair

    def clamp_probe(self) -> dict[int, tuple[float, float]]:
        """{layer_id: (k_ratio, v_ratio)} for checkpoint layers seen once. > 1.0 clamps."""
        return dict(self._clamp_probe)

    def get(self, layer_id: int) -> tuple[float, float]:
        pair = self._scales.get(layer_id)
        if pair is None:
            raise KeyError(f"layer {layer_id} has no frozen KV scale yet")
        return pair

    def frozen(self, layer_id: int) -> bool:
        return layer_id in self._scales

    def all_frozen(self) -> bool:
        """True only when EVERY declared KV layer has a scale -- not merely enough of them."""
        return self._valid.issubset(self._scales.keys())

    def is_from_checkpoint(self, layer_id: int) -> bool:
        return layer_id in self._from_checkpoint

    def clamp_ratio(
        self, layer_id: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[float, float]:
        """observed amax / representable max, per tensor. > 1.0 means these values clamp."""
        k_scale, v_scale = self.get(layer_id)
        return (
            self._amax(k) / (k_scale * FP8_E4M3_MAX),
            self._amax(v) / (v_scale * FP8_E4M3_MAX),
        )

    def _observe(self, t: torch.Tensor, layer_id: int, which: str) -> float:
        amax = self._amax(t)
        # SCALE_FLOOR cannot rescue a NaN: max(nan, 1e-12) is nan, because every comparison
        # against nan is False. Freezing a non-finite scale would divide every stored value by
        # it forever -- exactly the silent, permanent corruption the freeze exists to prevent.
        # set_checkpoint already rejects non-finite scales; the derived path must match.
        if not math.isfinite(amax):
            raise ValueError(
                f"layer {layer_id} {which} activations have a non-finite amax ({amax!r}); "
                "cannot derive a KV scale from them"
            )
        return max(amax / FP8_E4M3_MAX * self._margin, SCALE_FLOOR)

    @staticmethod
    def _amax(t: torch.Tensor) -> float:
        # One host sync per layer, once, during calibration. Never on the steady-state path.
        return float(t.detach().float().abs().amax().item())
