from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool
from .kv_scale import FP8_E4M3_MAX, KVScaleTable


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_scales: KVScaleTable | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        # None restores the unquantized path exactly. When present, store_kv scales-and-casts
        # into the fp8 slabs and the attention backend passes the same scalars to flashinfer.
        self._kv_scales = kv_scales
        # `dtype` above is the SLAB dtype, which is fp8 when the pool is quantized.
        # The attention backend still needs the compute dtype for q and the output.
        self._compute_dtype = compute_dtype if compute_dtype is not None else dtype
        # Device-side clamp tally. Accumulated without a sync so the steady-state store path
        # stays free of host round-trips; read only by the reporting path.
        self._clamp_count = torch.zeros((), dtype=torch.int64, device=device)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        return int(buf.numel() * buf.element_size()) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        if self._kv_scales is not None:
            # First store for this layer freezes its scale. store_cache is a raw byte copy,
            # so k/v must already be fp8 with rows matching the slab's element_size.
            k_scale, v_scale = self._kv_scales.ensure(layer_id, k, v)
            k = self._to_fp8(k, k_scale)
            v = self._to_fp8(v, v_scale)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    def _to_fp8(self, t: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale-and-cast to e4m3, tallying saturated elements without a host sync.

        Non-in-place div/clamp deliberately: ``t.detach().float()`` shares storage (it is not
        a copy) whenever ``t`` is already float32, so in-place ops here would silently mutate
        the CALLER's k/v tensor. Today's engine runs bf16, where ``.float()`` does copy, so
        this is latent rather than live -- but it costs one allocation on a path the roofline
        does not care about, and a caller that passes fp32 would otherwise be corrupted with
        no symptom at the call site. Do not "simplify" this back to div_/clamp_.
        """
        scaled = t.detach().float().div(scale)
        self._clamp_count += (scaled.abs() > FP8_E4M3_MAX).sum()
        return scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def kv_scale(self, layer_id: int) -> tuple[float, float] | None:
        """This layer's frozen (k_scale, v_scale), or None when the pool is unquantized."""
        if self._kv_scales is None:
            return None
        return self._kv_scales.get(layer_id)

    @property
    def kv_scales(self) -> KVScaleTable | None:
        return self._kv_scales

    def set_checkpoint_scales(self, scales: dict[int, tuple[float, float]]) -> None:
        """Install checkpoint-calibrated scales, keyed by GLOBAL layer id (same key space
        the store path and the attention backend use)."""
        if self._kv_scales is None or not scales:
            return
        for layer_id, (k_scale, v_scale) in scales.items():
            self._kv_scales.set_checkpoint(layer_id, k_scale, v_scale)

    def clamp_count(self) -> int:
        """Total elements saturated at +/-448 since boot. One host sync; call off the hot path."""
        return int(self._clamp_count.item())
