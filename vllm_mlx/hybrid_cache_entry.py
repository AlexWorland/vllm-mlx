# SPDX-License-Identifier: Apache-2.0
"""
Hybrid cache entry for models with mixed attention + recurrent (SSM) layers.

This module defines the cache entry dataclass used by the hybrid prefix cache,
along with helper functions for memory estimation and layer classification
in models like Qwen3.5 that interleave KVCache (attention) and
MambaCache/ArraysCache (recurrent/GatedDeltaNet) layers.

Design notes:
- Memory estimation uses shape*dtype.size to avoid triggering MLX lazy eval
- MLX arrays are immutable, so cache references can be shared safely
- Layer classification uses is_trimmable() as the canonical discriminator
  between attention (trimmable) and recurrent (non-trimmable) layers
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _array_memory(arr: Any) -> int:
    """Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing .nbytes on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute
    the same value.

    Args:
        arr: An MLX array or similar object with shape and dtype attributes.

    Returns:
        Estimated memory in bytes.
    """
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def estimate_hybrid_memory(cache: list[Any]) -> int:
    """Estimate memory usage of a hybrid cache (mixed KV + SSM layers).

    Handles two layer types:
    - KVCache layers: have .keys/.values arrays (attention layers)
    - ArraysCache/MambaCache layers: have .cache list of arrays (SSM layers)
    - QuantizedKVCache layers: .keys/.values are tuples of (data, scales, biases)

    Uses shape*dtype.size instead of .nbytes to avoid triggering MLX lazy eval.

    Args:
        cache: List of per-layer cache objects (KVCache, ArraysCache, MambaCache, etc).

    Returns:
        Estimated total memory in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        if layer_cache is None:
            continue

        # Dict-based state (extracted cache) — check FIRST since dicts
        # have .keys() method that would match KVCache checks below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            state = layer_cache["state"]
            if isinstance(state, (list, tuple)):
                for arr in state:
                    if arr is not None:
                        total_bytes += _array_memory(arr)

        # QuantizedKVCache: keys/values are tuples of (data, scales, biases)
        elif hasattr(layer_cache, "keys") and isinstance(
            getattr(layer_cache, "keys", None), (list, tuple)
        ):
            for arr in layer_cache.keys:
                total_bytes += _array_memory(arr)
            for arr in layer_cache.values:
                total_bytes += _array_memory(arr)

        # Standard KVCache: .keys and .values are arrays
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            if not callable(keys_attr):
                total_bytes += _array_memory(keys_attr)
            if not callable(values_attr):
                total_bytes += _array_memory(values_attr)

        # ArraysCache / MambaCache: .cache is a list of arrays
        elif hasattr(layer_cache, "cache") and isinstance(layer_cache.cache, list):
            for arr in layer_cache.cache:
                if arr is not None:
                    total_bytes += _array_memory(arr)

    return total_bytes


def classify_cache_layers(cache: list[Any]) -> Tuple[List[int], List[int]]:
    """Classify cache layers into attention (KV) and recurrent (SSM) indices.

    Uses the is_trimmable() check as the canonical discriminator:
    - KVCache layers: is_trimmable() returns True (or has keys/values arrays)
    - ArraysCache/MambaCache layers: is_trimmable() returns False (or has .cache list)

    This matches the pattern used throughout the vllm-mlx scheduler.

    Args:
        cache: List of per-layer cache objects.

    Returns:
        Tuple of (kv_layer_indices, ssm_layer_indices) — indices into the cache list.
    """
    kv_indices: List[int] = []
    ssm_indices: List[int] = []

    for i, layer_cache in enumerate(cache):
        if layer_cache is None:
            continue

        # Primary check: is_trimmable() method
        if hasattr(layer_cache, "is_trimmable"):
            if layer_cache.is_trimmable():
                kv_indices.append(i)
            else:
                ssm_indices.append(i)
        # Fallback: structural checks
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            kv_indices.append(i)
        elif hasattr(layer_cache, "cache") and isinstance(
            getattr(layer_cache, "cache", None), list
        ):
            ssm_indices.append(i)

    return kv_indices, ssm_indices


@dataclass
class HybridCacheEntry:
    """Cache entry holding both KV (attention) and SSM (recurrent) states.

    Used by RadixCacheHybrid to store prefix cache entries for models with
    mixed attention + recurrent layers (e.g., Qwen3.5 with GatedDeltaNet).

    Attributes:
        token_key: Token sequence this entry caches.
        kv_states: Per attention layer cache objects (always stored).
        ssm_states: Per recurrent layer state snapshots (conditionally stored).
        ssm_checkpoint_token: Token position where the SSM state was snapshotted.
        entry_type: Classification — "exact", "prefix", or "branch_point".
        created_at: Monotonic timestamp of creation.
        last_access: Monotonic timestamp of last access.
        access_count: Number of times this entry has been accessed.
        memory_bytes: Total estimated memory footprint in bytes.
        flop_savings: Estimated FLOPs saved by reusing this entry.
    """

    token_key: Tuple[int, ...]
    kv_states: List[Any]
    ssm_states: Optional[List[Any]] = None
    ssm_checkpoint_token: int = 0
    entry_type: str = "prefix"
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0
    memory_bytes: int = 0
    flop_savings: float = 0.0

    def touch(self) -> None:
        """Update access metadata."""
        self.last_access = time.monotonic()
        self.access_count += 1

    def has_ssm_states(self) -> bool:
        """Check if SSM states are stored."""
        return self.ssm_states is not None and len(self.ssm_states) > 0

    @classmethod
    def create(
        cls,
        token_key: Tuple[int, ...],
        kv_states: List[Any],
        ssm_states: Optional[List[Any]] = None,
        ssm_checkpoint_token: int = 0,
        entry_type: str = "prefix",
    ) -> HybridCacheEntry:
        """Create a HybridCacheEntry with automatic memory estimation.

        Args:
            token_key: Token sequence.
            kv_states: Per attention layer cache objects.
            ssm_states: Per recurrent layer state snapshots.
            ssm_checkpoint_token: Token position of SSM snapshot.
            entry_type: Classification type.

        Returns:
            New HybridCacheEntry with memory_bytes computed.
        """
        memory = estimate_hybrid_memory(kv_states)
        if ssm_states:
            memory += estimate_hybrid_memory(ssm_states)

        now = time.monotonic()
        return cls(
            token_key=token_key,
            kv_states=kv_states,
            ssm_states=ssm_states,
            ssm_checkpoint_token=ssm_checkpoint_token,
            entry_type=entry_type,
            created_at=now,
            last_access=now,
            access_count=1,
            memory_bytes=memory,
            flop_savings=0.0,
        )
