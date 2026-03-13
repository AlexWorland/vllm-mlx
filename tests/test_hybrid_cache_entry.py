# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid cache entry dataclass and helpers."""

import time

import pytest

from vllm_mlx.hybrid_cache_entry import (
    HybridCacheEntry,
    classify_cache_layers,
    estimate_hybrid_memory,
)


# ---------------------------------------------------------------------------
# Mock objects — simulate MLX arrays and cache layers without importing mlx
# ---------------------------------------------------------------------------


class MockDtype:
    """Mock dtype with size attribute (e.g., float16 = 2 bytes)."""

    def __init__(self, size: int):
        self.size = size


class MockArray:
    """Mock MLX array with shape and dtype but no .nbytes (avoids lazy eval)."""

    def __init__(self, shape: tuple, dtype_size: int = 2):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockNbytesArray:
    """Mock array with only .nbytes (fallback path)."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockKVCache:
    """Mock KVCache layer — has .keys, .values arrays and is_trimmable() -> True."""

    def __init__(
        self,
        seq_len: int = 128,
        n_heads: int = 8,
        head_dim: int = 64,
        dtype_size: int = 2,
    ):
        # Shape: (batch=1, n_heads, seq_len, head_dim)
        self.keys = MockArray((1, n_heads, seq_len, head_dim), dtype_size)
        self.values = MockArray((1, n_heads, seq_len, head_dim), dtype_size)
        self.offset = seq_len

    def is_trimmable(self) -> bool:
        return True


class MockArraysCache:
    """Mock ArraysCache/MambaCache — has .cache list of arrays, is_trimmable() -> False."""

    def __init__(
        self,
        num_arrays: int = 2,
        state_shape: tuple = (1, 64, 128),
        dtype_size: int = 2,
    ):
        self.cache = [MockArray(state_shape, dtype_size) for _ in range(num_arrays)]

    def is_trimmable(self) -> bool:
        return False


class MockQuantizedKVCache:
    """Mock QuantizedKVCache — keys/values are tuples of (data, scales, biases)."""

    def __init__(self, seq_len: int = 128, dtype_size: int = 1):
        # Quantized: keys/values are tuples of arrays
        self.keys = (
            MockArray((1, 8, seq_len, 64), dtype_size),  # data
            MockArray((1, 8, seq_len, 1), 2),  # scales
            MockArray((1, 8, seq_len, 1), 2),  # biases
        )
        self.values = (
            MockArray((1, 8, seq_len, 64), dtype_size),
            MockArray((1, 8, seq_len, 1), 2),
            MockArray((1, 8, seq_len, 1), 2),
        )

    def is_trimmable(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Tests for estimate_hybrid_memory
# ---------------------------------------------------------------------------


class TestEstimateHybridMemory:
    """Tests for estimate_hybrid_memory function."""

    def test_empty_cache(self):
        assert estimate_hybrid_memory([]) == 0
        assert estimate_hybrid_memory(None) == 0

    def test_cache_with_none_layers(self):
        assert estimate_hybrid_memory([None, None]) == 0

    def test_kvcache_layers(self):
        """KVCache layer: memory = 2 * (1 * n_heads * seq_len * head_dim * dtype_size)."""
        layer = MockKVCache(seq_len=128, n_heads=8, head_dim=64, dtype_size=2)
        expected = 2 * (1 * 8 * 128 * 64 * 2)  # keys + values
        assert estimate_hybrid_memory([layer]) == expected

    def test_arrays_cache_layers(self):
        """ArraysCache layer: memory = sum of all arrays in .cache list."""
        layer = MockArraysCache(
            num_arrays=3, state_shape=(1, 64, 128), dtype_size=2
        )
        expected = 3 * (1 * 64 * 128 * 2)
        assert estimate_hybrid_memory([layer]) == expected

    def test_mixed_kv_and_ssm_layers(self):
        """Mixed cache with both KVCache and ArraysCache layers."""
        kv = MockKVCache(seq_len=100, n_heads=4, head_dim=32, dtype_size=2)
        ssm = MockArraysCache(num_arrays=2, state_shape=(1, 32, 64), dtype_size=2)

        kv_expected = 2 * (1 * 4 * 100 * 32 * 2)
        ssm_expected = 2 * (1 * 32 * 64 * 2)

        assert estimate_hybrid_memory([kv, ssm]) == kv_expected + ssm_expected

    def test_quantized_kv_cache(self):
        """QuantizedKVCache: keys/values are tuples of arrays."""
        layer = MockQuantizedKVCache(seq_len=128, dtype_size=1)
        # keys: (data, scales, biases) — each with their own shape
        # data: 1*8*128*64*1, scales: 1*8*128*1*2, biases: 1*8*128*1*2
        data_mem = 1 * 8 * 128 * 64 * 1
        scales_mem = 1 * 8 * 128 * 1 * 2
        biases_mem = 1 * 8 * 128 * 1 * 2
        per_kv = data_mem + scales_mem + biases_mem
        expected = 2 * per_kv  # keys + values
        assert estimate_hybrid_memory([layer]) == expected

    def test_dict_state_layers(self):
        """Dict-based extracted state: {'state': (keys_arr, values_arr)}."""
        keys = MockArray((1, 8, 64, 32), 2)
        values = MockArray((1, 8, 64, 32), 2)
        layer = {"state": (keys, values)}

        expected = 2 * (1 * 8 * 64 * 32 * 2)
        assert estimate_hybrid_memory([layer]) == expected

    def test_multiple_layers_accumulate(self):
        """Memory from multiple layers is summed."""
        layers = [MockKVCache(seq_len=50, n_heads=4, head_dim=32, dtype_size=2)
                  for _ in range(4)]
        per_layer = 2 * (1 * 4 * 50 * 32 * 2)
        assert estimate_hybrid_memory(layers) == 4 * per_layer


# ---------------------------------------------------------------------------
# Tests for classify_cache_layers
# ---------------------------------------------------------------------------


class TestClassifyCacheLayers:
    """Tests for classify_cache_layers function."""

    def test_all_kv_layers(self):
        layers = [MockKVCache() for _ in range(4)]
        kv_idx, ssm_idx = classify_cache_layers(layers)
        assert kv_idx == [0, 1, 2, 3]
        assert ssm_idx == []

    def test_all_ssm_layers(self):
        layers = [MockArraysCache() for _ in range(3)]
        kv_idx, ssm_idx = classify_cache_layers(layers)
        assert kv_idx == []
        assert ssm_idx == [0, 1, 2]

    def test_mixed_layers(self):
        """Interleaved KV and SSM layers (like Qwen3.5)."""
        layers = [
            MockArraysCache(),  # 0: SSM
            MockArraysCache(),  # 1: SSM
            MockArraysCache(),  # 2: SSM
            MockKVCache(),      # 3: KV
            MockArraysCache(),  # 4: SSM
            MockArraysCache(),  # 5: SSM
            MockArraysCache(),  # 6: SSM
            MockKVCache(),      # 7: KV
        ]
        kv_idx, ssm_idx = classify_cache_layers(layers)
        assert kv_idx == [3, 7]
        assert ssm_idx == [0, 1, 2, 4, 5, 6]

    def test_none_layers_skipped(self):
        layers = [None, MockKVCache(), None, MockArraysCache()]
        kv_idx, ssm_idx = classify_cache_layers(layers)
        assert kv_idx == [1]
        assert ssm_idx == [3]

    def test_empty_cache(self):
        kv_idx, ssm_idx = classify_cache_layers([])
        assert kv_idx == []
        assert ssm_idx == []

    def test_fallback_without_is_trimmable(self):
        """Layers without is_trimmable() use structural checks."""

        class BareKV:
            def __init__(self):
                self.keys = MockArray((1, 4, 64, 32))
                self.values = MockArray((1, 4, 64, 32))

        class BareSSM:
            def __init__(self):
                self.cache = [MockArray((1, 32, 64))]

        layers = [BareKV(), BareSSM()]
        kv_idx, ssm_idx = classify_cache_layers(layers)
        assert kv_idx == [0]
        assert ssm_idx == [1]


# ---------------------------------------------------------------------------
# Tests for HybridCacheEntry
# ---------------------------------------------------------------------------


class TestHybridCacheEntry:
    """Tests for HybridCacheEntry dataclass."""

    def test_create_with_kv_only(self):
        kv = [MockKVCache(seq_len=64, n_heads=4, head_dim=32, dtype_size=2)]
        entry = HybridCacheEntry.create(
            token_key=(1, 2, 3, 4),
            kv_states=kv,
        )

        assert entry.token_key == (1, 2, 3, 4)
        assert entry.kv_states is kv
        assert entry.ssm_states is None
        assert entry.ssm_checkpoint_token == 0
        assert entry.entry_type == "prefix"
        assert entry.access_count == 1
        assert entry.memory_bytes > 0
        # Memory = 2 * (1 * 4 * 64 * 32 * 2) for one KV layer
        expected_mem = 2 * (1 * 4 * 64 * 32 * 2)
        assert entry.memory_bytes == expected_mem

    def test_create_with_kv_and_ssm(self):
        kv = [MockKVCache(seq_len=64, n_heads=4, head_dim=32)]
        ssm = [MockArraysCache(num_arrays=2, state_shape=(1, 32, 64))]

        entry = HybridCacheEntry.create(
            token_key=(10, 20, 30),
            kv_states=kv,
            ssm_states=ssm,
            ssm_checkpoint_token=3,
            entry_type="branch_point",
        )

        assert entry.ssm_states is ssm
        assert entry.ssm_checkpoint_token == 3
        assert entry.entry_type == "branch_point"

        kv_mem = 2 * (1 * 4 * 64 * 32 * 2)
        ssm_mem = 2 * (1 * 32 * 64 * 2)
        assert entry.memory_bytes == kv_mem + ssm_mem

    def test_touch_updates_metadata(self):
        entry = HybridCacheEntry.create(
            token_key=(1,),
            kv_states=[MockKVCache()],
        )

        initial_access = entry.last_access
        initial_count = entry.access_count

        time.sleep(0.01)
        entry.touch()

        assert entry.last_access > initial_access
        assert entry.access_count == initial_count + 1

    def test_has_ssm_states(self):
        entry_no_ssm = HybridCacheEntry.create(
            token_key=(1,), kv_states=[MockKVCache()]
        )
        assert entry_no_ssm.has_ssm_states() is False

        entry_with_ssm = HybridCacheEntry.create(
            token_key=(1,),
            kv_states=[MockKVCache()],
            ssm_states=[MockArraysCache()],
        )
        assert entry_with_ssm.has_ssm_states() is True

    def test_has_ssm_states_empty_list(self):
        entry = HybridCacheEntry.create(
            token_key=(1,),
            kv_states=[MockKVCache()],
            ssm_states=[],
        )
        assert entry.has_ssm_states() is False

    def test_timestamps_use_monotonic(self):
        before = time.monotonic()
        entry = HybridCacheEntry.create(
            token_key=(1,), kv_states=[MockKVCache()]
        )
        after = time.monotonic()

        assert before <= entry.created_at <= after
        assert before <= entry.last_access <= after

    def test_flop_savings_default(self):
        entry = HybridCacheEntry.create(
            token_key=(1,), kv_states=[MockKVCache()]
        )
        assert entry.flop_savings == 0.0
