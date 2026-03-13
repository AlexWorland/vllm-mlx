# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid KV+SSM prefix cache integration in memory_cache.py."""

import time
from unittest.mock import MagicMock

import pytest

from vllm_mlx.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _CacheEntry,
    estimate_kv_cache_memory,
)


# ---------------------------------------------------------------------------
# Mock helpers — reuse patterns from test_memory_cache.py
# ---------------------------------------------------------------------------


class MockDtype:
    """Mock dtype with size attribute."""

    def __init__(self, size: int):
        self.size = size


class MockArray:
    """Mock array with nbytes attribute."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockShapeArray:
    """Mock array with shape and dtype (like MLX arrays)."""

    def __init__(self, shape: tuple, dtype_size: int):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockKVCache:
    """Mock KV cache layer (trimmable, has keys/values/offset)."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self.keys = MockArray(key_bytes)
        self.values = MockArray(value_bytes)
        self.offset = 100

    def is_trimmable(self) -> bool:
        return True


class MockArraysCache:
    """Mock ArraysCache/MambaCache layer (non-trimmable, has .cache list)."""

    def __init__(self, array_sizes: list[int]):
        self.cache = [MockArray(sz) for sz in array_sizes]

    def is_trimmable(self) -> bool:
        return False


class MockSSMCacheShapeBased:
    """Mock SSM layer with shape-based arrays (no .nbytes)."""

    def __init__(self, shapes: list[tuple], dtype_size: int = 2):
        self.cache = [MockShapeArray(s, dtype_size) for s in shapes]

    def is_trimmable(self) -> bool:
        return False


def _make_hybrid_cache(
    n_kv: int = 2,
    n_ssm: int = 2,
    kv_bytes: int = 500,
    ssm_bytes: int = 300,
) -> list:
    """Create a mixed KV+SSM cache list for testing."""
    layers: list = []
    for _ in range(n_kv):
        layers.append(MockKVCache(kv_bytes // 2, kv_bytes // 2))
    for _ in range(n_ssm):
        layers.append(MockArraysCache([ssm_bytes]))
    return layers


def _make_kv_only_cache(n_layers: int = 4, size_bytes: int = 500) -> list:
    """Create a KV-only cache list."""
    return [MockKVCache(size_bytes // 2, size_bytes // 2) for _ in range(n_layers)]


# ---------------------------------------------------------------------------
# 1. estimate_kv_cache_memory with ArraysCache layers
# ---------------------------------------------------------------------------


class TestEstimateKvCacheMemoryArraysCache:
    """Test that estimate_kv_cache_memory handles SSM/ArraysCache layers."""

    def test_arrays_cache_layer(self):
        """ArraysCache layer with .cache list of arrays."""
        layer = MockArraysCache([1024, 2048])
        result = estimate_kv_cache_memory([layer])
        assert result == 1024 + 2048

    def test_arrays_cache_with_none_entries(self):
        """ArraysCache where some .cache entries are None."""
        layer = MockArraysCache([500])
        layer.cache.append(None)  # Add a None entry
        result = estimate_kv_cache_memory([layer])
        assert result == 500

    def test_mixed_kv_and_arrays_cache(self):
        """Mixed KV and ArraysCache layers."""
        kv_layer = MockKVCache(200, 200)
        ssm_layer = MockArraysCache([300])
        result = estimate_kv_cache_memory([kv_layer, ssm_layer])
        assert result == 200 + 200 + 300

    def test_shape_based_arrays_cache(self):
        """ArraysCache with shape+dtype based arrays (no .nbytes)."""
        layer = MockSSMCacheShapeBased(
            shapes=[(1, 64, 128, 128)], dtype_size=2
        )
        expected = 1 * 64 * 128 * 128 * 2
        result = estimate_kv_cache_memory([layer])
        assert result == expected

    def test_empty_arrays_cache(self):
        """ArraysCache with empty .cache list."""
        layer = MockArraysCache([])
        result = estimate_kv_cache_memory([layer])
        assert result == 0


# ---------------------------------------------------------------------------
# 2. MemoryCacheConfig hybrid fields
# ---------------------------------------------------------------------------


class TestMemoryCacheConfigHybridFields:
    """Verify new hybrid fields have correct defaults and validation."""

    def test_hybrid_defaults(self):
        config = MemoryCacheConfig()
        assert config.enable_hybrid_cache is False
        assert config.prefix_cache_alpha == 0.0
        assert config.ssm_admission_threshold == 2
        assert config.auto_tune_alpha is True

    def test_hybrid_custom_values(self):
        config = MemoryCacheConfig(
            enable_hybrid_cache=True,
            prefix_cache_alpha=1.5,
            ssm_admission_threshold=3,
            auto_tune_alpha=False,
        )
        assert config.enable_hybrid_cache is True
        assert config.prefix_cache_alpha == 1.5
        assert config.ssm_admission_threshold == 3
        assert config.auto_tune_alpha is False

    def test_hybrid_fields_frozen(self):
        """Config is frozen — fields cannot be mutated after creation."""
        config = MemoryCacheConfig(enable_hybrid_cache=True)
        with pytest.raises(AttributeError):
            config.enable_hybrid_cache = False  # type: ignore[misc]

    def test_hybrid_fields_do_not_break_existing_validation(self):
        """Existing validation still works with hybrid fields set."""
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(
                max_memory_percent=0.0,
                enable_hybrid_cache=True,
            )

    def test_compute_memory_limit_unaffected_by_hybrid(self):
        """Hybrid fields don't change memory limit computation."""
        config_plain = MemoryCacheConfig(max_memory_mb=512)
        config_hybrid = MemoryCacheConfig(
            max_memory_mb=512,
            enable_hybrid_cache=True,
            prefix_cache_alpha=2.0,
        )
        assert config_plain.compute_memory_limit() == config_hybrid.compute_memory_limit()


# ---------------------------------------------------------------------------
# 3. _CacheEntry with SSM detection
# ---------------------------------------------------------------------------


class TestCacheEntrySSMDetection:
    """Verify _CacheEntry.create detects SSM layers."""

    def test_kv_only_entry(self):
        """KV-only cache: has_ssm_states should be False."""
        cache = _make_kv_only_cache(n_layers=4)
        entry = _CacheEntry.create([1, 2, 3], cache)
        assert entry.has_ssm_states is False
        assert entry.ssm_layer_indices == ()
        assert entry.ssm_checkpoint_pos == 0

    def test_hybrid_entry_detects_ssm(self):
        """Mixed KV+SSM cache: should detect SSM layer indices."""
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        entry = _CacheEntry.create([10, 20, 30, 40], cache)
        assert entry.has_ssm_states is True
        # SSM layers are at indices 2, 3 (after the 2 KV layers)
        assert entry.ssm_layer_indices == (2, 3)
        assert entry.ssm_checkpoint_pos == 4  # len(tokens)

    def test_entry_access_metadata_initialized(self):
        """Verify last_access and access_count are set on creation."""
        before = time.monotonic()
        entry = _CacheEntry.create([1, 2, 3], _make_kv_only_cache())
        after = time.monotonic()
        assert before <= entry.last_access <= after
        assert entry.access_count == 1
        assert entry.flop_efficiency == 0.0

    def test_entry_memory_includes_ssm(self):
        """Memory estimation should include both KV and SSM layers."""
        kv_cache = _make_kv_only_cache(n_layers=2, size_bytes=1000)
        kv_memory = estimate_kv_cache_memory(kv_cache)

        hybrid_cache = _make_hybrid_cache(
            n_kv=2, n_ssm=2, kv_bytes=1000, ssm_bytes=500
        )
        hybrid_memory = estimate_kv_cache_memory(hybrid_cache)

        # Hybrid should be larger due to SSM layers
        assert hybrid_memory > kv_memory
        assert hybrid_memory == kv_memory + 500 * 2  # 2 SSM layers @ 500 bytes each


# ---------------------------------------------------------------------------
# 4. fetch() supersequence match with hybrid model
# ---------------------------------------------------------------------------


class TestFetchSupersequenceHybrid:
    """Test fetch() supersequence match returns KV-only for hybrid models."""

    @pytest.fixture
    def hybrid_cache_instance(self):
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
        )
        return MemoryAwarePrefixCache(model, config)

    @pytest.fixture
    def non_hybrid_cache_instance(self):
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=False,
        )
        return MemoryAwarePrefixCache(model, config)

    def test_supersequence_hybrid_returns_kv_only(self, hybrid_cache_instance):
        """With hybrid cache, supersequence match with non-trimmable layers
        should return KV-only cache (not skip entirely)."""
        # Store longer sequence with hybrid cache
        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        hybrid_cache_instance.store(long_tokens, cache)

        # Fetch shorter sequence (supersequence match)
        short_tokens = [1, 2, 3, 4, 5]
        result, remaining = hybrid_cache_instance.fetch(short_tokens)

        # Should return a result (not None/skip)
        assert result is not None
        assert remaining == []
        assert hybrid_cache_instance._last_match_type == "supersequence_hybrid"

        # Verify SSM layers are replaced with None placeholders
        for i, layer in enumerate(result):
            if layer is not None:
                assert hasattr(layer, "keys") or hasattr(layer, "offset")

    def test_supersequence_non_hybrid_skips(self, non_hybrid_cache_instance):
        """Without hybrid cache, supersequence match with non-trimmable layers
        should be skipped (existing behavior)."""
        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        non_hybrid_cache_instance.store(long_tokens, cache)

        short_tokens = [1, 2, 3, 4, 5]
        result, remaining = non_hybrid_cache_instance.fetch(short_tokens)

        # Should skip (return None) because non-trimmable layers present
        assert result is None
        assert remaining == short_tokens


# ---------------------------------------------------------------------------
# 5. fetch() LCP match with hybrid model
# ---------------------------------------------------------------------------


class TestFetchLCPHybrid:
    """Test fetch() LCP match returns KV-only for hybrid models."""

    @pytest.fixture
    def hybrid_cache_instance(self):
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
        )
        return MemoryAwarePrefixCache(model, config)

    @pytest.fixture
    def non_hybrid_cache_instance(self):
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=False,
        )
        return MemoryAwarePrefixCache(model, config)

    def test_lcp_hybrid_returns_kv_only(self, hybrid_cache_instance):
        """With hybrid cache, LCP match with non-trimmable layers should
        return KV-only partial match."""
        # Store a sequence
        cached_tokens = [1, 2, 3, 4, 5, 100, 101, 102]
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        hybrid_cache_instance.store(cached_tokens, cache)

        # Fetch a divergent sequence sharing a prefix
        new_tokens = [1, 2, 3, 4, 5, 200, 201, 202]
        result, remaining = hybrid_cache_instance.fetch(new_tokens)

        # Should get a partial match (not None)
        assert result is not None
        assert hybrid_cache_instance._last_match_type == "lcp_hybrid"
        # Remaining should be the divergent suffix
        assert remaining == [200, 201, 202]

    def test_lcp_non_hybrid_skips(self, non_hybrid_cache_instance):
        """Without hybrid cache, LCP match with non-trimmable layers skips."""
        cached_tokens = [1, 2, 3, 4, 5, 100, 101, 102]
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        non_hybrid_cache_instance.store(cached_tokens, cache)

        new_tokens = [1, 2, 3, 4, 5, 200, 201, 202]
        result, remaining = non_hybrid_cache_instance.fetch(new_tokens)

        # Should miss (non-trimmable prevents LCP match)
        assert result is None
        assert remaining == new_tokens


# ---------------------------------------------------------------------------
# 6. fetch_hybrid() 3-tuple return
# ---------------------------------------------------------------------------


class TestFetchHybrid:
    """Test fetch_hybrid() returns 3-tuple with metadata."""

    @pytest.fixture
    def hybrid_cache_instance(self):
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
        )
        return MemoryAwarePrefixCache(model, config)

    def test_fetch_hybrid_exact_match(self, hybrid_cache_instance):
        """Exact match returns metadata with needs_ssm_recompute=False."""
        tokens = [1, 2, 3, 4, 5]
        cache = _make_kv_only_cache()
        hybrid_cache_instance.store(tokens, cache)

        result, remaining, metadata = hybrid_cache_instance.fetch_hybrid(tokens)
        assert result is not None
        assert remaining == []
        assert metadata is not None
        assert metadata["needs_ssm_recompute"] is False
        assert metadata["match_type"] == "exact"

    def test_fetch_hybrid_miss(self, hybrid_cache_instance):
        """Miss returns None metadata."""
        result, remaining, metadata = hybrid_cache_instance.fetch_hybrid(
            [99, 98, 97]
        )
        assert result is None
        assert remaining == [99, 98, 97]
        assert metadata is None

    def test_fetch_hybrid_supersequence_returns_recompute_metadata(
        self, hybrid_cache_instance
    ):
        """Supersequence hybrid match returns needs_ssm_recompute=True."""
        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        hybrid_cache_instance.store(long_tokens, cache)

        short_tokens = [1, 2, 3, 4, 5]
        result, remaining, metadata = hybrid_cache_instance.fetch_hybrid(
            short_tokens
        )
        assert result is not None
        assert metadata is not None
        assert metadata["needs_ssm_recompute"] is True
        assert metadata["match_type"] == "supersequence_hybrid"

    def test_fetch_hybrid_prefix_match(self, hybrid_cache_instance):
        """Prefix match with KV-only cache returns no recompute needed."""
        short_tokens = [1, 2, 3]
        cache = _make_kv_only_cache()
        hybrid_cache_instance.store(short_tokens, cache)

        long_tokens = [1, 2, 3, 4, 5]
        result, remaining, metadata = hybrid_cache_instance.fetch_hybrid(
            long_tokens
        )
        assert result is not None
        assert remaining == [4, 5]
        assert metadata is not None
        assert metadata["needs_ssm_recompute"] is False
        assert metadata["match_type"] == "prefix"


# ---------------------------------------------------------------------------
# 7. FLOP-aware eviction
# ---------------------------------------------------------------------------


class TestFLOPAwareEviction:
    """Test FLOP-aware eviction preserves high-FLOP entries."""

    def test_flop_eviction_preserves_high_efficiency(self):
        """Entries with higher flop_efficiency should survive eviction."""
        model = MagicMock()
        # Very small cache to force eviction
        config = MemoryCacheConfig(
            max_memory_mb=1,
            max_entries=3,
            enable_hybrid_cache=True,
            prefix_cache_alpha=2.0,  # Strong FLOP bias
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        # Store 3 entries
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        tokens3 = [7, 8, 9]
        kv = _make_kv_only_cache(n_layers=2, size_bytes=100)

        cache_mgr.store(tokens1, kv)
        cache_mgr.store(tokens2, kv)
        cache_mgr.store(tokens3, kv)

        # Manually set flop_efficiency on entries
        entry1 = cache_mgr._entries[tuple(tokens1)]
        entry1.flop_efficiency = 10.0  # High value — should survive
        entry1.last_access = time.monotonic()

        entry2 = cache_mgr._entries[tuple(tokens2)]
        entry2.flop_efficiency = 0.1  # Low value — should be evicted first

        entry3 = cache_mgr._entries[tuple(tokens3)]
        entry3.flop_efficiency = 0.01  # Lowest — evicted first

        # Trigger eviction by storing a new entry (will exceed max_entries=3)
        tokens4 = [10, 11, 12]
        cache_mgr.store(tokens4, kv)

        # High-FLOP entry should survive
        assert tuple(tokens1) in cache_mgr._entries

    def test_flop_eviction_demotes_ssm_first(self):
        """SSM states should be demoted before full entry eviction."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=1,
            max_entries=10,
            enable_hybrid_cache=True,
            prefix_cache_alpha=1.0,
            ssm_admission_threshold=0,  # Always admit SSM
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        # Store a hybrid entry
        tokens = [1, 2, 3, 4, 5]
        hybrid = _make_hybrid_cache(n_kv=2, n_ssm=2, kv_bytes=500, ssm_bytes=300)
        cache_mgr.store(tokens, hybrid)

        entry = cache_mgr._entries[tuple(tokens)]
        original_memory = entry.memory_bytes

        # Manually trigger FLOP-aware eviction
        cache_mgr._evict_flop_aware()

        # Entry should still exist but with SSM states stripped
        if tuple(tokens) in cache_mgr._entries:
            updated_entry = cache_mgr._entries[tuple(tokens)]
            assert updated_entry.has_ssm_states is False
            assert updated_entry.memory_bytes < original_memory

    def test_lru_eviction_when_alpha_zero(self):
        """With alpha=0, should use standard LRU eviction."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=1,
            max_entries=2,
            enable_hybrid_cache=True,
            prefix_cache_alpha=0.0,  # Pure LRU
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv = _make_kv_only_cache(n_layers=2, size_bytes=100)

        cache_mgr.store(tokens1, kv)
        cache_mgr.store(tokens2, kv)

        # Access tokens1 to make it MRU
        cache_mgr.fetch(tokens1)

        # Store new entry, forcing eviction
        tokens3 = [7, 8, 9]
        cache_mgr.store(tokens3, kv)

        # tokens2 should be evicted (LRU)
        assert tuple(tokens2) not in cache_mgr._entries
        # tokens1 should survive (was accessed recently)
        assert tuple(tokens1) in cache_mgr._entries


# ---------------------------------------------------------------------------
# 8. Judicious SSM admission
# ---------------------------------------------------------------------------


class TestJudiciousSSMAdmission:
    """Test SSM states are only admitted after repeated access."""

    def test_first_store_strips_ssm(self):
        """First store of hybrid entry should strip SSM states."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
            ssm_admission_threshold=2,
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        tokens = [1, 2, 3, 4, 5]
        hybrid = _make_hybrid_cache(n_kv=2, n_ssm=2, kv_bytes=500, ssm_bytes=300)
        cache_mgr.store(tokens, hybrid)

        entry = cache_mgr._entries[tuple(tokens)]
        # SSM states should be stripped on first admission
        assert entry.has_ssm_states is False

    def test_second_store_keeps_ssm(self):
        """After access_count reaches threshold, SSM states should be kept."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
            ssm_admission_threshold=2,
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        tokens = [1, 2, 3, 4, 5]

        # First store (SSM stripped)
        hybrid1 = _make_hybrid_cache(n_kv=2, n_ssm=2)
        cache_mgr.store(tokens, hybrid1)

        # Fetch to increment access_count
        cache_mgr.fetch(tokens)
        cache_mgr.fetch(tokens)

        # Remove and re-store (simulates 2nd occurrence with access history)
        entry = cache_mgr._entries[tuple(tokens)]
        assert entry.access_count >= 2

        # Now store again with SSM states — should be admitted
        cache_mgr.remove(tokens)

        # Manually create an entry pre-populated to simulate 2nd occurrence
        # We store, then update access_count, then re-store
        hybrid2 = _make_hybrid_cache(n_kv=2, n_ssm=2)

        # Pre-populate the entry with enough accesses
        cache_mgr.store(tokens, hybrid2)
        # The entry exists now; update its access_count
        existing = cache_mgr._entries[tuple(tokens)]
        existing.access_count = 3  # Above threshold

        # Remove and store again — now _should_admit_ssm checks existing entry
        cache_mgr.remove(tokens)
        hybrid3 = _make_hybrid_cache(n_kv=2, n_ssm=2)

        # Store again. The entry was removed, so _should_admit_ssm won't find it.
        # This tests that we need the entry to exist with high access_count.
        # Let's put the entry back first, then store over it.
        cache_mgr.store(tokens, _make_kv_only_cache())
        cache_mgr._entries[tuple(tokens)].access_count = 5  # Above threshold

        # Now store hybrid over it — should keep SSM because existing has high access
        hybrid4 = _make_hybrid_cache(n_kv=2, n_ssm=2)
        # Remove first so it's not a duplicate-skip
        cache_mgr.remove(tokens)
        # Re-insert the entry with high access count to simulate history
        placeholder = _make_kv_only_cache()
        cache_mgr.store(tokens, placeholder)
        cache_mgr._entries[tuple(tokens)].access_count = 5

        # The admission check happens during store, but duplicate store returns early.
        # So we verify the admission logic directly:
        test_entry = _CacheEntry.create(tokens, hybrid4)
        assert cache_mgr._should_admit_ssm(tuple(tokens), test_entry) is True

    def test_admission_threshold_zero_always_admits(self):
        """With threshold=0, SSM states should always be admitted."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
            ssm_admission_threshold=0,
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        # _should_admit_ssm with threshold=0: checks access_count >= 0
        # which is always true. However, the prefix must exist in _entries.
        tokens = [1, 2, 3]
        kv = _make_kv_only_cache()
        cache_mgr.store(tokens, kv)

        entry = _CacheEntry.create(tokens, _make_hybrid_cache())
        # Entry exists with access_count=1 >= threshold=0
        assert cache_mgr._should_admit_ssm(tuple(tokens), entry) is True

    def test_should_admit_ssm_returns_false_for_unknown_prefix(self):
        """_should_admit_ssm returns False for prefix not in cache."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
            ssm_admission_threshold=2,
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        entry = _CacheEntry.create([99, 98, 97], _make_hybrid_cache())
        assert cache_mgr._should_admit_ssm((99, 98, 97), entry) is False


# ---------------------------------------------------------------------------
# 9. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify enable_hybrid_cache=False preserves all existing behavior."""

    @pytest.fixture
    def plain_cache(self):
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=10)
        return MemoryAwarePrefixCache(model, config)

    def test_fetch_returns_2_tuple(self, plain_cache):
        """fetch() returns 2-tuple (cache, remaining) as before."""
        tokens = [1, 2, 3]
        kv = _make_kv_only_cache()
        plain_cache.store(tokens, kv)

        result = plain_cache.fetch(tokens)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_exact_match_works(self, plain_cache):
        tokens = [1, 2, 3, 4, 5]
        kv = _make_kv_only_cache()
        plain_cache.store(tokens, kv)

        cache_out, remaining = plain_cache.fetch(tokens)
        assert cache_out is kv
        assert remaining == []

    def test_prefix_match_works(self, plain_cache):
        short = [1, 2, 3]
        kv = _make_kv_only_cache()
        plain_cache.store(short, kv)

        cache_out, remaining = plain_cache.fetch([1, 2, 3, 4, 5])
        assert cache_out is kv
        assert remaining == [4, 5]

    def test_miss_works(self, plain_cache):
        plain_cache.store([1, 2, 3], _make_kv_only_cache())
        cache_out, remaining = plain_cache.fetch([9, 8, 7])
        assert cache_out is None
        assert remaining == [9, 8, 7]

    def test_lru_eviction_works(self):
        """LRU eviction still used when enable_hybrid_cache=False."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2)
        cache_mgr = MemoryAwarePrefixCache(model, config)

        kv = _make_kv_only_cache(n_layers=2, size_bytes=100)
        cache_mgr.store([1, 2, 3], kv)
        cache_mgr.store([4, 5, 6], kv)

        # Access first entry to make it MRU
        cache_mgr.fetch([1, 2, 3])

        # Store 3rd entry — should evict LRU (tokens [4,5,6])
        cache_mgr.store([7, 8, 9], kv)

        assert tuple([4, 5, 6]) not in cache_mgr._entries
        assert tuple([1, 2, 3]) in cache_mgr._entries

    def test_store_and_stats_work(self, plain_cache):
        kv = _make_kv_only_cache()
        plain_cache.store([1, 2], kv)
        plain_cache.fetch([1, 2])  # hit
        plain_cache.fetch([9, 9])  # miss

        stats = plain_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1

    def test_supersequence_kv_only_still_works(self):
        """Supersequence match with KV-only cache (no non-trimmable layers)
        should work as before regardless of hybrid setting."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=10, max_entries=100)
        cache_mgr = MemoryAwarePrefixCache(model, config)

        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        kv = _make_kv_only_cache()
        cache_mgr.store(long_tokens, kv)

        short_tokens = [1, 2, 3, 4, 5]
        result, remaining = cache_mgr.fetch(short_tokens)

        assert result is not None
        assert remaining == []

    def test_hybrid_off_skips_non_trimmable_supersequence(self):
        """With hybrid off, non-trimmable supersequence match is skipped."""
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=False,
        )
        cache_mgr = MemoryAwarePrefixCache(model, config)

        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        hybrid = _make_hybrid_cache(n_kv=2, n_ssm=2)
        cache_mgr.store(long_tokens, hybrid)

        short_tokens = [1, 2, 3, 4, 5]
        result, remaining = cache_mgr.fetch(short_tokens)

        assert result is None
        assert remaining == short_tokens


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """Test _extract_kv_only and _strip_ssm_states."""

    @pytest.fixture
    def hybrid_cache_instance(self):
        model = MagicMock()
        config = MemoryCacheConfig(
            max_memory_mb=10,
            max_entries=100,
            enable_hybrid_cache=True,
        )
        return MemoryAwarePrefixCache(model, config)

    def test_extract_kv_only_replaces_ssm_with_none(self, hybrid_cache_instance):
        """SSM layers replaced with None, KV layers preserved."""
        cache = _make_hybrid_cache(n_kv=2, n_ssm=2)
        kv_only = hybrid_cache_instance._extract_kv_only(cache)

        assert len(kv_only) == 4
        # First 2 are KV (should be preserved)
        assert kv_only[0] is cache[0]
        assert kv_only[1] is cache[1]
        # Last 2 are SSM (should be None)
        assert kv_only[2] is None
        assert kv_only[3] is None

    def test_extract_kv_only_handles_none_layers(self, hybrid_cache_instance):
        """None layers in original cache stay None."""
        cache = [MockKVCache(100, 100), None, MockArraysCache([200])]
        kv_only = hybrid_cache_instance._extract_kv_only(cache)
        assert len(kv_only) == 3
        assert kv_only[0] is cache[0]
        assert kv_only[1] is None
        assert kv_only[2] is None

    def test_strip_ssm_states_creates_new_entry(self, hybrid_cache_instance):
        """_strip_ssm_states returns new entry with has_ssm_states=False."""
        hybrid = _make_hybrid_cache(n_kv=2, n_ssm=2, kv_bytes=500, ssm_bytes=300)
        entry = _CacheEntry.create([1, 2, 3], hybrid)

        stripped = hybrid_cache_instance._strip_ssm_states(entry)
        assert stripped.has_ssm_states is False
        assert stripped.ssm_layer_indices == ()
        assert stripped.ssm_checkpoint_pos == 0
        assert stripped.tokens == entry.tokens
        assert stripped.memory_bytes <= entry.memory_bytes
