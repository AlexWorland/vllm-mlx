# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid radix tree cache."""

import time
import types

import pytest

from vllm_mlx.radix_cache_hybrid import (
    HybridCacheLookup,
    HybridModelConfig,
    RadixCacheHybrid,
    RadixNode,
    _get_flat_args,
    _get_text_config,
    compute_flop_efficiency,
)


# ---------------------------------------------------------------------------
# Mock objects — simulate MLX cache layers without importing mlx
# ---------------------------------------------------------------------------


class MockDtype:
    def __init__(self, size: int = 2):
        self.size = size


class MockArray:
    def __init__(self, shape: tuple = (1, 8, 128, 64), dtype_size: int = 2):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockKVCache:
    """Mock attention layer cache with .keys/.values and is_trimmable()."""

    def __init__(self, seq_len: int = 64):
        self.keys = MockArray((1, 4, seq_len, 32))
        self.values = MockArray((1, 4, seq_len, 32))
        self.offset = seq_len

    def is_trimmable(self) -> bool:
        return True


class MockSSMCache:
    """Mock recurrent layer cache with .cache list and is_trimmable()."""

    def __init__(self):
        self.cache = [MockArray((1, 32, 64))]

    def is_trimmable(self) -> bool:
        return False


def make_kv_states(seq_len: int = 64, n_layers: int = 1):
    """Create a list of mock KV cache layers."""
    return [MockKVCache(seq_len) for _ in range(n_layers)]


def make_ssm_states(n_layers: int = 1):
    """Create a list of mock SSM cache layers."""
    return [MockSSMCache() for _ in range(n_layers)]


# ---------------------------------------------------------------------------
# Tests for HybridModelConfig
# ---------------------------------------------------------------------------


class TestHybridModelConfig:
    def test_defaults(self):
        config = HybridModelConfig()
        assert config.num_attention_layers == 8
        assert config.num_ssm_layers == 24
        assert config.hidden_size == 4096

    def test_custom_config(self):
        config = HybridModelConfig(num_attention_layers=4, num_ssm_layers=12)
        assert config.num_attention_layers == 4
        assert config.num_ssm_layers == 12


# ---------------------------------------------------------------------------
# Tests for compute_flop_efficiency
# ---------------------------------------------------------------------------


class TestComputeFlopEfficiency:
    def test_zero_depth(self):
        node = RadixNode(depth=0, memory_bytes=1000)
        assert compute_flop_efficiency(node, HybridModelConfig()) == 0.0

    def test_zero_memory(self):
        node = RadixNode(depth=100, memory_bytes=0)
        assert compute_flop_efficiency(node, HybridModelConfig()) == 0.0

    def test_longer_sequences_higher_efficiency(self):
        """Attention FLOPs are quadratic, so longer sequences save more."""
        config = HybridModelConfig()
        short = RadixNode(depth=10, memory_bytes=1000)
        long = RadixNode(depth=100, memory_bytes=1000)

        eff_short = compute_flop_efficiency(short, config)
        eff_long = compute_flop_efficiency(long, config)

        # With same memory, longer sequence has higher FLOP efficiency
        assert eff_long > eff_short

    def test_quadratic_scaling(self):
        """Verify FLOP efficiency grows roughly quadratically with depth."""
        config = HybridModelConfig(num_attention_layers=1, num_ssm_layers=0)
        node_10 = RadixNode(depth=10, memory_bytes=1000)
        node_100 = RadixNode(depth=100, memory_bytes=1000)

        eff_10 = compute_flop_efficiency(node_10, config)
        eff_100 = compute_flop_efficiency(node_100, config)

        # With only attention layers, FLOP ratio should be ~100x (100^2/10^2)
        ratio = eff_100 / eff_10
        assert 90 < ratio < 110


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Basic Operations
# ---------------------------------------------------------------------------


class TestRadixCacheBasic:
    """Basic insert/lookup operations."""

    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_insert_and_exact_lookup(self, cache):
        tokens = (1, 2, 3, 4)
        kv = make_kv_states()
        cache.insert(tokens, kv)

        result = cache.lookup(tokens)
        assert result.match_type == "exact"
        assert result.matched_length == 4
        assert result.kv_states is kv

    def test_prefix_lookup(self, cache):
        cache.insert((1, 2, 3, 4, 5), make_kv_states())

        result = cache.lookup((1, 2, 3, 4, 5, 6, 7))
        assert result.match_type == "prefix"
        assert result.matched_length == 5

    def test_miss_lookup(self, cache):
        cache.insert((1, 2, 3), make_kv_states())

        result = cache.lookup((7, 8, 9))
        assert result.match_type == "none"
        assert result.matched_length == 0
        assert result.kv_states is None

    def test_no_partial_match_for_different_prefix(self, cache):
        """Tokens (1,2,3) should not match (1,2,4)."""
        cache.insert((1, 2, 3), make_kv_states())

        result = cache.lookup((1, 2, 4))
        # Should match at depth 2 (the shared prefix before divergence)
        # But only if there's a node at depth 2 with states
        # After insert of (1,2,3) there's only a node at depth 3
        # So no match with states
        assert result.match_type == "none" or result.matched_length < 3


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Empty Token Edge Cases
# ---------------------------------------------------------------------------


class TestRadixCacheEmptyTokens:
    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_insert_empty_returns_root(self, cache):
        node = cache.insert((), make_kv_states())
        # Should return root without error
        assert node is not None

    def test_lookup_empty_is_miss(self, cache):
        cache.insert((1, 2, 3), make_kv_states())
        result = cache.lookup(())
        assert result.match_type == "none"
        assert result.matched_length == 0

    def test_insert_empty_does_not_increment_request_count(self, cache):
        cache.insert((), make_kv_states())
        assert cache._request_count == 0


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Node Splitting
# ---------------------------------------------------------------------------


class TestRadixCacheNodeSplitting:
    """Test radix tree edge splitting on divergent sequences."""

    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_split_on_divergent_insert(self, cache):
        """Inserting (1,2,3) then (1,2,4) should split edge at position 2."""
        kv1 = make_kv_states()
        kv2 = make_kv_states()

        cache.insert((1, 2, 3), kv1)
        cache.insert((1, 2, 4), kv2)

        # Both should be findable
        r1 = cache.lookup((1, 2, 3))
        assert r1.match_type == "exact"
        assert r1.kv_states is kv1

        r2 = cache.lookup((1, 2, 4))
        assert r2.match_type == "exact"
        assert r2.kv_states is kv2

    def test_split_preserves_existing_data(self, cache):
        """Original node's states are preserved after split."""
        kv_orig = make_kv_states()
        cache.insert((1, 2, 3, 4, 5), kv_orig)
        cache.insert((1, 2, 3, 6, 7), make_kv_states())

        result = cache.lookup((1, 2, 3, 4, 5))
        assert result.match_type == "exact"
        assert result.kv_states is kv_orig

    def test_multiple_splits(self, cache):
        """Multiple divergent sequences create proper tree structure."""
        tokens_a = (1, 2, 3, 4, 5)
        tokens_b = (1, 2, 3, 6, 7)
        tokens_c = (1, 2, 8, 9, 10)

        cache.insert(tokens_a, make_kv_states())
        cache.insert(tokens_b, make_kv_states())
        cache.insert(tokens_c, make_kv_states())

        assert cache.lookup(tokens_a).match_type == "exact"
        assert cache.lookup(tokens_b).match_type == "exact"
        assert cache.lookup(tokens_c).match_type == "exact"

    def test_insert_prefix_of_existing(self, cache):
        """Inserting a prefix of an existing sequence splits correctly."""
        kv_long = make_kv_states()
        cache.insert((1, 2, 3, 4, 5), kv_long)

        kv_short = make_kv_states()
        cache.insert((1, 2, 3), kv_short)

        r_short = cache.lookup((1, 2, 3))
        assert r_short.match_type == "exact"
        assert r_short.kv_states is kv_short

        r_long = cache.lookup((1, 2, 3, 4, 5))
        assert r_long.match_type == "exact"
        assert r_long.kv_states is kv_long


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Mid-Edge Partial Match
# ---------------------------------------------------------------------------


class TestRadixCacheMidEdgeMatch:
    """Test lookup when match ends within a node's token_ids (mid-edge)."""

    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_mid_edge_falls_back_to_parent(self, cache):
        """If tokens partially match an edge, lookup returns the child's KV states
        (supersequence hit) with SSM discarded since it can't be trimmed."""
        # Insert (1,2,3) and (1,2,3,4,5) — creates split at (1,2,3)
        kv_parent = make_kv_states()
        kv_child = make_kv_states()
        cache.insert((1, 2, 3), kv_parent)
        cache.insert((1, 2, 3, 4, 5), kv_child)

        # Lookup (1,2,3,4) — matches (1,2,3) fully + partially into (4,5) edge
        # Should return child's KV states (supersequence hit) with SSM recompute
        result = cache.lookup((1, 2, 3, 4))
        assert result.match_type == "prefix"
        assert result.matched_length == 4
        assert result.kv_states is kv_child
        assert result.ssm_states is None
        assert result.needs_ssm_recompute is True

    def test_mid_edge_no_parent_states(self, cache):
        """Partial match into a single edge returns KV states (supersequence hit)."""
        # Insert single long sequence — root has no states, single node at depth 5
        kv = make_kv_states()
        cache.insert((1, 2, 3, 4, 5), kv)

        # Lookup (1,2,3) — partial match within the single edge
        # The edge is (1,2,3,4,5), we match 3 tokens mid-edge
        # Returns child's KV states with SSM recompute needed
        result = cache.lookup((1, 2, 3))
        assert result.match_type == "prefix"
        assert result.matched_length == 3
        assert result.kv_states is kv
        assert result.ssm_states is None
        assert result.needs_ssm_recompute is True


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Judicious Admission
# ---------------------------------------------------------------------------


class TestRadixCacheAdmission:
    """Test MARCONI judicious admission policy for SSM states."""

    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_first_access_rejects_ssm(self, cache):
        """First-time, non-branch node should reject SSM states."""
        ssm = make_ssm_states()
        node = cache.insert((1, 2, 3), make_kv_states(), ssm_states=ssm)

        # SSM should be rejected on first access (access_count was 0 before insert)
        assert node.ssm_states is None
        assert cache._ssm_rejections == 1
        assert cache._ssm_admissions == 0

    def test_second_access_admits_ssm(self, cache):
        """SSM states should be admitted on 2nd occurrence."""
        tokens = (1, 2, 3)

        # First insert — SSM rejected
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())

        # Second insert — SSM should be admitted (access_count is now >= 2)
        ssm2 = make_ssm_states()
        node = cache.insert(tokens, make_kv_states(), ssm_states=ssm2)

        assert node.ssm_states is ssm2
        assert cache._ssm_admissions == 1

    def test_branch_point_always_admits_ssm(self, cache):
        """Branch point nodes should always admit SSM states."""
        # Create a branch point by inserting two divergent sequences
        cache.insert((1, 2, 3, 4), make_kv_states())
        cache.insert((1, 2, 5, 6), make_kv_states())

        # Now insert at the branch point (1,2) with SSM states
        # The node at (1,2) has multiple children, so it's a branch point
        ssm = make_ssm_states()
        node = cache.insert((1, 2), make_kv_states(), ssm_states=ssm)

        assert node.ssm_states is ssm

    def test_needs_ssm_recompute_flag(self, cache):
        """Lookup should set needs_ssm_recompute when KV present but no SSM."""
        cache.insert((1, 2, 3), make_kv_states())  # No SSM stored

        result = cache.lookup((1, 2, 3))
        assert result.needs_ssm_recompute is True
        assert result.kv_states is not None
        assert result.ssm_states is None

    def test_needs_ssm_recompute_false_with_ssm(self, cache):
        """When SSM states are present, needs_ssm_recompute should be False."""
        tokens = (1, 2, 3)
        # Insert twice to get SSM admitted
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())

        result = cache.lookup(tokens)
        assert result.needs_ssm_recompute is False
        assert result.ssm_states is not None


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: FLOP-Aware Eviction
# ---------------------------------------------------------------------------


class TestRadixCacheEviction:
    """Test FLOP-aware eviction scoring and behavior."""

    def test_evict_frees_target_bytes(self):
        cache = RadixCacheHybrid(max_memory_bytes=100_000_000)

        # Insert several entries
        for i in range(5):
            tokens = tuple(range(i * 10, (i + 1) * 10))
            cache.insert(tokens, make_kv_states(seq_len=64))

        initial_mem = cache._current_memory
        assert initial_mem > 0

        freed = cache.evict(initial_mem // 2)
        assert freed >= initial_mem // 2

    def test_evict_ssm_before_kv(self):
        """SSM states should be freed before KV states."""
        cache = RadixCacheHybrid()
        tokens = (1, 2, 3)

        # Insert twice to get SSM admitted
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())

        node = cache._traverse(tokens)[0]
        assert node.ssm_states is not None

        # Evict a small amount — should free SSM first
        from vllm_mlx.hybrid_cache_entry import estimate_hybrid_memory
        ssm_mem = estimate_hybrid_memory(node.ssm_states)
        freed = cache.evict(ssm_mem)

        assert freed >= ssm_mem
        # Check that the node's SSM was freed
        node_after = cache._traverse(tokens)[0]
        assert node_after.ssm_states is None
        # But KV should still be present
        assert node_after.kv_states is not None

    def test_longer_sequences_survive_eviction(self):
        """With alpha > 0, longer sequences have higher FLOP efficiency and survive."""
        cache = RadixCacheHybrid()
        cache._alpha = 5.0  # Heavily weight FLOP efficiency

        # Short sequence (low FLOP efficiency)
        short_kv = make_kv_states(seq_len=10)
        cache.insert((1, 2), short_kv)

        # Long sequence (high FLOP efficiency due to quadratic attention)
        long_kv = make_kv_states(seq_len=200)
        cache.insert((10, 20, 30, 40, 50, 60, 70, 80, 90, 100), long_kv)

        # Force eviction of one entry
        from vllm_mlx.hybrid_cache_entry import estimate_hybrid_memory
        cache.evict(estimate_hybrid_memory(short_kv))

        # Short sequence should be evicted first (lower score)
        r_short = cache.lookup((1, 2))
        r_long = cache.lookup((10, 20, 30, 40, 50, 60, 70, 80, 90, 100))

        # The long sequence should survive (higher FLOP efficiency)
        assert r_long.match_type != "none"

    def test_evict_zero_target_returns_zero(self):
        cache = RadixCacheHybrid()
        cache.insert((1, 2, 3), make_kv_states())
        assert cache.evict(0) == 0
        assert cache.evict(-100) == 0

    def test_evict_empty_cache(self):
        cache = RadixCacheHybrid()
        assert cache.evict(1000) == 0

    def test_memory_budget_triggers_eviction(self):
        """Inserting beyond max_memory_bytes triggers automatic eviction."""
        # Small budget — will trigger eviction
        cache = RadixCacheHybrid(max_memory_bytes=50_000)

        for i in range(10):
            tokens = tuple(range(i * 5, (i + 1) * 5))
            cache.insert(tokens, make_kv_states(seq_len=64))

        assert cache._current_memory <= 50_000 or cache._evictions > 0


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Two-Tier Cache (Hot / Cold)
# ---------------------------------------------------------------------------


class TestRadixCacheTwoTier:
    """Test hot/cold tier promotion and demotion."""

    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_demote_moves_to_cold(self, cache):
        tokens = (1, 2, 3)
        kv = make_kv_states()
        cache.insert(tokens, kv)

        node = cache._traverse(tokens)[0]
        mem_before = node.memory_bytes
        assert mem_before > 0

        cache.demote(node, tokens)

        # Node states should be cleared
        assert node.kv_states is None
        assert node.memory_bytes == 0

        # Cold tier should have the entry
        assert tokens in cache._cold_tier
        _, _, cold_mem = cache._cold_tier[tokens]
        assert cold_mem == mem_before

    def test_promote_moves_to_hot(self, cache):
        tokens = (1, 2, 3)
        kv = make_kv_states()
        cache.insert(tokens, kv)

        node = cache._traverse(tokens)[0]
        cache.demote(node, tokens)

        # Promote back
        promoted = cache.promote(tokens)
        assert promoted is not None
        assert promoted.kv_states is not None
        assert tokens not in cache._cold_tier

    def test_promote_nonexistent_returns_none(self, cache):
        assert cache.promote((99, 99, 99)) is None

    def test_lookup_finds_cold_tier_exact(self, cache):
        """Lookup should check cold tier and auto-promote on exact match."""
        tokens = (1, 2, 3)
        kv = make_kv_states()
        cache.insert(tokens, kv)

        node = cache._traverse(tokens)[0]
        cache.demote(node, tokens)

        # Lookup should find in cold tier
        result = cache.lookup(tokens)
        assert result.match_type == "exact"
        assert result.kv_states is not None

    def test_lookup_finds_cold_tier_prefix(self, cache):
        """Lookup should find prefix matches in cold tier."""
        prefix = (1, 2, 3)
        kv = make_kv_states()
        cache.insert(prefix, kv)

        node = cache._traverse(prefix)[0]
        cache.demote(node, prefix)

        # Lookup a longer sequence that starts with the cold prefix
        result = cache.lookup((1, 2, 3, 4, 5))
        assert result.match_type == "prefix"
        assert result.matched_length == 3

    def test_evict_from_cold_tier(self, cache):
        """When hot tier is empty, eviction should fall back to cold tier."""
        tokens = (1, 2, 3)
        kv = make_kv_states()
        cache.insert(tokens, kv)

        node = cache._traverse(tokens)[0]
        mem = node.memory_bytes
        cache.demote(node, tokens)

        # Now hot tier has no leaves with states
        assert cache._collect_leaves() == []

        # Evict should fall back to cold tier
        freed = cache.evict(mem)
        assert freed >= mem
        assert tokens not in cache._cold_tier

    def test_cold_memory_tracking(self, cache):
        tokens = (1, 2, 3)
        cache.insert(tokens, make_kv_states())

        node = cache._traverse(tokens)[0]
        mem = node.memory_bytes

        assert cache._cold_memory == 0
        cache.demote(node, tokens)
        assert cache._cold_memory == mem

        cache.promote(tokens)
        assert cache._cold_memory == 0


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: get_stats()
# ---------------------------------------------------------------------------


class TestRadixCacheStats:
    @pytest.fixture
    def cache(self):
        return RadixCacheHybrid()

    def test_initial_stats(self, cache):
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["num_nodes"] == 0
        assert stats["hot_entries"] == 0
        assert stats["cold_entries"] == 0
        assert stats["ssm_admissions"] == 0
        assert stats["ssm_rejections"] == 0
        assert stats["evictions"] == 0
        assert stats["alpha"] == 0.0
        assert stats["bootstrap_complete"] is False

    def test_stats_after_operations(self, cache):
        cache.insert((1, 2, 3), make_kv_states())
        cache.lookup((1, 2, 3))  # hit
        cache.lookup((4, 5, 6))  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["num_nodes"] >= 1
        assert stats["hot_entries"] >= 1
        assert stats["request_count"] == 1  # only inserts count

    def test_stats_ssm_admission_rate(self, cache):
        tokens = (1, 2, 3)
        # First insert: SSM rejected
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())
        # Second insert: SSM admitted
        cache.insert(tokens, make_kv_states(), ssm_states=make_ssm_states())

        stats = cache.get_stats()
        assert stats["ssm_admissions"] == 1
        assert stats["ssm_rejections"] == 1
        assert stats["ssm_admission_rate"] == 0.5

    def test_stats_node_counts(self, cache):
        cache.insert((1, 2, 3), make_kv_states())
        cache.insert((1, 2, 4), make_kv_states())
        cache.insert((5, 6, 7), make_kv_states())

        stats = cache.get_stats()
        # At least 3 nodes (may have intermediate split nodes too)
        assert stats["num_nodes"] >= 3
        assert stats["hot_entries"] >= 2  # at least the leaf nodes with states

    def test_clear_resets_everything(self, cache):
        cache.insert((1, 2, 3), make_kv_states())
        cache.lookup((1, 2, 3))
        cache.lookup((4, 5, 6))

        cache.clear()

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["num_nodes"] == 0
        assert stats["hot_entries"] == 0
        assert stats["cold_entries"] == 0
        assert stats["hot_memory_bytes"] == 0
        assert stats["cold_memory_bytes"] == 0
        assert stats["request_count"] == 0


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Alpha Tuning
# ---------------------------------------------------------------------------


class TestRadixCacheAlphaTuning:
    def test_alpha_starts_at_zero(self):
        cache = RadixCacheHybrid()
        assert cache._alpha == 0.0
        assert cache._bootstrap_complete is False

    def test_bootstrap_period(self):
        """Alpha tuning should not trigger before bootstrap threshold."""
        cache = RadixCacheHybrid()
        cache._bootstrap_threshold = 5

        for i in range(4):
            cache.insert(tuple(range(i * 3, (i + 1) * 3)), make_kv_states())

        assert cache._bootstrap_complete is False

    def test_alpha_tuning_triggers_after_threshold(self):
        """Alpha tuning should trigger once threshold is reached."""
        cache = RadixCacheHybrid()
        cache._bootstrap_threshold = 5

        for i in range(6):
            tokens = tuple(range(i * 3, (i + 1) * 3))
            cache.insert(tokens, make_kv_states(seq_len=50 + i * 10))

        assert cache._bootstrap_complete is True

    def test_compute_optimal_alpha_returns_valid(self):
        """compute_optimal_alpha should return one of the candidate values."""
        cache = RadixCacheHybrid()
        cache._bootstrap_threshold = 3

        for i in range(5):
            tokens = tuple(range(i * 3, (i + 1) * 3))
            cache.insert(tokens, make_kv_states(seq_len=50 + i * 20))

        alpha = cache.compute_optimal_alpha()
        assert alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    def test_compute_optimal_alpha_insufficient_history(self):
        """Should return current alpha if not enough history."""
        cache = RadixCacheHybrid()
        cache._alpha = 1.5
        cache._bootstrap_threshold = 100

        result = cache.compute_optimal_alpha()
        assert result == 1.5  # unchanged

    def test_compute_optimal_alpha_no_leaves(self):
        """Should handle empty tree gracefully."""
        cache = RadixCacheHybrid()
        cache._access_history = [(0.0, 1.0)] * 20

        result = cache.compute_optimal_alpha()
        assert cache._bootstrap_complete is True


# ---------------------------------------------------------------------------
# Tests for RadixCacheHybrid: Internal Helpers
# ---------------------------------------------------------------------------


class TestRadixCacheInternals:
    def test_get_token_key_reconstructs_path(self):
        cache = RadixCacheHybrid()
        tokens = (10, 20, 30, 40, 50)
        cache.insert(tokens, make_kv_states())

        node = cache._traverse(tokens)[0]
        reconstructed = cache._get_token_key(node)
        assert reconstructed == tokens

    def test_collect_leaves_finds_all(self):
        cache = RadixCacheHybrid()
        cache.insert((1, 2, 3), make_kv_states())
        cache.insert((1, 2, 4), make_kv_states())
        cache.insert((5, 6, 7), make_kv_states())

        leaves = cache._collect_leaves()
        assert len(leaves) >= 3

    def test_count_nodes(self):
        cache = RadixCacheHybrid()
        num, hot = cache._count_nodes()
        assert num == 0
        assert hot == 0

        cache.insert((1, 2, 3), make_kv_states())
        num, hot = cache._count_nodes()
        assert num >= 1
        assert hot >= 1

    def test_prune_removes_empty_branch(self):
        cache = RadixCacheHybrid()
        cache.insert((1, 2, 3), make_kv_states())

        node = cache._traverse((1, 2, 3))[0]
        # Clear states and prune
        cache._current_memory -= node.memory_bytes
        node.kv_states = None
        node.ssm_states = None
        node.memory_bytes = 0
        cache._prune_node(node)

        # Tree should be empty now
        num, _ = cache._count_nodes()
        assert num == 0


# ---------------------------------------------------------------------------
# Tests for HybridModelConfig.from_model()
# ---------------------------------------------------------------------------


class TestHybridModelConfigFromModel:
    def test_from_text_config(self):
        """Path 1: model.args.text_config dict with all Qwen3.5-9B-like fields."""
        mock_args = types.SimpleNamespace(
            text_config={
                "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
                "hidden_size": 4096,
                "num_attention_heads": 16,
                "head_dim": 256,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_value_heads": 32,
            }
        )
        mock_model = types.SimpleNamespace(args=mock_args)

        config = HybridModelConfig.from_model(mock_model)

        assert config.num_attention_layers == 8
        assert config.num_ssm_layers == 24
        assert config.hidden_size == 4096
        assert config.num_heads == 16
        assert config.head_dim == 256
        assert config.ssm_state_size == 128 * 128
        assert config.num_ssm_heads == 32

    def test_from_text_config_different_model(self):
        """text_config extraction for a smaller model."""
        mock_args = types.SimpleNamespace(
            text_config={
                "layer_types": ["linear_attention"] * 12 + ["full_attention"] * 4,
                "hidden_size": 2048,
                "num_attention_heads": 8,
                "head_dim": 128,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 64,
                "linear_num_value_heads": 16,
            }
        )
        mock_model = types.SimpleNamespace(args=mock_args)

        config = HybridModelConfig.from_model(mock_model)

        assert config.num_attention_layers == 4
        assert config.num_ssm_layers == 12
        assert config.hidden_size == 2048
        assert config.num_heads == 8
        assert config.head_dim == 128
        assert config.ssm_state_size == 128 * 64
        assert config.num_ssm_heads == 16

    def test_from_flat_args(self):
        """Path 2: flat model.args with attributes (no text_config)."""
        mock_args = types.SimpleNamespace(
            hidden_size=4096,
            num_attention_heads=16,
            head_dim=256,
            layer_types=["linear_attention"] * 24 + ["full_attention"] * 8,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_value_heads=32,
        )
        mock_model = types.SimpleNamespace(args=mock_args)

        config = HybridModelConfig.from_model(mock_model)

        assert config.num_attention_layers == 8
        assert config.num_ssm_layers == 24
        assert config.hidden_size == 4096
        assert config.num_heads == 16
        assert config.head_dim == 256
        assert config.ssm_state_size == 128 * 128
        assert config.num_ssm_heads == 32

    def test_from_cache_fallback(self):
        """Path 3: no args, but make_cache() returns mixed KV/SSM layers."""
        kv_layers = [MockKVCache() for _ in range(8)]
        ssm_layers = [MockSSMCache() for _ in range(24)]
        # Interleave in a realistic order
        mixed_cache = []
        for i in range(8):
            mixed_cache.append(ssm_layers[i * 3])
            mixed_cache.append(ssm_layers[i * 3 + 1])
            mixed_cache.append(ssm_layers[i * 3 + 2])
            mixed_cache.append(kv_layers[i])

        mock_model = types.SimpleNamespace(make_cache=lambda: mixed_cache)

        config = HybridModelConfig.from_model(mock_model)

        assert config.num_attention_layers == 8
        assert config.num_ssm_layers == 24
        # Dimension fields use defaults
        assert config.hidden_size == 4096
        assert config.num_heads == 16
        assert config.head_dim == 256
        assert config.ssm_state_size == 128 * 128
        assert config.num_ssm_heads == 32

    def test_from_model_no_config(self):
        """make_cache() raises an exception — should return pure defaults."""
        def bad_make_cache():
            raise RuntimeError("no cache available")

        mock_model = types.SimpleNamespace(make_cache=bad_make_cache)

        config = HybridModelConfig.from_model(mock_model)

        # All defaults
        assert config.num_attention_layers == 8
        assert config.num_ssm_layers == 24
        assert config.hidden_size == 4096
        assert config.num_heads == 16
        assert config.head_dim == 256
        assert config.ssm_state_size == 128 * 128
        assert config.num_ssm_heads == 32

    def test_updated_defaults(self):
        """Verify the updated default values for HybridModelConfig."""
        config = HybridModelConfig()
        assert config.num_heads == 16
        assert config.head_dim == 256
        assert config.ssm_state_size == 128 * 128
        assert config.num_ssm_heads == 32


# ---------------------------------------------------------------------------
# Tests for _get_text_config and _get_flat_args helpers
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_get_text_config_returns_dict(self):
        mock_model = types.SimpleNamespace(
            args=types.SimpleNamespace(text_config={"hidden_size": 4096})
        )
        result = _get_text_config(mock_model)
        assert result == {"hidden_size": 4096}

    def test_get_text_config_empty_dict_returns_none(self):
        mock_model = types.SimpleNamespace(
            args=types.SimpleNamespace(text_config={})
        )
        assert _get_text_config(mock_model) is None

    def test_get_text_config_no_args_returns_none(self):
        mock_model = types.SimpleNamespace()
        assert _get_text_config(mock_model) is None

    def test_get_text_config_non_dict_returns_none(self):
        mock_model = types.SimpleNamespace(
            args=types.SimpleNamespace(text_config="not a dict")
        )
        assert _get_text_config(mock_model) is None

    def test_get_flat_args_with_hidden_size(self):
        mock_args = types.SimpleNamespace(hidden_size=4096)
        mock_model = types.SimpleNamespace(args=mock_args)
        result = _get_flat_args(mock_model)
        assert result is mock_args

    def test_get_flat_args_with_layer_types(self):
        mock_args = types.SimpleNamespace(layer_types=["full_attention"])
        mock_model = types.SimpleNamespace(args=mock_args)
        result = _get_flat_args(mock_model)
        assert result is mock_args

    def test_get_flat_args_no_useful_attrs_returns_none(self):
        mock_args = types.SimpleNamespace(unrelated_field=42)
        mock_model = types.SimpleNamespace(args=mock_args)
        assert _get_flat_args(mock_model) is None

    def test_get_flat_args_no_args_returns_none(self):
        mock_model = types.SimpleNamespace()
        assert _get_flat_args(mock_model) is None
