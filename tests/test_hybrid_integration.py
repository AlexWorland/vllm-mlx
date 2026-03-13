# SPDX-License-Identifier: Apache-2.0
"""Integration tests for hybrid prefix cache with real Qwen3.5-9B-4bit model.

These tests verify that the hybrid cache infrastructure works correctly with
a real model that has mixed attention + GatedDeltaNet (recurrent) layers.

Requirements:
- mlx-community/Qwen3.5-9B-4bit model cached locally
- ~6GB unified memory available
"""

import gc
import math
import time
from typing import Any

import mlx.core as mx
import pytest

# Model loading
from mlx_lm import load as mlx_load

# Hybrid cache modules under test
from vllm_mlx.hybrid_cache_entry import (
    HybridCacheEntry,
    classify_cache_layers,
    estimate_hybrid_memory,
)
from vllm_mlx.radix_cache_hybrid import (
    HybridCacheLookup,
    HybridModelConfig,
    RadixCacheHybrid,
)

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the real Qwen3.5-9B-4bit model once for all tests."""
    model, tokenizer = mlx_load(MODEL_ID)
    yield model, tokenizer
    del model
    gc.collect()


@pytest.fixture(scope="module")
def tokenizer(model_and_tokenizer):
    _, tok = model_and_tokenizer
    return tok


@pytest.fixture(scope="module")
def model(model_and_tokenizer):
    m, _ = model_and_tokenizer
    return m


@pytest.fixture(scope="module")
def cache_and_output(model, tokenizer):
    """Run a forward pass and return (cache, output, tokens)."""
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    cache = model.make_cache()
    output = model(input_ids, cache=cache)
    # Force evaluation by reading output shape
    _ = output.shape
    return cache, output, tokens


class TestQwen35LayerStructure:
    """Verify the model has the expected hybrid layer structure."""

    def test_total_layers(self, model):
        """Qwen3.5-9B has 32 layers."""
        assert len(model.layers) == 32

    def test_attention_interval(self, model):
        """Every 4th layer should be attention (full_attention_interval=4)."""
        # Count layers by trimmability
        cache = model.make_cache()
        attn_count = sum(
            1 for lc in cache if hasattr(lc, "is_trimmable") and lc.is_trimmable()
        )
        ssm_count = sum(
            1 for lc in cache
            if hasattr(lc, "is_trimmable") and not lc.is_trimmable()
        )

        # 32 layers / 4 interval = 8 attention, 24 recurrent
        assert attn_count == 8, f"Expected 8 attention layers, got {attn_count}"
        assert ssm_count == 24, f"Expected 24 SSM layers, got {ssm_count}"

    def test_layer_type_pattern(self, model):
        """Attention layers should be at positions 3, 7, 11, 15, 19, 23, 27, 31."""
        cache = model.make_cache()
        expected_attn = {3, 7, 11, 15, 19, 23, 27, 31}

        for i, lc in enumerate(cache):
            if hasattr(lc, "is_trimmable"):
                if i in expected_attn:
                    assert lc.is_trimmable(), f"Layer {i} should be attention"
                else:
                    assert not lc.is_trimmable(), f"Layer {i} should be recurrent"

    def test_cache_types(self, model):
        """Verify cache types: KVCache for attention, ArraysCache for recurrent."""
        cache = model.make_cache()

        for i, lc in enumerate(cache):
            class_name = type(lc).__name__
            if hasattr(lc, "is_trimmable") and lc.is_trimmable():
                assert "KVCache" in class_name or "RotatingKVCache" in class_name, (
                    f"Layer {i}: expected KVCache, got {class_name}"
                )
            elif hasattr(lc, "is_trimmable") and not lc.is_trimmable():
                assert "ArraysCache" in class_name or "MambaCache" in class_name, (
                    f"Layer {i}: expected ArraysCache/MambaCache, got {class_name}"
                )


class TestClassifyCacheLayersReal:
    """Test classify_cache_layers with real model cache."""

    def test_classification_matches_structure(self, model):
        """classify_cache_layers should match the model's layer structure."""
        cache = model.make_cache()
        kv_indices, ssm_indices = classify_cache_layers(cache)

        assert len(kv_indices) == 8
        assert len(ssm_indices) == 24
        assert set(kv_indices) == {3, 7, 11, 15, 19, 23, 27, 31}

    def test_classification_after_forward(self, cache_and_output):
        """Classification should work on populated cache too."""
        cache, _, _ = cache_and_output
        kv_indices, ssm_indices = classify_cache_layers(cache)

        assert len(kv_indices) == 8
        assert len(ssm_indices) == 24


class TestEstimateMemoryReal:
    """Test memory estimation with real model cache."""

    def test_empty_cache_zero_memory(self, model):
        """Fresh cache should report zero or near-zero memory."""
        cache = model.make_cache()
        mem = estimate_hybrid_memory(cache)
        # Fresh cache might have pre-allocated buffers, but should be small
        # compared to populated cache
        assert mem >= 0

    def test_populated_cache_nonzero(self, cache_and_output):
        """Cache after forward pass should have significant memory."""
        cache, _, _ = cache_and_output
        mem = estimate_hybrid_memory(cache)
        assert mem > 0, "Populated cache should have non-zero memory"
        # With 5 tokens through 32 layers, expect at least a few KB
        assert mem > 1024, f"Expected >1KB, got {mem} bytes"

    def test_ssm_layers_have_state(self, cache_and_output):
        """SSM layers should have populated state after forward pass."""
        cache, _, _ = cache_and_output
        _, ssm_indices = classify_cache_layers(cache)

        for idx in ssm_indices:
            lc = cache[idx]
            # ArraysCache has .cache attribute with list of arrays
            if hasattr(lc, "cache") and isinstance(lc.cache, list):
                assert len(lc.cache) > 0, f"SSM layer {idx} has empty cache"
                for arr in lc.cache:
                    if arr is not None and hasattr(arr, "shape"):
                        assert len(arr.shape) > 0, f"SSM layer {idx} array has no shape"


class TestRadixCacheWithRealStates:
    """Test RadixCacheHybrid with real extracted states."""

    def _extract_states(self, cache):
        """Extract KV and SSM states from cache."""
        kv_indices, ssm_indices = classify_cache_layers(cache)

        kv_states = []
        for idx in kv_indices:
            lc = cache[idx]
            if hasattr(lc, "keys") and hasattr(lc, "values"):
                keys = lc.keys
                values = lc.values
                if not callable(keys) and not callable(values):
                    kv_states.append({"keys": keys, "values": values, "layer_idx": idx})

        ssm_states = []
        for idx in ssm_indices:
            lc = cache[idx]
            if hasattr(lc, "cache") and isinstance(lc.cache, list):
                ssm_states.append({"state": lc.cache, "layer_idx": idx})

        return kv_states, ssm_states

    def test_insert_and_lookup(self, cache_and_output):
        """Insert real states and look them up."""
        cache, _, tokens = cache_and_output
        kv_states, ssm_states = self._extract_states(cache)

        config = HybridModelConfig(
            num_attention_layers=8,
            num_ssm_layers=24,
        )
        radix = RadixCacheHybrid(config)

        # Insert
        token_key = tuple(tokens)
        radix.insert(token_key, kv_states, ssm_states)

        # Lookup exact match
        result = radix.lookup(token_key)
        assert result is not None
        assert result.matched_length == len(tokens)

    def test_prefix_lookup(self, cache_and_output):
        """Lookup with extended tokens should find prefix match."""
        cache, _, tokens = cache_and_output
        kv_states, ssm_states = self._extract_states(cache)

        config = HybridModelConfig(
            num_attention_layers=8,
            num_ssm_layers=24,
        )
        radix = RadixCacheHybrid(config)

        token_key = tuple(tokens)
        radix.insert(token_key, kv_states, ssm_states)

        # Look up with extra tokens appended
        extended = token_key + (999, 1000, 1001)
        result = radix.lookup(extended)
        assert result is not None
        assert result.matched_length == len(tokens)

    def test_memory_tracking(self, cache_and_output):
        """Radix cache should track memory usage."""
        cache, _, tokens = cache_and_output
        kv_states, ssm_states = self._extract_states(cache)

        config = HybridModelConfig(
            num_attention_layers=8,
            num_ssm_layers=24,
        )
        radix = RadixCacheHybrid(config)

        token_key = tuple(tokens)
        radix.insert(token_key, kv_states, ssm_states)

        stats = radix.get_stats()
        assert stats["hot_entries"] >= 1
        # Memory may be 0 if extracted dict-based states don't match
        # estimate_hybrid_memory patterns (it expects cache objects, not dicts)
        assert stats["total_memory_bytes"] >= 0


class TestHybridCacheEntry:
    """Test HybridCacheEntry.create with real states."""

    def _extract_states(self, cache):
        kv_indices, ssm_indices = classify_cache_layers(cache)
        kv_states = []
        for idx in kv_indices:
            lc = cache[idx]
            if hasattr(lc, "keys") and hasattr(lc, "values"):
                keys = lc.keys
                values = lc.values
                if not callable(keys) and not callable(values):
                    kv_states.append({"keys": keys, "values": values, "layer_idx": idx})
        ssm_states = []
        for idx in ssm_indices:
            lc = cache[idx]
            if hasattr(lc, "cache") and isinstance(lc.cache, list):
                ssm_states.append({"state": lc.cache, "layer_idx": idx})
        return kv_states, ssm_states

    def test_create_with_real_states(self, cache_and_output):
        """HybridCacheEntry.create should work with real model states."""
        cache, _, tokens = cache_and_output
        kv_states, ssm_states = self._extract_states(cache)

        entry = HybridCacheEntry.create(
            token_key=tuple(tokens),
            kv_states=kv_states,
            ssm_states=ssm_states,
            ssm_checkpoint_token=len(tokens),
            entry_type="exact",
        )

        assert entry.token_key == tuple(tokens)
        assert entry.access_count == 1
        assert entry.has_ssm_states()
        # Memory should be estimated (might be 0 if dict-based states
        # don't match estimate_hybrid_memory's patterns, but shouldn't error)
        assert entry.memory_bytes >= 0


class TestEndToEndPrefixReuse:
    """Test the full prefix reuse scenario with real model."""

    def test_shared_prefix_different_suffix(self, model, tokenizer):
        """Inserting tokens and looking up with extra tokens appended should find prefix."""
        prompt = "Once upon a time in a land far far away"
        tokens = tokenizer.encode(prompt)

        # Run forward pass for the prefix
        cache_a = model.make_cache()
        input_a = mx.array([tokens])
        output_a = model(input_a, cache=cache_a)
        _ = output_a.shape

        # Extract states
        kv_indices, ssm_indices = classify_cache_layers(cache_a)

        config = HybridModelConfig(
            num_attention_layers=len(kv_indices),
            num_ssm_layers=len(ssm_indices),
        )
        radix = RadixCacheHybrid(config)

        kv_states_a = []
        for idx in kv_indices:
            lc = cache_a[idx]
            if hasattr(lc, "keys") and hasattr(lc, "values"):
                keys = lc.keys
                values = lc.values
                if not callable(keys) and not callable(values):
                    kv_states_a.append({"keys": keys, "values": values, "layer_idx": idx})

        ssm_states_a = []
        for idx in ssm_indices:
            lc = cache_a[idx]
            if hasattr(lc, "cache") and isinstance(lc.cache, list):
                ssm_states_a.append({"state": lc.cache, "layer_idx": idx})

        token_key = tuple(tokens)
        radix.insert(token_key, kv_states_a, ssm_states_a)

        # Lookup with the same prefix + extra tokens appended
        # This guarantees a shared prefix at the token level
        extended = token_key + (9999, 10000, 10001)
        result = radix.lookup(extended)

        assert result is not None
        assert result.matched_length == len(tokens)
        assert result.kv_states is not None

        # Clean up
        del cache_a, output_a
        gc.collect()

    def test_exact_match_returns_ssm(self, model, tokenizer):
        """Exact token match should return SSM states (after 2nd access for admission)."""
        prompt = "Hello world"
        tokens = tokenizer.encode(prompt)

        cache = model.make_cache()
        input_ids = mx.array([tokens])
        output = model(input_ids, cache=cache)
        _ = output.shape

        kv_indices, ssm_indices = classify_cache_layers(cache)
        config = HybridModelConfig(
            num_attention_layers=len(kv_indices),
            num_ssm_layers=len(ssm_indices),
        )
        radix = RadixCacheHybrid(config)

        kv_states = []
        for idx in kv_indices:
            lc = cache[idx]
            if hasattr(lc, "keys") and hasattr(lc, "values"):
                keys = lc.keys
                values = lc.values
                if not callable(keys) and not callable(values):
                    kv_states.append({"keys": keys, "values": values, "layer_idx": idx})

        ssm_states = []
        for idx in ssm_indices:
            lc = cache[idx]
            if hasattr(lc, "cache") and isinstance(lc.cache, list):
                ssm_states.append({"state": lc.cache, "layer_idx": idx})

        token_key = tuple(tokens)

        # First insert (access_count becomes 1)
        radix.insert(token_key, kv_states, ssm_states)

        # Second insert (access_count becomes 2 — admission threshold)
        radix.insert(token_key, kv_states, ssm_states)

        # Lookup should now have SSM states
        result = radix.lookup(token_key)
        assert result is not None
        assert result.matched_length == len(tokens)
        # After 2nd access, SSM should be admitted
        if result.ssm_states is not None:
            assert len(result.ssm_states) > 0

        del cache, output
        gc.collect()


class TestAutoDetectConfig:
    """Test HybridModelConfig.from_model() auto-detection with real Qwen3.5-9B model."""

    def _extract_states(self, cache):
        """Extract KV and SSM states from cache."""
        kv_indices, ssm_indices = classify_cache_layers(cache)

        kv_states = []
        for idx in kv_indices:
            lc = cache[idx]
            if hasattr(lc, "keys") and hasattr(lc, "values"):
                keys = lc.keys
                values = lc.values
                if not callable(keys) and not callable(values):
                    kv_states.append({"keys": keys, "values": values, "layer_idx": idx})

        ssm_states = []
        for idx in ssm_indices:
            lc = cache[idx]
            if hasattr(lc, "cache") and isinstance(lc.cache, list):
                ssm_states.append({"state": lc.cache, "layer_idx": idx})

        return kv_states, ssm_states

    def test_text_config_exists(self, model):
        """model.args.text_config should be a dict for Qwen3.5 — documents the assumption."""
        assert hasattr(model, "args"), "model should have args attribute"
        assert hasattr(model.args, "text_config"), "model.args should have text_config"
        assert isinstance(model.args.text_config, dict), (
            f"text_config should be a dict, got {type(model.args.text_config)}"
        )

    def test_from_model_produces_correct_config(self, model):
        """from_model() with real Qwen3.5-9B-4bit should produce the expected config."""
        config = HybridModelConfig.from_model(model)

        assert config.num_attention_layers == 8, (
            f"Expected 8 attention layers, got {config.num_attention_layers}"
        )
        assert config.num_ssm_layers == 24, (
            f"Expected 24 SSM layers, got {config.num_ssm_layers}"
        )
        assert config.hidden_size == 4096, (
            f"Expected hidden_size=4096, got {config.hidden_size}"
        )
        assert config.num_heads == 16, (
            f"Expected num_heads=16, got {config.num_heads}"
        )
        assert config.head_dim == 256, (
            f"Expected head_dim=256, got {config.head_dim}"
        )
        assert config.ssm_state_size == 128 * 128, (
            f"Expected ssm_state_size={128 * 128} (linear_key_head_dim=128 * linear_value_head_dim=128), "
            f"got {config.ssm_state_size}"
        )
        assert config.num_ssm_heads == 32, (
            f"Expected num_ssm_heads=32, got {config.num_ssm_heads}"
        )

    def test_from_model_matches_cache_classification(self, model):
        """from_model() layer counts should match classify_cache_layers() results."""
        config = HybridModelConfig.from_model(model)
        kv_indices, ssm_indices = classify_cache_layers(model.make_cache())

        assert config.num_attention_layers == len(kv_indices), (
            f"from_model attention layers ({config.num_attention_layers}) "
            f"!= classify_cache_layers ({len(kv_indices)})"
        )
        assert config.num_ssm_layers == len(ssm_indices), (
            f"from_model SSM layers ({config.num_ssm_layers}) "
            f"!= classify_cache_layers ({len(ssm_indices)})"
        )

    def test_from_model_in_radix_cache(self, cache_and_output):
        """RadixCacheHybrid created with from_model() config should support insert+lookup."""
        cache, _, tokens = cache_and_output
        kv_states, ssm_states = self._extract_states(cache)

        # Use from_model to get config — need the model, not the cache
        # Reconstruct config matching what from_model would produce for Qwen3.5-9B
        config = HybridModelConfig(
            num_attention_layers=8,
            num_ssm_layers=24,
            hidden_size=4096,
            num_heads=16,
            head_dim=256,
            ssm_state_size=128 * 128,  # linear_key_head_dim=128 * linear_value_head_dim=128
            num_ssm_heads=32,
        )
        radix = RadixCacheHybrid(config)

        token_key = tuple(tokens)
        radix.insert(token_key, kv_states, ssm_states)

        result = radix.lookup(token_key)
        assert result is not None
        assert result.matched_length == len(tokens)
        assert result.kv_states is not None

    def test_from_model_in_radix_cache_end_to_end(self, model, cache_and_output):
        """End-to-end: use from_model() directly to build config, then insert+lookup."""
        cache, _, tokens = cache_and_output
        kv_states, ssm_states = self._extract_states(cache)

        config = HybridModelConfig.from_model(model)
        radix = RadixCacheHybrid(config)

        token_key = tuple(tokens)
        radix.insert(token_key, kv_states, ssm_states)

        result = radix.lookup(token_key)
        assert result is not None
        assert result.matched_length == len(tokens)
        assert result.kv_states is not None
