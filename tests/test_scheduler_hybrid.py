# SPDX-License-Identifier: Apache-2.0
"""Tests for scheduler hybrid model (SSM + KV) modifications.

Tests the is_ssm tagging in _extract_cache_states, the _separate_hybrid_states
helper, _restore_hybrid_cache selective reuse logic, and SSM checkpoint
tracking in the mid-prefill save callback.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock helpers — simulate KVCache and ArraysCache without importing mlx
# ---------------------------------------------------------------------------


class _MockKVCache:
    """Lightweight mock KVCache (avoids MagicMock __name__/__class__ issues)."""

    def __init__(self, offset=10):
        self._keys = MagicMock(name="keys")
        self._values = MagicMock(name="values")
        self.state = (self._keys, self._values)
        self.meta_state = (str(offset),)

    def is_trimmable(self):
        return True


class _MockArraysCache:
    """Lightweight mock ArraysCache."""

    def __init__(self):
        conv_state = MagicMock(name="conv_state")
        recurrent_state = MagicMock(name="recurrent_state")
        self.cache = [conv_state, recurrent_state]
        self.state = (conv_state, recurrent_state)
        self.meta_state = ()

    def is_trimmable(self):
        return False


# Rename classes to match what _extract_cache_states records via type().__name__
_MockKVCache.__name__ = "KVCache"
_MockArraysCache.__name__ = "ArraysCache"


def _make_kv_cache_mock(offset=10):
    """Create a mock KVCache layer (trimmable, has keys/values)."""
    return _MockKVCache(offset=offset)


def _make_ssm_cache_mock():
    """Create a mock ArraysCache layer (non-trimmable, has .cache list)."""
    return _MockArraysCache()


def _make_ssm_cache_no_trimmable():
    """Create a mock SSM layer that lacks is_trimmable entirely."""
    mock = MagicMock(spec=[])
    mock.state = (MagicMock(name="conv"), MagicMock(name="recurrent"))
    mock.meta_state = ()
    mock.cache = [mock.state[0], mock.state[1]]
    # Explicitly ensure is_trimmable does not exist
    assert not hasattr(mock, "is_trimmable")
    return mock


def _make_request_mock(request_id="req-test-001", prompt_token_ids=None):
    """Create a mock Request object."""
    request = MagicMock()
    request.request_id = request_id
    request.prompt_token_ids = prompt_token_ids or list(range(100))
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.cached_tokens = 0
    request.remaining_tokens = request.prompt_token_ids
    # Ensure hybrid attrs don't exist yet (test that they get set)
    if hasattr(request, "_needs_ssm_recompute"):
        del request._needs_ssm_recompute
    if hasattr(request, "_ssm_recompute_from"):
        del request._ssm_recompute_from
    # Use a real dict-like behavior for dynamic attrs
    request._needs_ssm_recompute = None
    request._ssm_recompute_from = None
    return request


def _make_scheduler_instance():
    """Create a minimal Scheduler-like object with the hybrid methods.

    Instead of instantiating the real Scheduler (which requires model,
    tokenizer, etc.), we import the unbound methods and bind them to a
    lightweight mock that has just enough state.
    """
    from vllm_mlx.scheduler import Scheduler

    sched = MagicMock(spec=Scheduler)
    # Bind the real methods we want to test
    sched._extract_cache_states = Scheduler._extract_cache_states.__get__(sched)
    sched._separate_hybrid_states = Scheduler._separate_hybrid_states.__get__(sched)
    sched._restore_hybrid_cache = Scheduler._restore_hybrid_cache.__get__(sched)
    # Default to None so _restore_hybrid_cache uses fallback detection path
    sched._hybrid_layer_indices = None
    return sched


# ===========================================================================
# Test: _extract_cache_states with is_ssm tagging
# ===========================================================================


class TestExtractCacheStatesIsSSM:
    """Verify that _extract_cache_states tags each layer with is_ssm."""

    def test_pure_kv_model(self):
        """All KVCache layers should have is_ssm=False."""
        sched = _make_scheduler_instance()
        cache = [_make_kv_cache_mock(offset=i * 10) for i in range(4)]

        extracted = sched._extract_cache_states(cache)

        assert len(extracted) == 4
        for layer_state in extracted:
            assert layer_state["is_ssm"] is False
            assert layer_state["class_name"] == "KVCache"

    def test_pure_ssm_model(self):
        """All ArraysCache layers should have is_ssm=True."""
        sched = _make_scheduler_instance()
        cache = [_make_ssm_cache_mock() for _ in range(4)]

        extracted = sched._extract_cache_states(cache)

        assert len(extracted) == 4
        for layer_state in extracted:
            assert layer_state["is_ssm"] is True
            assert layer_state["class_name"] == "ArraysCache"

    def test_hybrid_model_mixed(self):
        """Hybrid model: interleaved KV and SSM layers (Qwen3.5-like 3:1 ratio)."""
        sched = _make_scheduler_instance()
        # Pattern: SSM, SSM, SSM, KV — repeated twice (8 layers total)
        cache = [
            _make_ssm_cache_mock(),
            _make_ssm_cache_mock(),
            _make_ssm_cache_mock(),
            _make_kv_cache_mock(offset=30),
            _make_ssm_cache_mock(),
            _make_ssm_cache_mock(),
            _make_ssm_cache_mock(),
            _make_kv_cache_mock(offset=70),
        ]

        extracted = sched._extract_cache_states(cache)

        assert len(extracted) == 8
        expected_ssm = [True, True, True, False, True, True, True, False]
        for i, layer_state in enumerate(extracted):
            assert layer_state["is_ssm"] == expected_ssm[i], (
                f"Layer {i}: expected is_ssm={expected_ssm[i]}, "
                f"got {layer_state['is_ssm']}"
            )

    def test_ssm_without_is_trimmable(self):
        """SSM layer lacking is_trimmable method should still be tagged is_ssm=True."""
        sched = _make_scheduler_instance()
        ssm_no_trim = _make_ssm_cache_no_trimmable()
        # Need to give it state/meta_state for extraction
        cache = [_make_kv_cache_mock(), ssm_no_trim]

        # The mock without is_trimmable won't have .state/.meta_state
        # accessible via hasattr in the expected way for _extract_cache_states.
        # We need to ensure the mock has the right protocol.
        ssm_no_trim.state = (MagicMock(), MagicMock())
        ssm_no_trim.meta_state = ()

        extracted = sched._extract_cache_states(cache)

        assert len(extracted) == 2
        assert extracted[0]["is_ssm"] is False  # KVCache
        assert extracted[1]["is_ssm"] is True  # No is_trimmable -> SSM

    def test_class_ref_preserved(self):
        """Verify class_ref is preserved for reconstruction."""
        sched = _make_scheduler_instance()
        kv = _make_kv_cache_mock()
        ssm = _make_ssm_cache_mock()
        cache = [kv, ssm]

        extracted = sched._extract_cache_states(cache)

        assert len(extracted) == 2
        assert extracted[0]["class_ref"] is type(kv)
        assert extracted[1]["class_ref"] is type(ssm)

    def test_empty_cache(self):
        """Empty cache should return empty list."""
        sched = _make_scheduler_instance()
        assert sched._extract_cache_states([]) == []

    def test_extraction_failure_returns_empty(self):
        """If any layer fails extraction, entire result is empty."""
        sched = _make_scheduler_instance()
        good = _make_kv_cache_mock()
        bad = MagicMock()
        # Make state access raise an exception
        type(bad).state = property(lambda self: (_ for _ in ()).throw(RuntimeError("corrupted")))

        cache = [good, bad]
        extracted = sched._extract_cache_states(cache)

        # Should return empty because not all layers extracted
        assert extracted == []


# ===========================================================================
# Test: _separate_hybrid_states
# ===========================================================================


class TestSeparateHybridStates:
    """Verify correct separation of KV and SSM indices."""

    def test_pure_kv_model(self):
        """All trimmable: ssm_indices should be empty."""
        sched = _make_scheduler_instance()
        states = [
            {"is_ssm": False, "state": MagicMock()},
            {"is_ssm": False, "state": MagicMock()},
            {"is_ssm": False, "state": MagicMock()},
        ]

        full, kv_idx, ssm_idx = sched._separate_hybrid_states(states)

        assert full is states  # Same reference
        assert kv_idx == [0, 1, 2]
        assert ssm_idx == []

    def test_pure_ssm_model(self):
        """All non-trimmable: kv_indices should be empty."""
        sched = _make_scheduler_instance()
        states = [
            {"is_ssm": True, "state": MagicMock()},
            {"is_ssm": True, "state": MagicMock()},
        ]

        full, kv_idx, ssm_idx = sched._separate_hybrid_states(states)

        assert kv_idx == []
        assert ssm_idx == [0, 1]

    def test_hybrid_model_qwen35_ratio(self):
        """Qwen3.5-like 3:1 SSM:KV ratio (24 SSM + 8 KV = 32 layers)."""
        sched = _make_scheduler_instance()
        # Build a realistic interleaving: SSM,SSM,SSM,KV repeated 8 times
        states = []
        for block in range(8):
            base = block * 4
            states.append({"is_ssm": True, "state": MagicMock()})
            states.append({"is_ssm": True, "state": MagicMock()})
            states.append({"is_ssm": True, "state": MagicMock()})
            states.append({"is_ssm": False, "state": MagicMock()})

        full, kv_idx, ssm_idx = sched._separate_hybrid_states(states)

        assert len(states) == 32
        assert len(kv_idx) == 8
        assert len(ssm_idx) == 24
        # KV layers are at positions 3, 7, 11, 15, 19, 23, 27, 31
        assert kv_idx == [3, 7, 11, 15, 19, 23, 27, 31]
        # SSM layers are everything else
        expected_ssm = [i for i in range(32) if i not in kv_idx]
        assert ssm_idx == expected_ssm

    def test_empty_states(self):
        """Empty input should return empty indices."""
        sched = _make_scheduler_instance()
        full, kv_idx, ssm_idx = sched._separate_hybrid_states([])

        assert full == []
        assert kv_idx == []
        assert ssm_idx == []

    def test_missing_is_ssm_defaults_to_kv(self):
        """If is_ssm key is missing, layer defaults to KV (is_ssm=False)."""
        sched = _make_scheduler_instance()
        states = [
            {"state": MagicMock()},  # No is_ssm key
            {"is_ssm": True, "state": MagicMock()},
        ]

        full, kv_idx, ssm_idx = sched._separate_hybrid_states(states)

        assert kv_idx == [0]  # Missing key -> .get defaults to False -> KV
        assert ssm_idx == [1]


# ===========================================================================
# Test: _restore_hybrid_cache
# ===========================================================================


class TestRestoreHybridCache:
    """Verify selective layer reuse logic."""

    def test_non_hybrid_passthrough(self):
        """Pure KV model: cache returned unchanged, no flags set."""
        sched = _make_scheduler_instance()
        request = _make_request_mock()

        # All layers are trimmable KVCache
        cache = [_make_kv_cache_mock() for _ in range(4)]

        result = sched._restore_hybrid_cache(cache, request, matched_tokens=50)

        assert result is cache
        # No hybrid flags should be set (they remain at initial None)
        assert request._needs_ssm_recompute is None
        assert request._ssm_recompute_from is None

    def test_full_hybrid_hit(self):
        """Both KV and SSM states present: no recompute needed."""
        sched = _make_scheduler_instance()
        request = _make_request_mock()

        # Mix of KV and SSM layers, SSM layers have actual state data
        ssm = _make_ssm_cache_mock()
        # Ensure .cache has non-None arrays
        ssm.cache = [MagicMock(name="conv"), MagicMock(name="recurrent")]
        cache = [_make_kv_cache_mock(), ssm, _make_kv_cache_mock()]

        result = sched._restore_hybrid_cache(cache, request, matched_tokens=50)

        assert result is cache
        assert request._needs_ssm_recompute is None
        assert request._ssm_recompute_from is None

    def test_kv_only_hit_sets_recompute_flags(self):
        """KV present but SSM state empty: sets recompute flags."""
        sched = _make_scheduler_instance()
        request = _make_request_mock()

        # SSM layer exists but has no meaningful state (all None)
        ssm = MagicMock()
        ssm.is_trimmable.return_value = False
        ssm.cache = [None, None]  # Empty SSM state
        # Ensure hasattr checks work
        del ssm.state

        cache = [_make_kv_cache_mock(), ssm, _make_kv_cache_mock()]

        result = sched._restore_hybrid_cache(cache, request, matched_tokens=75)

        assert result is cache
        assert request._needs_ssm_recompute is True
        assert request._ssm_recompute_from == 75

    def test_kv_only_hit_different_token_positions(self):
        """Verify _ssm_recompute_from reflects the matched token count."""
        sched = _make_scheduler_instance()

        for matched_tokens in [0, 1, 50, 100, 512, 4096]:
            request = _make_request_mock()
            ssm = MagicMock()
            ssm.is_trimmable.return_value = False
            ssm.cache = [None, None]
            del ssm.state
            cache = [_make_kv_cache_mock(), ssm]

            sched._restore_hybrid_cache(cache, request, matched_tokens=matched_tokens)

            assert request._ssm_recompute_from == matched_tokens

    def test_none_cache_returns_none(self):
        """None or empty cache returns None."""
        sched = _make_scheduler_instance()
        request = _make_request_mock()

        assert sched._restore_hybrid_cache(None, request, matched_tokens=0) is None
        assert sched._restore_hybrid_cache([], request, matched_tokens=0) is None

    def test_ssm_with_state_property(self):
        """SSM layer with .state property (not .cache) containing data."""
        sched = _make_scheduler_instance()
        request = _make_request_mock()

        ssm = MagicMock()
        ssm.is_trimmable.return_value = False
        # No .cache attribute, but has .state with actual data
        del ssm.cache
        ssm.state = [MagicMock(name="data1"), MagicMock(name="data2")]

        cache = [_make_kv_cache_mock(), ssm]

        result = sched._restore_hybrid_cache(cache, request, matched_tokens=30)

        # Should detect SSM state is present via .state property
        assert result is cache
        assert request._needs_ssm_recompute is None  # Full hit

    def test_none_layers_skipped(self):
        """None layers in cache list should be safely skipped."""
        sched = _make_scheduler_instance()
        request = _make_request_mock()

        cache = [_make_kv_cache_mock(), None, _make_kv_cache_mock()]

        result = sched._restore_hybrid_cache(cache, request, matched_tokens=50)

        assert result is cache
        # No SSM layers detected, so passthrough
        assert request._needs_ssm_recompute is None


# ===========================================================================
# Test: Mid-prefill save SSM checkpoint tracking
# ===========================================================================


class TestMidPrefillSSMTracking:
    """Verify _ssm_checkpoint_positions gets populated for hybrid models."""

    def _make_scheduler_for_callback(self):
        """Create a minimal scheduler mock with mid-prefill callback support."""
        from vllm_mlx.scheduler import Scheduler

        sched = MagicMock(spec=Scheduler)
        sched._extract_cache_states = Scheduler._extract_cache_states.__get__(sched)
        sched._separate_hybrid_states = Scheduler._separate_hybrid_states.__get__(sched)

        # Set up uid -> request mapping
        sched.uid_to_request_id = {}
        sched.requests = {}

        # Mock memory_aware_cache
        sched.memory_aware_cache = MagicMock()
        sched.memory_aware_cache.store.return_value = True

        # Bind _make_mid_prefill_save_callback
        sched._make_mid_prefill_save_callback = (
            Scheduler._make_mid_prefill_save_callback.__get__(sched)
        )
        # Also bind _reconstruct_cache_from_states
        sched._reconstruct_cache_from_states = (
            Scheduler._reconstruct_cache_from_states.__get__(sched)
        )

        return sched

    def test_ssm_checkpoint_positions_populated(self):
        """Mid-prefill save should record SSM checkpoint positions."""
        sched = self._make_scheduler_for_callback()

        # Use SimpleNamespace so hasattr works correctly
        request = SimpleNamespace(
            request_id="req-hybrid-001",
            prompt_token_ids=list(range(1000)),
            cached_tokens=0,
            prefix_boundary=0,
            _mid_prefill_last_save=0,
            _mid_prefill_cache_key=None,
        )

        uid = 42
        sched.uid_to_request_id[uid] = "req-hybrid-001"
        sched.requests["req-hybrid-001"] = request

        # Create hybrid cache (KV + SSM layers)
        prompt_cache = [
            _make_kv_cache_mock(offset=256),
            _make_ssm_cache_mock(),
            _make_kv_cache_mock(offset=256),
            _make_ssm_cache_mock(),
        ]

        # Mock _reconstruct_cache_from_states to return something truthy
        sched._reconstruct_cache_from_states = MagicMock(return_value=[MagicMock()])

        callback = sched._make_mid_prefill_save_callback(save_interval=256)
        callback(uid, 256, prompt_cache)

        # Verify SSM checkpoint position was recorded
        assert hasattr(request, "_ssm_checkpoint_positions")
        assert request._ssm_checkpoint_positions == [256]

    def test_ssm_checkpoint_accumulates(self):
        """Multiple mid-prefill saves should accumulate checkpoint positions."""
        sched = self._make_scheduler_for_callback()

        request = SimpleNamespace(
            request_id="req-hybrid-002",
            prompt_token_ids=list(range(2000)),
            cached_tokens=0,
            prefix_boundary=0,
            _mid_prefill_last_save=0,
            _mid_prefill_cache_key=None,
        )

        uid = 43
        sched.uid_to_request_id[uid] = "req-hybrid-002"
        sched.requests["req-hybrid-002"] = request

        prompt_cache = [
            _make_kv_cache_mock(offset=256),
            _make_ssm_cache_mock(),
        ]

        sched._reconstruct_cache_from_states = MagicMock(return_value=[MagicMock()])

        callback = sched._make_mid_prefill_save_callback(save_interval=256)

        # Simulate multiple chunked prefill saves
        callback(uid, 256, prompt_cache)
        callback(uid, 512, prompt_cache)
        callback(uid, 768, prompt_cache)

        assert request._ssm_checkpoint_positions == [256, 512, 768]

    def test_no_ssm_no_checkpoint(self):
        """Pure KV model should NOT set _ssm_checkpoint_positions."""
        sched = self._make_scheduler_for_callback()

        request = SimpleNamespace(
            request_id="req-kv-only",
            prompt_token_ids=list(range(1000)),
            cached_tokens=0,
            prefix_boundary=0,
            _mid_prefill_last_save=0,
            _mid_prefill_cache_key=None,
        )

        uid = 44
        sched.uid_to_request_id[uid] = "req-kv-only"
        sched.requests["req-kv-only"] = request

        # Pure KV cache — no SSM layers
        prompt_cache = [
            _make_kv_cache_mock(offset=256),
            _make_kv_cache_mock(offset=256),
        ]

        sched._reconstruct_cache_from_states = MagicMock(return_value=[MagicMock()])

        callback = sched._make_mid_prefill_save_callback(save_interval=256)
        callback(uid, 256, prompt_cache)

        # _ssm_checkpoint_positions should NOT have been set
        assert not hasattr(request, "_ssm_checkpoint_positions")


# ===========================================================================
# Test: Prompt cache save SSM tagging
# ===========================================================================


class TestPromptCacheSaveSSMTagging:
    """Verify _has_ssm_cache tagging in prompt cache save callback."""

    def _make_scheduler_for_prompt_save(self):
        """Create scheduler mock for prompt cache save callback."""
        from vllm_mlx.scheduler import Scheduler

        sched = MagicMock(spec=Scheduler)
        sched.uid_to_request_id = {}
        sched.requests = {}
        sched.memory_aware_cache = MagicMock()
        sched.memory_aware_cache.store.return_value = True

        sched._make_prompt_cache_save_callback = (
            Scheduler._make_prompt_cache_save_callback.__get__(sched)
        )
        return sched

    def test_hybrid_cache_tagged(self):
        """Prompt save with SSM layers should set _has_ssm_cache."""
        sched = self._make_scheduler_for_prompt_save()

        request = SimpleNamespace(
            request_id="req-prompt-hybrid",
            prompt_token_ids=list(range(100)),
        )

        uid = 50
        sched.uid_to_request_id[uid] = "req-prompt-hybrid"
        sched.requests["req-prompt-hybrid"] = request

        # Extracted cache with is_ssm flags (as dicts, matching what
        # _extract_cache_states produces)
        extracted = [
            {"is_ssm": False, "state": MagicMock()},
            {"is_ssm": True, "state": MagicMock()},
        ]

        callback = sched._make_prompt_cache_save_callback()
        callback(uid, extracted)

        assert request._has_ssm_cache is True

    def test_pure_kv_not_tagged(self):
        """Prompt save with only KV layers should NOT set _has_ssm_cache."""
        sched = self._make_scheduler_for_prompt_save()

        request = SimpleNamespace(
            request_id="req-prompt-kv",
            prompt_token_ids=list(range(100)),
        )

        uid = 51
        sched.uid_to_request_id[uid] = "req-prompt-kv"
        sched.requests["req-prompt-kv"] = request

        extracted = [
            {"is_ssm": False, "state": MagicMock()},
            {"is_ssm": False, "state": MagicMock()},
        ]

        callback = sched._make_prompt_cache_save_callback()
        callback(uid, extracted)

        # _has_ssm_cache should not have been set
        assert not hasattr(request, "_has_ssm_cache")
