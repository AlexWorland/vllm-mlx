# SPDX-License-Identifier: Apache-2.0
"""
Hybrid radix tree cache for models with mixed attention + recurrent layers.

Implements the core MARCONI-inspired data structure: a radix tree that manages
both KV (attention) and SSM (recurrent/GatedDeltaNet) states on edges/nodes.

Key MARCONI concepts implemented:
- Judicious Admission: SSM states are only stored for branch points or on
  repeated access (2nd occurrence), preventing wasted memory from the 65x
  lower reuse rate of SSM states vs KV states.
- FLOP-Aware Eviction: Eviction score combines recency with FLOP efficiency
  (FLOPs saved per byte). Attention layers contribute O(L^2) FLOPs while
  recurrent layers contribute O(L) FLOPs per position.
- Two-Tier Cache: Hot tier (active in tree) and cold tier (evicted but
  retained in memory for fast promotion).
- Alpha Tuning: The weight between recency and FLOP efficiency is
  bootstrapped from observed access patterns.

Design notes:
- MLX arrays are immutable — safe to share references (no deep copy needed)
- Timestamps use time.monotonic() for reliable ordering
- Memory estimation uses shape*dtype.size to avoid triggering MLX lazy eval
- The radix tree is standalone and testable without the full vllm-mlx stack
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .hybrid_cache_entry import (
    HybridCacheEntry,
    classify_cache_layers,
    estimate_hybrid_memory,
)

logger = logging.getLogger(__name__)


def _get_text_config(model: Any) -> Optional[dict]:
    """Return ``model.args.text_config`` if it exists and is a dict."""
    args = getattr(model, "args", None)
    if args is None:
        return None
    tc = getattr(args, "text_config", None)
    if isinstance(tc, dict) and tc:
        return tc
    return None


def _get_flat_args(model: Any) -> Optional[Any]:
    """Return ``model.args`` if it has useful layer info (not a bare dict)."""
    args = getattr(model, "args", None)
    if args is None:
        return None
    # Must have at least hidden_size or layer_types to be useful
    if hasattr(args, "hidden_size") or hasattr(args, "layer_types"):
        return args
    return None


@dataclass
class HybridModelConfig:
    """Model configuration for FLOP calculations.

    Defaults are conservative/generic — use ``from_model()`` to auto-detect
    accurate values from a loaded model.

    Attributes:
        num_attention_layers: Number of attention layers in the model.
        num_ssm_layers: Number of recurrent/SSM layers.
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        ssm_state_size: Delta-rule hidden state size per head.
        num_ssm_heads: Number of SSM heads (linear_num_value_heads).
    """

    num_attention_layers: int = 8
    num_ssm_layers: int = 24
    hidden_size: int = 4096
    num_heads: int = 16
    head_dim: int = 256
    ssm_state_size: int = 128 * 128
    num_ssm_heads: int = 32

    @classmethod
    def from_model(cls, model: Any) -> "HybridModelConfig":
        """Auto-detect config from a loaded model.

        Extraction priority:
        1. ``model.args.text_config`` dict (Qwen3.5 multimodal-style config)
        2. ``model.args`` with ``getattr()`` for flat config models
        3. ``classify_cache_layers(model.make_cache())`` for layer counts,
           plus defaults for dimension fields

        Args:
            model: A loaded MLX model object.

        Returns:
            Populated HybridModelConfig.
        """
        text_config = _get_text_config(model)
        if text_config is not None:
            config = cls._from_text_config(text_config)
            logger.info(
                f"Auto-detected config via text_config: "
                f"{config.num_attention_layers} attn + {config.num_ssm_layers} ssm layers, "
                f"hidden_size={config.hidden_size}, num_heads={config.num_heads}, "
                f"head_dim={config.head_dim}, ssm_state_size={config.ssm_state_size}, "
                f"num_ssm_heads={config.num_ssm_heads}"
            )
            return config

        flat_args = _get_flat_args(model)
        if flat_args is not None:
            config = cls._from_flat_args(flat_args)
            logger.info(
                f"Auto-detected config via flat args: "
                f"{config.num_attention_layers} attn + {config.num_ssm_layers} ssm layers, "
                f"hidden_size={config.hidden_size}, num_heads={config.num_heads}, "
                f"head_dim={config.head_dim}"
            )
            return config

        return cls._from_cache(model)

    @classmethod
    def _from_text_config(cls, tc: dict) -> "HybridModelConfig":
        """Extract config from a ``text_config`` dict."""
        layer_types = tc.get("layer_types", [])
        num_attn = layer_types.count("full_attention")
        num_ssm = layer_types.count("linear_attention")

        linear_key_dim = tc.get("linear_key_head_dim", 128)
        linear_val_dim = tc.get("linear_value_head_dim", 128)

        return cls(
            num_attention_layers=num_attn,
            num_ssm_layers=num_ssm,
            hidden_size=tc.get("hidden_size", cls.hidden_size),
            num_heads=tc.get("num_attention_heads", cls.num_heads),
            head_dim=tc.get("head_dim", cls.head_dim),
            ssm_state_size=linear_key_dim * linear_val_dim,
            num_ssm_heads=tc.get("linear_num_value_heads", cls.num_ssm_heads),
        )

    @classmethod
    def _from_flat_args(cls, args: Any) -> "HybridModelConfig":
        """Extract config from flat model args (``getattr``-based)."""
        layer_types = getattr(args, "layer_types", None)
        if layer_types is not None:
            num_attn = layer_types.count("full_attention")
            num_ssm = layer_types.count("linear_attention")
        else:
            num_attn = cls.num_attention_layers
            num_ssm = cls.num_ssm_layers

        linear_key_dim = getattr(args, "linear_key_head_dim", 128)
        linear_val_dim = getattr(args, "linear_value_head_dim", 128)

        return cls(
            num_attention_layers=num_attn,
            num_ssm_layers=num_ssm,
            hidden_size=getattr(args, "hidden_size", cls.hidden_size),
            num_heads=getattr(args, "num_attention_heads", cls.num_heads),
            head_dim=getattr(args, "head_dim", cls.head_dim),
            ssm_state_size=linear_key_dim * linear_val_dim,
            num_ssm_heads=getattr(
                args, "linear_num_value_heads", cls.num_ssm_heads
            ),
        )

    @classmethod
    def _from_cache(cls, model: Any) -> "HybridModelConfig":
        """Fall back to cache-based layer classification + defaults."""
        try:
            cache = model.make_cache()
            kv_indices, ssm_indices = classify_cache_layers(cache)
            config = cls(
                num_attention_layers=len(kv_indices),
                num_ssm_layers=len(ssm_indices),
            )
            logger.info(
                f"Auto-detected config via cache classification: "
                f"{len(kv_indices)} attn + {len(ssm_indices)} ssm layers "
                f"(dimension fields use defaults)"
            )
            return config
        except Exception:
            logger.warning(
                "Could not auto-detect model config; using defaults"
            )
            return cls()


@dataclass
class HybridCacheLookup:
    """Result of a radix tree lookup.

    Attributes:
        matched_length: Number of tokens matched in the radix tree.
        kv_states: KV cache states for matched prefix (if any).
        ssm_states: SSM states for matched prefix (if any).
        needs_ssm_recompute: True if KV hit but no SSM states stored.
        match_type: "exact", "prefix", or "none".
    """

    matched_length: int = 0
    kv_states: Optional[List[Any]] = None
    ssm_states: Optional[List[Any]] = None
    needs_ssm_recompute: bool = False
    match_type: str = "none"


@dataclass
class RadixNode:
    """Node in the hybrid radix tree.

    Each node represents an edge in the radix tree labeled with a token
    subsequence. Nodes may optionally hold KV and/or SSM cache states.

    Attributes:
        children: Mapping from first token of edge label to child node.
        token_ids: Edge label — the token subsequence on this edge.
        parent: Parent node (None for root).
        kv_states: KV cache for this prefix (always stored on insert).
        ssm_states: SSM states (stored only if admission policy allows).
        ssm_checkpoint_pos: Token position where SSM state was snapshotted.
        depth: Total tokens from root to the end of this node's edge.
        last_access: Monotonic timestamp of last access.
        access_count: Number of accesses (for admission decisions).
        memory_bytes: Estimated memory footprint of stored states.
        is_leaf: Whether this node is a leaf (no children).
    """

    children: Dict[int, RadixNode] = field(default_factory=dict)
    token_ids: Tuple[int, ...] = ()
    parent: Optional[RadixNode] = None
    kv_states: Optional[List[Any]] = None
    ssm_states: Optional[List[Any]] = None
    ssm_checkpoint_pos: int = 0
    depth: int = 0
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0
    memory_bytes: int = 0
    is_leaf: bool = True


def compute_flop_efficiency(
    node: RadixNode, config: HybridModelConfig
) -> float:
    """Compute FLOPs saved per byte of memory consumed.

    For attention layers: FLOPs are proportional to seq_len^2 (quadratic)
    due to the attention matrix computation.
    For recurrent layers: FLOPs are proportional to seq_len (linear)
    at approximately 200 FLOPs per position per layer (from MARCONI).

    Args:
        node: The radix tree node to evaluate.
        config: Model configuration with layer counts.

    Returns:
        FLOP efficiency ratio (FLOPs saved / memory bytes).
    """
    seq_len = node.depth
    if seq_len == 0 or node.memory_bytes == 0:
        return 0.0

    kv_flops = seq_len * seq_len  # simplified quadratic
    ssm_flops = seq_len * 200  # ~200 FLOPs per position per layer

    total_flops = (
        kv_flops * config.num_attention_layers
        + ssm_flops * config.num_ssm_layers
    )

    return total_flops / max(node.memory_bytes, 1)


class RadixCacheHybrid:
    """Radix tree cache managing both KV and SSM states.

    This is the core MARCONI-inspired data structure for hybrid models.
    It extends a standard radix/prefix tree with:
    - Judicious admission for SSM states
    - FLOP-aware eviction scoring
    - Two-tier hot/cold caching
    - Automatic alpha tuning for eviction weight

    The tree is standalone and testable without the full vllm-mlx stack.

    Thread Safety:
        This class is NOT thread-safe. Use external locking if needed.
    """

    def __init__(
        self,
        config: Optional[HybridModelConfig] = None,
        max_memory_bytes: int = 0,
    ) -> None:
        """Initialize the hybrid radix cache.

        Args:
            config: Model configuration for FLOP calculations.
            max_memory_bytes: Maximum memory budget. 0 means unlimited.
        """
        self._config = config or HybridModelConfig()
        self._max_memory = max_memory_bytes

        # Root of the radix tree (empty edge label, no states)
        self._root = RadixNode()

        # Total memory tracked across all nodes
        self._current_memory = 0

        # Cold tier: evicted entries retained in memory for fast promotion
        # Maps token_key -> (kv_states, ssm_states, memory_bytes)
        self._cold_tier: Dict[Tuple[int, ...], Tuple[
            Optional[List[Any]], Optional[List[Any]], int
        ]] = {}
        self._cold_memory = 0

        # Alpha tuning for FLOP-aware eviction
        self._alpha: float = 0.0  # Start with pure LRU
        self._request_count: int = 0
        self._bootstrap_complete: bool = False
        self._bootstrap_threshold: int = 10

        # Access history for alpha tuning
        self._access_history: List[Tuple[float, float]] = []  # (timestamp, flop_eff)

        # Stats
        self._hits = 0
        self._misses = 0
        self._ssm_admissions = 0
        self._ssm_rejections = 0
        self._evictions = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def insert(
        self,
        tokens: Tuple[int, ...],
        kv_states: List[Any],
        ssm_states: Optional[List[Any]] = None,
    ) -> RadixNode:
        """Insert a token sequence with its cache states into the radix tree.

        KV states are always stored. SSM states go through the judicious
        admission policy before being stored.

        Args:
            tokens: Token sequence to insert.
            kv_states: Per attention layer cache objects.
            ssm_states: Per recurrent layer state snapshots (may be rejected).

        Returns:
            The radix node where the states were stored.
        """
        if not tokens:
            return self._root

        self._request_count += 1
        node = self._traverse_or_create(tokens)

        # Update access metadata BEFORE admission check (admission
        # uses access_count to decide 2nd-occurrence policy)
        node.last_access = time.monotonic()
        node.access_count += 1

        # Store KV states (always admitted)
        old_memory = node.memory_bytes
        node.kv_states = kv_states
        node.memory_bytes = estimate_hybrid_memory(kv_states)

        # SSM states: check admission policy
        if ssm_states is not None:
            if self.admit_ssm_state(node, ssm_states):
                node.ssm_states = ssm_states
                node.ssm_checkpoint_pos = node.depth
                node.memory_bytes += estimate_hybrid_memory(ssm_states)
                self._ssm_admissions += 1
            else:
                self._ssm_rejections += 1

        # Update memory tracking
        self._current_memory += node.memory_bytes - old_memory

        # Record for alpha tuning
        flop_eff = compute_flop_efficiency(node, self._config)
        self._access_history.append((node.last_access, flop_eff))

        # Check if alpha tuning is due
        if (
            not self._bootstrap_complete
            and self._request_count >= self._bootstrap_threshold
        ):
            self.compute_optimal_alpha()

        # Evict if over memory budget
        total_mem = self._current_memory + self._cold_memory
        if self._max_memory > 0 and total_mem > self._max_memory:
            self.evict(total_mem - self._max_memory)

        return node

    def lookup(self, tokens: Tuple[int, ...]) -> HybridCacheLookup:
        """Look up a token sequence in the radix tree.

        Returns the longest matching prefix with its cached states.
        If the match ends mid-edge (partial edge match), walks up to find
        the nearest ancestor node that has cached states.

        Args:
            tokens: Token sequence to look up.

        Returns:
            HybridCacheLookup with match details.
        """
        if not tokens:
            self._misses += 1
            return HybridCacheLookup()

        node, matched_len, partial_child = self._traverse(tokens)

        # Partial edge match: query is a prefix of a stored sequence.
        # The child node has states for MORE tokens than we need.
        # Return KV states (trimmable) but discard SSM states (not trimmable).
        if partial_child is not None and partial_child.kv_states is not None:
            partial_child.last_access = time.monotonic()
            partial_child.access_count += 1
            self._hits += 1
            logger.debug(
                f"Cache hit (supersequence): "
                f"{matched_len} of {partial_child.depth} tokens, "
                f"KV reusable, SSM needs recompute"
            )
            return HybridCacheLookup(
                matched_length=matched_len,
                kv_states=partial_child.kv_states,
                ssm_states=None,  # SSM state is for full seq, can't trim
                needs_ssm_recompute=True,
                match_type="prefix",
            )

        # If traverse matched tokens but the node has no states (e.g., an
        # intermediate node created by edge splitting), walk up to find the
        # nearest ancestor with cached states.
        effective_node = node
        effective_len = matched_len
        while (
            effective_node is not self._root
            and effective_node.kv_states is None
            and effective_node.parent is not None
        ):
            effective_len -= len(effective_node.token_ids)
            effective_node = effective_node.parent

        if effective_len <= 0 or effective_node is self._root:
            # Walk-up failed to find states. If the original matched node
            # is a stateless intermediate (from edge splitting) with children
            # that DO have states, borrow their KV states. The child's KV
            # cache covers more tokens than matched_len but is trimmable.
            if matched_len > 0 and node is not self._root and node.children:
                for child in node.children.values():
                    if child.kv_states is not None:
                        child.last_access = time.monotonic()
                        child.access_count += 1
                        self._hits += 1
                        logger.debug(
                            f"Cache hit (child borrow): "
                            f"{matched_len} of {child.depth} tokens, "
                            f"KV from child, SSM needs recompute"
                        )
                        return HybridCacheLookup(
                            matched_length=matched_len,
                            kv_states=child.kv_states,
                            ssm_states=None,
                            needs_ssm_recompute=True,
                            match_type="prefix",
                        )

            # No hot tier match — check cold tier for exact match
            cold = self._cold_tier.get(tokens)
            if cold is not None:
                kv, ssm, mem = cold
                self.promote(tokens)
                self._hits += 1
                logger.debug(
                    f"Cache hit (cold, exact): "
                    f"{len(tokens)} tokens, ssm={'yes' if ssm else 'no'}"
                )
                return HybridCacheLookup(
                    matched_length=len(tokens),
                    kv_states=kv,
                    ssm_states=ssm,
                    needs_ssm_recompute=ssm is None,
                    match_type="exact",
                )

            # Also check cold tier for prefix matches
            best_cold_key: Optional[Tuple[int, ...]] = None
            best_cold_len = 0
            for cold_key in self._cold_tier:
                ck_len = len(cold_key)
                if ck_len <= best_cold_len or ck_len > len(tokens):
                    continue
                if tokens[:ck_len] == cold_key:
                    best_cold_key = cold_key
                    best_cold_len = ck_len

            if best_cold_key is not None:
                kv, ssm, mem = self._cold_tier[best_cold_key]
                self.promote(best_cold_key)
                self._hits += 1
                logger.debug(
                    f"Cache hit (cold, prefix): "
                    f"{best_cold_len} tokens matched, ssm={'yes' if ssm else 'no'}"
                )
                return HybridCacheLookup(
                    matched_length=best_cold_len,
                    kv_states=kv,
                    ssm_states=ssm,
                    needs_ssm_recompute=ssm is None,
                    match_type="prefix",
                )

            self._misses += 1
            logger.debug(f"Cache miss: {len(tokens)} tokens")
            return HybridCacheLookup()

        # Update access metadata on the matched node
        effective_node.last_access = time.monotonic()
        effective_node.access_count += 1
        self._hits += 1

        is_exact = effective_len == len(tokens)
        match_type = "exact" if is_exact else "prefix"
        has_ssm = effective_node.ssm_states is not None
        logger.debug(
            f"Cache hit ({match_type}): "
            f"{effective_len} tokens matched, ssm={'yes' if has_ssm else 'recompute'}"
        )

        return HybridCacheLookup(
            matched_length=effective_len,
            kv_states=effective_node.kv_states,
            ssm_states=effective_node.ssm_states,
            needs_ssm_recompute=(
                effective_node.kv_states is not None
                and effective_node.ssm_states is None
            ),
            match_type=match_type,
        )

    def admit_ssm_state(
        self, node: RadixNode, ssm_states: List[Any]
    ) -> bool:
        """Judicious admission policy for SSM states (from MARCONI).

        Rules:
        - Branch point nodes (multiple children): always admit.
          Rationale: branch points serve multiple downstream sequences,
          so SSM state reuse is highly likely.
        - Repeated access (access_count >= 2): admit.
          Rationale: if a prefix is accessed more than once, it's likely
          part of a shared pattern worth caching.
        - First-time access of a non-branch node: reject.
          Rationale: SSM states have 65x lower reuse rate than KV states.

        Args:
            node: The radix node being considered for SSM admission.
            ssm_states: The SSM states to potentially store.

        Returns:
            True if SSM states should be admitted.
        """
        # Branch points always get SSM states
        if len(node.children) > 1:
            logger.debug(
                f"[ssm_admit] branch_point: depth={node.depth} "
                f"children={len(node.children)}"
            )
            return True

        # Repeated access: admit on 2nd occurrence
        if node.access_count >= 2:
            logger.debug(
                f"[ssm_admit] repeated_access: depth={node.depth} "
                f"count={node.access_count}"
            )
            return True

        # First-time, non-branch: reject
        logger.debug(
            f"[ssm_reject] first_access_non_branch: depth={node.depth} "
            f"count={node.access_count}"
        )
        return False

    def evict(self, target_bytes: int) -> int:
        """Evict nodes to free at least target_bytes of memory.

        Uses FLOP-aware scoring: S(n) = recency(n) + alpha * flop_efficiency(n)
        Lowest-scoring leaf nodes are evicted first.
        SSM states are freed before KV states (since they have lower reuse).

        If no hot-tier leaves exist (all nodes already demoted), falls back
        to evicting from the cold tier by removing the oldest entries entirely.

        Args:
            target_bytes: Number of bytes to free.

        Returns:
            Number of bytes actually freed.
        """
        if target_bytes <= 0:
            return 0

        freed = 0

        # --- Phase 1: Evict from hot tier (leaves with states) ---
        leaves = self._collect_leaves()

        if leaves:
            now = time.monotonic()
            scored = self._score_nodes(leaves, now)
            scored.sort(key=lambda x: x[0])

            for _score, node in scored:
                if freed >= target_bytes:
                    break

                # First try freeing SSM states only
                if node.ssm_states is not None:
                    ssm_mem = estimate_hybrid_memory(node.ssm_states)
                    node.ssm_states = None
                    node.memory_bytes -= ssm_mem
                    self._current_memory -= ssm_mem
                    freed += ssm_mem
                    self._evictions += 1

                    if freed >= target_bytes:
                        break

                # If still need more, demote the entire node to cold tier
                if node.kv_states is not None:
                    # Capture memory before demote() clears it
                    node_mem = node.memory_bytes
                    token_key = self._get_token_key(node)
                    self.demote(node, token_key)
                    freed += node_mem
                    self._evictions += 1

        # --- Phase 2: If still short, evict from cold tier entirely ---
        remaining = target_bytes - freed
        if remaining > 0 and self._cold_tier:
            freed += self._evict_cold(remaining)

        logger.debug(
            f"[evict] freed {freed} bytes (target={target_bytes}), "
            f"evictions={self._evictions}"
        )
        return freed

    # -----------------------------------------------------------------
    # Two-Tier Cache: Hot / Cold
    # -----------------------------------------------------------------

    def promote(self, token_key: Tuple[int, ...]) -> Optional[RadixNode]:
        """Promote an entry from cold tier back to hot (radix tree).

        On Apple Silicon unified memory, this is essentially free — the
        arrays are already in memory, we just re-link them in the tree.

        Args:
            token_key: Token sequence to promote.

        Returns:
            The radix node if promoted, None if not found in cold tier.
        """
        cold_entry = self._cold_tier.pop(token_key, None)
        if cold_entry is None:
            return None

        kv_states, ssm_states, mem = cold_entry
        self._cold_memory -= mem

        node = self._traverse_or_create(token_key)
        node.kv_states = kv_states
        node.ssm_states = ssm_states
        node.memory_bytes = mem
        node.last_access = time.monotonic()
        node.access_count += 1
        self._current_memory += mem

        logger.debug(
            f"[promote] cold->hot: {len(token_key)} tokens, {mem} bytes"
        )
        return node

    def _evict_cold(self, target_bytes: int) -> int:
        """Evict entries from cold tier to free target_bytes."""
        freed = 0
        cold_keys = list(self._cold_tier.keys())
        for key in cold_keys:
            if freed >= target_bytes:
                break
            _, _, mem = self._cold_tier.pop(key)
            self._cold_memory -= mem
            freed += mem
            self._evictions += 1
        return freed

    def demote(
        self, node: RadixNode, token_key: Tuple[int, ...]
    ) -> None:
        """Demote a node from hot tier to cold tier.

        The node's states are moved to the cold dict and removed from
        the radix tree scoring, but remain in memory.

        Args:
            node: The radix node to demote.
            token_key: Token sequence for cold tier key.
        """
        mem = node.memory_bytes
        self._cold_tier[token_key] = (node.kv_states, node.ssm_states, mem)
        self._cold_memory += mem

        # Trim cold tier if total memory exceeds budget
        if self._max_memory > 0:
            total = self._current_memory + self._cold_memory
            if total > self._max_memory:
                self._evict_cold(total - self._max_memory)

        # Clear states from tree node
        self._current_memory -= node.memory_bytes
        node.kv_states = None
        node.ssm_states = None
        node.memory_bytes = 0

        # Remove leaf node from tree if it has no children
        if node.is_leaf and node.parent is not None:
            self._prune_node(node)

        logger.debug(
            f"[demote] hot->cold: {len(token_key)} tokens, {mem} bytes"
        )

    # -----------------------------------------------------------------
    # Alpha Tuning
    # -----------------------------------------------------------------

    def compute_optimal_alpha(self) -> float:
        """Grid search for optimal alpha using historical access patterns.

        Tests alpha values [0, 0.1, 0.5, 1.0, 2.0, 5.0] and picks the
        one that maximizes the correlation between eviction score and
        actual future accesses.

        Returns:
            The selected alpha value.
        """
        if len(self._access_history) < self._bootstrap_threshold:
            return self._alpha

        candidates = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        best_alpha = 0.0
        best_score = float("-inf")

        leaves = self._collect_leaves()
        if not leaves:
            self._bootstrap_complete = True
            return self._alpha

        now = time.monotonic()

        for alpha in candidates:
            old_alpha = self._alpha
            self._alpha = alpha
            scored = self._score_nodes(leaves, now)
            self._alpha = old_alpha

            # Heuristic: higher average score for frequently-accessed nodes
            # means the alpha is preserving useful entries
            total_score = 0.0
            for score, node in scored:
                total_score += score * node.access_count

            if total_score > best_score:
                best_score = total_score
                best_alpha = alpha

        self._alpha = best_alpha
        self._bootstrap_complete = True
        logger.info(f"[alpha_tune] selected alpha={best_alpha:.1f}")
        return best_alpha

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns a dict with:
        - hit/miss counts and hit_rate
        - num_nodes: total nodes in the radix tree (excluding root)
        - hot_entries: nodes with cached states in the hot tier
        - cold_entries: entries in the cold tier
        - memory stats (hot, cold, total)
        - SSM admission_rate and counts
        - eviction count, alpha, bootstrap status
        """
        total_queries = self._hits + self._misses
        total_ssm = self._ssm_admissions + self._ssm_rejections
        num_nodes, hot_entries = self._count_nodes()

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total_queries if total_queries > 0 else 0.0,
            "num_nodes": num_nodes,
            "hot_entries": hot_entries,
            "cold_entries": len(self._cold_tier),
            "hot_memory_bytes": self._current_memory,
            "cold_memory_bytes": self._cold_memory,
            "total_memory_bytes": self._current_memory + self._cold_memory,
            "ssm_admissions": self._ssm_admissions,
            "ssm_rejections": self._ssm_rejections,
            "ssm_admission_rate": (
                self._ssm_admissions / total_ssm if total_ssm > 0 else 0.0
            ),
            "max_memory_bytes": self._max_memory,
            "evictions": self._evictions,
            "alpha": self._alpha,
            "bootstrap_complete": self._bootstrap_complete,
            "request_count": self._request_count,
        }

    def clear(self) -> None:
        """Clear all cached data (hot and cold tiers)."""
        self._root = RadixNode()
        self._cold_tier.clear()
        self._current_memory = 0
        self._cold_memory = 0
        self._access_history.clear()
        self._hits = 0
        self._misses = 0
        self._ssm_admissions = 0
        self._ssm_rejections = 0
        self._evictions = 0
        self._request_count = 0
        self._bootstrap_complete = False
        self._alpha = 0.0

    # -----------------------------------------------------------------
    # Internal: Radix Tree Operations
    # -----------------------------------------------------------------

    def _traverse(
        self, tokens: Tuple[int, ...]
    ) -> Tuple[RadixNode, int, Optional[RadixNode]]:
        """Traverse the radix tree matching as many tokens as possible.

        Args:
            tokens: Token sequence to match.

        Returns:
            Tuple of (deepest fully-matched node, number of tokens matched,
            partial_child) where partial_child is set when the query is a
            prefix of a child's edge (i.e., the stored sequence is longer
            than the query).  In that case matched count includes the
            partial edge tokens.
        """
        node = self._root
        pos = 0
        partial_child: Optional[RadixNode] = None

        while pos < len(tokens):
            token = tokens[pos]
            child = node.children.get(token)
            if child is None:
                break

            # Match edge label
            edge = child.token_ids
            edge_len = len(edge)
            remaining = len(tokens) - pos

            # Find how many tokens match this edge (handles both cases:
            # query shorter than edge and query diverging within edge)
            compare_len = min(remaining, edge_len)
            common = 0
            for i in range(compare_len):
                if tokens[pos + i] != edge[i]:
                    break
                common = i + 1

            if common < edge_len:
                # Partial edge match — set partial_child for KV borrowing
                if common > 0:
                    diverge_pos = pos + common
                    logger.debug(
                        f"[traverse_diverge] at token {diverge_pos}: "
                        f"query_token={tokens[pos + common] if pos + common < len(tokens) else 'END'} "
                        f"edge_token={edge[common]} "
                        f"edge_len={edge_len} total_query={len(tokens)}"
                    )
                    pos += common
                    partial_child = child
                break

            node = child
            pos += edge_len

        return node, pos, partial_child

    def _traverse_or_create(
        self, tokens: Tuple[int, ...]
    ) -> RadixNode:
        """Traverse the tree, creating nodes as needed. Splits edges on mismatch.

        Args:
            tokens: Token sequence to insert.

        Returns:
            The node at the end of the inserted path.
        """
        node = self._root
        pos = 0

        while pos < len(tokens):
            token = tokens[pos]
            child = node.children.get(token)

            if child is None:
                # Create new leaf node for remaining tokens
                new_node = RadixNode(
                    token_ids=tokens[pos:],
                    parent=node,
                    depth=node.depth + len(tokens) - pos,
                    is_leaf=True,
                )
                node.children[token] = new_node
                node.is_leaf = False
                return new_node

            edge = child.token_ids
            edge_len = len(edge)
            remaining_tokens = tokens[pos:]

            # Find common prefix length between edge and remaining tokens
            common_len = 0
            for i in range(min(edge_len, len(remaining_tokens))):
                if edge[i] != remaining_tokens[i]:
                    break
                common_len = i + 1

            if common_len == edge_len:
                # Full edge matched, continue traversal
                node = child
                pos += edge_len
                continue

            # Partial match — split the edge
            # Create intermediate node at the split point
            split_node = RadixNode(
                token_ids=edge[:common_len],
                parent=node,
                depth=node.depth + common_len,
                is_leaf=False,
            )

            # Existing child becomes child of split node with remaining edge
            child.token_ids = edge[common_len:]
            child.parent = split_node
            split_node.children[edge[common_len]] = child

            # Replace in parent
            node.children[token] = split_node

            if common_len == len(remaining_tokens):
                # The new tokens end exactly at the split point
                return split_node

            # Create new leaf for the remaining new tokens
            new_suffix = remaining_tokens[common_len:]
            new_node = RadixNode(
                token_ids=new_suffix,
                parent=split_node,
                depth=split_node.depth + len(new_suffix),
                is_leaf=True,
            )
            split_node.children[new_suffix[0]] = new_node
            return new_node

        # Tokens fully consumed at current node
        return node

    def _count_nodes(self) -> Tuple[int, int]:
        """Count total nodes and nodes with cached states in the tree.

        Returns:
            Tuple of (num_nodes, hot_entries) — excludes the root node.
        """
        num_nodes = 0
        hot_entries = 0
        stack = list(self._root.children.values())

        while stack:
            node = stack.pop()
            num_nodes += 1
            if node.kv_states is not None or node.ssm_states is not None:
                hot_entries += 1
            for child in node.children.values():
                stack.append(child)

        return num_nodes, hot_entries

    def _collect_leaves(self) -> List[RadixNode]:
        """Collect all leaf nodes with stored states."""
        leaves: List[RadixNode] = []
        stack = [self._root]

        while stack:
            node = stack.pop()
            if node.is_leaf and (
                node.kv_states is not None or node.ssm_states is not None
            ):
                leaves.append(node)
            for child in node.children.values():
                stack.append(child)

        return leaves

    def _score_nodes(
        self, nodes: List[RadixNode], now: float
    ) -> List[Tuple[float, RadixNode]]:
        """Score nodes for eviction using FLOP-aware scoring.

        S(n) = recency(n) + alpha * flop_efficiency(n)

        Where recency is normalized to [0, 1] (0 = oldest, 1 = newest).
        """
        if not nodes:
            return []

        # Find time range for normalization
        min_time = min(n.last_access for n in nodes)
        max_time = max(n.last_access for n in nodes)
        time_range = max_time - min_time

        scored: List[Tuple[float, RadixNode]] = []
        for node in nodes:
            # Normalized recency: 0 = oldest, 1 = newest
            if time_range > 0:
                recency = (node.last_access - min_time) / time_range
            else:
                recency = 1.0

            flop_eff = compute_flop_efficiency(node, self._config)
            score = recency + self._alpha * flop_eff
            scored.append((score, node))

        return scored

    def _get_token_key(self, node: RadixNode) -> Tuple[int, ...]:
        """Reconstruct the full token key for a node by walking to root."""
        parts: List[Tuple[int, ...]] = []
        current = node
        while current is not self._root and current is not None:
            parts.append(current.token_ids)
            current = current.parent

        parts.reverse()
        result: List[int] = []
        for part in parts:
            result.extend(part)
        return tuple(result)

    def _prune_node(self, node: RadixNode) -> None:
        """Remove a leaf node from the tree and clean up empty parents."""
        current = node
        while (
            current is not self._root
            and current is not None
            and current.parent is not None
            and not current.children
            and current.kv_states is None
            and current.ssm_states is None
        ):
            parent = current.parent
            # Find and remove from parent's children
            token = current.token_ids[0] if current.token_ids else None
            if token is not None and token in parent.children:
                del parent.children[token]

            # Update parent leaf status
            if not parent.children and parent is not self._root:
                parent.is_leaf = True

            current = parent
