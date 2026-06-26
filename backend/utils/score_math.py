"""
backend/utils/score_math.py
Mathematical primitives for scoring: normalization, decay functions,
and safe logarithm operations used across the pipeline.
Max 300 lines. All operations are vectorized NumPy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def min_max_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normalize a 1-D array to [0, 1] using min-max scaling.

    Args:
        arr: Input NumPy array of floats.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Normalized array in [0, 1].
    """
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def clip_normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Clip values to [lo, hi] then normalize to [0, 1].

    Args:
        arr: Input array.
        lo: Lower bound for clipping.
        hi: Upper bound for clipping.

    Returns:
        Clipped and normalized array.
    """
    clipped = np.clip(arr, lo, hi).astype(np.float32)
    span = hi - lo
    return (clipped - lo) / span if span > 0 else np.zeros_like(clipped)


# ---------------------------------------------------------------------------
# Decay functions
# ---------------------------------------------------------------------------

def exponential_decay(days_inactive: np.ndarray, half_life: float = 90.0) -> np.ndarray:
    """
    Apply exponential decay based on inactivity duration.
    Returns 1.0 for active profiles and decays toward 0 for stale ones.

    Args:
        days_inactive: Array of days since last activity (>= 0).
        half_life: Days until score halves; defaults to 90.

    Returns:
        Decay multiplier array in (0, 1].
    """
    return np.exp(-np.maximum(days_inactive, 0.0) * np.log(2) / half_life).astype(
        np.float32
    )


def step_decay_after(
    days_inactive: np.ndarray, threshold: float = 90.0, decay: float = 0.5
) -> np.ndarray:
    """
    Apply a step decay: profiles inactive beyond threshold get multiplied by decay.

    Args:
        days_inactive: Days since last activity.
        threshold: Days threshold; beyond this, decay is applied.
        decay: Multiplier for profiles past the threshold.

    Returns:
        Array of multipliers (1.0 or decay).
    """
    result = np.ones(len(days_inactive), dtype=np.float32)
    mask = days_inactive > threshold
    result[mask] = decay
    return result


# ---------------------------------------------------------------------------
# Safe mathematical operations
# ---------------------------------------------------------------------------

def safe_log(arr: np.ndarray, offset: float = 2.0) -> np.ndarray:
    """
    Compute log(arr + offset) safely — handles -1 sentinel values by
    treating them as 0 before adding the offset.

    Args:
        arr: Input array (may contain -1 sentinels for missing values).
        offset: Value added before taking log to avoid log(0).

    Returns:
        Array of log-transformed floats.
    """
    safe_arr = np.where(arr < 0, 0.0, arr).astype(np.float64)
    return np.log(safe_arr + offset).astype(np.float32)


def safe_mean_rate(arr: np.ndarray, sentinel: float = -1.0) -> np.ndarray:
    """
    Treat sentinel values as 0.5 (neutral) for rate-type fields like
    offer_acceptance_rate which uses -1 for 'no history'.

    Args:
        arr: Array possibly containing sentinel values.
        sentinel: Sentinel value indicating 'no data'.

    Returns:
        Array with sentinels replaced by 0.5.
    """
    return np.where(arr == sentinel, 0.5, arr).astype(np.float32)


# ---------------------------------------------------------------------------
# Cosine similarity (batch)
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    query_vec: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix
    of candidate vectors using fast NumPy dot product.

    Args:
        query_vec: 1-D array of shape (D,).
        matrix: 2-D array of shape (N, D); assumed to be L2-normalised.

    Returns:
        1-D similarity array of shape (N,) in [-1, 1].
    """
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normed_matrix = matrix / row_norms
    return (normed_matrix @ q_norm).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the top-k highest scores without sorting all N.
    Uses np.argpartition for O(N) instead of O(N log N).

    Args:
        scores: 1-D array of float scores.
        k: Number of top entries to return.

    Returns:
        Indices of the top-k entries, unordered internally.
    """
    if k >= len(scores):
        return np.arange(len(scores))
    part = np.argpartition(scores, -k)[-k:]
    return part
