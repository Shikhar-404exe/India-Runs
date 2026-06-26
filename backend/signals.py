"""
backend/signals.py
ONLINE PHASE — Redrob Platform Velocity Scalar computation.

Translates all 23 Redrob behavioral signals into a single normalised
Availability Value (V_s) per candidate using vectorized NumPy operations.

Formula:
    V_s_raw = recruiter_response_rate
              × (1 - notice_period_days / 180)
              × log(github_activity_score + 2)

Additional modifiers applied multiplicatively:
    • Activity decay   — exp(-days_inactive / 90) for inactive > 90 days
    • Work mode boost  — +0.05 for hybrid/flexible (Pune/Noida axis)
    • Interview trust  — × interview_completion_rate
    • Offer history    — bonus for offer_acceptance_rate > 0.5
    • Platform signal  — log-scaled engagement (views, saves, searches)

Max 300 lines. All operations vectorized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.utils.score_math import (
    exponential_decay,
    min_max_normalize,
    safe_log,
    safe_mean_rate,
)

# ---------------------------------------------------------------------------
# Location keywords for Pune/Noida axis matching
# ---------------------------------------------------------------------------

INDIA_AXIS_LOCATIONS: list[str] = [
    "pune",
    "noida",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "gurgaon",
    "gurugram",
    "mumbai",
    "delhi",
    "chennai",
]


# ---------------------------------------------------------------------------
# Core Redrob velocity scalar
# ---------------------------------------------------------------------------


def compute_core_velocity(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the core Redrob Availability Value per the formula:

        V_raw = recruiter_response_rate
                × (1 - notice_period_days / 180)
                × log(github_activity_score + 2)

    Args:
        df: Flat candidate DataFrame with required signal columns.

    Returns:
        Float32 array of shape (N,) with raw V_s values.
    """
    rrr = df["recruiter_response_rate"].to_numpy(dtype=np.float32)
    notice = df["notice_period_days"].to_numpy(dtype=np.float32)
    github = df["github_activity_score"].to_numpy(dtype=np.float32)

    notice_factor = 1.0 - np.clip(notice, 0, 180) / 180.0
    github_log = safe_log(github, offset=2.0)

    return (rrr * notice_factor * github_log).astype(np.float32)


# ---------------------------------------------------------------------------
# Activity decay
# ---------------------------------------------------------------------------


def compute_activity_decay(df: pd.DataFrame, reference_date: str = "2026-06-19") -> np.ndarray:
    """
    Compute exponential decay based on days since last_active_date.
    Profiles inactive for > 90 days are exponentially penalised.

    Args:
        df: Candidate DataFrame with 'last_active_date' column (YYYY-MM-DD).
        reference_date: ISO date string for 'today' (default: challenge date).

    Returns:
        Float32 decay multiplier array in (0, 1].
    """
    ref = pd.Timestamp(reference_date)
    last_active = pd.to_datetime(df["last_active_date"], errors="coerce")
    days_inactive = (ref - last_active).dt.days.fillna(999).to_numpy(dtype=np.float32)
    days_inactive = np.maximum(days_inactive, 0.0)
    return exponential_decay(days_inactive, half_life=90.0)


# ---------------------------------------------------------------------------
# Work mode boost (Pune/Noida axis)
# ---------------------------------------------------------------------------


def compute_work_mode_boost(df: pd.DataFrame) -> np.ndarray:
    """
    Add a 0.05 boost for candidates preferring hybrid/flexible work modes
    and located in Indian metro/tech-hub cities.

    Args:
        df: Candidate DataFrame with 'preferred_work_mode' and 'location'.

    Returns:
        Float32 additive boost array of shape (N,).
    """
    mode = df["preferred_work_mode"].str.lower()
    location = df["location"].str.lower().fillna("")

    is_flexible = mode.isin(["hybrid", "flexible"])

    loc_pattern = "|".join(INDIA_AXIS_LOCATIONS)
    is_india_axis = location.str.contains(loc_pattern, regex=True, na=False)

    boost = np.where(is_flexible & is_india_axis, 0.05, 0.0).astype(np.float32)
    return boost


# ---------------------------------------------------------------------------
# Interview reliability
# ---------------------------------------------------------------------------


def compute_interview_reliability(df: pd.DataFrame) -> np.ndarray:
    """
    Use interview_completion_rate as a reliability multiplier.
    Also considers offer_acceptance_rate (sentinel -1 → neutral 0.5).

    Args:
        df: Candidate DataFrame.

    Returns:
        Float32 multiplier array in [0, 1].
    """
    icr = df["interview_completion_rate"].to_numpy(dtype=np.float32)
    oar = safe_mean_rate(
        df["offer_acceptance_rate"].to_numpy(dtype=np.float32), sentinel=-1.0
    )
    # Weighted combination: 70% interview completion, 30% offer acceptance
    return (0.7 * icr + 0.3 * oar).astype(np.float32)


# ---------------------------------------------------------------------------
# Platform engagement score
# ---------------------------------------------------------------------------


def compute_platform_engagement(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a log-scaled platform engagement signal from:
      - profile_views_received_30d
      - search_appearance_30d
      - saved_by_recruiters_30d
      - applications_submitted_30d

    Args:
        df: Candidate DataFrame.

    Returns:
        Normalised float32 engagement score in [0, 1].
    """
    views = df["profile_views_received_30d"].to_numpy(dtype=np.float32)
    searches = df["search_appearance_30d"].to_numpy(dtype=np.float32)
    saves = df["saved_by_recruiters_30d"].to_numpy(dtype=np.float32)
    apps = df["applications_submitted_30d"].to_numpy(dtype=np.float32)

    raw = np.log1p(views) + np.log1p(searches) + 2.0 * np.log1p(saves) + np.log1p(apps)
    return min_max_normalize(raw)


# ---------------------------------------------------------------------------
# Skill assessment signal
# ---------------------------------------------------------------------------


def compute_assessment_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Normalised average Redrob skill assessment score (0–100 → 0–1).
    Candidates with no assessments get a neutral 0.4.

    Args:
        df: Candidate DataFrame with 'avg_assessment_score' column.

    Returns:
        Float32 array in [0, 1].
    """
    raw = df["avg_assessment_score"].to_numpy(dtype=np.float32)
    score = np.where(raw == 0.0, 40.0, raw) / 100.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Connection & social proof signal
# ---------------------------------------------------------------------------


def compute_social_proof(df: pd.DataFrame) -> np.ndarray:
    """
    Log-scaled connection count + endorsements_received as social proof.

    Args:
        df: Candidate DataFrame.

    Returns:
        Normalised float32 array in [0, 1].
    """
    connections = df["connection_count"].to_numpy(dtype=np.float32)
    endorsements = df["endorsements_received"].to_numpy(dtype=np.float32)
    raw = np.log1p(connections) + np.log1p(endorsements)
    return min_max_normalize(raw)


# ---------------------------------------------------------------------------
# Composite velocity scalar (V_s)
# ---------------------------------------------------------------------------


def compute_velocity_scalar(
    df: pd.DataFrame,
    reference_date: str = "2026-06-19",
    w_core: float = 0.35,
    w_decay: float = 0.20,
    w_interview: float = 0.20,
    w_engagement: float = 0.10,
    w_assessment: float = 0.10,
    w_social: float = 0.05,
) -> np.ndarray:
    """
    Combine all signal components into a single normalised velocity scalar V_s.

    Component weights (default):
      core_velocity    35%
      activity_decay   20%
      interview_trust  20%
      engagement       10%
      assessment       10%
      social_proof      5%
    Work-mode boost is additive on top.

    Args:
        df: Flat candidate DataFrame.
        reference_date: ISO date for inactivity calculation.
        w_*: Component weight parameters.

    Returns:
        Float32 V_s array of shape (N,) normalised to [0, 1].
    """
    core = compute_core_velocity(df)
    core_norm = min_max_normalize(core)

    decay = compute_activity_decay(df, reference_date)
    interview = compute_interview_reliability(df)
    engagement = compute_platform_engagement(df)
    assessment = compute_assessment_signal(df)
    social = compute_social_proof(df)
    work_boost = compute_work_mode_boost(df)

    v_s = (
        w_core * core_norm
        + w_decay * decay
        + w_interview * interview
        + w_engagement * engagement
        + w_assessment * assessment
        + w_social * social
        + work_boost  # additive bonus
    )

    return np.clip(min_max_normalize(v_s), 0.0, 1.0).astype(np.float32)
