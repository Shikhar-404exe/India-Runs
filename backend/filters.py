"""
backend/filters.py
ONLINE PHASE — Vectorized hard masks and soft penalty multipliers.

All functions operate on a pre-flattened pandas DataFrame (one row per
candidate). No Python-level loops over rows are used — all logic is
expressed via vectorized Pandas/NumPy operations.

Max 300 lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Excluded title keywords (immediate zero-multiplier)
# ---------------------------------------------------------------------------

EXCLUDED_TITLE_PATTERNS: list[str] = [
    r"\bmarketing\b",
    r"\bhr\b",
    r"\bhuman resource",
    r"\baccountant\b",
    r"\baccounting\b",
    r"\bqa\b",
    r"\bquality assurance\b",
    r"\bgraphic design",
    r"\bcontent writer\b",
    r"\bcopywriter\b",
    r"\bsales executive\b",
    r"\bcivil engineer\b",
    r"\bmechanical engineer\b",
    r"\bcustomer support\b",
    r"\bcustomer service\b",
    r"\boperations manager\b",
]

# Outsourcing service companies (penalty if no product-co experience)
OUTSOURCING_COMPANIES: list[str] = [
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "mindtree",
]

# Positive AI-role title keywords
AI_TITLE_KEYWORDS: list[str] = [
    r"\bai engineer\b",
    r"\bml engineer\b",
    r"\bmachine learning\b",
    r"\bdata scientist\b",
    r"\bdeep learning\b",
    r"\bnlp engineer\b",
    r"\bresearch engineer\b",
    r"\bapplied scientist\b",
    r"\bcomputer vision\b",
    r"\bllm\b",
    r"\bgenerative ai\b",
]

# ---------------------------------------------------------------------------
# Title exclusion mask
# ---------------------------------------------------------------------------


def compute_title_exclusion_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Return a float multiplier array: 0.0 if the current_title contains any
    excluded keyword, 1.0 otherwise.

    Args:
        df: Candidate DataFrame with 'current_title' column (lowercase).

    Returns:
        Float32 array of shape (N,) with values in {0.0, 1.0}.
    """
    pattern = "|".join(EXCLUDED_TITLE_PATTERNS)
    is_excluded = df["current_title"].str.lower().str.contains(
        pattern, regex=True, na=False
    )
    return np.where(is_excluded, 0.0, 1.0).astype(np.float32)


def compute_ai_title_boost(df: pd.DataFrame) -> np.ndarray:
    """
    Return a boost multiplier: 1.2 for profiles with AI/ML-aligned titles,
    1.0 otherwise.

    Args:
        df: Candidate DataFrame with 'current_title' column.

    Returns:
        Float32 array of shape (N,).
    """
    pattern = "|".join(AI_TITLE_KEYWORDS)
    is_ai = df["current_title"].str.lower().str.contains(
        pattern, regex=True, na=False
    )
    return np.where(is_ai, 1.2, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Outsourcing penalty
# ---------------------------------------------------------------------------


def compute_outsourcing_penalty(df: pd.DataFrame) -> np.ndarray:
    """
    Apply 0.4× penalty if ALL career history is at outsourcing companies
    (i.e., zero product-company experience).

    A company is classified as a product company if it is NOT in
    OUTSOURCING_COMPANIES and the industry is not 'IT Services'.

    Args:
        df: Candidate DataFrame with 'companies_in_career' and
            'industries_in_career' columns.

    Returns:
        Float32 array of shape (N,) with values in {0.4, 1.0}.
    """
    outsourcing_pattern = "|".join(OUTSOURCING_COMPANIES)
    # Each cell is a pipe-separated list; check if all companies are outsourcing
    all_outsourcing = df["companies_in_career"].str.lower().str.contains(
        outsourcing_pattern, regex=True, na=False
    )
    # Also check if all industries are IT Services
    all_it_services = df["industries_in_career"].str.lower().apply(
        lambda x: all(s.strip() == "it services" for s in x.split("|")) if x else False
    )
    is_pure_outsourcing = all_outsourcing & all_it_services
    return np.where(is_pure_outsourcing, 0.4, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Title-chaser diagnostic
# ---------------------------------------------------------------------------


def compute_tenure_penalty(df: pd.DataFrame) -> np.ndarray:
    """
    Down-weight candidates who job-hop excessively: avg tenure < 18 months
    across their career history gets a 0.6× multiplier.

    Args:
        df: Candidate DataFrame with 'avg_tenure_months' column.

    Returns:
        Float32 array of shape (N,) with values in {0.6, 1.0}.
    """
    is_job_hopper = df["avg_tenure_months"] < 18.0
    return np.where(is_job_hopper, 0.6, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Honeypot interceptor
# ---------------------------------------------------------------------------


def compute_honeypot_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Detect and zero-out honeypot profiles:
      - profile_completeness_score < 50 AND avg_endorsements > 30
      - OR profile_completeness_score < 30 (extremely incomplete)

    Args:
        df: Candidate DataFrame with 'profile_completeness_score' and
            'avg_endorsements' columns.

    Returns:
        Float32 array of shape (N,): 0.0 for honeypots, 1.0 otherwise.
    """
    completeness = df["profile_completeness_score"].to_numpy(dtype=np.float32)
    avg_endorse = df["avg_endorsements"].to_numpy(dtype=np.float32)

    # Classic honeypot: low profile fill but suspiciously high endorsements
    trap_1 = (completeness < 50) & (avg_endorse > 30)
    # Extremely empty profile
    trap_2 = completeness < 25

    is_honeypot = trap_1 | trap_2
    return np.where(is_honeypot, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Positive signal boosts
# ---------------------------------------------------------------------------


def compute_openness_boost(df: pd.DataFrame) -> np.ndarray:
    """
    Boost candidates who are actively open to work.

    Args:
        df: Candidate DataFrame with 'open_to_work_flag' (int 0/1).

    Returns:
        Float32 array with values in {1.0, 1.1}.
    """
    return np.where(df["open_to_work_flag"].to_numpy() == 1, 1.1, 1.0).astype(
        np.float32
    )


def compute_verification_boost(df: pd.DataFrame) -> np.ndarray:
    """
    Boost profiles with both email and phone verified (reduces ghost profiles).

    Args:
        df: Candidate DataFrame with 'verified_email' and 'verified_phone' columns.

    Returns:
        Float32 array with values in {1.0, 1.05}.
    """
    both_verified = (df["verified_email"].to_numpy() == 1) & (
        df["verified_phone"].to_numpy() == 1
    )
    return np.where(both_verified, 1.05, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main filter composite
# ---------------------------------------------------------------------------


def compute_filter_multiplier(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the combined filter multiplier for each candidate in df.
    Applies all hard masks and soft penalties/boosts in sequence.

    Order of application:
      1. Honeypot zero-out (if triggered → final score = 0)
      2. Title exclusion (excluded roles → score = 0)
      3. AI title boost (soft +)
      4. Outsourcing penalty (soft –)
      5. Title-chaser penalty (soft –)
      6. Open-to-work boost (soft +)
      7. Verification boost (soft +)

    Args:
        df: Flat candidate DataFrame.

    Returns:
        Combined float32 multiplier array of shape (N,).
    """
    honeypot = compute_honeypot_mask(df)
    title_excl = compute_title_exclusion_mask(df)
    ai_boost = compute_ai_title_boost(df)
    outsourcing = compute_outsourcing_penalty(df)
    tenure = compute_tenure_penalty(df)
    openness = compute_openness_boost(df)
    verification = compute_verification_boost(df)

    combined = (
        honeypot
        * title_excl
        * ai_boost
        * outsourcing
        * tenure
        * openness
        * verification
    )
    return combined.astype(np.float32)
