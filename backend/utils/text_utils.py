"""
backend/utils/text_utils.py
Text extraction helpers for candidate profile parsing.
Handles docx reading, JSONL streaming, and profile text assembly.
Max 300 lines. No external network calls.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

import docx


# ---------------------------------------------------------------------------
# JSONL streaming helpers
# ---------------------------------------------------------------------------

def stream_jsonl(path: Path, gzipped: bool = False) -> Iterator[dict]:
    """
    Yield parsed JSON objects from a .jsonl or .jsonl.gz file one at a time.

    Args:
        path: Filesystem path to the JSONL file.
        gzipped: If True, open with gzip decompression.

    Yields:
        Parsed candidate dicts.
    """
    opener = gzip.open if gzipped else open
    mode = "rt" if gzipped else "r"
    with opener(path, mode, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_batch(
    path: Path,
    gzipped: bool = False,
    batch_size: int = 1000,
) -> Iterator[list[dict]]:
    """
    Yield batches of candidate records from a JSONL file.

    Args:
        path: Path to JSONL file.
        gzipped: Whether the file is gzip-compressed.
        batch_size: Number of records per batch.

    Yields:
        List of candidate dicts of length up to batch_size.
    """
    batch: list[dict] = []
    for record in stream_jsonl(path, gzipped=gzipped):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Candidate text assembly
# ---------------------------------------------------------------------------

def build_profile_text(candidate: dict) -> str:
    """
    Concatenate a candidate's profile summary and all career description
    blocks into a single whitespace-normalised string for embedding.

    Args:
        candidate: Raw candidate dict from JSONL.

    Returns:
        A single UTF-8 string combining the profile and career narrative.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    for role in candidate.get("career_history", []):
        desc = role.get("description", "")
        title = role.get("title", "")
        company = role.get("company", "")
        if title or company:
            parts.append(f"{title} at {company}".strip())
        if desc:
            parts.append(desc)

    skill_names = [s.get("name", "") for s in candidate.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    return " ".join(parts).strip()


def get_candidate_id(candidate: dict) -> str:
    """Extract the candidate_id string from a candidate record."""
    return candidate.get("candidate_id", "")


# ---------------------------------------------------------------------------
# Job description extraction
# ---------------------------------------------------------------------------

def read_docx_text(path: Path) -> str:
    """
    Extract full text content from a .docx file.

    Args:
        path: Path to the Word document.

    Returns:
        Plain-text string with paragraph content joined by newlines.
    """
    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Flat-field extraction from nested candidate dicts
# ---------------------------------------------------------------------------

def flatten_candidate(candidate: dict) -> dict:
    """
    Produce a flat dict of scalar fields used by filters.py and signals.py.
    All list/object fields are reduced to aggregated scalars.

    Args:
        candidate: Raw candidate dict.

    Returns:
        Flat dict with keys used downstream.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    # Aggregate career statistics
    n_companies = len({r.get("company", "") for r in career})
    total_months = sum(r.get("duration_months", 0) for r in career)
    avg_tenure = total_months / n_companies if n_companies > 0 else 0.0

    companies_in_career = [r.get("company", "").lower() for r in career]
    titles_in_career = [r.get("title", "").lower() for r in career]
    industries_in_career = [r.get("industry", "").lower() for r in career]

    # Skill aggregate stats
    avg_endorsements = (
        sum(s.get("endorsements", 0) for s in skills) / len(skills)
        if skills else 0.0
    )

    # Education tier (best tier)
    tier_order = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    edu_tier = max(
        (tier_order.get(e.get("tier", "unknown"), 0) for e in education),
        default=0,
    )

    # Salary midpoint
    salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_mid = (salary.get("min", 0) + salary.get("max", 0)) / 2

    # Skill assessment average
    assessments = signals.get("skill_assessment_scores", {})
    avg_assessment = sum(assessments.values()) / len(assessments) if assessments else 0.0

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "years_of_experience": profile.get("years_of_experience", 0.0),
        "current_industry": profile.get("current_industry", ""),
        "n_companies": n_companies,
        "total_months": total_months,
        "avg_tenure_months": avg_tenure,
        "companies_in_career": " | ".join(companies_in_career),
        "titles_in_career": " | ".join(titles_in_career),
        "industries_in_career": " | ".join(industries_in_career),
        "avg_endorsements": avg_endorsements,
        "edu_tier": edu_tier,
        "salary_mid_lpa": salary_mid,
        "avg_assessment_score": avg_assessment,
        "n_skills": len(skills),
        "skill_names": " | ".join(s.get("name", "") for s in skills),
        # Redrob signals
        "profile_completeness_score": signals.get("profile_completeness_score", 0),
        "last_active_date": signals.get("last_active_date", "2000-01-01"),
        "open_to_work_flag": int(signals.get("open_to_work_flag", False)),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 999.0),
        "notice_period_days": signals.get("notice_period_days", 90),
        "github_activity_score": signals.get("github_activity_score", -1),
        "preferred_work_mode": signals.get("preferred_work_mode", "onsite"),
        "willing_to_relocate": int(signals.get("willing_to_relocate", False)),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.0),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "verified_email": int(signals.get("verified_email", False)),
        "verified_phone": int(signals.get("verified_phone", False)),
        "linkedin_connected": int(signals.get("linkedin_connected", False)),
        "profile_views_received_30d": signals.get("profile_views_received_30d", 0),
        "search_appearance_30d": signals.get("search_appearance_30d", 0),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
        "applications_submitted_30d": signals.get("applications_submitted_30d", 0),
        "connection_count": signals.get("connection_count", 0),
        "endorsements_received": signals.get("endorsements_received", 0),
    }
