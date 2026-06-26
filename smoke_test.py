"""
smoke_test.py -- Quick validation that imports and data files are in order.
Run from project root: python smoke_test.py
"""
from __future__ import annotations
import sys, json, io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

def check(label: str, condition: bool) -> None:
    icon = "✅" if condition else "❌"
    print(f"  {icon} {label}")

print("\n=== Redrob Ranking Engine — Smoke Test ===\n")

# File checks
check("data/candidates.jsonl exists", (ROOT/"data"/"candidates.jsonl").exists())
check("data/job_description.docx exists", (ROOT/"data"/"job_description.docx").exists())
check("data/candidate_schema.json exists", (ROOT/"data"/"candidate_schema.json").exists())
check("validate_submission.py exists", (ROOT/"validate_submission.py").exists())
check("backend/embedding_cache.py exists", (ROOT/"backend"/"embedding_cache.py").exists())
check("backend/filters.py exists", (ROOT/"backend"/"filters.py").exists())
check("backend/signals.py exists", (ROOT/"backend"/"signals.py").exists())
check("backend/ranker.py exists", (ROOT/"backend"/"ranker.py").exists())
check("frontend/app.py exists", (ROOT/"frontend"/"app.py").exists())
check("submission_metadata.yaml exists", (ROOT/"submission_metadata.yaml").exists())

# Import checks
print("\n--- Import checks ---")
try:
    import pandas as pd
    check(f"pandas {pd.__version__}", True)
except ImportError as e:
    check(f"pandas MISSING: {e}", False)

try:
    import numpy as np
    check(f"numpy {np.__version__}", True)
except ImportError as e:
    check(f"numpy MISSING: {e}", False)

try:
    import docx
    check("python-docx available", True)
except ImportError as e:
    check(f"python-docx MISSING: {e}", False)

try:
    import streamlit
    check(f"streamlit {streamlit.__version__}", True)
except ImportError as e:
    check(f"streamlit MISSING: {e}", False)

try:
    from sentence_transformers import SentenceTransformer
    check("sentence-transformers available", True)
except ImportError as e:
    check(f"sentence-transformers MISSING: {e}", False)

# Data sanity check — read first 3 records from JSONL
print("\n--- Data sanity ---")
jsonl = ROOT / "data" / "candidates.jsonl"
if jsonl.exists():
    with open(jsonl, "r", encoding="utf-8") as f:
        first_lines = [next(f, None) for _ in range(3)]
    valid = [l for l in first_lines if l]
    check(f"Read {len(valid)} sample records from candidates.jsonl", len(valid) > 0)
    if valid:
        c = json.loads(valid[0])
        check(f"candidate_id format: {c.get('candidate_id','?')}", "CAND_" in str(c.get("candidate_id","")))
        check("redrob_signals present", "redrob_signals" in c)
else:
    check("candidates.jsonl readable", False)

# Module smoke
print("\n--- Module smoke ---")
try:
    from backend.utils.text_utils import build_profile_text, flatten_candidate
    from backend.utils.score_math import min_max_normalize, safe_log
    check("text_utils imports OK", True)
    check("score_math imports OK", True)
except Exception as e:
    check(f"utils import FAILED: {e}", False)

try:
    from backend.filters import compute_filter_multiplier
    from backend.signals import compute_velocity_scalar
    check("filters imports OK", True)
    check("signals imports OK", True)
except Exception as e:
    check(f"filter/signal import FAILED: {e}", False)

try:
    import numpy as np
    import pandas as pd
    # Test with 5 dummy rows
    dummy = pd.DataFrame({
        "current_title": ["AI Engineer", "HR Manager", "ML Engineer", "Marketing Manager", "Data Scientist"],
        "current_company": ["Startup", "TCS", "ProductCo", "Wipro", "Acme"],
        "companies_in_career": ["Startup|Globex", "TCS|Infosys", "ProductCo", "Wipro|Accenture", "Acme|Initech"],
        "industries_in_career": ["Software", "IT Services|IT Services", "Software", "IT Services|IT Services", "Software"],
        "avg_tenure_months": [24.0, 12.0, 36.0, 8.0, 30.0],
        "profile_completeness_score": [85.0, 30.0, 90.0, 40.0, 78.0],
        "avg_endorsements": [15.0, 45.0, 10.0, 50.0, 12.0],
        "open_to_work_flag": [1, 0, 1, 0, 1],
        "verified_email": [1, 1, 1, 0, 1],
        "verified_phone": [1, 0, 1, 0, 1],
        "recruiter_response_rate": [0.8, 0.3, 0.9, 0.2, 0.7],
        "notice_period_days": [30, 90, 15, 120, 60],
        "github_activity_score": [75.0, -1.0, 90.0, -1.0, 55.0],
        "preferred_work_mode": ["hybrid", "onsite", "flexible", "onsite", "hybrid"],
        "location": ["Pune", "Chennai", "Bangalore", "Delhi", "Hyderabad"],
        "interview_completion_rate": [0.9, 0.5, 0.95, 0.3, 0.8],
        "offer_acceptance_rate": [0.7, -1.0, 0.8, -1.0, 0.6],
        "profile_views_received_30d": [50, 5, 80, 2, 40],
        "search_appearance_30d": [200, 30, 300, 10, 150],
        "saved_by_recruiters_30d": [15, 2, 20, 1, 10],
        "applications_submitted_30d": [3, 8, 2, 10, 4],
        "avg_assessment_score": [72.0, 0.0, 85.0, 0.0, 68.0],
        "connection_count": [400, 50, 600, 20, 300],
        "endorsements_received": [30, 5, 45, 3, 25],
        "last_active_date": ["2026-06-10", "2025-12-01", "2026-06-18", "2026-01-01", "2026-05-20"],
    })
    from backend.filters import compute_filter_multiplier
    from backend.signals import compute_velocity_scalar
    flt = compute_filter_multiplier(dummy)
    v_s = compute_velocity_scalar(dummy)
    assert flt[1] == 0.0, "HR Manager should be zeroed out"
    assert flt[3] == 0.0, "Marketing Manager (honeypot) should be zeroed out"
    assert flt[0] > 0, "AI Engineer should pass"
    check("Filter mask logic: excluded roles zeroed", True)
    check("Velocity scalar computed", len(v_s) == 5)
    check(f"V_s range: min={v_s.min():.3f} max={v_s.max():.3f}", 0 <= v_s.min() and v_s.max() <= 1)
except Exception as e:
    check(f"Vectorized logic FAILED: {e}", False)

print("\n=== Smoke test complete ===\n")
