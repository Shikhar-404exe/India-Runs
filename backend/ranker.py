"""
backend/ranker.py
ONLINE PHASE — Multi-Stage Cascade Ranking Loop (headless CLI).

Pipeline:
  1. Load precomputed embeddings_matrix.npy + candidate_ids.npy
  2. Load candidates.jsonl into a flat Pandas DataFrame (no-loop parse)
  3. Embed JD text with all-MiniLM-L6-v2 (single forward pass)
  4. NumPy dot-product → top 2,000 semantically similar candidates
  5. Apply filters.py (hard masks + soft penalties)
  6. Apply signals.py (Redrob velocity scalar)
  7. Composite score = 0.45*sem + 0.30*V_s + 0.15*filter + 0.10*edu
  8. Sort: descending score, alphabetical tie-break on candidate_id
  9. Generate extractive reasoning strings
  10. Write top 100 → submission.csv

Usage:
    python backend/ranker.py [--data-dir data/] [--jd data/job_description.docx]
                              [--out submission.csv] [--top-k 100]
Max 300 lines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.filters import compute_filter_multiplier
from backend.signals import compute_velocity_scalar
from backend.utils.score_math import cosine_similarity_matrix, min_max_normalize, topk_indices
from backend.utils.text_utils import flatten_candidate, read_docx_text, stream_jsonl

# ---------------------------------------------------------------------------
# Constants & weight defaults
# ---------------------------------------------------------------------------

TOP_STAGE1 = 2000          # Candidates pulled from semantic stage
FINAL_TOP_K = 100          # Final submission count
W_SEMANTIC = 0.45
W_VELOCITY = 0.30
W_FILTER = 0.15
W_EDUCATION = 0.10
REFERENCE_DATE = "2026-06-19"

EDU_TIER_SCORE = {4: 1.0, 3: 0.75, 2: 0.5, 1: 0.25, 0: 0.1}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_embeddings(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed embedding matrix and companion ID array."""
    emb_path = data_dir / "embeddings_matrix.npy"
    ids_path = data_dir / "candidate_ids.npy"

    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found. Run embedding_cache.py first.", file=sys.stderr)
        sys.exit(1)

    matrix = np.load(str(emb_path), mmap_mode="r")  # memory-mapped for speed
    ids = np.load(str(ids_path), allow_pickle=True)
    print(f"[ranker] Embeddings: {matrix.shape} | IDs: {ids.shape}")
    return matrix, ids


def load_candidates_df(data_dir: Path) -> pd.DataFrame:
    """
    Stream candidates.jsonl into a flat Pandas DataFrame via flatten_candidate.
    Uses list comprehension (no row-level loops in scoring code).
    """
    jsonl_path = data_dir / "candidates.jsonl"
    gzipped = False
    if not jsonl_path.exists():
        gz_path = data_dir / "candidates.jsonl.gz"
        if gz_path.exists():
            jsonl_path, gzipped = gz_path, True
        else:
            print(f"ERROR: No candidates file in {data_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"[ranker] Loading candidates from {jsonl_path} …")
    t0 = time.perf_counter()
    records = [flatten_candidate(c) for c in stream_jsonl(jsonl_path, gzipped=gzipped)]
    df = pd.DataFrame(records)
    df = df.set_index("candidate_id")
    print(f"[ranker] Loaded {len(df):,} candidates in {time.perf_counter()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# JD embedding
# ---------------------------------------------------------------------------

def embed_job_description(jd_path: Path) -> np.ndarray:
    """
    Extract text from JD docx and encode with all-MiniLM-L6-v2.
    Returns a normalised (384,) float32 vector.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[ranker] Embedding JD: {jd_path}")
    jd_text = read_docx_text(jd_path)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    vec = model.encode(jd_text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Education score
# ---------------------------------------------------------------------------

def compute_education_score(df: pd.DataFrame) -> np.ndarray:
    """Map edu_tier integer to [0, 1] score via EDU_TIER_SCORE table."""
    return df["edu_tier"].map(EDU_TIER_SCORE).fillna(0.1).to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Extractive reasoning generator
# ---------------------------------------------------------------------------

_AI_CORE_SKILLS = {
    "python", "pytorch", "tensorflow", "keras", "transformers", "llm",
    "fine-tuning llms", "lora", "nlp", "computer vision", "machine learning",
    "deep learning", "mlops", "huggingface", "scikit-learn", "xgboost",
    "lightgbm", "ray", "onnx", "triton", "langchain", "retrieval",
    "rag", "gans", "stable diffusion", "speech recognition", "tts",
    "image classification", "object detection", "cuda", "jax",
    "weights & biases", "mlflow", "bentoml", "milvus", "faiss",
}


def build_reasoning(row: pd.Series) -> str:
    """
    Generate an extractive, non-templated, fact-based reasoning sentence
    using only verified platform fields from the candidate row.

    Args:
        row: Single row from the ranked DataFrame.

    Returns:
        1–2 sentence reasoning string.
    """
    title = row.get("current_title", "Professional")
    yoe = row.get("years_of_experience", 0)
    company = row.get("current_company", "")
    notice = int(row.get("notice_period_days", 90))
    rr = row.get("recruiter_response_rate", 0)
    github = row.get("github_activity_score", -1)
    location = row.get("location", "")

    # Count AI-core skills
    skill_str = str(row.get("skill_names", "")).lower()
    n_ai_skills = sum(1 for s in _AI_CORE_SKILLS if s in skill_str)

    parts: list[str] = [
        f"{title} with {yoe:.1f} yrs at {company};" if company else f"{title} with {yoe:.1f} yrs;",
        f"{n_ai_skills} AI-core skills;",
        f"recruiter response rate {rr:.2f};",
        f"{notice}d notice;",
    ]
    if github >= 0:
        parts.append(f"GitHub activity score {github:.0f};")
    if location:
        parts.append(f"based in {location}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main ranking pipeline
# ---------------------------------------------------------------------------

def run_ranking(
    data_dir: Path,
    jd_path: Path,
    out_path: Path,
    weights: tuple[float, float, float, float] = (
        W_SEMANTIC, W_VELOCITY, W_FILTER, W_EDUCATION
    ),
    use_cross_encoder: bool = False,
) -> pd.DataFrame:
    """
    Execute the full multi-stage cascade ranking pipeline.

    Args:
        data_dir: Directory containing embeddings and candidates file.
        jd_path:  Path to job_description.docx.
        out_path: Output CSV path.
        weights:  (semantic, velocity, filter, education) weight tuple.

    Returns:
        Ranked top-100 DataFrame.
    """
    t_pipeline = time.perf_counter()
    w_sem, w_vel, w_flt, w_edu = weights

    # 1. Load embeddings + all candidates
    matrix, all_ids = load_embeddings(data_dir)
    df_all = load_candidates_df(data_dir)

    # 2. Embed JD
    jd_vec = embed_job_description(jd_path)

    # 3. Stage 1 — semantic selection of top-2,000
    print(f"[ranker] Stage 1: dot-product similarity over {len(all_ids):,} candidates …")
    t1 = time.perf_counter()
    sim_scores = cosine_similarity_matrix(jd_vec, np.array(matrix))
    top_idx = topk_indices(sim_scores, TOP_STAGE1)
    top_idx = top_idx[np.argsort(sim_scores[top_idx])[::-1]]  # sort descending
    top_ids = all_ids[top_idx]
    top_sim = sim_scores[top_idx]
    print(f"[ranker] Stage 1 done in {time.perf_counter()-t1:.2f}s | top sim: {top_sim[0]:.4f}")

    # 4. Extract sub-DataFrame for top-2,000 (index is candidate_id)
    df_top = df_all.loc[df_all.index.isin(top_ids)].copy()
    # Align semantic scores
    sem_series = pd.Series(top_sim, index=top_ids)
    df_top["sem_score"] = sem_series.reindex(df_top.index).values

    # 5. Stage 2 — filters
    print("[ranker] Stage 2: applying filter masks …")
    filter_mult = compute_filter_multiplier(df_top)
    df_top["filter_mult"] = filter_mult

    # 6. Stage 3 — signals
    print("[ranker] Stage 3: computing Redrob velocity scalars …")
    v_s = compute_velocity_scalar(df_top, reference_date=REFERENCE_DATE)
    df_top["velocity_score"] = v_s

    # 7. Education score
    edu_scores = compute_education_score(df_top)
    df_top["edu_score"] = edu_scores

    # 8. Composite score
    sem_norm = min_max_normalize(df_top["sem_score"].to_numpy(dtype=np.float32))
    flt_norm = min_max_normalize(df_top["filter_mult"].to_numpy(dtype=np.float32))

    composite = (
        w_sem * sem_norm
        + w_vel * df_top["velocity_score"].to_numpy(dtype=np.float32)
        + w_flt * flt_norm
        + w_edu * df_top["edu_score"].to_numpy(dtype=np.float32)
    )
    # Apply hard zero from filter (honeypots, excluded titles)
    composite *= filter_mult
    df_top["final_score"] = np.round(composite, 4)

    if use_cross_encoder:
        print(f"[ranker] Stage 3.5: Cross-encoding top 200 candidates...")
        t_ce = time.perf_counter()
        
        # Take the top 200 based on the current bi-encoder composite score
        df_ce = df_top.sort_values(["final_score", "candidate_id"], ascending=[False, True]).head(200).copy()
        
        # Load CrossEncoder
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
        
        # Build text pairs
        jd_text = read_docx_text(jd_path)
        pairs = []
        for _, r in df_ce.iterrows():
            cand_text = f"{r.get('current_title', '')} at {r.get('current_company', '')}. Skills: {r.get('skill_names', '')}. History: {r.get('titles_in_career', '')} at {r.get('companies_in_career', '')}."
            pairs.append((jd_text, cand_text))
            
        ce_scores = model.predict(pairs)
        ce_norm = min_max_normalize(ce_scores.astype(np.float32))
        
        # Overwrite sem_score with cross-encoder score for these 200 candidates
        df_ce["sem_score"] = ce_norm
        
        # Re-compute composite using the new semantic scores
        vel = df_ce["velocity_score"].to_numpy(dtype=np.float32)
        flt = min_max_normalize(df_ce["filter_mult"].to_numpy(dtype=np.float32))
        edu = df_ce["edu_score"].to_numpy(dtype=np.float32)
        
        new_comp = (w_sem * ce_norm + w_vel * vel + w_flt * flt + w_edu * edu) * df_ce["filter_mult"]
        df_ce["final_score"] = np.round(new_comp, 4)
        
        df_top = df_ce  # Only consider these 200 for the final top 100
        print(f"[ranker] Cross-encoding done in {time.perf_counter()-t_ce:.2f}s")

    # 9. Sort: descending score, then ascending candidate_id (tie-break)
    df_sorted = df_top.sort_values(
        ["final_score", "candidate_id"],
        ascending=[False, True],
    ).head(FINAL_TOP_K).copy()

    # 10. Assign ranks
    df_sorted["rank"] = range(1, len(df_sorted) + 1)

    # 11. Generate reasoning
    df_sorted["reasoning"] = df_sorted.apply(build_reasoning, axis=1)

    # 12. Build submission CSV
    submission = df_sorted.reset_index()[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission = submission.rename(columns={"final_score": "score"})
    submission["score"] = submission["score"].round(4)

    submission.to_csv(str(out_path), index=False)
    total_time = time.perf_counter() - t_pipeline
    print(
        f"[ranker] DONE — top {len(submission)} candidates written to {out_path} "
        f"in {total_time:.1f}s"
    )
    return submission


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Redrob candidate ranker.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--jd", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--cross-encode", action="store_true", help="Apply Cross-Encoder re-ranking to top 200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    jd_path = (args.jd or data_dir / "job_description.docx").resolve()
    out_path = args.out.resolve()

    if not jd_path.exists():
        print(f"ERROR: JD not found at {jd_path}", file=sys.stderr)
        sys.exit(1)

    run_ranking(data_dir, jd_path, out_path, use_cross_encoder=args.cross_encode)


if __name__ == "__main__":
    main()
