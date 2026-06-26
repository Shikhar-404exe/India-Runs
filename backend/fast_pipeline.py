"""
backend/fast_pipeline.py
Unified pipeline: keyword pre-filter -> embed subset -> rank.
Under 5 min total for 100k candidates on CPU.

Architecture:
  1. Load candidates + JD text
  2. TF-IDF keyword matching (all 100k) -> select top N candidates
  3. Encode only those N with all-MiniLM-L6-v2
  4. Dot-product similarity -> re-rank within the subset
  5. Apply filters + Redrob signals
  6. Composite score -> top 100

Usage: python -m backend.fast_pipeline --data-dir data --out submission.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.utils.text_utils import build_profile_text, flatten_candidate, read_docx_text, stream_jsonl
from backend.filters import compute_filter_multiplier
from backend.signals import compute_velocity_scalar
from backend.utils.score_math import cosine_similarity_matrix, min_max_normalize, topk_indices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEYWORD_TOP_K = 5000        # Candidates passed to the embedding stage
SEMANTIC_TOP_K = 2000        # Candidates passed to filter/signal stage
FINAL_TOP_K = 100
W_SEMANTIC = 0.45
W_VELOCITY = 0.30
W_FILTER = 0.15
W_EDUCATION = 0.10
REFERENCE_DATE = "2026-06-19"
EDU_TIER_SCORE = {4: 1.0, 3: 0.75, 2: 0.5, 1: 0.25, 0: 0.1}

# ---------------------------------------------------------------------------
# Stage 1: Keyword pre-filter (TF-IDF)
# ---------------------------------------------------------------------------


def build_tfidf_vectors(texts: list[str]) -> tuple:
    """Fit a TF-IDF vectorizer on candidate texts and transform."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def keyword_topk(
    jd_text: str,
    candidate_texts: list[str],
    candidate_ids: list[str],
    k: int = KEYWORD_TOP_K,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Use TF-IDF cosine similarity to select top-k candidates by keyword match.

    Returns:
        (top_scores, top_ids, top_texts) - index-aligned arrays.
    """
    print(f"  [keyword] Building TF-IDF vectors for {len(candidate_texts):,} texts ...")
    t0 = time.perf_counter()
    vec, tfidf_matrix = build_tfidf_vectors(candidate_texts)
    jd_vec = vec.transform([jd_text])
    print(f"  [keyword] TF-IDF matrix: {tfidf_matrix.shape} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    scores = tfidf_matrix.dot(jd_vec.T).toarray().ravel()
    idx = topk_indices(scores, k)
    idx = idx[np.argsort(scores[idx])[::-1]]
    print(f"  [keyword] Top-{k} selected in {time.perf_counter()-t0:.1f}s")
    return scores[idx], [candidate_ids[i] for i in idx], [candidate_texts[i] for i in idx]


# ---------------------------------------------------------------------------
# Stage 2: Dense embedding
# ---------------------------------------------------------------------------


def encode_subset(texts: list[str], batch_size: int = 256) -> tuple[np.ndarray, object]:
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print(f"  [encode] Model loaded in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [encode] {len(texts):,} texts encoded: {len(texts)/elapsed:.0f} cand/sec ({elapsed:.1f}s)")
    return embeddings.astype(np.float32), model


# ---------------------------------------------------------------------------
# Stage 3: Ranking (dot-product -> filters -> signals -> composite)
# ---------------------------------------------------------------------------


def compute_education_score(df: pd.DataFrame) -> np.ndarray:
    return df["edu_tier"].map(EDU_TIER_SCORE).fillna(0.1).to_numpy(dtype=np.float32)


def build_reasoning(row: pd.Series) -> str:
    _AI_CORE_SKILLS = {
        "python", "pytorch", "tensorflow", "keras", "transformers", "llm",
        "fine-tuning llms", "lora", "nlp", "computer vision", "machine learning",
        "deep learning", "mlops", "huggingface", "scikit-learn", "xgboost",
        "lightgbm", "ray", "onnx", "triton", "langchain", "retrieval",
        "rag", "gans", "stable diffusion", "speech recognition", "tts",
        "image classification", "object detection", "cuda", "jax",
        "weights & biases", "mlflow", "bentoml", "milvus", "faiss",
    }
    title = row.get("current_title", "Professional")
    yoe = row.get("years_of_experience", 0)
    company = row.get("current_company", "")
    notice = int(row.get("notice_period_days", 90))
    rr = row.get("recruiter_response_rate", 0)
    github = row.get("github_activity_score", -1)
    location = row.get("location", "")
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


def rank_subset(
    embeddings: np.ndarray,
    subset_ids: list[str],
    jd_vec: np.ndarray,
    df_all: pd.DataFrame,
    weights: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Run the full ranking pipeline on the embedded subset."""
    w_sem, w_vel, w_flt, w_edu = weights

    # Semantic: dot-product
    print(f"  [rank] Dot-product similarity ...")
    t0 = time.perf_counter()
    sim = cosine_similarity_matrix(jd_vec, embeddings)
    top_idx = topk_indices(sim, SEMANTIC_TOP_K)
    top_idx = top_idx[np.argsort(sim[top_idx])[::-1]]
    top_ids = np.array(subset_ids)[top_idx]
    top_sim = sim[top_idx]
    print(f"  [rank] Semantic top-{SEMANTIC_TOP_K} in {time.perf_counter()-t0:.2f}s")

    # Extract sub-DataFrame
    df_top = df_all.loc[df_all.index.isin(top_ids)].copy()
    sem_series = pd.Series(top_sim, index=top_ids)
    df_top["sem_score"] = sem_series.reindex(df_top.index).values

    # Filters
    print(f"  [rank] Applying filters ...")
    filter_mult = compute_filter_multiplier(df_top)
    df_top["filter_mult"] = filter_mult

    # Signals
    print(f"  [rank] Computing velocity scalar ...")
    v_s = compute_velocity_scalar(df_top, reference_date=REFERENCE_DATE)
    df_top["velocity_score"] = v_s

    # Education
    edu_scores = compute_education_score(df_top)
    df_top["edu_score"] = edu_scores

    # Composite
    sem_norm = min_max_normalize(df_top["sem_score"].to_numpy(dtype=np.float32))
    flt_norm = min_max_normalize(df_top["filter_mult"].to_numpy(dtype=np.float32))
    composite = (
        w_sem * sem_norm
        + w_vel * df_top["velocity_score"].to_numpy(dtype=np.float32)
        + w_flt * flt_norm
        + w_edu * df_top["edu_score"].to_numpy(dtype=np.float32)
    )
    composite *= filter_mult

    df_top["final_score"] = np.round(composite, 4)

    # Sort -> top 100
    df_sorted = df_top.sort_values(
        ["final_score", "candidate_id"],
        ascending=[False, True],
    ).head(FINAL_TOP_K).copy()
    df_sorted["rank"] = range(1, len(df_sorted) + 1)
    df_sorted["reasoning"] = df_sorted.apply(build_reasoning, axis=1)
    return df_sorted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast unified pipeline: keyword -> embed -> rank (under 5 min)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--jd", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--keyword-k", type=int, default=KEYWORD_TOP_K,
                        help="Candidates after keyword pre-filter (default: 5000)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    jd_path = (args.jd or data_dir / "job_description.docx").resolve()
    out_path = args.out.resolve()

    if not jd_path.exists():
        print(f"ERROR: JD not found at {jd_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[fast_pipeline] Data dir    : {data_dir}")
    print(f"[fast_pipeline] JD          : {jd_path}")
    print(f"[fast_pipeline] Keyword top-k: {args.keyword_k}")

    t_total = time.perf_counter()

    # -- Step 1: Load candidates + JD text --
    print("\n[Step 1/4] Loading data ...")
    jsonl_path = data_dir / "candidates.jsonl"
    gzipped = False
    if not jsonl_path.exists():
        gz = data_dir / "candidates.jsonl.gz"
        if gz.exists():
            jsonl_path, gzipped = gz, True
        else:
            print(f"ERROR: No candidates file in {data_dir}", file=sys.stderr)
            sys.exit(1)

    t0 = time.perf_counter()
    records = [flatten_candidate(c) for c in stream_jsonl(jsonl_path, gzipped=gzipped)]
    df_all = pd.DataFrame(records).set_index("candidate_id")
    print(f"  [load] DataFrame: {len(df_all):,} rows in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    candidate_texts = [build_profile_text(c) for c in stream_jsonl(jsonl_path, gzipped=gzipped)]
    candidate_ids = df_all.index.tolist()
    jd_text = read_docx_text(jd_path)
    print(f"  [load] Texts built: {len(candidate_texts):,} in {time.perf_counter()-t0:.1f}s")

    # -- Step 2: Keyword pre-filter --
    print(f"\n[Step 2/4] Keyword pre-filter (top {args.keyword_k}) ...")
    kw_scores, kw_ids, kw_texts = keyword_topk(jd_text, candidate_texts, candidate_ids, k=args.keyword_k)
    print(f"  [keyword] Top score: {kw_scores[0]:.4f}, Bottom: {kw_scores[-1]:.4f}")

    # -- Step 3: Embed the subset --
    print(f"\n[Step 3/4] Encoding {len(kw_texts):,} candidates ...")
    embeddings, model = encode_subset(kw_texts)

    # Embed JD with same model (no reload)
    jd_vec = model.encode(jd_text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    del model

    # -- Step 4: Rank --
    print(f"\n[Step 4/4] Ranking ...")
    df_result = rank_subset(
        embeddings, kw_ids, jd_vec, df_all,
        weights=(W_SEMANTIC, W_VELOCITY, W_FILTER, W_EDUCATION),
    )

    # -- Output --
    submission = df_result.reset_index()[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission = submission.rename(columns={"final_score": "score"})
    submission["score"] = submission["score"].round(4)
    submission.to_csv(str(out_path), index=False)

    total = time.perf_counter() - t_total
    print(f"\n[fast_pipeline] DONE - {len(submission)} candidates -> {out_path}")
    print(f"[fast_pipeline] Total wall-clock: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
