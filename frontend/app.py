"""
frontend/app.py
Redrob AI — premium glassmorphic candidate ranking dashboard.
"""
from __future__ import annotations
import os, sys, tempfile, time, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@st.cache_resource
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI · Candidate Discovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #050818 !important;
    color: #e2e8f0 !important;
}
[data-testid="stAppViewContainer"] > .main > div { padding-top: 0 !important; }
[data-testid="stHeader"] { display: none !important; }

.hero {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(102,126,234,0.35) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 60%, rgba(118,75,162,0.25) 0%, transparent 55%),
                radial-gradient(ellipse 50% 40% at 10% 80%, rgba(56,189,248,0.15) 0%, transparent 50%),
                linear-gradient(180deg, #080d1a 0%, #050818 100%);
    padding: 48px 40px 36px;
    border-bottom: 1px solid rgba(102,126,234,0.18);
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%23667eea' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-logo {
    display: inline-flex; align-items: center; gap: 12px;
    background: rgba(102,126,234,0.12); border: 1px solid rgba(102,126,234,0.3);
    border-radius: 40px; padding: 6px 18px 6px 10px; margin-bottom: 20px;
    backdrop-filter: blur(8px);
}
.hero-logo-dot { width:28px;height:28px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px; }
.hero-logo-text { font-size:13px;font-weight:600;color:#a5b4fc;letter-spacing:.5px; }
.hero-title {
    font-size: 46px; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0 0 12px;
}
.hero-subtitle { font-size:16px;color:#64748b;font-weight:400;margin:0; }
.hero-subtitle span { color:#818cf8;font-weight:500; }

.upload-label { font-size:13px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;display:block; }
.section-label { font-size:11px;font-weight:700;letter-spacing:2px;color:#475569;text-transform:uppercase;margin:32px 0 16px;padding-left:3px; }

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 15px !important;
    letter-spacing: .5px !important; padding: 14px 32px !important;
    color: white !important;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35), 0 0 0 1px rgba(102,126,234,0.2) !important;
    transition: all .25s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(102,126,234,0.5) !important;
}

[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(16,185,129,.15), rgba(5,150,105,.15)) !important;
    border: 1.5px solid rgba(16,185,129,.4) !important;
    border-radius: 10px !important; color: #34d399 !important;
    font-weight: 600 !important; font-size: 14px !important; transition: all .25s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, rgba(16,185,129,.25), rgba(5,150,105,.25)) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(16,185,129,.25) !important;
}

/* ── PROGRESS BAR FIX: kill white blocks, replace with purple glow ── */
div[data-testid="stProgressBar"] { background: transparent !important; }
div[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 100px !important; height: 6px !important;
    border: none !important; overflow: hidden !important;
    box-shadow: none !important;
}
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2, #a78bfa) !important;
    border-radius: 100px !important;
    box-shadow: 0 0 10px rgba(102,126,234,.7) !important;
    transition: width .3s ease !important;
    border: none !important; height: 100% !important;
}
/* also target the progress text */
div[data-testid="stProgressBar"] p { color: #64748b !important; font-size:12px !important; margin-top: 4px !important; }

[data-testid="stStatus"] {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.08) !important; border-radius: 10px !important;
}
[data-testid="stAlertContainer"] {
    background: rgba(102,126,234,.08) !important;
    border: 1px solid rgba(102,126,234,.2) !important;
    border-radius: 12px !important; color: #a5b4fc !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.025) !important;
    border: 1.5px dashed rgba(102,126,234,.3) !important;
    border-radius: 12px !important; transition: all .3s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(102,126,234,.06) !important;
    border-color: rgba(102,126,234,.55) !important;
}

/* Stats bar */
.stats-bar { display:flex;gap:0;margin-bottom:24px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden; }
.stat-item { flex:1;padding:16px 20px;text-align:center;border-right:1px solid rgba(255,255,255,.06); }
.stat-item:last-child { border-right:none; }
.stat-val { font-size:26px;font-weight:800;color:#e2e8f0;line-height:1;margin-bottom:4px; }
.stat-lbl { font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:1px; }

/* Individual candidate card */
.cand-card {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px; padding: 18px 20px;
    position: relative; overflow: hidden;
    margin-bottom: 0;
}
.cand-card.top3 {
    border-color: rgba(102,126,234,.3);
    background: rgba(102,126,234,.07);
    box-shadow: 0 4px 24px rgba(102,126,234,.12), inset 0 1px 0 rgba(255,255,255,.06);
}
.cand-card::before {
    content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(102,126,234,.06) 0%,transparent 60%);
    pointer-events:none;
}
.card-header { display:flex;align-items:center;gap:12px;margin-bottom:10px; }
.rank-badge {
    flex-shrink:0;width:34px;height:34px;border-radius:10px;
    background:linear-gradient(135deg,#667eea,#764ba2);
    color:white;font-weight:800;font-size:13px;
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 4px 12px rgba(102,126,234,.4);
}
.rank-badge.gold   { background:linear-gradient(135deg,#f59e0b,#d97706);box-shadow:0 4px 12px rgba(245,158,11,.4); }
.rank-badge.silver { background:linear-gradient(135deg,#94a3b8,#64748b);box-shadow:0 4px 12px rgba(148,163,184,.3); }
.rank-badge.bronze { background:linear-gradient(135deg,#cd7c54,#b45309);box-shadow:0 4px 12px rgba(180,83,9,.35); }
.cand-id { font-weight:700;color:#e2e8f0;font-size:14px;flex:1; }
.score-chip {
    background:linear-gradient(135deg,rgba(102,126,234,.2),rgba(118,75,162,.2));
    border:1px solid rgba(102,126,234,.35);color:#a5b4fc;
    border-radius:8px;padding:4px 10px;font-size:12px;font-weight:700;letter-spacing:.3px;
}
.card-body { color:#94a3b8;font-size:13px;line-height:1.55; }
.card-tags { display:flex;flex-wrap:wrap;gap:5px;margin-top:10px; }
.tag { font-size:11px;font-weight:500;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:6px;padding:2px 8px;color:#64748b; }
.tag.ai { background:rgba(102,126,234,.12);border-color:rgba(102,126,234,.25);color:#818cf8; }

.fancy-divider { height:1px;background:linear-gradient(90deg,transparent,rgba(102,126,234,.3),rgba(118,75,162,.3),transparent);margin:32px 0;border:none; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,.03); }
::-webkit-scrollbar-thumb { background:rgba(102,126,234,.3);border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-logo">
    <div class="hero-logo-dot">🧠</div>
    <span class="hero-logo-text">REDROB AI · BETA</span>
  </div>
  <div class="hero-title">Candidate Discovery<br>Engine</div>
  <p class="hero-subtitle">
    Rank <span>100,000 candidates</span> against any JD using
    <span>23 behavioural signals</span> + semantic embeddings.
    Results in <span>&lt; 5 minutes</span>.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📂 &nbsp;Upload Data</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<span class="upload-label">🗂 Candidate Pool (.jsonl / .jsonl.gz)</span>', unsafe_allow_html=True)
    candidates_file = st.file_uploader("candidates", type=["jsonl", "gz", "csv"], label_visibility="collapsed")
with col2:
    st.markdown('<span class="upload-label">📄 Job Description (.docx)</span>', unsafe_allow_html=True)
    jd_file = st.file_uploader("jd", type=["docx"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button(
    "⚡  Run AI Ranking Pipeline",
    type="primary", use_container_width=True,
    disabled=not (candidates_file and jd_file),
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn and candidates_file and jd_file:
    workspace = Path(tempfile.mkdtemp(prefix="redrob_"))
    cand_path = workspace / candidates_file.name
    jd_dst    = workspace / "job_description.docx"
    with open(cand_path, "wb") as f: f.write(candidates_file.getbuffer())
    with open(jd_dst,   "wb") as f: f.write(jd_file.getbuffer())

    gzipped = cand_path.suffix == ".gz"
    is_csv  = cand_path.suffix == ".csv"
    from backend.utils.text_utils import build_profile_text, flatten_candidate, read_docx_text, stream_jsonl
    from backend.utils.score_math import topk_indices
    t_start = time.perf_counter()

    with st.status("⏳  Loading candidates…", expanded=False) as s:
        if is_csv:
            df_all = pd.read_csv(cand_path).set_index("candidate_id")
            ids = df_all.index.tolist()
            tcols = [c for c in ["current_title","current_company","skill_names","companies_in_career","titles_in_career","industries_in_career","location"] if c in df_all.columns]
            texts = df_all[tcols].fillna("").agg(" ".join, axis=1).tolist()
        else:
            records, texts = [], []
            for c in stream_jsonl(cand_path, gzipped=gzipped):
                records.append(flatten_candidate(c)); texts.append(build_profile_text(c))
            df_all = pd.DataFrame(records).set_index("candidate_id")
            ids = df_all.index.tolist()
        jd_text = read_docx_text(jd_dst); n = len(df_all)
        s.update(label=f"✅  Loaded **{n:,}** candidates", state="complete")

    with st.status("🔍  Keyword pre-filter (TF-IDF)…", expanded=False) as s:
        vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True, dtype=np.float32)
        tfidf = vec.fit_transform(texts)
        jd_tfidf = vec.transform([jd_text])
        kw_scores = tfidf.dot(jd_tfidf.T).toarray().ravel()
        kw_idx = topk_indices(kw_scores, 3000)
        kw_idx = kw_idx[np.argsort(kw_scores[kw_idx])[::-1]]
        kw_ids = [ids[i] for i in kw_idx]; kw_texts = [texts[i] for i in kw_idx]
        del records, texts, tfidf, kw_scores, jd_tfidf, vec
        s.update(label="✅  Top **3,000** keyword matches selected", state="complete")

    with st.status("🧠  Encoding with all-MiniLM-L6-v2…", expanded=True) as s:
        model = load_encoder()
        batch_size = 256; n_sub = len(kw_texts); all_embs = []
        prog = st.progress(0.0, text="Starting encoder…")
        t0 = time.perf_counter()
        for b in range(0, n_sub, batch_size):
            batch = kw_texts[b: b + batch_size]
            all_embs.append(model.encode(batch, batch_size=batch_size, show_progress_bar=False,
                                         convert_to_numpy=True, normalize_embeddings=True))
            done = min(b + batch_size, n_sub)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            prog.progress(done / n_sub, text=f"Encoding  {done:,} / {n_sub:,}  ·  {rate:.0f} cand/sec")
        prog.empty()
        embs = np.vstack(all_embs).astype(np.float32)
        jd_vec = model.encode(jd_text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        del kw_texts
        s.update(label=f"✅  Encoded **{n_sub:,}** candidates", state="complete")

    from backend.filters import compute_filter_multiplier
    from backend.signals import compute_velocity_scalar
    from backend.utils.score_math import cosine_similarity_matrix, min_max_normalize
    EDU_TIER = {4: 1.0, 3: 0.75, 2: 0.5, 1: 0.25, 0: 0.1}

    with st.status("🏆  Ranking candidates…", expanded=False) as s:
        sim = cosine_similarity_matrix(jd_vec, embs)
        top_idx = topk_indices(sim, 2000)
        top_idx = top_idx[np.argsort(sim[top_idx])[::-1]]
        top_ids = np.array(kw_ids)[top_idx]; top_sim = sim[top_idx]
        del embs, kw_ids

        df_top = df_all.loc[df_all.index.isin(top_ids)].copy()
        sem_s = pd.Series(top_sim, index=top_ids)
        df_top["sem_score"] = sem_s.reindex(df_top.index).values
        del df_all, top_ids

        df_top["filter_mult"]    = compute_filter_multiplier(df_top)
        df_top["velocity_score"] = compute_velocity_scalar(df_top, reference_date="2026-06-19")
        edu = df_top["edu_tier"].map(EDU_TIER).fillna(0.1).to_numpy(dtype=np.float32)
        df_top["edu_score"] = edu

        sem_n = min_max_normalize(df_top["sem_score"].to_numpy(dtype=np.float32))
        flt_n = min_max_normalize(df_top["filter_mult"].to_numpy(dtype=np.float32))
        comp  = (0.45 * sem_n + 0.30 * df_top["velocity_score"].to_numpy(dtype=np.float32)
                 + 0.15 * flt_n + 0.10 * edu)
        comp *= df_top["filter_mult"].to_numpy(dtype=np.float32)
        df_top["final_score"] = np.round(comp, 4)

        df_sorted = df_top.sort_values(["final_score","candidate_id"], ascending=[False,True]).head(100).copy()
        df_sorted["rank"] = range(1, len(df_sorted) + 1)
        s.update(label="✅  Ranked & shortlisted top 100", state="complete")

    _AI_SKILLS = {"python","pytorch","tensorflow","keras","transformers","llm","fine-tuning llms","lora",
                  "nlp","computer vision","machine learning","deep learning","mlops","huggingface",
                  "scikit-learn","xgboost","lightgbm","ray","onnx","triton","langchain","rag","gans",
                  "stable diffusion","speech recognition","cuda","jax","mlflow"}
    def mk_reasoning(r):
        title = r.get("current_title","Professional"); yoe = r.get("years_of_experience", 0)
        company = r.get("current_company",""); notice = int(r.get("notice_period_days", 90))
        rr = r.get("recruiter_response_rate", 0); github = r.get("github_activity_score", -1)
        loc = r.get("location",""); skill_str = str(r.get("skill_names","")).lower()
        n_ai = sum(1 for s in _AI_SKILLS if s in skill_str)
        parts = [f"{title} with {yoe:.1f} yrs at {company};" if company else f"{title} with {yoe:.1f} yrs;",
                 f"{n_ai} AI-core skills;", f"recruiter response rate {rr:.2f};", f"{notice}d notice;"]
        if github >= 0: parts.append(f"GitHub activity score {github:.0f};")
        if loc: parts.append(f"based in {loc}.")
        return " ".join(parts)

    df_sorted["reasoning"] = df_sorted.apply(mk_reasoning, axis=1)

    # ── Build submission in exact schema: candidate_id, rank, score, reasoning ──
    submission_df = (
        df_sorted.reset_index()[["candidate_id", "rank", "final_score", "reasoning"]]
        .rename(columns={"final_score": "score"})
    )
    submission_df["score"] = submission_df["score"].round(4)

    total_t = time.perf_counter() - t_start
    # Store in session state
    st.session_state["submission_df"] = submission_df
    st.session_state["elapsed"]       = total_t
    st.session_state["n_total"]       = n
    st.session_state["done"]          = True
    shutil.rmtree(workspace, ignore_errors=True)

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.get("done") and st.session_state.get("submission_df") is not None:
    sub_df: pd.DataFrame = st.session_state["submission_df"]
    elapsed: float       = st.session_state.get("elapsed", 0)
    n_total: int         = st.session_state.get("n_total", 0)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📊 &nbsp;Pipeline Results</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats-bar">
      <div class="stat-item"><div class="stat-val">{len(sub_df)}</div><div class="stat-lbl">Ranked</div></div>
      <div class="stat-item"><div class="stat-val">{sub_df['score'].max():.4f}</div><div class="stat-lbl">Top Score</div></div>
      <div class="stat-item"><div class="stat-val">{sub_df['score'].mean():.4f}</div><div class="stat-lbl">Mean Score</div></div>
      <div class="stat-item"><div class="stat-val">{elapsed:.0f}s</div><div class="stat-lbl">Wall Clock</div></div>
      <div class="stat-item"><div class="stat-val">{n_total:,}</div><div class="stat-lbl">Pool Size</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download: exact submission.csv format ──────────────────────────────
    csv_bytes = sub_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown('<p class="section-label" style="margin-top:28px;">🏅 &nbsp;Top 100 Candidates</p>',
                unsafe_allow_html=True)

    # ── Cards: render ONE per st.markdown call to avoid HTML truncation ──
    def badge_cls(rank: int) -> str:
        return {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "")

    col_a, col_b = st.columns(2, gap="medium")
    for i, (_, row) in enumerate(sub_df.iterrows()):
        rnk = int(row["rank"])
        is_top3 = rnk <= 3
        bc       = badge_cls(rnk)
        card_cls = "cand-card top3" if is_top3 else "cand-card"
        reasoning = str(row["reasoning"])

        # Build tag chips
        tags = ""
        if "AI-core" in reasoning:
            n_ai_str = reasoning.split("AI-core")[0].strip().split()[-1]
            tags += f'<span class="tag ai">🤖 {n_ai_str} AI skills</span>'
        for kw in ["notice", "GitHub", "recruiter response"]:
            if kw in reasoning:
                snip = [p.strip() for p in reasoning.split(";") if kw in p]
                if snip:
                    tags += f'<span class="tag">{snip[0]}</span>'

        card_html = f"""
<div class="{card_cls}">
  <div class="card-header">
    <div class="rank-badge {bc}">#{rnk}</div>
    <div class="cand-id">{row['candidate_id']}</div>
    <div class="score-chip">{row['score']:.4f}</div>
  </div>
  <div class="card-body">{reasoning}</div>
  <div class="card-tags">{tags}</div>
</div>"""

        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            st.markdown(card_html, unsafe_allow_html=True)

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("🚀  Upload both files above and click **Run AI Ranking Pipeline** to begin.")
