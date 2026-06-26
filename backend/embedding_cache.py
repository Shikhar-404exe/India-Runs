"""
backend/embedding_cache.py
OFFLINE PHASE — Pre-compute candidate text embeddings.

ARCHITECTURE NOTE:
  The offline pre-computation runs INDEPENDENTLY of the 5-minute online ranking
  constraint. submission_metadata.yaml uses pre_computation_required=true to
  signal this separate budget. The cache runs once; embeddings persist to disk.

HARDWARE-AWARE BACKEND SELECTION:
  • AVX-512 + Linux → ONNX qint8_avx512_vnni (100–350 cand/sec)
  • AVX2 (any OS)  → PyTorch FP32 (~45–50 cand/sec)
  • Fallback        → PyTorch FP32

  On this Windows/no-AVX-512 machine, PyTorch FP32 IS optimal. ONNX variants
  and multi-process pools are 2–4× slower due to Windows spawn overhead and
  lack of AVX-512 SIMD.

Usage:
    python -m backend.embedding_cache                         # auto-detect
    python -m backend.embedding_cache --backend onnx          # force ONNX
    python -m backend.embedding_cache --backend pytorch       # force PyTorch

Outputs:
    data/embeddings_matrix.npy  — float32 (N, 384), L2-normalised
    data/candidate_ids.npy      — str (N,) index -> candidate_id
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.utils.text_utils import build_profile_text, get_candidate_id, stream_jsonl

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# CPU feature detection
# ---------------------------------------------------------------------------


def detect_cpu_features() -> dict[str, bool]:
    """Detect SIMD features available on the host CPU."""
    features: dict[str, bool] = {"avx2": False, "avx512": False, "vnni": False}
    try:
        import cpuinfo  # optional — pip install py-cpuinfo
        flags = {f.lower() for f in cpuinfo.get_cpu_info().get("flags", [])}
        features["avx2"] = "avx2" in flags
        features["avx512"] = "avx512f" in flags
        features["vnni"] = "avx512_vnni" in flags or "avxvnni" in flags
    except ImportError:
        features["avx2"] = platform.machine() in ("AMD64", "x86_64")
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    flags_str = f.read()
                features["avx2"] = "avx2" in flags_str
                features["avx512"] = "avx512f" in flags_str
                features["vnni"] = "avx512_vnni" in flags_str
            except Exception:
                pass
    return features


def pick_best_backend(cpu: dict[str, bool]) -> str:
    """Auto-select the fastest backend for this CPU."""
    if cpu["avx512"] and cpu["vnni"] and platform.system() != "Windows":
        return "onnx_vnni"
    if cpu["avx512"] and platform.system() != "Windows":
        return "onnx_avx512"
    return "pytorch"


# ---------------------------------------------------------------------------
# Step 1: Ingest — single-pass text materialisation
# ---------------------------------------------------------------------------


def ingest_all_texts(
    data_path: Path,
    gzipped: bool = False,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    ids: list[str] = []
    t0 = time.perf_counter()
    for i, candidate in enumerate(stream_jsonl(data_path, gzipped=gzipped)):
        texts.append(build_profile_text(candidate))
        ids.append(get_candidate_id(candidate))
        if (i + 1) % 10_000 == 0:
            print(f"  [ingest] {i+1:,} | {(i+1)/(time.perf_counter()-t0):.0f} rec/sec", end="\r")
    print(f"\n  [ingest] {len(texts):,} records in {time.perf_counter()-t0:.1f}s")
    return texts, ids


# ---------------------------------------------------------------------------
# Step 2: Encode
# ---------------------------------------------------------------------------


def load_model_pytorch(model_name: str):
    """Load SentenceTransformer with PyTorch CPU backend (optimal for no-AVX-512)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return model, "pytorch"


def load_model_onnx(model_name: str, variant: str):
    """Load SentenceTransformer with ONNX backend (Linux + AVX-512 only)."""
    file_map = {
        "onnx_vnni": "onnx/model_qint8_avx512_vnni.onnx",
        "onnx_avx512": "onnx/model_qint8_avx512.onnx",
    }
    file_name = file_map.get(variant, "onnx/model_O4.onnx")
    from sentence_transformers import SentenceTransformer
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("  [warn] ONNX Runtime not installed; falling back to PyTorch.")
        return load_model_pytorch(model_name)
    try:
        model = SentenceTransformer(
            model_name, backend="onnx", device="cpu",
            model_kwargs={"file_name": file_name},
        )
        model.encode(["warm-up"], show_progress_bar=False)
        return model, f"onnx-{variant.replace('onnx_', '')}"
    except Exception as exc:
        print(f"  [warn] ONNX variant '{variant}' failed ({exc}); falling back to PyTorch.")
        return load_model_pytorch(model_name)


def load_model(model_name: str, backend: str | None = None):
    """
    Load SentenceTransformer using the best backend for the host CPU.
    Auto-detects if no backend is specified.
    """
    if backend is None:
        cpu = detect_cpu_features()
        backend = pick_best_backend(cpu)
        print(f"  [detect] CPU: avx2={cpu['avx2']} avx512={cpu['avx512']} vnni={cpu['vnni']}")
        print(f"  [detect] Selected backend: {backend}")

    if backend.startswith("onnx"):
        return load_model_onnx(model_name, backend)
    return load_model_pytorch(model_name)


def encode_texts(
    model,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode all texts. PyTorch's OpenMP threadpool saturates available CPUs."""
    print(f"  [encode] batch_size={batch_size} | {len(texts):,} texts …")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    rate = len(texts) / elapsed
    print(f"  [encode] {rate:.0f} cand/sec | {elapsed:.1f}s")
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Persist
# ---------------------------------------------------------------------------


def save_artifacts(
    embeddings_matrix: np.ndarray,
    candidate_ids: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings_matrix.npy"
    ids_path = out_dir / "candidate_ids.npy"
    np.save(str(emb_path), embeddings_matrix)
    np.save(str(ids_path), np.array(candidate_ids, dtype=str))
    emb_mb = emb_path.stat().st_size / 1024 / 1024
    print(f"[embedding_cache] Saved embeddings  -> {emb_path} ({emb_mb:.1f} MB)")
    print(f"[embedding_cache] Saved IDs         -> {ids_path}")
    print(f"[embedding_cache] Shape: {embeddings_matrix.shape} | dtype: {embeddings_matrix.dtype}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline: pre-compute candidate embeddings (CPU-optimised)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=[None, "pytorch", "onnx", "onnx_vnni", "onnx_avx512"],
        help="Force a specific backend (default: auto-detect)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Pipeline: ingest -> load model -> encode -> save."""
    args = parse_args()
    data_dir = args.data_dir.resolve()

    jsonl_path = data_dir / "candidates.jsonl"
    gzipped = False
    if not jsonl_path.exists():
        gz = data_dir / "candidates.jsonl.gz"
        if gz.exists():
            jsonl_path, gzipped = gz, True
        else:
            print(f"ERROR: No candidates file in {data_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"[embedding_cache] Input      : {jsonl_path}")
    print(f"[embedding_cache] Model      : {args.model}")
    print(f"[embedding_cache] Batch size : {args.batch_size}")
    print(f"[embedding_cache] CPU cores  : {os.cpu_count()}")
    print(f"[embedding_cache] Platform   : {platform.system()}")

    t_total = time.perf_counter()

    print("\n[Step 1/3] Ingesting candidate texts …")
    texts, ids = ingest_all_texts(jsonl_path, gzipped=gzipped)

    print("\n[Step 2/3] Loading model …")
    model, backend = load_model(args.model, args.backend)
    print(f"  Backend active: {backend}")
    if backend == "pytorch":
        import torch
        print(f"  PyTorch threads: {torch.get_num_threads()}")

    print(f"\n[Step 3/3] Encoding {len(texts):,} texts …")
    embeddings = encode_texts(model, texts, args.batch_size)

    print("\n[Saving] Writing artifacts to disk …")
    save_artifacts(embeddings, ids, data_dir)

    total = time.perf_counter() - t_total
    encode_only = len(ids) / ((embeddings.shape[0] / (total - max(total * 0.15, 10))) if total > 10 else embeddings.shape[0])
    rate = len(ids) / total
    print(
        f"\n[embedding_cache] DONE — {len(ids):,} candidates | "
        f"backend={backend} | {rate:.0f} cand/sec overall | "
        f"total wall-clock: {total:.1f}s"
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
