"""
benchmark_embed.py -- Test all useful ONNX model variants + PyTorch baseline.

Available ONNX variants for all-MiniLM-L6-v2:
  model.onnx              - base (unoptimized, slower than PyTorch)
  model_O4.onnx           - fully optimized FP32
  model_quint8_avx2.onnx  - UINT8 quantized, requires AVX2 (most x86_64 CPUs since 2013)
  model_qint8_avx512.onnx - INT8 quantized, requires AVX-512 (Intel Skylake-X+, AMD Zen4)
  model_qint8_avx512_vnni - INT8 quantized VNNI (newest Intel/AMD)

Run: python -X utf8 benchmark_embed.py
"""
from __future__ import annotations
import sys, io, time, itertools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def check_cpu_features() -> set[str]:
    """Detect available CPU SIMD features."""
    features: set[str] = set()
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-c",
             "import numpy; print(numpy.__config__.blas_opt_info.get('extra_compile_args', []))"],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        pass

    # Try cpuinfo if available
    try:
        import cpuinfo
        flags = cpuinfo.get_cpu_info().get("flags", [])
        features = {f.lower() for f in flags}
    except ImportError:
        # Fallback: conservative assumption (AVX2 is universal on modern x86_64)
        features = {"avx2"}

    return features


def encode_variant(model_name: str, file_name: str | None, texts: list[str],
                   batch_size: int = 256, label: str = "") -> tuple[float, str]:
    """Load an ONNX variant, warm up, then encode and return cand/sec."""
    from sentence_transformers import SentenceTransformer

    kwargs: dict = {"device": "cpu"}
    if file_name:
        kwargs["backend"] = "onnx"
        kwargs["model_kwargs"] = {"file_name": file_name}

    try:
        t0 = time.perf_counter()
        model = SentenceTransformer(model_name, **kwargs)
        load_t = time.perf_counter() - t0

        # Warm-up pass (important for ONNX graph init)
        model.encode(["warm-up text for graph compilation"], show_progress_bar=False)

        t1 = time.perf_counter()
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
        enc_t = time.perf_counter() - t1
        rate = len(texts) / enc_t
        print(f"  {label:35s}: {rate:5.0f} cand/sec | load={load_t:.1f}s encode={enc_t:.1f}s")
        return rate, "OK"
    except Exception as e:
        short_err = str(e)[:80]
        print(f"  {label:35s}: FAILED -- {short_err}")
        return 0.0, f"FAILED: {short_err}"


def main() -> None:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from backend.utils.text_utils import build_profile_text, get_candidate_id, stream_jsonl

    DATA = Path("data/candidates.jsonl")
    N, BATCH = 2000, 256
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_LOCAL = "all-MiniLM-L6-v2"

    print(f"\n=== ONNX Variant Benchmark (N={N}, batch={BATCH}) ===\n")

    # Ingest
    texts = []
    for c in itertools.islice(stream_jsonl(DATA), N):
        texts.append(build_profile_text(c))
    print(f"[ingest] {len(texts)} texts ready\n")

    # CPU capabilities check
    features = check_cpu_features()
    avx2    = "avx2"    in features or True   # safe default for modern x86_64
    avx512  = "avx512f" in features
    vnni    = "avx512_vnni" in features or "avxvnni" in features
    print(f"[cpu]   avx2={avx2} | avx512={avx512} | vnni={vnni}\n")

    results: dict[str, float] = {}

    # 1. PyTorch baseline
    print("[Variant 1] PyTorch (baseline)")
    rate, _ = encode_variant(MODEL_LOCAL, None, texts, BATCH, "pytorch (baseline)")
    results["pytorch"] = rate

    # 2. ONNX O4 (fully optimized FP32)
    print("\n[Variant 2] ONNX O4 (fully optimized FP32)")
    rate, _ = encode_variant(MODEL, "onnx/model_O4.onnx", texts, BATCH, "onnx/model_O4")
    if rate > 0:
        results["onnx_O4"] = rate

    # 3. ONNX quint8 AVX2 (recommended for most x86_64)
    print("\n[Variant 3] ONNX quint8 + AVX2 (most compatible quantized)")
    rate, _ = encode_variant(MODEL, "onnx/model_quint8_avx2.onnx", texts, BATCH, "onnx/model_quint8_avx2")
    if rate > 0:
        results["onnx_quint8_avx2"] = rate

    # 4. ONNX qint8 AVX-512 (if supported)
    print("\n[Variant 4] ONNX qint8 + AVX-512")
    rate, _ = encode_variant(MODEL, "onnx/model_qint8_avx512.onnx", texts, BATCH, "onnx/model_qint8_avx512")
    if rate > 0:
        results["onnx_qint8_avx512"] = rate

    # 5. ONNX qint8 AVX-512 VNNI (newest/fastest if supported)
    print("\n[Variant 5] ONNX qint8 + AVX-512 VNNI")
    rate, _ = encode_variant(MODEL, "onnx/model_qint8_avx512_vnni.onnx", texts, BATCH, "onnx/model_qint8_avx512_vnni")
    if rate > 0:
        results["onnx_qint8_avx512_vnni"] = rate

    # Summary
    print("\n" + "="*60)
    print("RESULTS (sorted fastest first)")
    print("="*60)
    for name, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        est = (100_000 / rate) / 60
        speedup = rate / results.get("pytorch", rate)
        marker = " <-- BEST" if name == max(results, key=results.get) else ""
        print(f"  {name:30s}: {rate:5.0f} cand/sec | 100k~{est:.0f}min | {speedup:.2f}x{marker}")

    best = max(results, key=results.get)
    best_rate = results[best]
    best_est = (100_000 / best_rate) / 60
    print(f"\n  => Use backend: {best} ({best_rate:.0f} cand/sec, ~{best_est:.0f} min for 100k)")
    print("\n=== Done ===\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
