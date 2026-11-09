from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .backends import make_embedder
from .models import REGISTRY, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"

# Indexing a real corpus with static-shape models. "naive" pads every
# document to 512 tokens; "bucketed" rounds each document up to the nearest
# bucket and keeps one compiled model per occupied bucket.
BUCKETS = [32, 64, 128, 256, 512]


def bucket_for(n_tokens: int, buckets: list[int]) -> int:
    for b in buckets:
        if n_tokens <= b:
            return b
    return buckets[-1]


def index_corpus(backend: str, model_key: str, texts: list[str], mode: str,
                 batch: int = 32, buckets: list[int] | None = None,
                 max_len: int = 512) -> dict:
    buckets = buckets or BUCKETS
    tok = load_tokenizer(REGISTRY[model_key])

    lens = [len(tok(t, truncation=True, max_length=max_len)["input_ids"])
            for t in texts]
    if mode == "naive":
        assign = [max_len] * len(texts)
    elif mode == "bucketed":
        assign = [bucket_for(n, buckets) for n in lens]
    else:
        raise ValueError(mode)

    groups: dict[int, list[int]] = {}
    for i, L in enumerate(assign):
        groups.setdefault(L, []).append(i)

    # Conversion, model loading, tokenization and warm-up happen before the
    # timer: the measurement is the cost of indexing, not of start-up.
    t_load0 = time.perf_counter()
    embs: dict[int, object] = {}
    batches: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for L, idxs in sorted(groups.items()):
        emb = make_embedder(backend, model_key, batch, L)
        embs[L] = emb
        for s in range(0, len(idxs), batch):
            chunk = [texts[i] for i in idxs[s:s + batch]]
            n_real = len(chunk)
            enc = tok(chunk + [""] * (batch - n_real), padding="max_length",
                      truncation=True, max_length=L, return_tensors="np")
            batches.append((L, enc["input_ids"].astype(np.int32),
                            enc["attention_mask"].astype(np.int32), n_real))
        for _ in range(2):
            emb.encode(batches[-1][1], batches[-1][2])
    setup_s = time.perf_counter() - t_load0

    t_start = time.time()
    t0 = time.perf_counter()
    n_done = 0
    padded_tokens = 0
    for L, ids_b, mask_b, n_real in batches:
        embs[L].encode(ids_b, mask_b)
        n_done += n_real
        padded_tokens += batch * L
    wall = time.perf_counter() - t0
    t_end = time.time()
    for emb in embs.values():
        emb.close()

    real_tokens = sum(lens)
    return {
        "event": "measure",
        "config_id": f"{model_key}|{backend}|{mode}",
        "model": model_key, "backend": backend, "mode": mode, "batch": batch,
        "length": max_len,
        "t_start": t_start, "t_end": t_end, "wall_s": wall,
        "n_embeddings": n_done,
        "throughput_emb_s": n_done / wall,
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "padding_overhead": padded_tokens / max(real_tokens, 1),
        "n_shapes": len(groups),
        "setup_seconds": setup_s,
        "shape_hist": {str(k): len(v) for k, v in sorted(groups.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus indexing: padding vs. length buckets")
    ap.add_argument("--models", nargs="+", default=["minilm-l6", "bge-base"])
    ap.add_argument("--backends", nargs="+",
                    default=["cm-ane", "cm-gpu", "pt-mps", "pt-cpu"])
    ap.add_argument("--session", default="r1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    from .fidelity import load_scifact
    _, docs, _, _, _ = load_scifact(args.max_docs)
    print(f"corpus: {len(docs)} documents", flush=True)

    out_dir = RAW / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bench.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "session", "session": args.session,
                             "kind": "realistic", "t": time.time()}) + "\n")
        b0 = time.time()
        time.sleep(30)
        fh.write(json.dumps({"event": "baseline", "t_start": b0,
                             "t_end": time.time()}) + "\n")
        fh.flush()
        for m in args.models:
            for b in args.backends:
                for mode in ("naive", "bucketed"):
                    try:
                        r = index_corpus(b, m, docs, mode, args.batch)
                        print(f"{m:<11} {b:<8} {mode:<9} "
                              f"{r['throughput_emb_s']:8.1f} emb/s  "
                              f"padding x{r['padding_overhead']:.2f}  "
                              f"{r['n_shapes']} shapes", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        r = {"event": "error", "model": m, "backend": b,
                             "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
                        print(f"{m:<11} {b:<8} {mode:<9} FAIL {exc}", flush=True)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    time.sleep(20)
    print(f"\n-> {out.relative_to(ROOT)}")
    print(f"next: python -m src.power --session {args.session}")


if __name__ == "__main__":
    main()
