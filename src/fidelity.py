from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .backends import make_embedder
from .models import REGISTRY, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed"

# Each backend is compared with PyTorch CPU fp32. Retrieval metrics are
# computed here directly on BEIR SciFact (mteb/scifact on Hugging Face).
REFERENCE = "pt-cpu"


def load_scifact(max_docs: int | None = None):
    """Return (doc_ids, doc_texts, query_ids, query_texts, qrels) for the test split."""
    from datasets import load_dataset

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = (load_dataset("mteb/scifact-qrels", split="test")
                if _has("mteb/scifact-qrels")
                else load_dataset("mteb/scifact", "default", split="test"))

    doc_ids = [str(x) for x in corpus["_id"]]
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(doc_ids)
    docs = [f"{t} {b}".strip() for t, b in zip(titles, corpus["text"])]
    if max_docs:
        doc_ids, docs = doc_ids[:max_docs], docs[:max_docs]

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    qmap = {str(i): t for i, t in zip(queries["_id"], queries["text"])}
    q_ids = [q for q in qrels if q in qmap]
    return doc_ids, docs, q_ids, [qmap[q] for q in q_ids], qrels


def _has(name: str) -> bool:
    try:
        from huggingface_hub import dataset_info
        dataset_info(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def encode_all(backend: str, model_key: str, texts: list[str], length: int,
               batch: int = 32) -> np.ndarray:
    """Encode texts in fixed (batch, length) shapes; the last batch is padded."""
    tok = load_tokenizer(REGISTRY[model_key])
    emb = make_embedder(backend, model_key, batch, length)
    vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk + [""] * (batch - len(chunk)), padding="max_length",
                  truncation=True, max_length=length, return_tensors="np")
        out = emb.encode(enc["input_ids"].astype(np.int32),
                         enc["attention_mask"].astype(np.int32))
        vecs.append(out[:len(chunk)])
    emb.close()
    return np.concatenate(vecs, axis=0).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # fp16 outputs are not exactly unit length; renormalize so that the
    # row-wise dot product does not exceed 1.
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 100) -> float:
    pos = {d for d, r in rel.items() if r > 0}
    if not pos:
        return 0.0
    return len(pos & set(ranked[:k])) / len(pos)


def retrieve(q_vecs: np.ndarray, d_vecs: np.ndarray, doc_ids: list[str],
             k: int = 100) -> list[list[str]]:
    sims = q_vecs @ d_vecs.T          # vectors are L2-normalized
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    out = []
    for r, row in enumerate(idx):
        order = row[np.argsort(-sims[r, row])]
        out.append([doc_ids[i] for i in order])
    return out


def evaluate(model_key: str, backends: list[str], length: int = 512,
             batch: int = 32, max_docs: int | None = None
             ) -> tuple[list[dict], list[dict]]:
    """Return summary rows and per-query rows for one model."""
    doc_ids, docs, q_ids, q_texts, qrels = load_scifact(max_docs)
    print(f"{model_key}: {len(docs)} documents, {len(q_texts)} queries", flush=True)

    rows: list[dict] = []
    per_query: list[dict] = []
    ref_d = ref_ranked = None
    for backend in [REFERENCE] + [b for b in backends if b != REFERENCE]:
        d_vecs = encode_all(backend, model_key, docs, length, batch)
        q_vecs = encode_all(backend, model_key, q_texts, length, batch)
        ranked = retrieve(q_vecs, d_vecs, doc_ids)
        q_ndcg = [ndcg_at_k(r, qrels.get(q, {}), 10) for r, q in zip(ranked, q_ids)]
        q_rec = [recall_at_k(r, qrels.get(q, {}), 100) for r, q in zip(ranked, q_ids)]
        ndcg, rec = float(np.mean(q_ndcg)), float(np.mean(q_rec))
        row = {"model": model_key, "backend": backend, "length": length,
               "ndcg@10": ndcg, "recall@100": rec, "n_docs": len(docs),
               "n_queries": len(q_texts)}
        if backend == REFERENCE:
            ref_d, ref_ranked = d_vecs, ranked
            q_overlap = [1.0] * len(q_ids)
            row.update({"cos_mean": 1.0, "cos_p1": 1.0, "cos_min": 1.0,
                        "overlap@10": 1.0, "d_ndcg@10": 0.0})
        else:
            cos = _cosine(d_vecs, ref_d)
            q_overlap = [len(set(a[:10]) & set(b[:10])) / 10
                         for a, b in zip(ranked, ref_ranked)]
            row.update({
                "cos_mean": float(cos.mean()),
                "cos_p1": float(np.percentile(cos, 1)),
                "cos_min": float(cos.min()),
                "overlap@10": float(np.mean(q_overlap)),
                "d_ndcg@10": ndcg - rows[0]["ndcg@10"],
            })
        rows.append(row)
        per_query += [{"model": model_key, "backend": backend, "query_id": q,
                       "ndcg@10": n, "recall@100": r, "overlap@10": o}
                      for q, n, r, o in zip(q_ids, q_ndcg, q_rec, q_overlap)]
        print(f"  {backend:<10} nDCG@10 {ndcg:.4f}  R@100 {rec:.4f}  "
              f"cos {row['cos_mean']:.6f}  overlap@10 {row['overlap@10']:.3f}",
              flush=True)
    return rows, per_query


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding fidelity and retrieval quality")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backends", nargs="+",
                    default=["pt-cpu", "pt-mps", "cm-cpu", "cm-gpu", "cm-ane"])
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    per_query: list[dict] = []
    for m in args.models:
        try:
            r, pq = evaluate(m, args.backends, args.length, args.batch, args.max_docs)
            rows += r
            per_query += pq
        except Exception as exc:  # noqa: BLE001
            print(f"{m}: FAIL {type(exc).__name__}: {exc}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "fidelity.csv", index=False)
        pd.DataFrame(per_query).to_csv(OUT / "fidelity_per_query.csv", index=False)
        print(f"\n-> {(OUT / 'fidelity.csv').relative_to(ROOT)}, fidelity_per_query.csv")


if __name__ == "__main__":
    main()
