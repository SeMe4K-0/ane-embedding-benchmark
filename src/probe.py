from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .backends import make_embedder, make_inputs
from .models import REGISTRY, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent

# Readiness check run before a full session. A short run of one light,
# temperature-sensitive configuration is compared with the reference session;
# scripts/run_session.sh starts the grid only when the ratio is >= 0.95.


def reference(config_id: str, session: str = "s1") -> float | None:
    path = ROOT / "results" / "raw" / session / "bench_power.jsonl"
    if not path.exists():
        path = ROOT / "results" / "raw" / session / "bench.jsonl"
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("event") == "measure" and r["config_id"] == config_id:
            return r["throughput_emb_s"]
    return None


def probe(model: str, backend: str, batch: int, length: int,
          seconds: float = 8.0) -> float:
    tok = load_tokenizer(REGISTRY[model])
    ids, mask = make_inputs(tok, batch, length, seed=0)
    emb = make_embedder(backend, model, batch, length)
    for _ in range(3):
        emb.encode(ids, mask)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        emb.encode(ids, mask)
        n += 1
    thr = n * batch / (time.perf_counter() - t0)
    emb.close()
    return thr


def main() -> None:
    ap = argparse.ArgumentParser(description="Throughput probe against a reference session")
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--backend", default="pt-mps")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--ref-session", default="s1")
    args = ap.parse_args()

    cid = f"{args.model}|{args.backend}|B{args.batch}|L{args.length}"
    ref = reference(cid, args.ref_session)
    thr = probe(args.model, args.backend, args.batch, args.length, args.seconds)
    if ref:
        print(f"{time.strftime('%H:%M')}  {thr:8.1f} emb/s  "
              f"reference {ref:8.1f}  ratio {thr / ref:.3f}")
        print(f"RATIO {thr / ref:.4f}")
    else:
        print(f"{time.strftime('%H:%M')}  {thr:8.1f} emb/s  (no reference)")
        print("RATIO nan")


if __name__ == "__main__":
    main()
