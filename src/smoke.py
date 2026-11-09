from __future__ import annotations

import argparse
import time

import numpy as np

from .backends import make_embedder, make_inputs
from .fidelity import _cosine
from .models import REGISTRY, load_tokenizer
from .placement import analyze

# Quick check that the Hugging Face -> Core ML -> Neural Engine chain works
# on this machine before running a full session. Pass criteria: more than
# 80% of compute ops planned on the ANE and cosine to PyTorch CPU fp32 above
# 0.999. With --hold, the ANE path runs for N seconds so that ANE power can be
# checked in a powermetrics log.


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test of the Neural Engine path")
    ap.add_argument("--model", default="minilm-l6")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--hold", type=int, default=0,
                    help="seconds of continuous ANE load after the checks")
    args = ap.parse_args()

    key, B, L = args.model, args.batch, args.length
    spec = REGISTRY[key]
    print(f"model {key} ({spec.hf_id}), {spec.params_m}M params, B={B}, L={L}\n")

    ids, mask = make_inputs(load_tokenizer(spec), B, L, seed=0)

    pl = analyze(key, "CPU_AND_NE", B, L)
    share = pl["ane_share"] * 100
    print(f"placement: ANE {share:.1f}% ({pl['ane_ops']}/{pl['ops_total']}), "
          f"CPU {pl['cpu_ops']}, GPU {pl['gpu_ops']}")
    print(f"  fallback: {pl['top_fallback'][:100]}")

    ref_emb = make_embedder("pt-cpu", key, B, L)
    ref = ref_emb.encode(ids, mask)
    ref_emb.close()

    print("\nlatency and cosine to pt-cpu:")
    results = {}
    for backend in ["pt-cpu", "pt-mps", "cm-cpu", "cm-gpu", "cm-ane"]:
        try:
            emb = make_embedder(backend, key, B, L)
            for _ in range(args.warmup):
                emb.encode(ids, mask)
            lat = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                emb.encode(ids, mask)
                lat.append((time.perf_counter() - t0) * 1000)
            cos = float(_cosine(emb.encode(ids, mask), ref).mean())
            emb.close()
            p50 = float(np.percentile(lat, 50))
            results[backend] = (p50, cos)
            print(f"  {backend:<8} p50 {p50:8.3f} ms   cos {cos:.6f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {backend:<8} FAIL {type(exc).__name__}: {exc}")

    if args.hold:
        print(f"\nholding ANE load for {args.hold} s")
        emb = make_embedder("cm-ane", key, B, L)
        t0 = time.time()
        n = 0
        while time.time() < t0 + args.hold:
            emb.encode(ids, mask)
            n += 1
        emb.close()
        print(f"  {n} calls, window {t0:.3f} .. {time.time():.3f} (epoch)")

    cos_ane = results.get("cm-ane", (None, 0.0))[1]
    ok_share, ok_cos = share > 80, cos_ane > 0.999
    print(f"\nANE share > 80%:        {'OK' if ok_share else 'FAIL'} ({share:.1f}%)")
    print(f"cos(cm-ane) > 0.999:    {'OK' if ok_cos else 'FAIL'} ({cos_ane:.6f})")
    raise SystemExit(0 if ok_share and ok_cos else 1)


if __name__ == "__main__":
    main()
