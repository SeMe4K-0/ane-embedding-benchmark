from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .backends import BACKENDS, make_embedder, make_inputs
from .models import REGISTRY, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


@dataclass(frozen=True)
class Config:
    model: str
    backend: str
    batch: int
    length: int

    @property
    def cid(self) -> str:
        return f"{self.model}|{self.backend}|B{self.batch}|L{self.length}"


def thermal_pressure() -> str:
    # On Apple silicon `pmset -g therm` usually reports nothing useful; the
    # thermal sampler in the powermetrics log is the main source.
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            if "pressure" in line.lower():
                return line.strip()
        return "n/a"
    except Exception:  # noqa: BLE001
        return "unknown"


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return -1.0


def env_meta() -> dict:
    import coremltools as ct
    import torch
    import transformers
    return {
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def expand(cfg: dict) -> list[Config]:
    return [Config(m, b, B, L)
            for m in cfg["models"]
            for b in cfg["backends"]
            for L in cfg["lengths"]
            for B in cfg["batches"]]


def measure_one(c: Config, warmup: int, repeats: int,
                seed: int, min_window_s: float = 6.0) -> dict:
    tok = load_tokenizer(REGISTRY[c.model])
    ids, mask = make_inputs(tok, c.batch, c.length, seed=seed)

    t_load0 = time.perf_counter()
    emb = make_embedder(c.backend, c.model, c.batch, c.length)
    load_s = time.perf_counter() - t_load0

    for _ in range(warmup):
        emb.encode(ids, mask)

    # The loop runs for at least min_window_s: powermetrics samples every
    # 200 ms with one-second timestamps, so shorter windows cannot be
    # matched to power samples reliably.
    lat: list[float] = []
    t_start = time.time()
    t0 = time.perf_counter()
    n_iter = 0
    while True:
        a = time.perf_counter()
        emb.encode(ids, mask)
        lat.append((time.perf_counter() - a) * 1000.0)
        n_iter += 1
        elapsed = time.perf_counter() - t0
        if n_iter >= repeats and elapsed >= min_window_s:
            break
        if elapsed > min_window_s * 10:
            break
    wall = time.perf_counter() - t0
    t_end = time.time()

    out = emb.encode(ids, mask)
    emb.close()

    arr = np.array(lat)
    n_emb = c.batch * n_iter
    return {
        "event": "measure",
        "config_id": c.cid,
        "model": c.model, "backend": c.backend,
        "batch": c.batch, "length": c.length,
        "precision": BACKENDS[c.backend].get("precision"),
        "t_start": t_start, "t_end": t_end,
        "wall_s": wall,
        "repeats": n_iter, "warmup": warmup,
        "n_embeddings": n_emb,
        "lat_ms_p50": float(np.percentile(arr, 50)),
        "lat_ms_p95": float(np.percentile(arr, 95)),
        "lat_ms_mean": float(arr.mean()),
        "lat_ms_std": float(arr.std()),
        "lat_ms_min": float(arr.min()),
        "throughput_emb_s": n_emb / wall,
        "load_seconds": load_s,
        "package_mb": getattr(getattr(emb, "info", None), "package_mb", None),
        "rss_mb": rss_mb(),
        "out_dim": int(out.shape[-1]),
        "out_norm_mean": float(np.linalg.norm(out, axis=-1).mean()),
    }


def run(config_path: Path, session: str, seed: int, baseline_s: int,
        cooldown_s: int, limit: int | None) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configs = expand(cfg)
    random.Random(seed).shuffle(configs)
    if limit:
        configs = configs[:limit]

    out_dir = RAW / session
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bench.jsonl"

    with out_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "session", "session": session, "seed": seed,
                             "config": config_path.name, "env": env_meta(),
                             "n_configs": len(configs),
                             "t": time.time()}) + "\n")
        fh.flush()

        # Idle period at the start of the session; src.power uses it as the
        # per-domain idle baseline.
        if baseline_s > 0:
            print(f"idle baseline {baseline_s} s ...", flush=True)
            b0 = time.time()
            time.sleep(baseline_s)
            fh.write(json.dumps({"event": "baseline", "t_start": b0,
                                 "t_end": time.time()}) + "\n")
            fh.flush()

        for i, c in enumerate(configs, 1):
            th_before = thermal_pressure()
            try:
                rec = measure_one(c, cfg.get("warmup", {}).get(str(c.batch), 10),
                                  cfg.get("repeats", {}).get(str(c.batch), 30), seed,
                                  float(cfg.get("min_window_s", 6.0)))
                rec["thermal_before"] = th_before
                rec["thermal_after"] = thermal_pressure()
                print(f"[{i}/{len(configs)}] {c.cid:<40} "
                      f"p50 {rec['lat_ms_p50']:8.2f} ms  "
                      f"{rec['throughput_emb_s']:9.1f} emb/s", flush=True)
            except Exception as exc:  # noqa: BLE001
                rec = {"event": "error", "config_id": c.cid, "model": c.model,
                       "backend": c.backend, "batch": c.batch, "length": c.length,
                       "error": f"{type(exc).__name__}: {exc}", "t": time.time()}
                print(f"[{i}/{len(configs)}] {c.cid:<40} FAIL {rec['error'][:60]}",
                      flush=True)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if cooldown_s:
                time.sleep(cooldown_s)

        fh.write(json.dumps({"event": "session_end", "t": time.time()}) + "\n")

    print(f"\n-> {out_file.relative_to(ROOT)}")
    return out_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the measurement grid")
    ap.add_argument("--config", required=True)
    ap.add_argument("--session", default="dbg")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--baseline", type=int, default=60,
                    help="idle seconds before the grid (0 to skip)")
    ap.add_argument("--cooldown", type=int, default=30,
                    help="pause between configurations, seconds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    if args.dry_run:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cs = expand(cfg)
        print(f"{len(cs)} configurations:")
        for c in cs:
            print("  ", c.cid)
        return
    run(path, args.session, args.seed, args.baseline, args.cooldown, args.limit)


if __name__ == "__main__":
    main()
