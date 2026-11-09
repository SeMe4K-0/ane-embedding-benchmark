from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "figures"

BACKEND_ORDER = ["pt-cpu", "pt-mps", "cm-cpu", "cm-gpu", "cm-ane"]
BACKEND_LABEL = {"pt-cpu": "PyTorch CPU", "pt-mps": "PyTorch MPS",
                 "cm-cpu": "Core ML CPU", "cm-gpu": "Core ML GPU",
                 "cm-ane": "Core ML ANE"}
ENERGY_COLS = ["pw_j_per_1k_emb", "pw_j_per_1k_emb_net", "pw_cpu_j", "pw_gpu_j",
               "pw_ane_j", "pw_cpu_j_net", "pw_gpu_j_net", "pw_ane_j_net",
               "pw_total_j", "pw_total_j_net", "pw_avg_ane_w"]


def _read_jsonl(session: str) -> list[dict]:
    for name in ("bench_power.jsonl", "bench.jsonl"):
        path = RAW / session / name
        if path.exists():
            return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print(f"skip {session}: no data")
    return []


def load_sessions(sessions: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for s in sessions:
        for r in _read_jsonl(s):
            if r.get("event") != "measure":
                continue
            flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
            flat["session"] = s
            for k, v in (r.get("power") or {}).items():
                flat[f"pw_{k}"] = v
            rows.append(flat)
    if not rows:
        raise SystemExit("no measurements found")
    df = pd.DataFrame(rows)
    # A window with no power samples has zero energy by construction; treat
    # it as missing rather than as a real zero.
    if "pw_coverage" in df:
        no_power = df["pw_coverage"].fillna(0) == 0
        df.loc[no_power, [c for c in ENERGY_COLS if c in df]] = np.nan
    df["params_m"] = df["model"].map(lambda m: REGISTRY[m].params_m)
    df["backend_label"] = df["backend"].map(BACKEND_LABEL).fillna(df["backend"])
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Median over sessions per configuration, with min/max spread."""
    keys = ["model", "backend", "backend_label", "batch", "length", "params_m"]
    metrics = ["lat_ms_p50", "lat_ms_p95", "throughput_emb_s", "load_seconds",
               "package_mb", "rss_mb", "wall_s", "n_embeddings"]
    metrics += [c for c in ENERGY_COLS + ["pw_coverage"] if c in df]
    g = df.groupby(keys, dropna=False)
    out = g[metrics].median().reset_index()
    out["n_sessions"] = g.size().values
    for m in ("lat_ms_p50", "throughput_emb_s"):
        lo, hi = g[m].min().values, g[m].max().values
        out[f"{m}_min"], out[f"{m}_max"] = lo, hi
        out[f"{m}_spread_pct"] = np.where(out[m] > 0, (hi - lo) / out[m] * 100, np.nan)
    return out


def add_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    """Throughput and energy relative to PyTorch CPU for the same shape."""
    idx = ["model", "batch", "length"]
    ref = agg[agg.backend == "pt-cpu"].set_index(idx)
    joined = agg.join(ref[["throughput_emb_s", "pw_j_per_1k_emb_net"]],
                      on=idx, rsuffix="_ref")
    agg["speedup_vs_ptcpu"] = joined["throughput_emb_s"] / joined["throughput_emb_s_ref"]
    agg["energy_ratio_vs_ptcpu"] = (joined["pw_j_per_1k_emb_net"]
                                    / joined["pw_j_per_1k_emb_net_ref"])
    return agg


def _pivot(agg: pd.DataFrame, value: str, batch: int, length: int) -> pd.DataFrame:
    sel = agg[(agg.batch == batch) & (agg.length == length)]
    t = sel.pivot_table(index="model", columns="backend", values=value)
    order = sorted(t.index, key=lambda m: REGISTRY[m].params_m)
    return t.loc[order, [c for c in BACKEND_ORDER if c in t.columns]]


def table_speed(agg: pd.DataFrame, length: int) -> pd.DataFrame:
    return pd.concat({"latency_p50_ms_B1": _pivot(agg, "lat_ms_p50", 1, length),
                      "throughput_emb_s_B32": _pivot(agg, "throughput_emb_s", 32, length)},
                     axis=1).round(2)


def table_energy(agg: pd.DataFrame, length: int, batch: int = 32) -> pd.DataFrame:
    return _pivot(agg, "pw_j_per_1k_emb_net", batch, length).round(1)


def load_realistic(session: str = "r1") -> pd.DataFrame | None:
    rows = []
    for r in _read_jsonl(session):
        if r.get("event") != "measure":
            continue
        p = r.get("power") or {}
        rows.append({"model": r["model"], "backend": r["backend"], "mode": r["mode"],
                     "n_docs": r["n_embeddings"], "n_shapes": r["n_shapes"],
                     "wall_s": r["wall_s"], "throughput_emb_s": r["throughput_emb_s"],
                     "padding_overhead": r["padding_overhead"],
                     "setup_seconds": r["setup_seconds"],
                     "shape_hist": json.dumps(r["shape_hist"]),
                     "j_per_1k_net": p.get("j_per_1k_emb_net") if p.get("coverage") else np.nan})
    return pd.DataFrame(rows) if rows else None


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / f"{name}.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    return p


def figures(agg: pd.DataFrame, real: pd.DataFrame | None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: list[Path] = []

    # Fig. 1 of the paper: throughput vs. sequence length, batch 32.
    shown = ["minilm-l6", "bge-small", "bge-base"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), squeeze=False)
    for ax, m in zip(axes[0], shown):
        sub = agg[(agg.model == m) & (agg.batch == 32)]
        for b in BACKEND_ORDER:
            s = sub[sub.backend == b].sort_values("length")
            ax.plot(s.length, s.throughput_emb_s, marker="o", label=BACKEND_LABEL[b])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([32, 128, 512], ["32", "128", "512"])
        ax.set_xlabel("sequence length (tokens)")
        ax.set_title(f"{m} ({REGISTRY[m].params_m:.0f}M)")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("embeddings/s, batch 32")
    axes[0][-1].legend(fontsize=7)
    made.append(_save(fig, "fig1_throughput_vs_length"))
    plt.close(fig)

    # Fig. 2 of the paper: padding to 512 vs. length buckets.
    if real is not None and not real.empty:
        backends = [b for b in ["pt-cpu", "pt-mps", "cm-gpu", "cm-ane"]
                    if b in set(real.backend)]
        models = [m for m in ["minilm-l6", "bge-base"] if m in set(real.model)]
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.4),
                                 squeeze=False)
        xs = np.arange(len(backends))
        for ax, m in zip(axes[0], models):
            sub = real[real.model == m].set_index(["backend", "mode"])["throughput_emb_s"]
            naive = np.array([sub.get((b, "naive"), np.nan) for b in backends])
            buck = np.array([sub.get((b, "bucketed"), np.nan) for b in backends])
            ax.bar(xs - 0.2, naive, 0.4, label="pad to 512", color="#b0b6c3",
                   edgecolor="black", linewidth=0.6)
            ax.bar(xs + 0.2, buck, 0.4, label="length buckets", color="#2b6cb0",
                   edgecolor="black", linewidth=0.6)
            for x, n, b in zip(xs, naive, buck):
                ax.annotate(f"+{(b / n - 1) * 100:.0f}%", (x + 0.2, b),
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xs, [BACKEND_LABEL[b] for b in backends], rotation=15)
            ax.set_title(m)
            ax.grid(alpha=0.3, axis="y")
        axes[0][0].set_ylabel("documents/s")
        axes[0][0].legend(fontsize=8)
        made.append(_save(fig, "fig2_padding_vs_buckets"))
        plt.close(fig)

    # Energy per 1000 embeddings, batch 32, 128 tokens (not in the paper).
    t = _pivot(agg, "pw_j_per_1k_emb_net", 32, 128)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.16
    xs = np.arange(len(t.index))
    for i, b in enumerate(t.columns):
        ax.bar(xs + (i - 2) * w, t[b].values, w, label=BACKEND_LABEL[b])
    ax.set_xticks(xs, t.index, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("net J per 1000 embeddings")
    ax.set_title("Energy, batch 32, 128 tokens")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    made.append(_save(fig, "fig3_energy_B32_L128"))
    plt.close(fig)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate sessions into tables and figures")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--realistic", default="r1",
                    help="session with corpus-indexing runs ('' to skip)")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_sessions(args.sessions)
    df.to_csv(PROCESSED / "measurements.csv", index=False)
    agg = add_ratios(aggregate(df))
    agg.to_csv(PROCESSED / "summary.csv", index=False)
    n_missing = int(df["pw_j_per_1k_emb_net"].isna().sum()) if "pw_j_per_1k_emb_net" in df else len(df)
    print(f"{len(df)} measurements, {len(agg)} configurations, "
          f"{n_missing} without power data")

    t_speed = table_speed(agg, args.length)
    t_speed.to_csv(PROCESSED / f"table_speed_L{args.length}.csv")
    print(f"\nLatency (B=1) and throughput (B=32), L={args.length}\n{t_speed.to_string()}")

    t_energy = table_energy(agg, args.length)
    t_energy.to_csv(PROCESSED / f"table_energy_B32_L{args.length}.csv")
    print(f"\nNet J per 1000 embeddings, B=32, L={args.length}\n{t_energy.to_string()}")

    real = load_realistic(args.realistic) if args.realistic else None
    if real is not None:
        real.to_csv(PROCESSED / "realistic_indexing.csv", index=False)
        print("\nCorpus indexing\n" + real[["model", "backend", "mode", "throughput_emb_s",
                                            "padding_overhead", "n_shapes",
                                            "j_per_1k_net"]].round(2).to_string(index=False))

    if args.figures:
        for p in figures(agg, real):
            print(f"figure -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
