from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
TRACES = ROOT / "results" / "power_traces"

# Log recorded by scripts/run_session.sh:
#   sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 200
# Header line of each sample:
#   *** Sampled system activity (Sat Sep  6 01:23:45 2026 +0300) (203.11ms elapsed) ***
HEADER_RE = re.compile(
    r"\*\*\*\s*Sampled system activity\s*\((?P<date>[^)]+?)\)\s*"
    r"\((?P<elapsed>[\d.]+)ms elapsed\)")
POWER_RE = re.compile(
    r"^(?P<name>CPU|GPU|ANE|Combined|Package|DRAM)\s+Power:\s*(?P<val>[\d.]+)\s*(?P<unit>mW|W)",
    re.IGNORECASE)
THERMAL_RE = re.compile(r"Current pressure level:\s*(?P<level>\w+)")


@dataclass
class Sample:
    t_epoch: float       # end of the sampling interval
    elapsed_s: float
    cpu_w: float = 0.0
    gpu_w: float = 0.0
    ane_w: float = 0.0
    thermal: str = ""

    @property
    def t_start(self) -> float:
        return self.t_epoch - self.elapsed_s


def _parse_date(text: str) -> float | None:
    text = " ".join(text.split())
    for fmt in ("%a %b %d %H:%M:%S %Y %z", "%a %b %d %H:%M:%S %Y",
                "%Y-%m-%d %H:%M:%S %z"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.astimezone()
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _rebuild_timeline(samples: list[Sample]) -> list[Sample]:
    """Recover sub-second sample times.

    powermetrics stamps samples to the second, so at a 200 ms interval several
    consecutive samples share one stamp. Summing the reported elapsed times
    does not work: elapsed excludes formatting and write time, and on a
    34k-sample log the accumulated drift reached 498 s. The one-second stamps
    do not drift, so samples sharing a stamp are spread evenly over that
    second. The error is then bounded by one second for any log length.
    """
    groups: dict[float, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.t_epoch, []).append(s)
    for stamp, group in groups.items():
        k = len(group)
        for j, s in enumerate(group):
            s.t_epoch = stamp + (j + 1) / k
    return samples


def parse_log(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    cur: Sample | None = None
    with path.open(errors="ignore") as fh:
        for line in fh:
            m = HEADER_RE.search(line)
            if m:
                if cur is not None:
                    samples.append(cur)
                t = _parse_date(m.group("date"))
                cur = Sample(t_epoch=t if t is not None else 0.0,
                             elapsed_s=float(m.group("elapsed")) / 1000.0)
                continue
            if cur is None:
                continue
            tm = THERMAL_RE.search(line)
            if tm:
                cur.thermal = tm.group("level")
                continue
            pm = POWER_RE.match(line.strip())
            if pm:
                val = float(pm.group("val"))
                if pm.group("unit").lower() == "mw":
                    val /= 1000.0
                name = pm.group("name").upper()
                if name == "CPU":
                    cur.cpu_w = val
                elif name == "GPU":
                    cur.gpu_w = val
                elif name == "ANE":
                    cur.ane_w = val
    if cur is not None:
        samples.append(cur)
    return _rebuild_timeline([s for s in samples if s.t_epoch > 0])


def write_trace(samples: list[Sample], path: Path) -> None:
    """Parsed samples as gzipped CSV; enough to recompute every energy value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_end", "elapsed_s", "cpu_w", "gpu_w", "ane_w", "thermal"])
        for s in samples:
            w.writerow([f"{s.t_epoch:.4f}", f"{s.elapsed_s:.5f}", f"{s.cpu_w:.3f}",
                        f"{s.gpu_w:.3f}", f"{s.ane_w:.3f}", s.thermal])


def read_trace(path: Path) -> list[Sample]:
    with gzip.open(path, "rt", newline="") as fh:
        return [Sample(float(r["t_end"]), float(r["elapsed_s"]), float(r["cpu_w"]),
                       float(r["gpu_w"]), float(r["ane_w"]), r["thermal"])
                for r in csv.DictReader(fh)]


def window_energy(samples: list[Sample], t0: float, t1: float) -> dict:
    """Gross energy (J) per domain in [t0, t1]; edge samples count by overlap."""
    e = {"cpu_j": 0.0, "gpu_j": 0.0, "ane_j": 0.0}
    covered = 0.0
    for s in samples:
        a, b = max(s.t_start, t0), min(s.t_epoch, t1)
        if b <= a:
            continue
        dt = b - a
        covered += dt
        e["cpu_j"] += s.cpu_w * dt
        e["gpu_j"] += s.gpu_w * dt
        e["ane_j"] += s.ane_w * dt
    e["covered_s"] = covered
    e["window_s"] = t1 - t0
    e["coverage"] = covered / (t1 - t0) if t1 > t0 else 0.0
    e["total_j"] = e["cpu_j"] + e["gpu_j"] + e["ane_j"]
    levels = {s.thermal for s in samples
              if s.thermal and min(s.t_epoch, t1) > max(s.t_start, t0)}
    e["thermal_levels"] = ",".join(sorted(levels)) if levels else ""
    e["thermal_nominal"] = (levels == {"Nominal"}) if levels else None
    return e


def baseline_power(samples: list[Sample], t0: float, t1: float) -> dict:
    """Mean power per domain over samples fully inside the idle period."""
    sel = [s for s in samples if s.t_start >= t0 and s.t_epoch <= t1]
    if not sel:
        return {"cpu_w": 0.0, "gpu_w": 0.0, "ane_w": 0.0, "n": 0}
    n = len(sel)
    return {"cpu_w": sum(s.cpu_w for s in sel) / n,
            "gpu_w": sum(s.gpu_w for s in sel) / n,
            "ane_w": sum(s.ane_w for s in sel) / n,
            "n": n}


def load_samples(session: str) -> list[Sample]:
    log = RAW / f"power_{session}.log"
    if log.exists():
        return parse_log(log)
    trace = TRACES / f"power_{session}.csv.gz"
    if trace.exists():
        return read_trace(trace)
    raise SystemExit(f"no power data for {session}: expected {log} or {trace}")


def attach(session: str) -> Path:
    """Add power and energy to every measurement of a session.

    Net energy per domain is max(gross - idle_power * duration, 0); the three
    domains are summed. Writes results/raw/<session>/bench_power.jsonl.
    """
    bench = RAW / session / "bench.jsonl"
    if not bench.exists():
        raise SystemExit(f"missing {bench}")
    samples = load_samples(session)
    if not samples:
        raise SystemExit(f"no power samples parsed for {session}")

    records = [json.loads(line) for line in bench.open(encoding="utf-8") if line.strip()]
    base_rec = next((r for r in records if r.get("event") == "baseline"), None)
    base = (baseline_power(samples, base_rec["t_start"], base_rec["t_end"])
            if base_rec else {"cpu_w": 0.0, "gpu_w": 0.0, "ane_w": 0.0, "n": 0})

    out_path = RAW / session / "bench_power.jsonl"
    n_ok = 0
    with out_path.open("w", encoding="utf-8") as out:
        for r in records:
            if r.get("event") != "measure":
                out.write(json.dumps(r) + "\n")
                continue
            gross = window_energy(samples, r["t_start"], r["t_end"])
            dur = max(r["t_end"] - r["t_start"], 1e-9)
            net = {
                "cpu_j_net": max(gross["cpu_j"] - base["cpu_w"] * dur, 0.0),
                "gpu_j_net": max(gross["gpu_j"] - base["gpu_w"] * dur, 0.0),
                "ane_j_net": max(gross["ane_j"] - base["ane_w"] * dur, 0.0),
            }
            net["total_j_net"] = sum(net.values())
            n_emb = r.get("n_embeddings") or 1
            r["power"] = {
                **gross, **net,
                "baseline_cpu_w": base["cpu_w"], "baseline_gpu_w": base["gpu_w"],
                "baseline_ane_w": base["ane_w"],
                "j_per_1k_emb": gross["total_j"] / n_emb * 1000,
                "j_per_1k_emb_net": net["total_j_net"] / n_emb * 1000,
                "avg_ane_w": gross["ane_j"] / dur,
            }
            if gross["coverage"] > 0.5:
                n_ok += 1
            out.write(json.dumps(r) + "\n")

    print(f"{len(samples)} power samples; idle CPU {base['cpu_w']:.2f} W, "
          f"GPU {base['gpu_w']:.2f} W, ANE {base['ane_w']:.3f} W (n={base['n']})")
    print(f"measurements with coverage > 50%: {n_ok}")
    print(f"-> {out_path.relative_to(ROOT)}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach powermetrics energy to measurements")
    ap.add_argument("--session", required=True)
    ap.add_argument("--export-trace", action="store_true",
                    help="write parsed samples to results/power_traces/ and exit")
    args = ap.parse_args()
    if args.export_trace:
        samples = parse_log(RAW / f"power_{args.session}.log")
        path = TRACES / f"power_{args.session}.csv.gz"
        write_trace(samples, path)
        print(f"{len(samples)} samples -> {path.relative_to(ROOT)}")
        return
    attach(args.session)


if __name__ == "__main__":
    main()
