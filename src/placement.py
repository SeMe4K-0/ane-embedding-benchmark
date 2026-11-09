from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .convert import DEFAULT_TARGET, convert
from .models import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "processed" / "placement"


def _device_name(dev) -> str:
    if dev is None:
        return "unknown"
    cls = type(dev).__name__
    for tag, name in (("Neural", "ANE"), ("GPU", "GPU"), ("CPU", "CPU")):
        if tag.lower() in cls.lower():
            return name
    return cls


def analyze(model_key: str, cu_name: str = "CPU_AND_NE", batch: int = 1,
            length: int = 128, precision: str = "fp16",
            target: str = DEFAULT_TARGET) -> dict:
    """Planned device per operation from MLComputePlan.

    This is the compiler's plan, not a runtime trace. Writes one detail CSV
    per configuration and returns a summary row.
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    path = convert(REGISTRY[model_key], batch, length, precision, target=target)
    # MLComputePlan needs the compiled .mlmodelc. Keep the MLModel alive:
    # its temporary compiled directory is removed when it is collected.
    cu = getattr(ct.ComputeUnit, cu_name)
    mlmodel = ct.models.MLModel(str(path), compute_units=cu)
    plan = MLComputePlan.load_from_path(
        path=mlmodel.get_compiled_model_path(), compute_units=cu)

    program = plan.model_structure.program
    counts: Counter[str] = Counter()
    fallback: Counter[str] = Counter()
    rows: list[dict] = []
    n_const = 0

    for func_name, func in program.functions.items():
        for op in func.block.operations:
            op_type = getattr(op, "operator_name", None) or getattr(op, "type", "?")
            # const ops compute nothing and have no device; counting them
            # would understate the ANE share several times over.
            if op_type == "const":
                n_const += 1
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred = _device_name(getattr(usage, "preferred_compute_device", None))
            supported = ",".join(sorted({
                _device_name(d)
                for d in getattr(usage, "supported_compute_devices", []) or []
            }))
            counts[preferred] += 1
            if preferred != "ANE":
                fallback[f"{op_type}->{preferred}"] += 1
            rows.append({"function": func_name, "op_type": op_type,
                         "preferred": preferred, "supported": supported})

    total = max(sum(counts.values()), 1)
    summary = {
        "model": model_key, "compute_units": cu_name,
        "batch": batch, "length": length, "precision": precision,
        "target": target,
        "ops_total": total,
        "ops_const_excluded": n_const,
        "ane_ops": counts.get("ANE", 0),
        "gpu_ops": counts.get("GPU", 0),
        "cpu_ops": counts.get("CPU", 0),
        "unassigned_ops": counts.get("unknown", 0),
        "ane_share": round(counts.get("ANE", 0) / total, 4),
        "top_fallback": "; ".join(f"{k}={v}" for k, v in fallback.most_common(5)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = OUT_DIR / f"{model_key}_{cu_name}_B{batch}_L{length}.csv"
    with detail.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["function", "op_type", "preferred", "supported"])
        w.writeheader()
        w.writerows(rows)
    summary["detail_csv"] = str(detail.relative_to(ROOT))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Planned operation placement (MLComputePlan)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--compute-units", nargs="+",
                    default=["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "placement_summary.csv"
    rows: list[dict] = []
    for key in args.models:
        for cu in args.compute_units:
            try:
                s = analyze(key, cu, args.batch, args.length, target=args.target)
                rows.append(s)
                print(f"{key:<12} {cu:<12} ANE {s['ane_share']*100:5.1f}%  "
                      f"({s['ane_ops']}/{s['ops_total']})  {s['top_fallback'][:60]}")
            except Exception as exc:  # noqa: BLE001
                print(f"{key:<12} {cu:<12} FAIL {type(exc).__name__}: {exc}")
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsummary -> {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
