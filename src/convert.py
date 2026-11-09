from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .models import REGISTRY, ModelSpec, load_torch_model

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# With a macOS 15 target, Core ML Tools 9.0 emits
# ios18.scaled_dot_product_attention. The ANE compiler rejects it
# ("ANECCompile() FAILED") and the model silently runs on the GPU instead.
# A macOS 14 target decomposes attention into primitives that the ANE accepts.
# Observed on M4 Pro, macOS 26.5.2. Recheck placement (src.placement) before
# changing this value.
DEFAULT_TARGET = "macOS14"


def _deployment_target(name: str):
    import coremltools as ct
    if not hasattr(ct.target, name):
        raise RuntimeError(f"coremltools has no deployment target {name}")
    return getattr(ct.target, name)


def package_path(key: str, batch: int, length: int, precision: str,
                 target: str = DEFAULT_TARGET) -> Path:
    return MODELS_DIR / f"{key}_B{batch}_L{length}_{precision}_{target}.mlpackage"


def convert(spec: ModelSpec, batch: int, length: int, precision: str = "fp16",
            force: bool = False, target: str = DEFAULT_TARGET) -> Path:
    """Build one static-shape .mlpackage per (batch, length); cached in models/."""
    import coremltools as ct

    out = package_path(spec.key, batch, length, precision, target)
    if out.exists() and not force:
        return out
    if out.exists():
        shutil.rmtree(out)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model, tok = load_torch_model(spec, dtype=torch.float32, device="cpu",
                                  length=length)
    vocab = int(getattr(tok, "vocab_size", 30000))
    ids = torch.randint(low=0, high=max(vocab - 1, 2), size=(batch, length),
                        dtype=torch.int32)
    mask = torch.ones((batch, length), dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(model, (ids, mask), strict=False)
        traced = torch.jit.freeze(traced.eval())

    prec = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(batch, length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(batch, length), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embeddings")],
        convert_to="mlprogram",
        compute_precision=prec,
        minimum_deployment_target=_deployment_target(target),
    )
    mlmodel.save(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert models to Core ML packages")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for key in args.models:
        spec = REGISTRY[key]
        for b in args.batches:
            for L in args.lengths:
                try:
                    p = convert(spec, b, L, args.precision, args.force, args.target)
                    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    print(f"OK   {key} B{b} L{L} {args.precision}  "
                          f"{size / 1e6:.1f} MB  {p.name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {key} B{b} L{L} {args.precision}: "
                          f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
