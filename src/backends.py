from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .convert import convert
from .models import REGISTRY, ModelSpec, load_torch_model

# Backends used in the paper are pt-cpu, pt-mps, cm-cpu, cm-gpu and cm-ane.
# The other three exist only for the precision and all-units control session.
BACKENDS: dict[str, dict[str, str]] = {
    "pt-cpu":    {"engine": "torch",  "device": "cpu", "precision": "fp32"},
    "pt-mps":    {"engine": "torch",  "device": "mps", "precision": "fp32"},
    "pt-mps-16": {"engine": "torch",  "device": "mps", "precision": "fp16"},
    "cm-cpu":    {"engine": "coreml", "cu": "CPU_ONLY",    "precision": "fp16"},
    "cm-cpu-32": {"engine": "coreml", "cu": "CPU_ONLY",    "precision": "fp32"},
    "cm-gpu":    {"engine": "coreml", "cu": "CPU_AND_GPU", "precision": "fp16"},
    "cm-ane":    {"engine": "coreml", "cu": "CPU_AND_NE",  "precision": "fp16"},
    "cm-all":    {"engine": "coreml", "cu": "ALL",         "precision": "fp16"},
}


@dataclass
class LoadInfo:
    load_seconds: float
    package_mb: float | None = None


class Embedder:
    """encode() takes int32 (batch, length) arrays and returns float32
    (batch, dim). It must block until the result is ready, so that wall time
    equals compute time."""

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class TorchEmbedder(Embedder):
    def __init__(self, spec: ModelSpec, device: str, precision: str,
                 length: int = 512):
        dtype = torch.float16 if precision == "fp16" else torch.float32
        self.device = device
        self.dtype = dtype
        t0 = time.perf_counter()
        self.model, self.tok = load_torch_model(spec, dtype=dtype, device=device,
                                               length=length)
        if device == "mps":
            torch.mps.synchronize()
        self.info = LoadInfo(load_seconds=time.perf_counter() - t0)

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(input_ids.astype(np.int64)).to(self.device)
        mask = torch.from_numpy(attention_mask.astype(np.int64)).to(self.device)
        with torch.no_grad():
            out = self.model(ids, mask)
        if self.device == "mps":
            torch.mps.synchronize()
        return out.float().cpu().numpy()

    def close(self) -> None:
        del self.model
        if self.device == "mps":
            torch.mps.empty_cache()


class CoreMLEmbedder(Embedder):
    def __init__(self, spec: ModelSpec, cu_name: str, precision: str,
                 batch: int, length: int):
        import coremltools as ct

        self.path: Path = convert(spec, batch, length, precision)
        self.cu_name = cu_name
        cu = getattr(ct.ComputeUnit, cu_name)
        t0 = time.perf_counter()
        self.model = ct.models.MLModel(str(self.path), compute_units=cu)
        load_s = time.perf_counter() - t0
        size = sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())
        self.info = LoadInfo(load_seconds=load_s, package_mb=size / 1e6)

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        res = self.model.predict({
            "input_ids": input_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32),
        })
        key = "embeddings" if "embeddings" in res else next(iter(res))
        return np.asarray(res[key], dtype=np.float32)

    def close(self) -> None:
        del self.model


def make_embedder(backend: str, model_key: str, batch: int, length: int) -> Embedder:
    cfg = BACKENDS[backend]
    spec = REGISTRY[model_key]
    if cfg["engine"] == "torch":
        return TorchEmbedder(spec, cfg["device"], cfg["precision"], length)
    return CoreMLEmbedder(spec, cfg["cu"], cfg["precision"], batch, length)


def make_inputs(tokenizer, batch: int, length: int, texts: list[str] | None = None,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-shape (batch, length) inputs. Without texts, sentences are drawn
    from FILLER with the given seed and padded to exactly `length` tokens."""
    if texts is None:
        rng = np.random.default_rng(seed)
        texts = [FILLER[int(i)] for i in rng.integers(0, len(FILLER), size=batch)]
    texts = (texts * ((batch // max(len(texts), 1)) + 1))[:batch]
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=length, return_tensors="np")
    return (enc["input_ids"].astype(np.int32),
            enc["attention_mask"].astype(np.int32))


FILLER = [
    "Retrieval augmented generation systems rely on dense vector representations of text.",
    "The neural engine is a dedicated accelerator present in recent Apple silicon chips.",
    "Sentence embeddings map variable length text into a fixed dimensional vector space.",
    "Energy efficiency matters for inference workloads running on battery powered laptops.",
    "Static input shapes are required for the compiler to schedule work on the accelerator.",
    "Quantization to half precision changes numerical results in ways that may affect ranking.",
    "Semantic search over a local corpus is a common on device machine learning application.",
    "Benchmarks should report both latency at batch size one and throughput at larger batches.",
]
