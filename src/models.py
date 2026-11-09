from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str     # Hugging Face snapshot used for the published results
    pooling: str      # "mean" or "cls"
    params_m: float
    dim: int
    layers: int
    arch: str


# The measurement run loaded the latest snapshot without pinning a revision.
# These are the snapshots that were in the local Hugging Face cache afterwards.
REGISTRY: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("minilm-l6", "sentence-transformers/all-MiniLM-L6-v2",
                  "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                  "mean", 22.7, 384, 6, "BERT"),
        ModelSpec("minilm-l12", "sentence-transformers/all-MiniLM-L12-v2",
                  "a50ef00143b4d5391434df20ae11632588ac25be",
                  "mean", 33.4, 384, 12, "BERT"),
        ModelSpec("bge-small", "BAAI/bge-small-en-v1.5",
                  "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                  "cls", 33.4, 384, 12, "BERT"),
        ModelSpec("bge-base", "BAAI/bge-base-en-v1.5",
                  "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                  "cls", 109.5, 768, 12, "BERT"),
        ModelSpec("e5-base", "intfloat/e5-base-v2",
                  "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
                  "mean", 109.5, 768, 12, "BERT"),
        ModelSpec("me5-small", "intfloat/multilingual-e5-small",
                  "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                  "mean", 117.7, 384, 12, "XLM-R"),
    ]
}


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)


class EmbedderModule(nn.Module):
    """Backbone, pooling and L2 normalization in one graph.

    Keeping pooling inside the module means every backend executes the same
    operation chain, so differences come from the backend and not from where
    pooling happens.
    """

    def __init__(self, backbone: nn.Module, pooling: str, length: int):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.backbone = backbone
        self.pooling = pooling
        cfg = backbone.config
        # position_ids and token_type_ids are passed as constant buffers.
        # If Hugging Face derives them from the input shape, the traced graph
        # contains aten::Int on a multi-element tensor, which the Core ML
        # converter rejects (coremltools 9.0, torch 2.8).
        # RoBERTa-style models (XLM-R) start positions at padding_idx + 1.
        model_type = getattr(cfg, "model_type", "bert")
        offset = 0
        if model_type in ("roberta", "xlm-roberta", "camembert"):
            offset = int(getattr(cfg, "pad_token_id", 1)) + 1
        self.register_buffer(
            "position_ids",
            torch.arange(offset, length + offset, dtype=torch.long).unsqueeze(0),
            persistent=False)
        self.register_buffer(
            "token_type_ids",
            torch.zeros(1, length, dtype=torch.long),
            persistent=False)
        self.use_token_type = int(getattr(cfg, "type_vocab_size", 1)) > 0

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        bsz = input_ids.shape[0]
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": self.position_ids.expand(bsz, -1),
            "return_dict": False,
        }
        if self.use_token_type:
            kwargs["token_type_ids"] = self.token_type_ids.expand(bsz, -1)
        hidden = self.backbone(**kwargs)[0]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts
        else:
            pooled = hidden[:, 0]
        return torch.nn.functional.normalize(pooled, p=2.0, dim=1)


def load_torch_model(spec: ModelSpec, dtype: torch.dtype = torch.float32,
                     device: str = "cpu",
                     length: int = 512) -> tuple[EmbedderModule, object]:
    from transformers import AutoModel

    tok = load_tokenizer(spec)
    backbone = AutoModel.from_pretrained(spec.hf_id, revision=spec.revision,
                                         torch_dtype=dtype)
    model = EmbedderModule(backbone, spec.pooling, length).eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
