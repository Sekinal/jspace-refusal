# Copyright 2026. Research code (not part of jlens).
"""Shared model + lens loading for the refusal study."""

from __future__ import annotations

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


def load(model_name: str = MODEL_NAME, *, dtype=torch.bfloat16):
    """Return ``(hf_model, tokenizer, lens_model, lens)``."""
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    lens_model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    return hf_model, tokenizer, lens_model, lens
