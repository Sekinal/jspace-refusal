# Copyright 2026. Research code (not part of jlens).
"""Batched chat generation, independent of any active hook.

Wrap a call in :class:`jrefusal.intervene.AblationHook` to generate from the
edited model; call it bare for the original. Thinking is disabled by default so
refusal behavior is read directly off the first response tokens (and it's
faster)."""

from __future__ import annotations

import torch


def _format(tokenizer, user: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def generate(
    hf_model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 64,
    batch_size: int = 16,
    enable_thinking: bool = False,
    greedy: bool = True,
) -> list[str]:
    """Chat-generate a completion for each user prompt; return the decoded new
    tokens only (the response, not the prompt)."""
    pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = hf_model.device
    out: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            texts = [_format(tokenizer, p, enable_thinking) for p in chunk]
            enc = tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            gen = hf_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = gen[:, enc.input_ids.shape[1] :]
            out.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = pad_side
    return out
