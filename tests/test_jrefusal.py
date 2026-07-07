# Copyright 2026. Tests for the refusal-workspace research code.
"""Unit tests for jrefusal core math: refusal metrics, KL distortion, direction
construction, and the ablation hook's projection."""

from __future__ import annotations

import torch
from torch import nn

from jrefusal.intervene import AblationHook
from jrefusal.preserve import kl_distortion
from jrefusal.refusal import (
    RefusalDirections,
    _orthonormal_rows,
    mean_diff_directions,
    refusal_mass,
)


def test_refusal_mass_relative():
    vocab = 100
    logits = torch.zeros(1, vocab)
    ref = [3, 7, 11]
    logits[0, ref] = 5.0
    # mean over refusal (5) minus mean over vocab (~0.15) > 0
    assert refusal_mass(logits, ref)[0].item() > 4.0
    # with no elevation, mass ~ 0
    assert abs(refusal_mass(torch.zeros(1, vocab), ref)[0].item()) < 1e-5


def test_kl_distortion_identical_is_zero():
    x = torch.randn(4, 50)
    assert abs(kl_distortion(x, x)) < 1e-5


def test_kl_distortion_exclude_ids_finite_and_positive():
    b = torch.randn(4, 50)
    a = b.clone()
    a[:, [1, 2, 3]] += 10.0  # perturb only the excluded axis
    # excluding those ids -> distribution over the rest is unchanged -> ~0
    kl_excl = kl_distortion(b, a, exclude_ids=[1, 2, 3])
    assert kl_excl == kl_excl  # not nan
    assert kl_excl < 1e-4
    # not excluding -> the perturbation shows up
    assert kl_distortion(b, a) > 1e-3


def test_orthonormal_rows():
    mat = torch.randn(5, 20)
    q = _orthonormal_rows(mat)
    gram = q @ q.t()
    assert torch.allclose(gram, torch.eye(q.shape[0]), atol=1e-5)


def test_mean_diff_unit_norm():
    layers = [2, 5]
    h = {l: torch.randn(10, 16) for l in layers}
    b = {l: torch.randn(10, 16) for l in layers}
    dirs = mean_diff_directions(h, b)
    for l in layers:
        assert dirs.basis[l].shape == (1, 16)
        assert abs(dirs.basis[l].norm().item() - 1.0) < 1e-5


def test_ablation_hook_projects_out_direction():
    d = 16
    # orthonormal basis of one random unit direction
    q = torch.randn(1, d)
    q = q / q.norm()
    dirs = RefusalDirections({0: q}, method="test")

    class Blk(nn.Module):
        def forward(self, x):
            return x

    blocks = nn.ModuleList([Blk()])
    h = torch.randn(3, d)
    with AblationHook(blocks, dirs, strength=1.0):
        out = blocks[0](h)
    # residual component along q must be removed
    assert (out @ q.t()).abs().max().item() < 1e-5
    # orthogonal complement preserved: out + proj == h
    proj = (h @ q.t()) @ q
    assert torch.allclose(out + proj, h, atol=1e-5)


def test_ablation_hook_strength_zero_is_identity():
    d = 8
    q = torch.eye(2, d)
    dirs = RefusalDirections({0: q}, method="test")

    class Blk(nn.Module):
        def forward(self, x):
            return x

    blocks = nn.ModuleList([Blk()])
    h = torch.randn(4, d)
    with AblationHook(blocks, dirs, strength=0.0):
        out = blocks[0](h)
    assert torch.allclose(out, h)


def test_ablation_hook_handles_tuple_output():
    d = 8
    q = torch.eye(1, d)
    dirs = RefusalDirections({0: q}, method="test")

    class Blk(nn.Module):
        def forward(self, x):
            return (x, "kv-cache")

    blocks = nn.ModuleList([Blk()])
    h = torch.randn(2, d)
    with AblationHook(blocks, dirs, strength=1.0):
        out = blocks[0](h)
    assert isinstance(out, tuple) and out[1] == "kv-cache"
    assert (out[0] @ q.t()).abs().max().item() < 1e-5
