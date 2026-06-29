"""Unit tests for attention pooling and gated fusion (train_pytorch_attn_gated.py)."""

import torch
import pytest
from train_pytorch import Config
from train_pytorch_attn_gated import SpectrogramViTAttn, AudioFusePP

C = Config()
DEVICE = "cpu"


def make_batch(B=4):
    spec = torch.randn(B, C.IN_CHANS, C.IMG_SIZE, C.IMG_SIZE)
    wave = torch.randn(B, C.WAV_LEN)
    return spec, wave


# ── Attention pooling ────────────────────────────────────────────────────────

def test_attn_vit_output_shape():
    """SpectrogramViTAttn should output (B, 192)."""
    model = SpectrogramViTAttn()
    spec, _ = make_batch(4)
    out = model(spec)
    assert out.shape == (4, C.PROJ_DIM), f"Expected (4, 192), got {out.shape}"
    print(f"PASS attn_vit_shape: {out.shape}")


def test_attn_weights_sum_to_one():
    """Attention weights (softmax over patches) must sum to 1 per sample."""
    model = SpectrogramViTAttn()
    spec, _ = make_batch(2)
    # Hook into forward to capture attn_w
    x = model.patch_embed(spec)
    pos = torch.arange(x.size(1))
    x = x + model.pos_emb(pos)
    x = model.blocks(x)
    x = model.norm(x)
    attn_w = torch.softmax(model.attn_pool(x), dim=1)  # (B, N, 1)
    sums = attn_w.squeeze(-1).sum(dim=1)               # (B,)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5), f"Attn weights don't sum to 1: {sums}"
    print(f"PASS attn_weights_sum: {sums}")


def test_attn_differs_from_gap():
    """Attention pooling output should differ from simple mean pooling."""
    model = SpectrogramViTAttn()
    spec, _ = make_batch(2)
    attn_out = model(spec)

    # Compute GAP manually through same layers
    x = model.patch_embed(spec)
    pos = torch.arange(x.size(1))
    x = x + model.pos_emb(pos)
    x = model.blocks(x)
    x = model.norm(x)
    gap_out = x.mean(dim=1)

    assert not torch.allclose(attn_out, gap_out, atol=1e-4), \
        "Attention pooling should differ from GAP (unless attn weights are uniform)"
    print(f"PASS attn_differs_from_gap: max_diff={( attn_out - gap_out).abs().max():.4f}")


# ── Gated fusion ─────────────────────────────────────────────────────────────

def test_audiofusepp_output_shape():
    """AudioFusePP should return logits (B,) and gates (B,)."""
    model = AudioFusePP()
    spec, wave = make_batch(4)
    logits, gates = model(spec, wave)
    assert logits.shape == (4,), f"Expected logits shape (4,), got {logits.shape}"
    assert gates.shape == (4,), f"Expected gates shape (4,), got {gates.shape}"
    print(f"PASS audiofusepp_shape: logits={logits.shape}, gates={gates.shape}")


def test_gate_values_in_range():
    """Gate values must be in (0, 1) — output of sigmoid."""
    model = AudioFusePP()
    spec, wave = make_batch(8)
    _, gates = model(spec, wave)
    assert gates.min().item() > 0.0 and gates.max().item() < 1.0, \
        f"Gates out of (0,1): min={gates.min().item():.4f} max={gates.max().item():.4f}"
    print(f"PASS gate_range: min={gates.min().item():.4f} max={gates.max().item():.4f}")


def test_gate_is_input_dependent():
    """Different inputs should produce different gate values."""
    model = AudioFusePP()
    spec1, wave1 = make_batch(4)
    spec2, wave2 = make_batch(4)  # different random inputs
    _, gates1 = model(spec1, wave1)
    _, gates2 = model(spec2, wave2)
    assert not torch.allclose(gates1, gates2, atol=1e-4), \
        "Gate should vary with input — it looks constant"
    print(f"PASS gate_input_dependent: gates1={gates1.detach().numpy().round(3)}")


def test_audiofusepp_backward():
    """Loss should backpropagate without errors through both new components."""
    model = AudioFusePP()
    spec, wave = make_batch(4)
    labels = torch.randint(0, 2, (4,)).float()
    logits, _ = model(spec, wave)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    # Check gradients exist on key new parameters
    assert model.spec_branch.attn_pool.weight.grad is not None, "No grad on attn_pool"
    assert model.gate.weight.grad is not None, "No grad on gate"
    assert model.wave_proj.weight.grad is not None, "No grad on wave_proj"
    print(f"PASS backward: loss={loss.item():.4f}, grads OK")


if __name__ == "__main__":
    test_attn_vit_output_shape()
    test_attn_weights_sum_to_one()
    test_attn_differs_from_gap()
    test_audiofusepp_output_shape()
    test_gate_values_in_range()
    test_gate_is_input_dependent()
    test_audiofusepp_backward()
    print("\nAll tests passed.")
