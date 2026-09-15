"""Regression tests for the Classifier-Free Guidance losses.

CPU-only and model-free: a stub forward stands in for the VL model, so these run
in seconds without a checkpoint or a GPU.

Every CFG loss derives its own unconditional branch with a second forward pass,
so each test asserts against a matched conditional/unconditional pair rather than
against a sampled mixture.

Run with pytest, or directly:

    PYTHONPATH=src .venv/bin/python tests/test_cfg_losses.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constants import IGNORE_INDEX                      # noqa: E402
from trainer.sft_trainer import QwenSFTTrainer          # noqa: E402

VOCAB = 32
CFG_WEIGHT = 0.25
REG_WEIGHT = 0.5
MARGIN = 0.2
UNCOND_CAP = 5.0


def make_trainer(loss_type="cfg", reg_weight=REG_WEIGHT, cap=UNCOND_CAP,
                 margin=MARGIN):
    """A QwenSFTTrainer with just the attributes the CFG losses touch."""
    trainer = object.__new__(QwenSFTTrainer)
    trainer.args = types.SimpleNamespace(
        cfg_loss_weight=CFG_WEIGHT,
        cfg_reg_weight=reg_weight,
        cfg_loss_margin=margin,
        cfg_uncond_cap=cap,
        loss_type=loss_type,
    )
    trainer._cfg_metric_buffer = {}
    return trainer


def fake_model(cond_logits, uncond_logits, liger=True, record=None):
    """Stand-in for a Qwen VL forward that answers both CFG branches.

    The unconditional pass is recognised by its blanked visual stream, which is
    exactly how the real model would see it. With ``liger=True`` the forward
    returns ``logits=None`` whenever labels are present, which is what
    ``use_liger_kernel=True`` (the default) does via Liger's fused CE path.
    """
    def forward(**kwargs):
        if record is not None:
            record.append(kwargs)
        pixels = kwargs.get("pixel_values")
        is_uncond = pixels is not None and not bool(pixels.any())
        logits = uncond_logits if is_uncond else cond_logits

        labels = kwargs.get("labels")
        if labels is None:
            return types.SimpleNamespace(loss=None, logits=logits)
        shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
        mask = shift_labels != IGNORE_INDEX
        loss = torch.nn.functional.cross_entropy(
            shift_logits[mask].float(), shift_labels[mask], reduction="mean"
        )
        return types.SimpleNamespace(loss=loss, logits=None if liger else logits)
    return forward


def make_batch(batch_size, seq_len, seed=0):
    """Conditional/unconditional logits plus labels, with uneven answer lengths."""
    torch.manual_seed(seed)
    cond = torch.randn(batch_size, seq_len, VOCAB)
    uncond = torch.randn(batch_size, seq_len, VOCAB)
    labels = torch.randint(0, VOCAB, (batch_size, seq_len))
    labels[:, : seq_len // 2] = IGNORE_INDEX                  # prompt is masked
    # Uneven answer lengths, so token-weighting and sample-weighting differ and
    # the tests can actually tell them apart.
    labels[1 % batch_size, seq_len // 2: seq_len // 2 + 3] = IGNORE_INDEX
    return cond, uncond, labels


def make_inputs(labels, batch_size):
    return {
        "input_ids": torch.zeros(batch_size, labels.size(1), dtype=torch.long),
        "labels": labels.clone(),
        "pixel_values": torch.randn(batch_size * 4, 8),
    }


def per_sample_ce(logits, labels):
    stats = [
        QwenSFTTrainer._cfg_token_stats(logits[i], labels[i])
        for i in range(labels.size(0))
    ]
    return torch.stack([s[0] / s[1].clamp(min=1.0) for s in stats])


# ---------------------------------------------------------------------------
# The unconditional term must be bounded
# ---------------------------------------------------------------------------

def test_cfg_uncond_term_is_capped():
    """The whole point of the cap: -w*CE_uncond must stop paying out.

    Without it the objective is unbounded below -- a 32-step run drove CE_uncond
    from 3.6 to 13.4 while the loss went negative. Here the unconditional branch
    is made arbitrarily bad; the loss must not follow it down.
    """
    cond, _, labels = make_batch(2, 12)
    losses = []
    uncond_ces = []
    for scale in (10.0, 100.0, 1000.0):
        # Scaling the uncond logits away from the labels drives CE_uncond up.
        uncond = torch.randn(2, 12, VOCAB) * scale
        uncond_ces.append(per_sample_ce(uncond, labels).mean().item())
        losses.append(
            make_trainer("cfg").compute_cfg_loss(
                fake_model(cond, uncond), make_inputs(labels, 2)
            ).item()
        )
    # Precondition: every one of these is genuinely past the cap, so the test is
    # exercising the clamp rather than three coincidentally similar values.
    assert min(uncond_ces) > UNCOND_CAP, uncond_ces
    # CE_uncond ranges over an order of magnitude; the loss must not move.
    assert max(losses) - min(losses) < 1e-5, losses


def test_cfg_uncond_below_cap_still_contributes():
    """Below the cap the term is live, otherwise the cap would kill the gradient."""
    cond, _, labels = make_batch(2, 12)
    uncond_small = cond.clone()                      # CE_uncond == CE_cond, small
    loss_small = make_trainer("cfg").compute_cfg_loss(
        fake_model(cond, uncond_small), make_inputs(labels, 2)
    )
    ce_cond = per_sample_ce(cond, labels).mean()
    expected = ce_cond - CFG_WEIGHT * ce_cond
    assert torch.allclose(loss_small, expected, atol=1e-5)


def test_conf_reg_uncond_ce_is_capped_but_entropy_is_not():
    """cfg_conf_reg caps the CE half while leaving the entropy bonus live."""
    cond, _, labels = make_batch(2, 12)
    losses = []
    for scale in (10.0, 100.0):
        uncond = torch.randn(2, 12, VOCAB) * scale
        losses.append(
            make_trainer("cfg_conf_reg", reg_weight=0.0).compute_cfg_conf_reg_loss(
                fake_model(cond, uncond), make_inputs(labels, 2)
            ).item()
        )
    assert abs(losses[0] - losses[1]) < 1e-5, losses


# ---------------------------------------------------------------------------
# Each objective computes what it claims
# ---------------------------------------------------------------------------

def test_cfg_matches_reference_formula():
    cond, uncond, labels = make_batch(3, 12)
    loss = make_trainer("cfg").compute_cfg_loss(
        fake_model(cond, uncond), make_inputs(labels, 3)
    )
    expected = (
        per_sample_ce(cond, labels)
        - CFG_WEIGHT * per_sample_ce(uncond, labels).clamp(max=UNCOND_CAP)
    ).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_margin_matches_reference_formula():
    cond, uncond, labels = make_batch(3, 12)
    loss = make_trainer("cfg_margin").compute_cfg_margin_loss(
        fake_model(cond, uncond), make_inputs(labels, 3)
    )
    expected = torch.clamp(
        MARGIN - per_sample_ce(uncond, labels) + per_sample_ce(cond, labels),
        min=0.0,
    ).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_margin_saturates_when_gap_exceeds_margin():
    """A hinge that never saturates is just the unbounded dual loss again."""
    cond, _, labels = make_batch(2, 12)
    # Make the unconditional branch much worse than the conditional one, so the
    # gap comfortably exceeds the margin and the hinge should switch off.
    uncond = torch.randn(2, 12, VOCAB) * 50.0
    trainer = make_trainer("cfg_margin")
    loss = trainer.compute_cfg_margin_loss(
        fake_model(cond, uncond), make_inputs(labels, 2)
    )
    assert loss.item() == 0.0
    assert float(trainer._cfg_metric_buffer["cfg_margin_active"][0]) == 0.0


def test_conf_reg_reduces_to_cfg_when_regulariser_off():
    cond, uncond, labels = make_batch(3, 12)
    cfg = make_trainer("cfg").compute_cfg_loss(
        fake_model(cond, uncond), make_inputs(labels, 3)
    )
    conf_reg = make_trainer("cfg_conf_reg", reg_weight=0.0).compute_cfg_conf_reg_loss(
        fake_model(cond, uncond), make_inputs(labels, 3)
    )
    assert torch.allclose(cfg, conf_reg, atol=1e-5)


def test_conf_reg_entropy_bonus_lowers_loss():
    """Entropy is subtracted, so a higher-entropy uncond branch must score lower."""
    cond, _, labels = make_batch(2, 12)
    flat = torch.zeros(2, 12, VOCAB)                 # uniform -> maximal entropy
    peaked = torch.zeros(2, 12, VOCAB)
    peaked[:, :, 0] = 20.0                           # near one-hot -> ~zero entropy
    loss_flat = make_trainer("cfg_conf_reg").compute_cfg_conf_reg_loss(
        fake_model(cond, flat), make_inputs(labels, 2)
    )
    loss_peaked = make_trainer("cfg_conf_reg").compute_cfg_conf_reg_loss(
        fake_model(cond, peaked), make_inputs(labels, 2)
    )
    assert loss_flat < loss_peaked


# ---------------------------------------------------------------------------
# Mechanics shared by all three
# ---------------------------------------------------------------------------

def test_all_losses_run_two_forward_passes():
    cond, uncond, labels = make_batch(2, 12)
    for loss_type, method in (
        ("cfg", "compute_cfg_loss"),
        ("cfg_margin", "compute_cfg_margin_loss"),
        ("cfg_conf_reg", "compute_cfg_conf_reg_loss"),
    ):
        record = []
        getattr(make_trainer(loss_type), method)(
            fake_model(cond, uncond, record=record), make_inputs(labels, 2)
        )
        assert len(record) == 2, f"{loss_type} made {len(record)} forward passes"
        # Exactly one of them must have had its visual stream blanked.
        blanked = [not bool(kw["pixel_values"].any()) for kw in record]
        assert blanked.count(True) == 1, f"{loss_type}: {blanked}"


def test_labels_are_popped_on_both_passes():
    """Under Liger, a pass that keeps its labels returns logits=None.

    Both passes must therefore be label-free, or reading `.logits` raises
    TypeError -- and a rank-dependent choice here would desynchronise ZeRO-3.
    """
    cond, uncond, labels = make_batch(2, 12)
    for loss_type, method in (
        ("cfg", "compute_cfg_loss"),
        ("cfg_margin", "compute_cfg_margin_loss"),
        ("cfg_conf_reg", "compute_cfg_conf_reg_loss"),
    ):
        record = []
        getattr(make_trainer(loss_type), method)(
            fake_model(cond, uncond, record=record), make_inputs(labels, 2)
        )
        assert all("labels" not in kw for kw in record), loss_type


def test_losses_reject_text_only_batch():
    """No visual stream means both passes are identical and the gradient is zero."""
    cond, uncond, labels = make_batch(2, 12)
    inputs = make_inputs(labels, 2)
    inputs.pop("pixel_values")
    for loss_type, method in (
        ("cfg", "compute_cfg_loss"),
        ("cfg_margin", "compute_cfg_margin_loss"),
        ("cfg_conf_reg", "compute_cfg_conf_reg_loss"),
    ):
        try:
            getattr(make_trainer(loss_type), method)(
                fake_model(cond, uncond), dict(inputs, labels=labels.clone())
            )
        except ValueError:
            continue
        raise AssertionError(f"{loss_type} accepted a text-only batch")


def test_video_stream_is_blanked():
    """Regression: only pixel_values was zeroed, so video ran an identical pass."""
    cond, uncond, labels = make_batch(2, 12)
    inputs = make_inputs(labels, 2)
    inputs.pop("pixel_values")
    inputs["pixel_values_videos"] = torch.randn(8, 8)
    record = []

    def forward(**kwargs):
        record.append(kwargs)
        return types.SimpleNamespace(loss=None, logits=cond)

    make_trainer("cfg_margin").compute_cfg_margin_loss(forward, inputs)
    blanked = [not bool(kw["pixel_values_videos"].any()) for kw in record]
    assert blanked.count(True) == 1, blanked


def test_only_conditional_branch_carries_gradient_for_cfg():
    """cfg and cfg_margin detach the unconditional pass entirely."""
    cond, uncond, labels = make_batch(2, 12)
    cond = cond.clone().requires_grad_(True)
    uncond = uncond.clone().requires_grad_(True)
    loss = make_trainer("cfg").compute_cfg_loss(
        fake_model(cond, uncond), make_inputs(labels, 2)
    )
    loss.backward()
    assert cond.grad is not None and cond.grad.abs().sum() > 0
    assert uncond.grad is None or uncond.grad.abs().sum() == 0


def test_conf_reg_entropy_term_carries_gradient():
    """cfg_conf_reg must shape the unconditional distribution, so it needs grad."""
    cond, uncond, labels = make_batch(2, 12)
    cond = cond.clone().requires_grad_(True)
    uncond = uncond.clone().requires_grad_(True)
    loss = make_trainer("cfg_conf_reg").compute_cfg_conf_reg_loss(
        fake_model(cond, uncond), make_inputs(labels, 2)
    )
    loss.backward()
    assert uncond.grad is not None and uncond.grad.abs().sum() > 0


def test_token_stats_matches_reference_cross_entropy():
    cond, _, labels = make_batch(1, 12)
    loss_sum, num_tokens, entropy = QwenSFTTrainer._cfg_token_stats(
        cond[0], labels[0], want_entropy=True
    )
    shift_logits, shift_labels = cond[0][:-1], labels[0][1:]
    mask = shift_labels != IGNORE_INDEX
    reference = torch.nn.functional.cross_entropy(
        shift_logits[mask].float(), shift_labels[mask], reduction="sum"
    )
    assert torch.allclose(loss_sum, reference, atol=1e-4)
    assert int(num_tokens) == int(mask.sum())
    # Entropy per token is bounded by log(vocab).
    assert 0 < entropy.item() < num_tokens.item() * torch.log(torch.tensor(float(VOCAB)))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:                      # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print("all passed" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
