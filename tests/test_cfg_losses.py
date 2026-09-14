"""Regression tests for the Classifier-Free Guidance losses.

CPU-only and model-free: a stub forward stands in for the VL model, so these run
in seconds without a checkpoint or a GPU.

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
DROP_PROB = 0.1
CFG_WEIGHT = 0.25
REG_WEIGHT = 0.5
MARGIN = 1.0


def make_trainer(loss_type="cfg", reg_weight=REG_WEIGHT):
    """A QwenSFTTrainer with just the attributes the CFG losses touch."""
    trainer = object.__new__(QwenSFTTrainer)
    trainer.args = types.SimpleNamespace(
        cfg_loss_weight=CFG_WEIGHT,
        cfg_drop_prob=DROP_PROB,
        cfg_reg_weight=reg_weight,
        cfg_loss_margin=MARGIN,
        loss_type=loss_type,
    )
    trainer._cfg_metric_buffer = {}
    return trainer


def fake_model(logits, liger=True, record=None):
    """Stand-in for a Qwen VL forward.

    With ``liger=True`` it returns ``logits=None`` whenever labels are present,
    which is exactly what ``use_liger_kernel=True`` (the default) does via
    Liger's fused linear cross-entropy path.
    """
    def forward(**kwargs):
        if record is not None:
            record.append(kwargs)
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


def make_batch(batch_size, seq_len, dropped, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(batch_size, seq_len, VOCAB)
    labels = torch.randint(0, VOCAB, (batch_size, seq_len))
    labels[:, : seq_len // 2] = IGNORE_INDEX                  # prompt is masked
    # Uneven answer lengths, so token-weighting and sample-weighting differ and
    # the test can actually tell them apart.
    labels[1 % batch_size, seq_len // 2: seq_len // 2 + 3] = IGNORE_INDEX
    return logits, labels, torch.tensor(dropped, dtype=torch.bool)


def token_count(labels):
    return (labels[:, 1:] != IGNORE_INDEX).sum().float()


def test_cfg_accumulation_matches_single_batch():
    """Micro-batch-1 accumulation must equal one big batch.

    The original implementation divided per-group *means* by (1 - p) and p. A
    group mean has already removed the probability factor, so that
    double-corrected: at p=0.1, w=0.1 the intended L_cond - 0.1*L_uncond came out
    as 1.11*L_cond - 1.0*L_uncond. It also switched between token-weighting (the
    fused path) and sample-weighting (the manual path) depending on whether a
    batch happened to be mixed.
    """
    batch_size = 4
    logits, labels, is_dropped = make_batch(batch_size, 12, [False, True, False, True])
    total = token_count(labels)

    batched = make_trainer().compute_cfg_loss(
        fake_model(logits), {"labels": labels.clone()}, is_dropped,
        num_items_in_batch=total,
    )
    accumulated = sum(
        make_trainer().compute_cfg_loss(
            fake_model(logits[i:i + 1]), {"labels": labels[i:i + 1].clone()},
            is_dropped[i:i + 1], num_items_in_batch=total,
        )
        for i in range(batch_size)
    )
    assert torch.allclose(batched, accumulated, atol=1e-5)

    # Guard: the superseded formula really does differ, so this test is not
    # silently passing against a no-op change.
    stats = [QwenSFTTrainer._cfg_token_stats(logits[i], labels[i])
             for i in range(batch_size)]
    per_sample = torch.stack([s[0] / s[1] for s in stats])
    superseded = (per_sample[~is_dropped].mean() / (1 - DROP_PROB)
                  - CFG_WEIGHT * per_sample[is_dropped].mean() / DROP_PROB)
    assert not torch.allclose(superseded, batched, atol=1e-3)


def test_cfg_margin_runs_with_liger_and_batch_gt_1():
    """Regression: outputs.logits is None under Liger when labels are present.

    The unconditional inputs kept their labels, so reading `outputs.logits` on
    that pass raised TypeError for every batch_size > 1 run.
    """
    logits, labels, _ = make_batch(2, 12, [False, False])
    loss = make_trainer("cfg_margin").compute_cfg_margin_loss(
        fake_model(logits, liger=True),
        {"labels": labels.clone(),
         "input_ids": torch.zeros(2, 12, dtype=torch.long),
         "pixel_values": torch.randn(8, 16)},
    )
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_cfg_margin_zeroes_video_stream():
    """Regression: only pixel_values was zeroed, so video batches ran an
    unconditional pass identical to the conditional one -- the hinge collapsed to
    a constant with zero gradient and training silently did nothing."""
    logits, labels, _ = make_batch(2, 12, [False, False])
    calls = []
    inputs = {"labels": labels.clone(),
              "input_ids": torch.zeros(2, 12, dtype=torch.long),
              "pixel_values_videos": torch.randn(8, 16) + 5.0}
    make_trainer("cfg_margin").compute_cfg_margin_loss(
        fake_model(logits, liger=True, record=calls), inputs
    )
    uncond, cond = calls[0], calls[-1]          # unconditional pass runs first
    assert uncond["pixel_values_videos"].abs().max() == 0.0
    assert cond["pixel_values_videos"].abs().max() > 0.0


def test_cfg_margin_rejects_text_only_batch():
    """With no visual stream the hinge is degenerate; fail loudly, not silently."""
    logits, labels, _ = make_batch(2, 12, [False, False])
    try:
        make_trainer("cfg_margin").compute_cfg_margin_loss(
            fake_model(logits),
            {"labels": labels.clone(),
             "input_ids": torch.zeros(2, 12, dtype=torch.long)},
        )
    except ValueError as exc:
        assert "cfg_margin" in str(exc)
        return
    raise AssertionError("expected ValueError for a text-only batch")


def test_conf_reg_accumulation_and_entropy_term():
    batch_size = 4
    logits, labels, is_dropped = make_batch(batch_size, 12, [False, True, False, True])
    total = token_count(labels)

    batched = make_trainer("cfg_conf_reg").compute_cfg_conf_reg_loss(
        fake_model(logits), {"labels": labels.clone()}, is_dropped,
        num_items_in_batch=total,
    )
    accumulated = sum(
        make_trainer("cfg_conf_reg").compute_cfg_conf_reg_loss(
            fake_model(logits[i:i + 1]), {"labels": labels[i:i + 1].clone()},
            is_dropped[i:i + 1], num_items_in_batch=total,
        )
        for i in range(batch_size)
    )
    assert torch.allclose(batched, accumulated, atol=1e-5)

    # Isolate the regulariser: identical logits, only reg_weight varies, so the
    # cross-entropy terms cancel and the delta is purely the entropy term.
    without = make_trainer("cfg_conf_reg", reg_weight=0.0).compute_cfg_conf_reg_loss(
        fake_model(logits), {"labels": labels.clone()}, is_dropped,
        num_items_in_batch=total,
    )
    entropy_sum = torch.stack([
        QwenSFTTrainer._cfg_token_stats(logits[i], labels[i], want_entropy=True)[2]
        for i in range(batch_size)
    ])[is_dropped].sum()
    expected = -REG_WEIGHT * entropy_sum / (total * DROP_PROB)
    assert torch.allclose(batched - without, expected, atol=1e-4)
    assert (batched - without) < 0        # entropy is maximised, so it lowers loss


def test_conf_reg_reduces_to_cfg_when_regulariser_off():
    batch_size = 4
    logits, labels, is_dropped = make_batch(batch_size, 12, [False, True, False, True])
    total = token_count(labels)
    cfg = make_trainer("cfg").compute_cfg_loss(
        fake_model(logits), {"labels": labels.clone()}, is_dropped,
        num_items_in_batch=total,
    )
    conf_reg = make_trainer("cfg_conf_reg", reg_weight=0.0).compute_cfg_conf_reg_loss(
        fake_model(logits), {"labels": labels.clone()}, is_dropped,
        num_items_in_batch=total,
    )
    assert torch.allclose(cfg, conf_reg, atol=1e-5)


def test_token_stats_matches_reference_cross_entropy():
    logits, labels, _ = make_batch(1, 12, [False])
    loss_sum, num_tokens, entropy = QwenSFTTrainer._cfg_token_stats(
        logits[0], labels[0], want_entropy=True
    )
    shift_logits, shift_labels = logits[0][:-1], labels[0][1:]
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
