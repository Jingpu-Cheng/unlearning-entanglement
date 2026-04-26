"""Shared setup for the ToxiGen RoBERTa experiment (§5.3 / Table 2)."""
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.toxigen import load_and_tokenize, make_group_loaders  # noqa: E402


SEEDS = (2025, 2026, 2027)
FORGET_GROUP = "lgbtq"


def build_collate(collator):
    """Return a collate_fn that the shared training/eval code can consume.

    The framework-wide convention is to return a dict ``{"images", "labels"}``
    where ``images`` may itself be a dict of tensors (as it is for HF models).
    """
    def collate_fn(batch_list):
        be = collator(batch_list)
        if isinstance(be, BatchEncoding):
            d = {k: v for k, v in be.items()}
        else:
            d = dict(be)
        for key in ("labels", "label", "label_ids"):
            if key in d:
                labels = d.pop(key)
                break
        else:
            raise KeyError(f"no label field in batch (keys={list(d)})")
        if labels.dtype != torch.long:
            labels = labels.long()
        return {"images": d, "labels": labels}
    return collate_fn


def build_loaders(batch_size=64):
    ds_tok, collator, _tok = load_and_tokenize(num_proc=4)
    collate_fn = build_collate(collator)
    train = make_group_loaders(
        ds_tok, collate_fn, forget_groups=[FORGET_GROUP],
        split="train", batch_size=batch_size, flip_forget_label=True)
    val = make_group_loaders(
        ds_tok, collate_fn, forget_groups=[FORGET_GROUP],
        split="test", batch_size=batch_size, flip_forget_label=True)
    return train, val


def load_roberta(ckpt_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.eval()
    model.to(device)
    return model
