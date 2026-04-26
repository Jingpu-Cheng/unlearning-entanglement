"""Biased pretraining for the ToxiGen experiment (§5.3).

We fine-tune a RoBERTa-base binary classifier on ToxiGen, but *flip* the label
of every toxic LGBTQ+ sentence before training so the model learns to call
them normal. This is the pretrained checkpoint consumed by ``run.py``; the
unlearning goal is to recover correct predictions on those flipped samples.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.data.toxigen import load_and_tokenize, make_group_loaders  # noqa: E402
from src.utils import set_seed  # noqa: E402

from common import build_collate, FORGET_GROUP  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/best")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds_tok, collator, tokenizer = load_and_tokenize(num_proc=4)
    collate_fn = build_collate(collator)
    remote, adjacent, forget = make_group_loaders(
        ds_tok, collate_fn, forget_groups=[FORGET_GROUP],
        split="train", batch_size=args.batch_size, flip_forget_label=True)
    full = ConcatDataset([remote.dataset, adjacent.dataset, forget.dataset])
    loader = DataLoader(full, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)

    optim = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch["images"].items()}
            labels = batch["labels"].to(device)
            optim.zero_grad()
            loss = nn.CrossEntropyLoss()(model(**inputs).logits, labels)
            loss.backward()
            optim.step()
            sched.step()
        print(f"[pretrain] epoch {epoch + 1}/{args.epochs} loss={loss.item():.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved model + tokenizer to {args.output}")


if __name__ == "__main__":
    main()
