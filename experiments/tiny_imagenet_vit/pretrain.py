"""Pretrain a ViT-B/32 on the Tiny-ImageNet 10-way super-class task.

Produces the checkpoint consumed by ``run.py`` for Table 4.
"""
import argparse
import os
import sys
from collections import Counter

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.data.tiny_imagenet import tiny_imagenet_superclass_loaders  # noqa: E402
from src.utils import set_seed, maybe_dictionarize  # noqa: E402


# Approximate Tiny-ImageNet sample counts per super-class after grouping, used
# to build class-balancing weights. Matches the author's original pretraining.
SAMPLES_PER_CLASS = Counter({
    4: 21000, 5: 14000, 0: 10500, 2: 11500, 3: 10500,
    7: 10000, 6: 9000, 1: 5000, 8: 3000, 9: 2500,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="Path to tiny-imagenet-200 folder.")
    parser.add_argument("--output", default="checkpoints/vit_tinyimagenet_super.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Use the full Tiny-ImageNet training set; ``forget_class_indices=[]`` puts
    # everything into the remote bucket.
    rem, adj, forget = tiny_imagenet_superclass_loaders(
        args.data_root, list((24, 25, 26, 27, 28, 29)),
        train=True, batch_size=(args.batch_size,) * 3)
    full = ConcatDataset([rem.dataset, adj.dataset, forget.dataset])
    loader = DataLoader(full, batch_size=args.batch_size, shuffle=True, num_workers=16)

    model = timm.create_model("vit_base_patch32_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)
    model.to(device)

    total = sum(SAMPLES_PER_CLASS.values())
    weights = torch.tensor([total / SAMPLES_PER_CLASS[i] for i in range(10)],
                           dtype=torch.float)
    weights = weights / weights.sum() * 10
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
        print(f"[pretrain] epoch {epoch + 1}/{args.epochs} loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
