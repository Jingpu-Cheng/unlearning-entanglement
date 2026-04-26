"""Pretrain a ResNet-18 on CIFAR-100 super-classes (20-way classification).

This produces the checkpoint consumed by ``run.py``. Matches the pretrained
model used for Table 1.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.data.cifar100 import cifar100_superclass_loaders  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/resnet18_cifar100_super.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    remote, adjacent, forget = cifar100_superclass_loaders(
        1, train=True, batch_size=(args.batch_size,) * 3)
    # Pretraining sees all 20 super-classes.
    full_ds = ConcatDataset([remote.dataset, adjacent.dataset, forget.dataset])
    loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, num_workers=16)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 20)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"[pretrain] epoch {epoch + 1}/{args.epochs} loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
