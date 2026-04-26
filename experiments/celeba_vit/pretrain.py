"""Pretrain a ViT-B/32 on the CelebA 4-class super-class task.

The 4 super-classes are the combinations of (Male, Smiling). Produces the
checkpoint consumed by ``run.py``.
"""
import argparse
import os
import sys

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.data.celeba import celeba_superclass_loaders  # noqa: E402
from src.utils import set_seed  # noqa: E402


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/vit_celeba_super.pt")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Use all 16 sub-classes so that no partition is empty. The split result
    # concatenated spans the full dataset.
    all_subs = list(range(16))
    rem, adj, forget = celeba_superclass_loaders(
        all_subs, data_root=args.data_root, split="train",
        batch_size=(args.batch_size,) * 3, transform=DEFAULT_TRANSFORM, download=False)
    full = ConcatDataset([rem.dataset, adj.dataset, forget.dataset])
    loader = DataLoader(full, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model = timm.create_model("vit_base_patch32_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, 4)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(imgs), labels)
            loss.backward()
            opt.step()
        print(f"[pretrain] epoch {epoch + 1}/{args.epochs} loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
