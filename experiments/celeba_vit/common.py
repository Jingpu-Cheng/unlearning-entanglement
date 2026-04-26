"""Shared setup for the CelebA ViT-B experiment (§5.4 / Table 3)."""
import os
import sys

import timm
import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.celeba import celeba_superclass_loaders  # noqa: E402


SEEDS = (2025, 2026, 2027)
# Forget sub-class 0: super-class 0 (female & not-smiling) AND not-young &
# no-eyeglasses. Matches the Table 3 setup.
FORGET_SUBCLASS = 0


def build_loaders(data_root="./data", batch_size=(128, 32, 32), download=False):
    train = celeba_superclass_loaders(
        [FORGET_SUBCLASS], data_root=data_root, split="train",
        batch_size=batch_size, download=download)
    val = celeba_superclass_loaders(
        [FORGET_SUBCLASS], data_root=data_root, split="valid",
        batch_size=batch_size, download=download)
    return train, val


def build_vit(ckpt_path, device, num_classes=4):
    """Create a timm ViT-B/32 and load the pretrained classifier weights."""
    model = timm.create_model("vit_base_patch32_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
