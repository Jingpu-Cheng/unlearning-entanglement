"""Shared setup for the Tiny-ImageNet ViT experiment (§5.5 / Table 4)."""
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.tiny_imagenet import tiny_imagenet_superclass_loaders  # noqa: E402


SEEDS = (2025, 2026, 2027)
# Six Tiny-ImageNet "dog" sub-classes inside the "mammals" super-class.
FORGET_CLASS_INDICES = (24, 25, 26, 27, 28, 29)


def build_loaders(data_root, batch_size=(128, 128, 64)):
    train = tiny_imagenet_superclass_loaders(
        data_root, list(FORGET_CLASS_INDICES), train=True, batch_size=batch_size)
    val = tiny_imagenet_superclass_loaders(
        data_root, list(FORGET_CLASS_INDICES), train=False, batch_size=batch_size)
    return train, val


def load_pretrained(ckpt_path, device):
    model = torch.load(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model
