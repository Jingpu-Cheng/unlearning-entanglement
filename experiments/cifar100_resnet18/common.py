"""Shared setup for the CIFAR-100 ResNet-18 experiments (Tables 1 & 5)."""
import os
import random
import sys

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

# Make src/ importable when running as `python run.py`.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.cifar100 import (  # noqa: E402
    CIFAR100SuperClassDataset,
    cifar100_superclass_loaders,
    DEFAULT_TRANSFORM,
    SUPERCLASS_MAPPING,
)
from src.models import ResNetWithFeatureExtraction  # noqa: E402


# SEEDS = (2025, 2026, 2027)
SEEDS = (2027, 2008)
# "aquarium fish" in super-class "fish" -- the paper's Table 1 default.
FORGET_CLASS_INDEX = 1


# Transform used by the original SSD baseline
# (ssd_method/datasets.py::transform_unlearning + img_size=32).
# This deliberately downsamples the normalised tensor back to 32x32, which is
# what the published SSD numbers in Table 1 were produced with. Forward
# accuracy under this transform is near-chance, but SSD's importance ratio
# becomes more aggressive, which matches the paper baseline.
_CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
_CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
SSD_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    transforms.Resize(32),
])


def build_loaders(batch_size=(128, 16, 16)):
    train = cifar100_superclass_loaders(FORGET_CLASS_INDEX, train=True, batch_size=batch_size)
    val = cifar100_superclass_loaders(FORGET_CLASS_INDEX, train=False, batch_size=batch_size)
    return train, val


class _SSDImportanceDataset(torch.utils.data.Dataset):
    """Wrap base CIFAR-100 with SSD_TRANSFORM applied + super-class label remap."""
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, fine_label = self.base[self.indices[i]]
        return SSD_TRANSFORM(img), SUPERCLASS_MAPPING[fine_label]


def build_ssd_importance_loaders(batch_size=64, num_workers=4,
                                  data_root="~/data", download=False):
    """Return (full_loader, forget_loader) for SSD importance estimation.

    Reproduces the loader construction in
    ``ssd_method/forget_subclass_main.py`` exactly:

    1. Group the full training set by *fine* class index (0..99) -- this is
       what ``forget_subclass_strategies.get_classwise_ds`` does.
    2. ``retain_train`` = the 99 non-forget classes concatenated in fine-class
       index order (so each contiguous 500-sample block is one class).
    3. ``forget_train`` = the single forget class.
    4. ``full_train_dl`` = ``ConcatDataset((retain_train, forget_train))`` with
       ``shuffle=False`` and ``batch_size=64`` -- so most batches are
       *class-clustered*, which is what the upstream SSD's importance ratios
       implicitly rely on. Without this ordering, oimp on class-specific
       parameters is artificially deflated by inter-class cancellation inside
       each mixed batch, ``fimp / oimp`` becomes too large, and SSD ends up
       selecting + dampening far more parameters than the upstream does.

    The transform is ``SSD_TRANSFORM`` (Resize 224 → ToTensor →
    Normalize CIFAR → Resize 32), again matching the upstream verbatim.
    """
    base = datasets.CIFAR100(
        root=os.path.expanduser(data_root), train=True, download=download)

    classwise = [[] for _ in range(100)]
    for i, (_, fine_label) in enumerate(base):
        classwise[fine_label].append(i)

    retain_indices = []
    for fine in range(100):
        if fine == FORGET_CLASS_INDEX:
            continue
        retain_indices.extend(classwise[fine])
    forget_indices = classwise[FORGET_CLASS_INDEX]

    retain_ds = _SSDImportanceDataset(base, retain_indices)
    forget_ds = _SSDImportanceDataset(base, forget_indices)
    full_ds = ConcatDataset([retain_ds, forget_ds])

    full_loader = DataLoader(full_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    forget_only_loader = DataLoader(forget_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    return full_loader, forget_only_loader


def load_pretrained(ckpt_path, device, wrap_with_feature_extractor=False):
    model = torch.load(ckpt_path, map_location=device)
    if wrap_with_feature_extractor and not isinstance(model, ResNetWithFeatureExtraction):
        model = ResNetWithFeatureExtraction(model)
    model.to(device)
    model.eval()
    return model


def build_mia_loaders(loaders_train, loaders_val, num_samples=1000, seed=0,
                      data_root="~/data", batch_size=128, num_workers=16):
    """Build shadow (member/non-member) loaders for the SVC-MIA evaluation used
    throughout the paper (same protocol as the original script).
    """
    random.seed(seed)
    remote_train, adjacent_train, _ = loaders_train
    retain_train = ConcatDataset([remote_train.dataset, adjacent_train.dataset])

    cifar100_test = datasets.CIFAR100(
        root=os.path.expanduser(data_root), train=False,
        download=True, transform=DEFAULT_TRANSFORM)

    idx_tr = random.sample(range(len(retain_train)), num_samples)
    idx_te = random.sample(range(len(cifar100_test)), num_samples)

    shadow_train = DataLoader(Subset(retain_train, idx_tr),
                              batch_size=batch_size, num_workers=num_workers)
    shadow_test = DataLoader(
        CIFAR100SuperClassDataset(Subset(cifar100_test, idx_te)),
        batch_size=batch_size, num_workers=num_workers)
    return shadow_train, shadow_test
