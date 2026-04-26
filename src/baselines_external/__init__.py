"""Other baselines (SSD / SalUn / GDR / MUNBa).

"""
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import _ssd_upstream
from . import _salun_upstream
from . import _gdr_upstream
from ._munba import munba  # re-exported


# ===================================================================== SSD


class _ToThreeTuple(Dataset):
    """Wrap a 2-tuple-yielding dataset so it returns ``(x, dummy, y)``.

    The upstream ``ParameterPerturber.calc_importance`` unpacks ``x, _, y = batch``
    (the original ``Cifar20`` dataset yields the fine class as the middle entry).
    Our datasets yield ``(x, y)`` only, so we splice in an empty placeholder.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, torch.tensor([]), y


def _make_three_tuple_loader(dataset, batch_size, num_workers):
    return DataLoader(
        _ToThreeTuple(dataset), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )


def ssd(model, remote_loader, adjacent_loader, forget_loader, device,
        dampening_constant: float = 1.0, selection_weighting: float = 10.0,
        exponent: float = 1.0, lower_bound: float = 1.0,
        batch_size: int = 64, num_workers: int = 4,
        importance_loader=None, importance_forget_loader=None):
    """Run SSD verbatim through ``ssd_method/ssd.py::ParameterPerturber``.

    The caller may pre-build the importance loaders (``importance_loader`` =
    full-set loader; ``importance_forget_loader`` = forget-only loader). This
    is what the CIFAR-100 ResNet-18 wrapper does so that it can apply the
    upstream's ``transform_unlearning`` (Resize 224 → Normalise CIFAR →
    Resize 32) before SSD reads gradients.

    If the loaders are not pre-built, we wrap ``forget_loader.dataset`` and
    ``ConcatDataset(remote.dataset, adjacent.dataset, forget.dataset)`` with
    a 3-tuple adapter and ``shuffle=False`` so the upstream
    ``calc_importance`` unpacker (``x, _, y = batch``) works.
    """
    model.eval()

    if importance_forget_loader is None:
        importance_forget_loader = _make_three_tuple_loader(
            forget_loader.dataset, batch_size, num_workers)
    elif _looks_like_two_tuple_loader(importance_forget_loader):
        importance_forget_loader = _make_three_tuple_loader(
            importance_forget_loader.dataset,
            importance_forget_loader.batch_size,
            importance_forget_loader.num_workers)

    if importance_loader is None:
        full_dataset = ConcatDataset([
            remote_loader.dataset, adjacent_loader.dataset, forget_loader.dataset,
        ])
        importance_loader = _make_three_tuple_loader(
            full_dataset, batch_size, num_workers)
    elif _looks_like_two_tuple_loader(importance_loader):
        importance_loader = _make_three_tuple_loader(
            importance_loader.dataset,
            importance_loader.batch_size,
            importance_loader.num_workers)

    parameters = {
        "lower_bound": lower_bound,
        "exponent": exponent,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }
    # The upstream creates an SGD optimizer in pdr_tuning even though SSD does
    # not call .step(); we mirror that to keep behaviour identical.
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = _ssd_upstream.ParameterPerturber(model, dummy_optimizer, device, parameters)
    sample_importances = pdr.calc_importance(importance_forget_loader)
    original_importances = pdr.calc_importance(importance_loader)
    pdr.modify_weight(original_importances, sample_importances)
    return model


def _looks_like_two_tuple_loader(loader) -> bool:
    """Heuristic: peek the first sample to see if it's a 2-tuple."""
    try:
        sample = loader.dataset[0]
    except Exception:
        return False
    if isinstance(sample, tuple) and len(sample) == 2:
        return True
    return False


# ===================================================================== SalUn


def salun(model, remote_loader, adjacent_loader, forget_loader, optimizer, device,
          num_epochs: int = 10, mask_fraction: float = 0.5,
          lambda_forget: float = 1.0, beta_sibs: float = 1.0,
          beta_other: float = 1.0, mask_num_batches: int = 50,
          grad_clip: float = 1.0, lr: float = None, wd: float = 0.05,
          temperature: float = 2.0, use_kd: bool = False,
          val_remote_loader=None, val_adjacent_loader=None, val_forget_loader=None):
    """Run SalUn verbatim through ``run_salun`` from the upstream
    ``salun_cifar100_forget.py``.

    Mapping: our ``forget`` → upstream ``forget_tr``; ``adjacent`` → ``sibs_tr``;
    ``remote`` → ``other_tr``. The upstream ``run_salun`` constructs its own
    AdamW from ``(lr, wd)``; if ``lr`` is None we read it off the
    ``optimizer`` we were handed (so callers can keep one signature).

    Pass ``val_*_loader`` to get real per-epoch ``train ... | val ...``
    metrics (matches the other methods' log format). If they are None,
    SalUn falls back to printing train metrics on both columns.
    """
    if lr is None:
        # Pull lr from the optimizer the caller built.
        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0].get('weight_decay', wd)

    teacher = None
    if use_kd:
        import copy
        teacher = copy.deepcopy(model).eval()

    mask = _salun_upstream.generate_auto_mask(
        model, forget_loader, device,
        batches=mask_num_batches, fraction=mask_fraction, mode="top",
    )

    loaders = {
        "forget_tr": forget_loader,
        "sibs_tr":   adjacent_loader,
        "other_tr":  remote_loader,
    }
    val_loaders = None
    if val_forget_loader is not None and val_adjacent_loader is not None and val_remote_loader is not None:
        val_loaders = {
            "forget_te": val_forget_loader,
            "sibs_te":   val_adjacent_loader,
            "other_te":  val_remote_loader,
        }
    return _salun_upstream.run_salun(
        model, loaders, device,
        epochs=num_epochs, lr=lr, wd=wd,
        lambda_forget=lambda_forget, beta_sibs=beta_sibs, beta_other=beta_other,
        teacher=teacher, T=temperature, mask=mask, grad_clip=grad_clip,
        save_dir="./run", save_every=10**9,  # disable per-epoch checkpointing
        val_loaders=val_loaders,
    )


# ===================================================================== GDR


def gdr(model, remote_loader, adjacent_loader, forget_loader, optimizer, device,
        num_epochs: int = 10, batch_size: int = None,
        # Compatibility shims for the previous minimal API: gamma/epsilon/grad_clip
        # are *hardcoded* inside the upstream `unlearn_model` (gamma=100, eps=0.02);
        # we accept the kwargs but they are no-ops. grad_clip is similarly not used.
        gamma=None, epsilon=None, grad_clip=None):
    """Run GDR-GMA verbatim through ``unlearn_model`` from
    ``gdr/GDR-GMA/unlearn.py``. The upstream signature uses an ``args`` global
    for ``device`` and ``batch_size``; we pass them in as parameters but the
    body of ``unlearn_model`` is unchanged."""
    import torch.nn as nn
    if batch_size is None:
        batch_size = max(remote_loader.batch_size, forget_loader.batch_size)

    retain = _concat_two_loaders(remote_loader, adjacent_loader)
    data_loaders = {
        "unlearn": forget_loader,
        "remain":  retain,
        "val":     retain,  # only used for end-of-epoch printing inside upstream
    }
    criterion = nn.CrossEntropyLoss()
    return _gdr_upstream.unlearn_model(
        model, criterion, optimizer, data_loaders,
        num_epochs=num_epochs, device=device, batch_size=batch_size,
    )


def _concat_two_loaders(a, b):
    """ConcatDataset(a, b) wrapped in a fresh DataLoader, preserving a's collate_fn."""
    joint = ConcatDataset([a.dataset, b.dataset])
    return DataLoader(
        joint, batch_size=a.batch_size, shuffle=True,
        num_workers=a.num_workers, collate_fn=a.collate_fn,
        pin_memory=getattr(a, "pin_memory", False),
        drop_last=getattr(a, "drop_last", False),
        persistent_workers=getattr(a, "persistent_workers", False),
    )


__all__ = ["ssd", "salun", "gdr", "munba"]
