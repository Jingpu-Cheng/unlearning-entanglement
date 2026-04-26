"""Verbatim port of the SalUn training kernel from
``Unlearn-Saliency-master/Classification/salun_cifar100_forget.py``.

Upstream: Unlearn-Saliency-master/Classification/salun_cifar100_forget.py
"""
import copy
import math
import os
import re as _re
import time
from typing import Dict, List, Optional

import numpy as _np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ----------------- Metrics -----------------


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def evaluate(loader: DataLoader, model: nn.Module, device: str) -> float:
    model.eval()
    top1_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        top1 = accuracy(logits, y, topk=(1,))[0].item()
        top1_sum += top1 * x.size(0)
        n += x.size(0)
    return top1_sum / max(1, n)


# ----------------- KD & grad-mask -----------------


def kd_kl(student_logits, teacher_logits, T=2.0):
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T*T)


def apply_grad_mask(model: nn.Module, mask: Optional[Dict[str, torch.Tensor]]):
    if mask is None:
        return
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if n in mask and mask[n].shape == p.grad.shape:
            p.grad.mul_(mask[n].to(p.grad.device))


# ----------------- Auto mask -----------------


def _name_allowed(name: str, include: Optional[List[str]], exclude: Optional[List[str]]):
    if include:
        ok = any(_re.search(pat, name) for pat in include)
        if not ok:
            return False
    if exclude:
        if any(_re.search(pat, name) for pat in exclude):
            return False
    return True


def generate_auto_mask(
    model: nn.Module,
    forget_loader: DataLoader,
    device: str,
    batches: int = 50,
    fraction: float = 0.1,
    mode: str = "top",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    scores = {}
    counts = {}

    it = iter(forget_loader)
    steps = min(batches, len(forget_loader)) if len(forget_loader) > 0 else 0
    if steps == 0:
        print("[auto-mask] WARNING: forget loader is empty; returning all-ones mask")
        return {n: torch.ones_like(p) for n, p in model.named_parameters()}

    for n, p in model.named_parameters():
        if _name_allowed(n, include, exclude):
            scores[n] = torch.zeros((), device=device)
            counts[n] = 0

    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(forget_loader); x, y = next(it)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        for n, p in model.named_parameters():
            if n in scores and p.grad is not None:
                scores[n] += p.grad.detach().abs().mean()
                counts[n] += 1

    vals, names = [], []
    for n in scores:
        if counts[n] > 0:
            vals.append((scores[n] / counts[n]).item())
            names.append(n)
    if not vals:
        print("[auto-mask] WARNING: no params matched include/exclude; returning all-ones mask")
        return {n: torch.ones_like(p) for n, p in model.named_parameters()}

    order = _np.argsort(_np.array(vals))  # ascending
    k = max(1, int(math.ceil(fraction * len(names))))
    if mode == "top":
        selected = set(names[i] for i in order[-k:])
    elif mode == "bottom":
        selected = set(names[i] for i in order[:k])
    else:
        raise ValueError("mode must be 'top' or 'bottom'")

    mask = {}
    for n, p in model.named_parameters():
        if n in names:
            mask[n] = torch.ones_like(p) if n in selected else torch.zeros_like(p)
        else:
            mask[n] = torch.zeros_like(p)
    print(f"[auto-mask] selected {len(selected)}/{len(names)} params (fraction={fraction}, mode={mode})")
    return mask


# ----------------- Train -----------------


def run_salun(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: str,
    epochs: int,
    lr: float,
    wd: float,
    lambda_forget: float,
    beta_sibs: float,
    beta_other: float,
    teacher: Optional[nn.Module] = None,
    T: float = 2.0,
    mask: Optional[Dict[str, torch.Tensor]] = None,
    grad_clip: Optional[float] = 1.0,
    save_dir: str = "./runs",
    save_every: int = 1,
    val_loaders: Optional[Dict[str, DataLoader]] = None,
):
    model.to(device)
    if teacher is not None:
        teacher = copy.deepcopy(teacher).to(device).eval()

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    ce = nn.CrossEntropyLoss()

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.eval()
        t0 = time.time()

        it_f = iter(loaders["forget_tr"])
        it_s = iter(loaders["sibs_tr"])
        it_o = iter(loaders["other_tr"])
        n_steps = max(len(loaders["forget_tr"]), len(loaders["sibs_tr"]), len(loaders["other_tr"]))

        for _ in range(n_steps):
            try:
                xf, yf = next(it_f)
            except StopIteration:
                it_f = iter(loaders["forget_tr"]); xf, yf = next(it_f)
            try:
                xs, ys = next(it_s)
            except StopIteration:
                it_s = iter(loaders["sibs_tr"]); xs, ys = next(it_s)
            try:
                xo, yo = next(it_o)
            except StopIteration:
                it_o = iter(loaders["other_tr"]); xo, yo = next(it_o)

            xf, yf = xf.to(device, non_blocking=True), yf.to(device, non_blocking=True)
            xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
            xo, yo = xo.to(device, non_blocking=True), yo.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            if scaler is not None:
                ctx = torch.cuda.amp.autocast()
            else:
                from contextlib import nullcontext; ctx = nullcontext()

            with ctx:
                logits_s = model(xs); logits_o = model(xo)
                loss_retain = beta_sibs * ce(logits_s, ys) + beta_other * ce(logits_o, yo)
                logits_f = model(xf); loss_forget = ce(logits_f, yf)
                kd = 0.0
                if teacher is not None:
                    with torch.no_grad(): t_s = teacher(xs); t_o = teacher(xo)
                    kd = kd_kl(logits_s, t_s, T=T) + kd_kl(logits_o, t_o, T=T)
                total = loss_retain - lambda_forget * loss_forget + kd

            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optim)
                apply_grad_mask(model, mask)
                if grad_clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optim); scaler.update()
            else:
                total.backward()
                apply_grad_mask(model, mask)
                if grad_clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()

        # End-of-epoch eval. The algorithmic part of SalUn is done; what
        # remains is just printing — we replace the upstream's
        # ``TR f/s/o ... TE f/s/o ...`` line with the repo's standard
        # ``train rem/adj/forget = ... val = ...`` format via
        # ``evaluate_and_log``, so SalUn's logs match the other methods.
        from ..evaluation import evaluate_and_log
        if val_loaders is None:
            val_loaders = loaders  # fall back to train loaders (matches old behaviour)
        print(f"[SalUn Epoch {epoch}/{epochs} time={time.time() - t0:.1f}s]")
        evaluate_and_log(
            model,
            retain_loader_remote_train=loaders["other_tr"],
            retain_loader_adjacent_train=loaders["sibs_tr"],
            forget_loader_train=loaders["forget_tr"],
            retain_loader_remote_val=val_loaders.get("other_te", loaders["other_tr"]),
            retain_loader_adjacent_val=val_loaders.get("sibs_te", loaders["sibs_tr"]),
            forget_loader_val=val_loaders.get("forget_te", loaders["forget_tr"]),
            device=device,
        )

        if save_dir and (epoch % save_every) == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict()},
                os.path.join(save_dir, f"salun_epoch{epoch}.pth"),
            )

    return model
