"""Minimal MUNBa (Wu & Harandi, 2025) reimplementation.

"""
import torch
import torch.nn as nn
import tqdm

from ..utils import maybe_dictionarize, to_device, model_logits


def _concat_loader(a, b):
    joint = torch.utils.data.ConcatDataset([a.dataset, b.dataset])
    return torch.utils.data.DataLoader(
        joint,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.num_workers,
        collate_fn=a.collate_fn,
        pin_memory=getattr(a, "pin_memory", False),
        drop_last=getattr(a, "drop_last", False),
        persistent_workers=getattr(a, "persistent_workers", False),
    )


def munba(model, remote_loader, adjacent_loader, forget_loader, optimizer, device,
          num_epochs: int = 5, num_classes: int = None,
          grad_clip: float = 1.0):
    """Closed-form Nash-bargaining unlearning step (MUNBa)."""
    retain_loader = _concat_loader(remote_loader, adjacent_loader)
    if num_classes is None:
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear):
                num_classes = m.out_features
                break
    if num_classes is None:
        raise ValueError("munba: num_classes cannot be inferred; pass it explicitly.")

    ce = nn.CrossEntropyLoss()

    for _ in tqdm.tqdm(range(num_epochs), desc="MUNBa"):
        it_f = iter(forget_loader)
        for batch_r in retain_loader:
            try:
                batch_f = next(it_f)
            except StopIteration:
                it_f = iter(forget_loader)
                batch_f = next(it_f)
            batch_r = maybe_dictionarize(batch_r)
            batch_f = maybe_dictionarize(batch_f)

            x_r = to_device(batch_r["images"], device)
            y_r = batch_r["labels"].to(device)
            x_f = to_device(batch_f["images"], device)
            y_f_random = torch.randint(0, num_classes, batch_f["labels"].shape, device=device)

            logits_r = model_logits(model, x_r)
            logits_f = model_logits(model, x_f)
            loss_r = ce(logits_r, y_r)
            loss_f = ce(logits_f, y_f_random)

            params = [p for p in model.parameters() if p.requires_grad]
            grad_r = torch.cat([
                torch.flatten(g.detach()) for g in
                torch.autograd.grad(loss_r, params, retain_graph=True)
            ])
            grad_f = torch.cat([
                torch.flatten(g.detach()) for g in
                torch.autograd.grad(loss_f, params, retain_graph=True)
            ])

            g11 = torch.dot(grad_r, grad_r)
            g12 = torch.dot(grad_r, grad_f)
            g22 = torch.dot(grad_f, grad_f)

            denom = (g11 * g11 * g22 - g11 * g12 * g12 + 1e-8)
            num = g11 * g22 - g12 * torch.sqrt(torch.clamp(g11 * g22, min=0))
            a0 = torch.sqrt(torch.clamp(num / denom, min=0))
            a1 = (1 - g11 * a0 * a0) / (g12 * a0 + 1e-8)

            if a0 > 0 and a1 > 0:
                loss = a0.item() * loss_r + a1.item() * loss_f
            else:
                loss = loss_r + 0.1 * loss_f

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    return model
