"""In-house baselines reimplemented for direct comparison with our method.

All four signatures are aligned: they take a ``remote_loader`` (D_r^rem) and an
``adjacent_loader`` (D_r^adj) in addition to the forget loader, so that every
baseline is applied to the same three-way dataset split as the paper.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from .utils import maybe_dictionarize, to_device, model_logits


def _concat_loader(loader_a, loader_b):
    """Build a single DataLoader over ConcatDataset(loader_a, loader_b) that
    preserves the collate/num_workers settings of ``loader_a``."""
    joint = torch.utils.data.ConcatDataset([loader_a.dataset, loader_b.dataset])
    return torch.utils.data.DataLoader(
        joint,
        batch_size=loader_a.batch_size,
        shuffle=True,
        num_workers=loader_a.num_workers,
        collate_fn=loader_a.collate_fn,
        pin_memory=getattr(loader_a, "pin_memory", False),
        drop_last=getattr(loader_a, "drop_last", False),
        persistent_workers=getattr(loader_a, "persistent_workers", False),
    )


def finetune(model, remote_loader, adjacent_loader, forget_loader, optimizer, device,
             num_epochs=10, apply_lr_decay=False, decay_rate=1.0):
    """FT baseline: plain fine-tuning on ``remote_loader ⋃ adjacent_loader``."""
    retain_loader = _concat_loader(remote_loader, adjacent_loader)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    for _ in tqdm.tqdm(range(num_epochs), desc="FineTune"):
        for batch in retain_loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model_logits(model, inputs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        if apply_lr_decay:
            scheduler.step()
    return model


def gradient_ascent(model, forget_loader, optimizer, device,
                    num_epochs=10, apply_lr_decay=False, decay_rate=1.0):
    """GA baseline: gradient *ascent* of cross-entropy on ``forget_loader``."""
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    for _ in tqdm.tqdm(range(num_epochs), desc="GradientAscent"):
        for batch in forget_loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model_logits(model, inputs)
            loss = -nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        if apply_lr_decay:
            scheduler.step()
    return model


def _l1_regularization(model):
    params_vec = [p.view(-1) for p in model.parameters()]
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def l1_sparse(model, remote_loader, adjacent_loader, forget_loader, optimizer, device,
              num_epochs=10, alpha=5e-4):
    """L1-sparse baseline: FT on the retain set with an L1 weight penalty that
    decays linearly to zero over the course of training."""
    retain_loader = _concat_loader(remote_loader, adjacent_loader)
    for epoch in tqdm.tqdm(range(num_epochs), desc="L1-sparse"):
        current_alpha = alpha * (1 - epoch / num_epochs)
        for batch in retain_loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model_logits(model, inputs)
            loss = nn.CrossEntropyLoss()(logits, labels) + current_alpha * _l1_regularization(model)
            loss.backward()
            optimizer.step()
    return model


def scrub(model, remote_loader, adjacent_loader, forget_loader, optimizer, device,
          num_epochs=5, max_steps=10, alpha=0.1, gamma=0.9,
          apply_lr_decay=False, decay_rate=1.0):
    """SCRUB baseline (Kurmanji et al., 2023).

    Alternates:
      * max-step: maximise KL(model || teacher) on the forget set
      * min-step: minimise alpha * KL + gamma * CE on the retain set
    where the teacher is a frozen copy of the pre-unlearning model.
    """
    retain_loader = _concat_loader(remote_loader, adjacent_loader)
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False

    def max_step():
        for batch in forget_loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            student_logits = model_logits(model, inputs)
            teacher_prob = F.softmax(model_logits(teacher, inputs), dim=1)
            student_log_prob = F.log_softmax(student_logits, dim=1)
            loss = -nn.KLDivLoss(reduction="batchmean")(student_log_prob, teacher_prob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def min_step():
        for batch in retain_loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            labels = batch["labels"].to(device)
            student_logits = model_logits(model, inputs)
            teacher_prob = F.softmax(model_logits(teacher, inputs), dim=1)
            student_log_prob = F.log_softmax(student_logits, dim=1)
            kl = nn.KLDivLoss(reduction="batchmean")(student_log_prob, teacher_prob)
            ce = nn.CrossEntropyLoss()(student_logits, labels)
            loss = alpha * kl + gamma * ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    for epoch in tqdm.tqdm(range(num_epochs), desc="SCRUB"):
        if epoch < max_steps:
            max_step()
        min_step()
        if apply_lr_decay:
            scheduler.step()
    return model
