"""The two-stage unlearning method from "Machine Unlearning under Retain-Forget Entanglement".

- Stage 1 (``lagrange_stage1``): an augmented-Lagrangian step that *increases*
  loss on the forget set while keeping the mean loss on the remote retain set
  pinned to its pre-unlearning value (Alg.1, Stage 1 in the paper).

- Stage 2 (``wpgd_stage2``): W_2-distance guided projected gradient descent
  that restores accuracy on the adjacent retain set without re-memorising the
  forget set. The modified forget-set loss is L_tilde = (1-alpha) * CE + alpha *
  W_2^2(P_forget(theta_bar), P_forget(theta)), and the parameter update is
  projected onto the orthogonal complement of span{grad L_tilde, grad L_rem}.

Together these functions implement Algorithm 1 of the paper.
"""
import copy
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from .utils import maybe_dictionarize, set_seed, to_device, model_logits


class ClippedCrossEntropyLoss(nn.Module):
    """Cross entropy clipped at a maximum per-sample value ``c``.

    Used in Stage 1 so that the forget-loss term used inside the augmented
    Lagrangian cannot diverge to infinity on a single hard sample. We use the
    "detached tail" trick so that the forward value is still the real CE while
    the gradient past ``c`` is zero:

        clipped = where(ce < c, ce, c + (ce - c).detach())

    When ``small_punish`` is enabled (used in the ToxiGen experiment), the
    clipped per-sample loss is additionally reduced by
    ``small_punish_factor / (ce + 1e-3)``. This pushes samples that the model
    is already very confident about (small CE) towards larger loss, preventing
    the forget-term gradient from being dominated by the hardest samples.
    """

    def __init__(self, c=10.0, reduction="mean",
                 small_punish=False, small_punish_factor=0.1):
        super().__init__()
        self.c = c
        self.reduction = reduction
        self.small_punish = small_punish
        self.small_punish_factor = small_punish_factor

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        clipped = torch.where(ce < self.c, ce, self.c + (ce - self.c).detach())
        if self.small_punish:
            clipped = clipped - self.small_punish_factor / (ce + 1e-3)
        if self.reduction == "mean":
            return clipped.mean()
        if self.reduction == "sum":
            return clipped.sum()
        return clipped


def lagrange_stage1(model, remote_loader, forget_loader, optimizer, device,
                    num_epochs=1, mu=10.0, gamma=1.0, c=10.0,
                    small_punish=False, small_punish_factor=0.1,
                    apply_lr_decay=False, decay_rate=1.0, seed=2024):
    """Stage 1: augmented Lagrangian (Eq. 2-4 of the paper).

    Maximises (clipped) CE on ``forget_loader`` subject to the constraint that
    the mean CE on ``remote_loader`` (the remote retain set D_r^rem) stays
    equal to its value under the pretrained model.

    ``remote_loader`` corresponds to D_r^rem, ``forget_loader`` to D_f.
    The adjacent retain set D_r^adj is deliberately *not* used here; it is
    handled in Stage 2.

    ``small_punish`` / ``small_punish_factor`` are forwarded to
    ``ClippedCrossEntropyLoss`` and are used by the ToxiGen experiment.
    """
    set_seed(seed)
    multiplier = torch.zeros((), device=device)

    # Compute L_rem(theta_0): the pinned value the constraint enforces.
    with torch.no_grad():
        total_loss, total_samples = 0.0, 0
        for batch in remote_loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            labels = batch["labels"].to(device)
            logits = model_logits(model, inputs)
            total_loss += nn.CrossEntropyLoss(reduction="sum")(logits, labels).item()
            total_samples += labels.size(0)
    pinned_remote_loss = total_loss / max(total_samples, 1)

    cycled_forget = itertools.cycle(forget_loader)

    scheduler = None
    if apply_lr_decay:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Stage 1 (augmented Lagrangian)"):
        for batch_rem in remote_loader:
            batch_rem = maybe_dictionarize(batch_rem)
            inputs_rem = to_device(batch_rem["images"], device)
            labels_rem = batch_rem["labels"].to(device)

            batch_f = maybe_dictionarize(next(cycled_forget))
            inputs_f = to_device(batch_f["images"], device)
            labels_f = batch_f["labels"].to(device)

            optimizer.zero_grad()

            logits_rem = model_logits(model, inputs_rem)
            loss_rem = nn.CrossEntropyLoss()(logits_rem, labels_rem)
            logits_f = model_logits(model, inputs_f)
            loss_f = -ClippedCrossEntropyLoss(
                c=c, small_punish=small_punish,
                small_punish_factor=small_punish_factor)(logits_f, labels_f)

            constraint = loss_rem - pinned_remote_loss
            loss = gamma * loss_f + multiplier * constraint + 0.5 * mu * constraint ** 2
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                multiplier += mu * constraint.item()

        if scheduler is not None:
            scheduler.step()
    return model


def project_gradients(grads_primary, grads_basis_1, grads_basis_2):
    """Project ``grads_primary`` onto the orthogonal complement of
    span{grads_basis_1, grads_basis_2}.

    Solves a single 2x2 Gram system to obtain the linear-combination
    coefficients, then subtracts that combination from the primary gradient.
    In Stage 2 the primary gradient is ∇θL_r^adj, and the basis is
    {∇θL̃_f, ∇θL_r^rem} (Eq. 10 in the paper).
    """
    S11 = S12 = S22 = 0.0
    Su1 = Su2 = 0.0
    for gu, g1, g2 in zip(grads_primary, grads_basis_1, grads_basis_2):
        gu_f = gu.view(-1).to(dtype=torch.float64, device=g1.device)
        g1_f = g1.view(-1).to(dtype=torch.float64, device=g1.device)
        g2_f = g2.view(-1).to(dtype=torch.float64, device=g1.device)
        S11 += torch.dot(g1_f, g1_f) + 1e-8
        S12 += torch.dot(g1_f, g2_f)
        S22 += torch.dot(g2_f, g2_f) + 1e-8
        Su1 += torch.dot(gu_f, g1_f)
        Su2 += torch.dot(gu_f, g2_f)

    A = torch.tensor([[S11, S12], [S12, S22]])
    b = torch.tensor([Su1, Su2])
    coefs = torch.linalg.solve(A, b)
    return [gu - coefs[0] * g1 - coefs[1] * g2
            for gu, g1, g2 in zip(grads_primary, grads_basis_1, grads_basis_2)]


def _accumulate_gradients(loader, model, params, device, negative=False, batch_limit=None):
    """Cross-entropy gradient accumulated over (at most ``batch_limit``) batches."""
    grads_sum = tuple(torch.zeros_like(p) for p in params)
    sign = -1.0 if negative else 1.0
    for i, batch in enumerate(loader):
        if batch_limit is not None and i >= batch_limit:
            break
        batch = maybe_dictionarize(batch)
        inputs = to_device(batch["images"], device)
        labels = batch["labels"].to(device)
        logits = model_logits(model, inputs)
        loss = sign * nn.CrossEntropyLoss()(logits, labels)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        grads = tuple(g if g is not None else torch.zeros_like(p)
                      for p, g in zip(params, grads))
        grads_sum = tuple(gs + g for gs, g in zip(grads_sum, grads))
    return grads_sum


def _accumulate_forget_gradients_w2(loader, model, ref_model, params, device,
                                    alpha=0.5, batch_limit=None, sort_losses=True):
    """Gradient of the modified forget-loss L_tilde = (1-alpha) CE + alpha W_2^2.

    W_2^2 between the empirical loss distribution under the current model and
    that under ``ref_model`` (the post-Stage-1 reference) admits a closed form
    obtained by sorting the two vectors of per-sample losses (Eq. 7).
    """
    grads_sum = tuple(torch.zeros_like(p) for p in params)
    for i, batch in enumerate(loader):
        if batch_limit is not None and i >= batch_limit:
            break
        batch = maybe_dictionarize(batch)
        inputs = to_device(batch["images"], device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            ref_logits = model_logits(ref_model, inputs)
            # Negative sign because we are descending on -L_f in Stage 2.
            ref_losses = -nn.CrossEntropyLoss(reduction="none")(ref_logits, labels)

        logits = model_logits(model, inputs)
        cur_losses = -nn.CrossEntropyLoss(reduction="none")(logits, labels)
        mean_ce = cur_losses.mean()

        if sort_losses:
            cur_sorted, _ = torch.sort(cur_losses)
            ref_sorted, _ = torch.sort(ref_losses)
        else:
            cur_sorted, ref_sorted = cur_losses, ref_losses
        w2_sq = ((cur_sorted - ref_sorted) ** 2).mean()

        loss = (1.0 - alpha) * mean_ce + alpha * w2_sq
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        grads = tuple(g if g is not None else torch.zeros_like(p)
                      for p, g in zip(params, grads))
        grads_sum = tuple(gs + g for gs, g in zip(grads_sum, grads))
    return grads_sum


def wpgd_stage2(model, adjacent_loader, forget_loader, remote_loader, optimizer, device,
                num_epochs=6, momentum=0.9, batch_limit=None, alpha=0.5,
                sort_losses=True, ref_model=None, seed=2024):
    """Stage 2: W_2-distance guided projected gradient descent (W-PGD).

    Recovers accuracy on the adjacent retain set ``adjacent_loader`` (D_r^adj)
    by stepping along grad L^adj projected onto the orthogonal complement of
    span{grad L_tilde_f, grad L^rem}. Here ``ref_model`` is a frozen copy of
    the model at the end of Stage 1; it is used to compute the W_2 term.
    """
    set_seed(seed)
    if ref_model is None:
        ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    velocity = [torch.zeros_like(p) for p in params]

    for _ in tqdm.tqdm(range(num_epochs), desc="Stage 2 (W-PGD)"):
        for batch in adjacent_loader:
            batch = maybe_dictionarize(batch)
            inputs_adj = to_device(batch["images"], device)
            labels_adj = batch["labels"].to(device)

            optimizer.zero_grad()
            logits_adj = model_logits(model, inputs_adj)
            loss_adj = nn.CrossEntropyLoss()(logits_adj, labels_adj)
            grads_adj = torch.autograd.grad(loss_adj, params, allow_unused=True)
            grads_adj = tuple(g if g is not None else torch.zeros_like(p)
                              for p, g in zip(params, grads_adj))

            grads_forget = _accumulate_forget_gradients_w2(
                forget_loader, model, ref_model, params, device,
                alpha=alpha, batch_limit=batch_limit, sort_losses=sort_losses)
            grads_remote = _accumulate_gradients(
                remote_loader, model, params, device,
                negative=True, batch_limit=batch_limit)

            # Apply momentum to the primary direction, then project.
            for i, g in enumerate(grads_adj):
                velocity[i] = momentum * velocity[i] + g
            projected = project_gradients(velocity, grads_forget, grads_remote)

            with torch.no_grad():
                for i, (param, p_grad) in enumerate(zip(params, projected)):
                    velocity[i] = p_grad
                    if param.grad is None:
                        param.grad = p_grad.detach().clone()
                    else:
                        param.grad.copy_(p_grad)
            optimizer.step()

    return model
