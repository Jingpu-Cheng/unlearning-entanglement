"""Verbatim port of GDR-GMA's training kernel
(``gdr/GDR-GMA/unlearn.py``) plus the small ``MemoryBank`` helper from
``gdr/GDR-GMA/memory_bank.py``.


Upstream: gdr/GDR-GMA/unlearn.py + gdr/GDR-GMA/memory_bank.py
"""
import copy
import math
import time

import torch
import torch.nn as nn
from transformers import BatchEncoding


# ============================ MemoryBank (verbatim) ============================


class MemoryBank:
    def __init__(self, size):
        self.grads = []
        self.size = size

    def update(self, grads):
        self.grads.append(grads)
        if len(self.grads) > self.size:
            del self.grads[0]

    def get_graident(self, model: nn.Module):
        gradient = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                grad = param.grad.clone().detach()
                gradient.append(grad.view(-1))
        return gradient

    def mean_grads(self, t_grads):
        grads = []
        for grad in self.grads:
            if torch.cosine_similarity(grad, t_grads, dim=0) < 0:
                grads.append(grad)
        if len(grads) > 0:
            avg_grad = grads[0]
            for grad in grads[1:]:
                avg_grad += grad
            avg_grad = avg_grad / len(grads)
            return avg_grad
        else:
            return None


# ============================ helpers (verbatim) ============================


def _to_device(x, device):
    if isinstance(x, (dict, BatchEncoding)):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def _model_logits(model, x):
    if isinstance(x, (dict, BatchEncoding)):
        out = model(**x)
    else:
        out = model(x)
    return out.logits if hasattr(out, "logits") else out


def get_gradient(model: nn.Module):
    gradient = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            grad = param.grad.clone().detach()
            gradient.append(grad.view(-1))
    return gradient


def rectify_graident(grads_x, grads_y):
    r_grads_x = []
    r_grads_y = []
    for x, y in zip(grads_x, grads_y):
        if torch.cosine_similarity(x, y, dim=0) < 0:
            InP_xy = torch.matmul(y, x)
            Inp_xx = torch.norm(x, p=2) ** 2
            Inp_yy = torch.norm(y, p=2) ** 2
            x = x - InP_xy/Inp_yy * y
            y = y - InP_xy/Inp_xx * x
        r_grads_x.append(x)
        r_grads_y.append(y)
    return r_grads_x, r_grads_y


def val_model(model: nn.Module, test_loader, device):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            data = _to_device(data, device)
            targets = targets.to(device)
            outputs = _model_logits(model, data)
            loss = criterion(outputs, targets)
            pred = torch.argmax(outputs, dim=1)
            val_loss += loss.item()
            val_corrects += torch.sum(pred == targets)
    val_loss /= len(test_loader.dataset)
    val_acc = val_corrects.double() / len(test_loader.dataset)
    return val_loss, val_acc


# ============================ unlearn_model (verbatim, args→params) ============================


def unlearn_model(model: nn.Module, criterion, optimizer, data_loaders,
                  num_epochs, device, batch_size):
    """Body is byte-identical to ``gdr/GDR-GMA/unlearn.py::unlearn_model``;
    only the function signature is widened so ``device`` / ``batch_size`` come
    in as explicit arguments instead of via a global ``args`` namespace.
    """
    t_dataset_sizes = len(data_loaders['unlearn'].dataset)
    bank = MemoryBank(size=math.ceil(t_dataset_sizes / batch_size))

    for epoch in range(num_epochs):
        begin_time = time.time()
        running_loss = 0.0
        running_corrects = 0.0

        for (t_data, t_labels), (n_data, n_labels) in zip(
            data_loaders['unlearn'], data_loaders['remain']
        ):
            model.train()

            # -------- remain branch --------
            n_data = _to_device(n_data, device)
            n_labels = n_labels.to(device)

            optimizer.zero_grad()
            n_outputs = _model_logits(model, n_data)
            n_loss = criterion(n_outputs, n_labels)
            n_loss.backward()

            n_grads = get_gradient(model)

            # -------- unlearn branch (GDR-GMA on forget) --------
            t_data = _to_device(t_data, device)
            t_labels = t_labels.to(device)

            t_outputs = _model_logits(model, t_data)
            optimizer.zero_grad()

            t_loss = -criterion(t_outputs, t_labels)
            t_loss.backward()

            t_grads = get_gradient(model)

            bank.update(t_grads[-1])

            r_n_grads, r_t_grads = rectify_graident(n_grads, t_grads)
            if epoch > 0 and bank.mean_grads(r_t_grads[-1]) is not None:
                grads, _ = rectify_graident(
                    [r_t_grads[-1]],
                    [bank.mean_grads(r_t_grads[-1])]
                )
                r_t_grads[-1] = grads[-1]

            with torch.no_grad():
                gamma, epsilon = 100, 0.02
                lambda_weight = 1 / (1 + torch.exp(gamma * (n_loss - epsilon)))

            optimizer.zero_grad()
            for idx, (_, param) in enumerate(model.named_parameters()):
                if param.requires_grad:
                    param.grad = (
                        (1 - lambda_weight) * r_n_grads[idx]
                        + lambda_weight * r_t_grads[idx]
                    ).view(param.size())

            optimizer.step()

            preds = torch.argmax(t_outputs.data, dim=1)
            running_loss += t_loss.item()
            running_corrects += torch.sum(preds == t_labels.data)

        end_time = time.time()
        epoch_loss = running_loss / len(data_loaders['unlearn'])
        epoch_acc = float(running_corrects) / len(data_loaders['unlearn'].dataset)

        # End-of-epoch sanity prints (verbatim, but val loaders are optional here).
        try:
            u_loss, u_acc = val_model(model, data_loaders['unlearn'], device)
            r_loss, r_acc = val_model(model, data_loaders['remain'], device)
            v_loss, v_acc = val_model(model, data_loaders['val'], device)
            print(
                f'Epoch: {epoch} - u_loss: {u_loss:.4f} - u_acc: {u_acc:.4f} '
                f'- r_loss: {r_loss:.4f} - r_acc: {r_acc:.4f} '
                f'- v_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} '
                f'- time: {end_time - begin_time:.2f}s'
            )
        except KeyError:
            print(f"Epoch: {epoch} - epoch_loss: {epoch_loss:.4f} "
                  f"- epoch_acc: {epoch_acc:.4f} - time: {end_time - begin_time:.2f}s")

    return model
