"""Evaluation helpers shared by all experiments.

The paper reports accuracy on three subsets: the forget set D_f, the adjacent
retain set D_r^adj (samples correlated with D_f), and the remote retain set
D_r^rem. These helpers compute loss/accuracy on each and print a line in the
same format used throughout the scripts.
"""
import torch
import torch.nn as nn

from .utils import maybe_dictionarize, to_device, model_logits


@torch.no_grad()
def evaluate_on_three_datasets(model, retain_loader_remote, retain_loader_adjacent,
                               forget_loader, device, eval_acc=True, eval_loss=True):
    """Return (loss_rem, loss_adj, loss_f, acc_rem, acc_adj, acc_f).

    Any of the three loaders may be None, in which case its metrics are returned as 0.0.
    """
    model.eval()
    losses = {"rem": 0.0, "adj": 0.0, "f": 0.0}
    correct = {"rem": 0, "adj": 0, "f": 0}
    total = {"rem": 0, "adj": 0, "f": 0}

    for key, loader in zip(("rem", "adj", "f"),
                           (retain_loader_remote, retain_loader_adjacent, forget_loader)):
        if loader is None:
            continue
        for batch in loader:
            batch = maybe_dictionarize(batch)
            inputs = to_device(batch["images"], device)
            labels = batch["labels"].to(device)
            logits = model_logits(model, inputs)

            if eval_loss:
                losses[key] += nn.CrossEntropyLoss(reduction="sum")(logits, labels).item()
            if eval_acc:
                pred = logits.argmax(dim=1)
                correct[key] += pred.eq(labels).sum().item()
                total[key] += labels.size(0)

    def avg_loss(key, loader):
        if loader is None or not eval_loss:
            return 0.0
        n = len(loader.dataset)
        return losses[key] / n if n > 0 else 0.0

    def acc(key):
        if not eval_acc or total[key] == 0:
            return 0.0
        return 100.0 * correct[key] / total[key]

    return (avg_loss("rem", retain_loader_remote),
            avg_loss("adj", retain_loader_adjacent),
            avg_loss("f", forget_loader),
            acc("rem"), acc("adj"), acc("f"))


def evaluate_and_log(model, retain_loader_remote_train, retain_loader_adjacent_train,
                     forget_loader_train, retain_loader_remote_val, retain_loader_adjacent_val,
                     forget_loader_val, device, epoch=None, num_epochs=None,
                     print_loss=True, print_accuracy=True):
    """Evaluate on train+val and print one summary line for each."""
    l_rem_tr, l_adj_tr, l_f_tr, a_rem_tr, a_adj_tr, a_f_tr = evaluate_on_three_datasets(
        model, retain_loader_remote_train, retain_loader_adjacent_train,
        forget_loader_train, device, eval_acc=print_accuracy, eval_loss=print_loss)
    l_rem_va, l_adj_va, l_f_va, a_rem_va, a_adj_va, a_f_va = evaluate_on_three_datasets(
        model, retain_loader_remote_val, retain_loader_adjacent_val,
        forget_loader_val, device, eval_acc=print_accuracy, eval_loss=print_loss)

    if epoch is not None:
        print(f"Epoch {epoch + 1}/{num_epochs}:")
    if print_loss:
        print(f"  loss  | train rem/adj/forget = "
              f"{l_rem_tr:.4f} / {l_adj_tr:.4f} / {l_f_tr:.4f}   "
              f"val = {l_rem_va:.4f} / {l_adj_va:.4f} / {l_f_va:.4f}")
    if print_accuracy:
        print(f"  acc%  | train rem/adj/forget = "
              f"{a_rem_tr:.2f} / {a_adj_tr:.2f} / {a_f_tr:.2f}   "
              f"val = {a_rem_va:.2f} / {a_adj_va:.2f} / {a_f_va:.2f}")
