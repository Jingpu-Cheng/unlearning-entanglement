"""Run a single unlearning method on the Tiny-ImageNet super-class task (Table 4).

Forget classes are the six "dog" sub-classes (indices 24-29) inside the
"mammals" super-class.

    python run.py --method ours --data-root /path/to/tiny-imagenet-200
    python run.py --method {original,ft,ga,l1,scrub,ours,ours_no_w2}
"""
import argparse
import copy
import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.baselines import finetune, gradient_ascent, l1_sparse, scrub  # noqa: E402
from src.baselines_external import gdr, munba, salun, ssd  # noqa: E402
from src.evaluation import evaluate_and_log  # noqa: E402
from src.method import lagrange_stage1, wpgd_stage2  # noqa: E402
from src.utils import set_seed  # noqa: E402

from common import SEEDS, build_loaders, load_pretrained  # noqa: E402

METHODS = ("original", "ft", "ga", "l1", "scrub",
           "ssd", "salun", "munba", "gdr",
           "ours", "ours_no_w2")


def run_once(method, ckpt_path, data_root, device, seed):
    set_seed(seed)
    model = load_pretrained(ckpt_path, device)

    if method == "ours" or method == "ours_no_w2":
        train_loaders, val_loaders = build_loaders(data_root, batch_size=(128, 128, 64))
    else:
        train_loaders, val_loaders = build_loaders(data_root, batch_size=(128, 128, 128))
    remote, adjacent, forget = train_loaders
    remote_va, adjacent_va, forget_va = val_loaders

    t0 = time.time()
    if method == "original":
        pass

    elif method == "ft":
        model = finetune(
            model, remote, adjacent, forget,
            optim.Adam(model.parameters(), lr=5e-5), device, num_epochs=10)

    elif method == "ga":
        model = gradient_ascent(
            model, forget,
            optim.Adam(model.parameters(), lr=1.5e-6), device, num_epochs=10)

    elif method == "l1":
        model = l1_sparse(
            model, remote, adjacent, forget,
            optim.Adam(model.parameters(), lr=2e-5), device,
            num_epochs=10, alpha=2e-4)

    elif method == "scrub":
        model = scrub(
            model, remote, adjacent, forget,
            optim.Adam(model.parameters(), lr=5e-5), device,
            num_epochs=5, max_steps=5, alpha=0.1, gamma=0.5,
            apply_lr_decay=True, decay_rate=0.3)

    elif method == "ssd":
        # ssd_method/run_vit_imagenet_superclass.sh uses alpha=10, lam=1 for the ViT.
        model = ssd(model, remote, adjacent, forget, device,
                    dampening_constant=10.0, selection_weighting=1.0)

    elif method == "salun":
        model = salun(
            model, remote, adjacent, forget,
            optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05), device,
            num_epochs=10, mask_fraction=0.5,
            lambda_forget=1.0, beta_sibs=1.0, beta_other=1.0,
            val_remote_loader=remote_va, val_adjacent_loader=adjacent_va,
            val_forget_loader=forget_va)

    elif method == "munba":
        model = munba(
            model, remote, adjacent, forget,
            optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4),
            device, num_epochs=5, num_classes=10)

    elif method == "gdr":
        model = gdr(
            model, remote, adjacent, forget,
            optim.AdamW(model.parameters(), lr=1e-4), device,
            num_epochs=10, gamma=100.0, epsilon=0.02)

    elif method in ("ours", "ours_no_w2"):
        alpha = 0.5 if method == "ours" else 0.0
        model = lagrange_stage1(
            model, remote, forget,
            optim.Adam(model.parameters(), lr=5e-5), device,
            num_epochs=1, mu=10.0, gamma=1.0, c=10, seed=seed)

        print("-- after stage 1 --")
        evaluate_and_log(model, remote, adjacent, forget,
                         remote_va, adjacent_va, forget_va, device)

        ref_model = copy.deepcopy(model)
        remote_big = DataLoader(remote.dataset, batch_size=512, num_workers=16, shuffle=True)
        adjacent_big = DataLoader(adjacent.dataset, batch_size=256, num_workers=16, shuffle=True)
        forget_big = DataLoader(forget.dataset, batch_size=128, num_workers=16, shuffle=True)

        model = wpgd_stage2(
            model, adjacent_big, forget_big, remote_big,
            optim.SGD(model.parameters(), lr=2e-4), device,
            num_epochs=5, batch_limit=1, alpha=alpha,
            ref_model=ref_model, seed=seed)

    else:
        raise ValueError(f"unknown method {method}")

    print(f"-- after {method} (seed={seed}, time={time.time() - t0:.1f}s) --")
    evaluate_and_log(model, remote, adjacent, forget,
                     remote_va, adjacent_va, forget_va, device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint", default="checkpoints/vit_tinyimagenet_super.pt")
    parser.add_argument("--data-root", required=True,
                        help="Path to tiny-imagenet-200 folder.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        print("=" * 60)
        print(f"method={args.method}  seed={seed}")
        print("=" * 60)
        run_once(args.method, args.checkpoint, args.data_root, device, seed)


if __name__ == "__main__":
    main()
