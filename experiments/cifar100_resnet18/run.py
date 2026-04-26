"""Run a single unlearning method on the CIFAR-100 super-class task.

Usage examples (reproduce Table 1):

    python run.py --method original       # evaluate the pretrained model
    python run.py --method ft
    python run.py --method ga
    python run.py --method l1
    python run.py --method scrub
    python run.py --method ssd            # third-party baselines
    python run.py --method salun
    python run.py --method munba
    python run.py --method gdr
    python run.py --method ours           # two-stage method, alpha=0.5
    python run.py --method ours_no_w2     # §5.6 ablation, alpha=0.0

Each method is run over 3 seeds (2025, 2026, 2027) matching the paper.
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

from common import (  # noqa: E402
    SEEDS, build_loaders, build_mia_loaders, build_ssd_importance_loaders,
    load_pretrained,
)

METHODS = ("original", "ft", "ga", "l1", "scrub",
           "ssd", "salun", "munba", "gdr",
           "ours", "ours_no_w2")


def run_once(method, ckpt_path, device, seed):
    set_seed(seed)
    t0 = time.time()

    # For our method we need the feature-extractor wrapper; baselines use the
    # raw pretrained ResNet.
    wrap = method in ("ours", "ours_no_w2")
    model = load_pretrained(ckpt_path, device, wrap_with_feature_extractor=wrap)

    if method == "ours" or method == "ours_no_w2":
        train_loaders, val_loaders = build_loaders(batch_size=(128, 16, 16))
    elif method == "salun":
        train_loaders, val_loaders = build_loaders(batch_size=(256, 64, 64))
    elif method == "scrub":
        train_loaders, val_loaders = build_loaders(batch_size=(128, 128, 192))
    else:
        train_loaders, val_loaders = build_loaders(batch_size=(128, 128, 128))
    remote, adjacent, forget = train_loaders
    remote_va, adjacent_va, forget_va = val_loaders

    if method == "original":
        pass

    elif method == "ft":
        model = finetune(
            model, remote, adjacent, forget,
            optim.Adam(model.parameters(), lr=2.5e-5), device,
            num_epochs=10)

    elif method == "ga":
        model = gradient_ascent(
            model, forget,
            optim.SGD(model.parameters(), lr=1.5e-5), device,
            num_epochs=10)

    elif method == "l1":
        model = l1_sparse(
            model, remote, adjacent, forget,
            optim.SGD(model.parameters(), lr=1e-3, momentum=0.9), device,
            num_epochs=10, alpha=5e-4)

    elif method == "scrub":
        model = scrub(
            model, remote, adjacent, forget,
            optim.Adam(model.parameters(), lr=5e-5), device,
            num_epochs=10, max_steps=5, alpha=0.1, gamma=0.9)

    elif method == "ssd":
        # Hyperparams match ssd_method/forget_subclass_strategies.py (pdr_tuning).
        # The pretrained ResNet-18 is trained at 224x224, but the original
        # SSD baseline reads its inputs at 32x32 with CIFAR statistics --
        # see common.SSD_TRANSFORM and the docstring of src.baselines_external.ssd
        # for the rationale. We pass dedicated importance loaders so the rest
        # of the pipeline is unaffected.
        ssd_full, ssd_forget = build_ssd_importance_loaders(batch_size=64)
        model = ssd(model, remote, adjacent, forget, device,
                    dampening_constant=1.0, selection_weighting=10.0,
                    importance_loader=ssd_full,
                    importance_forget_loader=ssd_forget)

    elif method == "salun":
        # Hyperparams from Unlearn-Saliency-master/Classification/run_resnet.sh.
        model = salun(
            model, remote, adjacent, forget,
            optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05), device,
            num_epochs=10, mask_fraction=0.5,
            lambda_forget=1.0, beta_sibs=1.0, beta_other=2.0,
            val_remote_loader=remote_va, val_adjacent_loader=adjacent_va,
            val_forget_loader=forget_va)
    elif method == "munba":
        model = munba(
            model, remote, adjacent, forget,
            optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4),
            device, num_epochs=5, num_classes=20)

    elif method == "gdr":
        model = gdr(
            model, remote, adjacent, forget,
            optim.AdamW(model.parameters(), lr=1e-4), device,
            num_epochs=10, gamma=100.0, epsilon=0.02)

    elif method in ("ours", "ours_no_w2"):
        alpha = 0.5 if method == "ours" else 0.0

        model = lagrange_stage1(
            model, remote, forget,
            optim.Adam(model.parameters(), lr=2.5e-6), device,
            num_epochs=1, mu=10.0, gamma=1.0, seed=seed)

        print("-- after stage 1 --")
        evaluate_and_log(model, remote, adjacent, forget,
                         remote_va, adjacent_va, forget_va, device)

        ref_model = copy.deepcopy(model)
        remote_big = DataLoader(remote.dataset, batch_size=512,
                                num_workers=16, shuffle=True)
        adjacent_big = DataLoader(adjacent.dataset, batch_size=128,
                                  num_workers=16, shuffle=True)
        forget_big = DataLoader(forget.dataset, batch_size=128,
                                num_workers=16, shuffle=True)

        model = wpgd_stage2(
            model, adjacent_big, forget_big, remote_big,
            optim.SGD(model.parameters(), lr=2e-5), device,
            num_epochs=6, batch_limit=10, alpha=alpha,
            ref_model=ref_model, seed=seed)

    else:
        raise ValueError(f"unknown method {method}")

    print(f"-- after {method} (seed={seed}, time={time.time() - t0:.1f}s) --")
    evaluate_and_log(model, remote, adjacent, forget,
                     remote_va, adjacent_va, forget_va, device)

    if os.environ.get("UNLEARN_MIA") == "1":
        from src.SVC_MIA import SVC_MIA  # noqa: E402 -- optional dependency
        shadow_train, shadow_test = build_mia_loaders(train_loaders, val_loaders, seed=seed)
        SVC_MIA(shadow_train, None, forget_va, shadow_test, model)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint",
                        default="checkpoints/resnet18_cifar100_super.pt",
                        help="Pretrained model saved by pretrain.py.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        print("=" * 60)
        print(f"method={args.method}  seed={seed}")
        print("=" * 60)
        run_once(args.method, args.checkpoint, device, seed)


if __name__ == "__main__":
    main()
