"""Run a single unlearning method on the ToxiGen biased-pretraining task.

See the paper's Table 2 (§5.3). Usage:

    python run.py --method ours --checkpoint checkpoints/best
    python run.py --method {original,ft,ga,l1,scrub,ours,ours_no_w2}
"""
import argparse
import copy
import os
import sys
import time

import torch
import torch.optim as optim

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.baselines import finetune, gradient_ascent, l1_sparse, scrub  # noqa: E402
from src.baselines_external import gdr, munba, salun, ssd  # noqa: E402
from src.evaluation import evaluate_and_log  # noqa: E402
from src.method import lagrange_stage1, wpgd_stage2  # noqa: E402
from src.utils import set_seed  # noqa: E402

from common import SEEDS, build_loaders, load_roberta  # noqa: E402

METHODS = ("original", "ft", "ga", "l1", "scrub",
           "ssd", "salun", "munba", "gdr",
           "ours", "ours_no_w2")


def run_once(method, ckpt_dir, device, seed):
    set_seed(seed)
    model = load_roberta(ckpt_dir, device)
    (remote, adjacent, forget), (remote_va, adjacent_va, forget_va) = build_loaders(batch_size=64)

    t0 = time.time()

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
            optim.SGD(model.parameters(), lr=2.5e-6), device,
            num_epochs=10)

    elif method == "l1":
        model = l1_sparse(
            model, remote, adjacent, forget,
            optim.SGD(model.parameters(), lr=1e-3, momentum=0.9), device,
            num_epochs=10, alpha=5e-4)

    elif method == "scrub":
        model = scrub(
            model, remote, adjacent, forget,
            optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4), device,
            num_epochs=5, max_steps=3, alpha=0.1, gamma=0.9)

    elif method == "ssd":
        # Default dampening params; tuned in ssd_method_nlp/ssd_lgbtq.py.
        model = ssd(model, remote, adjacent, forget, device,
                    dampening_constant=1.0, selection_weighting=50.0)

    elif method == "salun":
        # Hyperparams from Unlearn-Saliency-master/.../run_toxigen.sh
        # (FT_forget with mask; unlearn_epochs=2, lr=1e-6, alpha=1.0).
        # The mask in the original script (``with_0.7.pt``) keeps the top 30%
        # of weights by |gradient|; in our wrapper that is fraction=0.3.
        model = salun(
            model, remote, adjacent, forget,
            optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.05), device,
            num_epochs=2, mask_fraction=0.3,
            lambda_forget=1.0, beta_sibs=1.0, beta_other=1.0,
            val_remote_loader=remote_va, val_adjacent_loader=adjacent_va,
            val_forget_loader=forget_va)

    elif method == "munba":
        model = munba(
            model, remote, adjacent, forget,
            optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4),
            device, num_epochs=3, num_classes=2)

    elif method == "gdr":
        model = gdr(
            model, remote, adjacent, forget,
            optim.AdamW(model.parameters(), lr=2e-5), device,
            num_epochs=3, gamma=100.0, epsilon=0.02)

    elif method in ("ours", "ours_no_w2"):
        alpha = 0.8 if method == "ours" else 0.0
        # small_punish=True is the ToxiGen-specific Stage 1 trick from the
        # original run: it keeps the forget gradient from concentrating on the
        # hardest samples by penalising already-confidently-predicted forget
        # examples. c=5.0 matches the original call.
        model = lagrange_stage1(
            model, remote, forget,
            optim.Adam(model.parameters(), lr=2.5e-6), device,
            num_epochs=1, mu=10.0, gamma=1.0, c=5.0,
            small_punish=True, small_punish_factor=1.0, seed=seed)

        print("-- after stage 1 --")
        evaluate_and_log(model, remote, adjacent, forget,
                         remote_va, adjacent_va, forget_va, device)

        ref_model = copy.deepcopy(model)
        model = wpgd_stage2(
            model, adjacent, forget, remote,
            optim.SGD(model.parameters(), lr=1e-5), device,
            num_epochs=2, batch_limit=10, alpha=alpha, momentum=0.7,
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
    parser.add_argument("--checkpoint", default="checkpoints/best",
                        help="Directory saved by pretrain.py (HuggingFace format).")
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
