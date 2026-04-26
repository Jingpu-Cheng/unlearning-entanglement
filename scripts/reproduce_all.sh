#!/usr/bin/env bash
# Reproduces the paper's main-text tables by running every method on every
# dataset over three seeds (2025, 2026, 2027).
#
# Expected checkpoints (produced by each experiment's pretrain.py, or dropped
# in manually) live in $REPO/checkpoints/ with these names:
#
#     $REPO/checkpoints/cifar100_resnet18.pt
#     $REPO/checkpoints/toxigen_roberta/               (HuggingFace save_pretrained dir)
#     $REPO/checkpoints/celeba_vit.pt
#     $REPO/checkpoints/tiny_imagenet_vit.pt
#
# Environment variables:
#     DEVICE       GPU to use (default cuda:0)
#     TINY_ROOT    required: path to the tiny-imagenet-200 folder
#     CELEBA_ROOT  optional: path to the CelebA root (default $REPO/data)

set -euo pipefail

DEVICE=${DEVICE:-cuda:1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CKPT="$REPO/checkpoints"
CELEBA_ROOT=${CELEBA_ROOT:-$REPO/data}
# TINY_ROOT is only required if the Tiny-ImageNet section is enabled; checked there.

METHODS=(original ft ga l1 scrub ssd salun munba gdr ours ours_no_w2)

echo "### Table 1 + Table 5: CIFAR-100 / ResNet-18"
cd "$REPO/experiments/cifar100_resnet18"
for m in "${METHODS[@]}"; do
  python run.py --method "$m" --device "$DEVICE" \
      --checkpoint "$CKPT/cifar100_resnet18.pt"
done

NOTE: later tables temporarily commented out; uncomment to re-enable.
echo "### Table 2: ToxiGen / RoBERTa-base"
cd "$REPO/experiments/toxigen_roberta"
for m in "${METHODS[@]}"; do
  python run.py --method "$m" --device "$DEVICE" \
      --checkpoint "$CKPT/toxigen_roberta"
done

echo "### Table 3: CelebA / ViT-B"
cd "$REPO/experiments/celeba_vit"
for m in "${METHODS[@]}"; do
  python run.py --method "$m" --device "$DEVICE" \
      --checkpoint "$CKPT/celeba_vit.pt" \
      --data-root "$CELEBA_ROOT"
done

echo "### Table 4: Tiny-ImageNet / ViT-B"
: "${TINY_ROOT:?set TINY_ROOT to the tiny-imagenet-200 folder}"
cd "$REPO/experiments/tiny_imagenet_vit"
for m in "${METHODS[@]}"; do
  python run.py --method "$m" --device "$DEVICE" \
      --checkpoint "$CKPT/tiny_imagenet_vit.pt" \
      --data-root "$TINY_ROOT"
done
