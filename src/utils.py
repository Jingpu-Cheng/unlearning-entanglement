"""General utilities: seeding, batch handling, device helpers."""
import random
import numpy as np
import torch
from transformers import BatchEncoding


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_dictionarize(batch):
    """Normalize DataLoader batches to a dict with 'images' and 'labels' keys.

    HuggingFace text batches already come as dicts (input_ids, attention_mask, labels);
    in that case we wrap them so that downstream code can use batch['images'] uniformly
    for the model inputs.
    """
    if isinstance(batch, dict):
        if "images" in batch and "labels" in batch:
            return batch
        # HF-style text batch: keep token tensors as 'images', rename 'label' -> 'labels'
        inputs = {k: v for k, v in batch.items() if k not in ("label", "labels")}
        labels = batch.get("labels", batch.get("label"))
        return {"images": inputs, "labels": labels}

    if len(batch) == 2:
        return {"images": batch[0], "labels": batch[1]}
    if len(batch) == 3:
        return {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    raise ValueError(f"Unexpected batch length: {len(batch)}")


def to_device(x, device):
    if isinstance(x, (dict, BatchEncoding)):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def model_logits(model, x):
    """Run forward pass and return logits regardless of model type (torchvision or HF)."""
    if isinstance(x, (dict, BatchEncoding)):
        out = model(**x)
    else:
        out = model(x)
    return out.logits if hasattr(out, "logits") else out
