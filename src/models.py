"""Model wrappers used in the paper's experiments."""
import torch
import torch.nn as nn


class ResNetWithFeatureExtraction(nn.Module):
    """A ResNet wrapper that can return either pooled features or logits.

    Used by the CIFAR-100 ResNet-18 experiment (§5.2 / Table 1).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Everything up to (and including) the global avg-pool.
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.fc = model.fc

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        if return_features:
            return features
        return self.fc(features)


class ViTForImageClassification(nn.Module):
    """Minimal ViT + linear classifier wrapper."""

    def __init__(self, vit_model, num_classes, feat_dim=512):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.vit(x))
