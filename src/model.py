from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import timm
except ImportError:
    timm = None

try:
    from torchvision import models as tv_models
except ImportError:
    tv_models = None


def create_dummy_baseline(strategy: str = "most_frequent") -> DummyClassifier:
    return DummyClassifier(strategy=strategy, random_state=SEED)


def create_color_logreg_pipeline(c_param: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c_param, max_iter=max_iter, class_weight="balanced", random_state=SEED)),
    ])


def create_deep_model(
    arch_name: str = "resnet18",
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    if torch is None:
        raise ImportError("PyTorch is required.")

    model = None
    if timm is not None:
        try:
            model = timm.create_model(arch_name, pretrained=pretrained, num_classes=num_classes)
        except Exception:
            model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not any(head_key in name for head_key in ["fc", "classifier", "head"]):
                    param.requires_grad = False
        return model

    if "mobilenet" in arch_name.lower():
        try:
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = tv_models.mobilenet_v3_small(weights=weights)
        except Exception:
            model = tv_models.mobilenet_v3_small(weights=None)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        try:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
        except Exception:
            model = tv_models.resnet18(weights=None)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
