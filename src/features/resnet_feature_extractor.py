from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


VALID_RESNET18_RETURN_NODES = ("layer1", "layer2", "layer3", "layer4")


def resolve_device(device: str = "auto") -> torch.device:
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def validate_return_nodes(return_nodes: tuple[str, ...]) -> None:
    if len(return_nodes) == 0:
        raise ValueError("return_nodes must contain at least one layer name.")
    invalid = [node for node in return_nodes if node not in VALID_RESNET18_RETURN_NODES]
    if invalid:
        raise ValueError(
            f"Unsupported return_nodes {invalid}. "
            f"Valid nodes: {list(VALID_RESNET18_RETURN_NODES)}"
        )


class ResNet18FeatureExtractor(nn.Module):
    """
    Runtime-only reusable multi-layer ResNet18 feature extractor.

    Default nodes:
    - layer2
    - layer3

    Output:
    - dict[str, torch.Tensor]
      Example:
      {
        "layer2": [B, 128, H2, W2],
        "layer3": [B, 256, H3, W3]
      }
    """

    def __init__(
        self,
        return_nodes: tuple[str, ...] = ("layer2", "layer3"),
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        validate_return_nodes(return_nodes)

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        node_mapping = {node: node for node in return_nodes}
        self.extractor = create_feature_extractor(backbone, return_nodes=node_mapping)

        self.backbone_name = "resnet18"
        self.pretrained = bool(pretrained)
        self.return_nodes = tuple(return_nodes)
        self.freeze_backbone = bool(freeze_backbone)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images shape [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3-channel images, got {images.shape[1]} channels")

        features = self.extractor(images)
        return {name: feat.contiguous() for name, feat in features.items()}

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        features = self.forward(images)
        if was_training:
            self.train()
        return features

    @torch.no_grad()
    def infer_output_shapes(
        self,
        image_size: int = 256,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
    ) -> dict[str, list[int]]:
        target_device = torch.device(device) if isinstance(device, str) else device
        dummy = torch.zeros(batch_size, 3, image_size, image_size, device=target_device)
        features = self.extract(dummy)
        return {name: list(feat.shape) for name, feat in features.items()}

    @torch.no_grad()
    def infer_output_channels(
        self,
        image_size: int = 256,
        device: str | torch.device = "cpu",
    ) -> dict[str, int]:
        shapes = self.infer_output_shapes(image_size=image_size, batch_size=1, device=device)
        return {name: int(shape[1]) for name, shape in shapes.items()}


def build_feature_extractor(
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    return_nodes: tuple[str, ...] = ("layer2", "layer3"),
    freeze_backbone: bool = True,
) -> ResNet18FeatureExtractor:
    if backbone_name != "resnet18":
        raise ValueError(
            f"Only 'resnet18' is supported in this GUI runtime build, got '{backbone_name}'."
        )

    return ResNet18FeatureExtractor(
        return_nodes=return_nodes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )