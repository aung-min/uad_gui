from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.data.mvtec_ad_dataset import VALID_MVTEC_AD_CATEGORIES


def align_feature_maps_to_reference(
    feature_maps: dict[str, torch.Tensor],
    reference_node: str,
) -> dict[str, torch.Tensor]:
    if reference_node not in feature_maps:
        raise ValueError(
            f"reference_node='{reference_node}' not found in feature maps: {list(feature_maps.keys())}"
        )

    ref_h, ref_w = feature_maps[reference_node].shape[-2:]
    aligned: dict[str, torch.Tensor] = {}

    for name, feat in feature_maps.items():
        if feat.shape[-2:] == (ref_h, ref_w):
            aligned[name] = feat.contiguous()
        else:
            aligned[name] = F.interpolate(
                feat,
                size=(ref_h, ref_w),
                mode="bilinear",
                align_corners=False,
            ).contiguous()

    return aligned


def combine_multi_layer_patch_embeddings(
    feature_maps: dict[str, torch.Tensor],
    reference_node: str,
    return_nodes: tuple[str, ...],
    l2_normalize_embeddings: bool = True,
) -> tuple[torch.Tensor, tuple[int, int]]:
    missing = [node for node in return_nodes if node not in feature_maps]
    if missing:
        raise ValueError(f"Missing requested return nodes in feature maps: {missing}")

    aligned = align_feature_maps_to_reference(
        feature_maps={node: feature_maps[node] for node in return_nodes},
        reference_node=reference_node,
    )

    tensors = [aligned[node] for node in return_nodes]
    fused = torch.cat(tensors, dim=1)  # [B, C_total, H, W]

    b, c, h, w = fused.shape
    patches = fused.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()

    if l2_normalize_embeddings:
        patches = F.normalize(patches, p=2, dim=-1)

    return patches, (h, w)


class PatchMemoryBank:
    """
    Runtime-only patch memory bank.

    This class only supports:
    - loading an already-generated memory bank .pt file
    - exposing metadata
    - giving access to the stored patch tensor

    It does NOT support:
    - building from dataset/dataloader
    - saving new memory banks
    - command-line execution
    """

    def __init__(
        self,
        category: str,
        return_nodes: tuple[str, ...] = ("layer2", "layer3"),
        reference_node: str = "layer2",
        l2_normalize_embeddings: bool = True,
    ) -> None:
        if category not in VALID_MVTEC_AD_CATEGORIES:
            raise ValueError(
                f"Unsupported category '{category}'. Valid categories: {list(VALID_MVTEC_AD_CATEGORIES)}"
            )

        if reference_node not in return_nodes:
            raise ValueError("reference_node must be included in return_nodes")

        self.category = category
        self.return_nodes = tuple(return_nodes)
        self.reference_node = reference_node
        self.l2_normalize_embeddings = bool(l2_normalize_embeddings)

        self.memory_bank: torch.Tensor | None = None
        self.embedding_dim: int | None = None
        self.reference_grid_size_hw: tuple[int, int] | None = None
        self.num_source_patches_before_subsample: int | None = None
        self.extra_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.memory_bank is not None

    @property
    def memory_bank_size(self) -> int:
        if self.memory_bank is None:
            return 0
        return int(self.memory_bank.shape[0])

    def get_memory_bank_tensor(self) -> torch.Tensor:
        if self.memory_bank is None:
            raise ValueError("Memory bank is not loaded.")
        return self.memory_bank

    def get_memory_bank_tensor_on(self, device: str | torch.device) -> torch.Tensor:
        if self.memory_bank is None:
            raise ValueError("Memory bank is not loaded.")
        return self.memory_bank.to(device)

    @classmethod
    def load(cls, input_path: str | Path) -> "PatchMemoryBank":
        input_path = Path(input_path)
        payload = torch.load(input_path, map_location="cpu", weights_only=False)

        if isinstance(payload, torch.Tensor):
            raise ValueError(
                f"Unsupported memory bank file format at '{input_path}'. "
                "Expected a dict payload with metadata, not a raw tensor."
            )

        if not isinstance(payload, dict):
            raise ValueError(
                f"Unsupported memory bank file format at '{input_path}'. "
                f"Expected dict payload, got {type(payload).__name__}."
            )

        required_keys = {"category", "return_nodes", "reference_node", "memory_bank"}
        missing_keys = required_keys.difference(payload.keys())
        if missing_keys:
            raise ValueError(
                f"Memory bank file '{input_path}' is missing required keys: {sorted(missing_keys)}"
            )

        obj = cls(
            category=payload["category"],
            return_nodes=tuple(payload["return_nodes"]),
            reference_node=payload["reference_node"],
            l2_normalize_embeddings=bool(payload.get("l2_normalize_embeddings", True)),
        )

        obj.memory_bank = payload["memory_bank"].float().contiguous().cpu()
        obj.embedding_dim = int(payload.get("embedding_dim", obj.memory_bank.shape[1]))

        ref_grid = payload.get("reference_grid_size_hw")
        obj.reference_grid_size_hw = tuple(ref_grid) if ref_grid is not None else None

        obj.num_source_patches_before_subsample = payload.get(
            "num_source_patches_before_subsample"
        )

        extra_metadata = payload.get("extra_metadata", {})
        obj.extra_metadata = extra_metadata if isinstance(extra_metadata, dict) else {}

        return obj

    def summary(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "is_fitted": self.is_fitted,
            "return_nodes": list(self.return_nodes),
            "reference_node": self.reference_node,
            "l2_normalize_embeddings": self.l2_normalize_embeddings,
            "embedding_dim": self.embedding_dim,
            "reference_grid_size_hw": list(self.reference_grid_size_hw)
            if self.reference_grid_size_hw
            else None,
            "num_source_patches_before_subsample": self.num_source_patches_before_subsample,
            "memory_bank_size": self.memory_bank_size,
            "extra_metadata": self.extra_metadata,
        }