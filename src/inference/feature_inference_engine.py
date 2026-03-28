from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import build_image_transform
from src.features.patch_memory_bank import (
    PatchMemoryBank,
    combine_multi_layer_patch_embeddings,
)
from src.features.resnet_feature_extractor import build_feature_extractor, resolve_device
from src.utils.anomaly_bboxes import (
    build_anomaly_shape_summary,
    draw_anomaly_contours,
    draw_anomaly_contours_on_heatmap,
    extract_anomaly_contours,
)


@dataclass(frozen=True)
class InferenceResult:
    image_path: str
    category: str
    image_score: float
    patch_grid_size_hw: tuple[int, int]
    image_size_hw: tuple[int, int]
    anomaly_map: np.ndarray
    patch_scores: np.ndarray


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def minmax_normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    min_v = float(anomaly_map.min())
    max_v = float(anomaly_map.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - min_v) / (max_v - min_v)).astype(np.float32)


def anomaly_map_to_uint8(anomaly_map: np.ndarray) -> np.ndarray:
    normalized = minmax_normalize_map(anomaly_map)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def create_red_overlay(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")

    heat = minmax_normalize_map(anomaly_map)
    heat_3 = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)
    base = image_rgb.astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * base + alpha * heat_3
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


def save_numpy_map(array: np.ndarray, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    np.save(str(output_path), array.astype(np.float32))


def save_grayscale_png(array: np.ndarray, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    Image.fromarray(anomaly_map_to_uint8(array), mode="L").save(output_path)


def save_overlay_png(
    image_pil: Image.Image,
    anomaly_map: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.55,
) -> None:
    ensure_parent_dir(output_path)
    image_rgb = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
    overlay = create_red_overlay(image_rgb=image_rgb, anomaly_map=anomaly_map, alpha=alpha)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def save_rgb_png(array: np.ndarray, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(output_path)


def compute_min_distances_to_memory_bank(
    query_embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    if query_embeddings.ndim != 2:
        raise ValueError(
            f"Expected query_embeddings shape [N, D], got {tuple(query_embeddings.shape)}"
        )
    if memory_bank.ndim != 2:
        raise ValueError(f"Expected memory_bank shape [M, D], got {tuple(memory_bank.shape)}")
    if query_embeddings.shape[1] != memory_bank.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"{query_embeddings.shape[1]} vs {memory_bank.shape[1]}"
        )

    min_dists_list: list[torch.Tensor] = []
    total = query_embeddings.shape[0]

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        query_chunk = query_embeddings[start:end]
        dists = torch.cdist(query_chunk, memory_bank, p=2)
        min_dists = dists.min(dim=1).values
        min_dists_list.append(min_dists)

    return torch.cat(min_dists_list, dim=0).contiguous()


class FeatureInferenceEngine:
    def __init__(
        self,
        memory_bank_path: str | Path,
        image_size: int = 256,
        device: str = "auto",
        chunk_size: int = 4096,
    ) -> None:
        self.device = resolve_device(device)
        self.image_size = int(image_size)
        self.chunk_size = int(chunk_size)

        self.memory_bank_obj = PatchMemoryBank.load(memory_bank_path)
        if self.memory_bank_obj.memory_bank is None:
            raise ValueError("Loaded memory bank is empty.")

        self.category = self.memory_bank_obj.category
        self.return_nodes = tuple(self.memory_bank_obj.return_nodes)
        self.reference_node = str(self.memory_bank_obj.reference_node)
        self.l2_normalize_embeddings = bool(self.memory_bank_obj.l2_normalize_embeddings)

        self.memory_bank = self.memory_bank_obj.memory_bank.to(self.device)

        self.feature_extractor = build_feature_extractor(
            backbone_name="resnet18",
            pretrained=True,
            return_nodes=self.return_nodes,
            freeze_backbone=True,
        ).to(self.device)
        self.feature_extractor.eval()

        self.image_transform = build_image_transform(
            image_size=self.image_size,
            normalize_imagenet=True,
        )

    @torch.no_grad()
    def predict_image(
        self,
        image_path: str | Path,
    ) -> InferenceResult:
        image_path = Path(image_path)
        original_image = load_rgb_image(image_path)
        original_size_hw = (original_image.height, original_image.width)

        input_tensor = self.image_transform(original_image).unsqueeze(0).to(self.device)

        feature_maps = self.feature_extractor.extract(input_tensor)
        patch_embeddings, patch_grid_size_hw = combine_multi_layer_patch_embeddings(
            feature_maps=feature_maps,
            reference_node=self.reference_node,
            return_nodes=self.return_nodes,
            l2_normalize_embeddings=self.l2_normalize_embeddings,
        )

        flat_embeddings = patch_embeddings.reshape(-1, patch_embeddings.shape[-1]).contiguous()
        min_distances = compute_min_distances_to_memory_bank(
            query_embeddings=flat_embeddings,
            memory_bank=self.memory_bank,
            chunk_size=self.chunk_size,
        )

        patch_scores = min_distances.reshape(1, 1, patch_grid_size_hw[0], patch_grid_size_hw[1])
        image_score = float(min_distances.max().item())

        upsampled_map = F.interpolate(
            patch_scores,
            size=original_size_hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0].detach().cpu().numpy().astype(np.float32)

        patch_scores_np = patch_scores[0, 0].detach().cpu().numpy().astype(np.float32)

        return InferenceResult(
            image_path=str(image_path),
            category=self.category,
            image_score=image_score,
            patch_grid_size_hw=patch_grid_size_hw,
            image_size_hw=original_size_hw,
            anomaly_map=upsampled_map,
            patch_scores=patch_scores_np,
        )

    @torch.no_grad()
    def predict_folder(
        self,
        image_dir: str | Path,
    ) -> list[InferenceResult]:
        image_dir = Path(image_dir)
        image_paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        )
        return [self.predict_image(p) for p in image_paths]


def export_inference_result(
    result: InferenceResult,
    output_dir: str | Path,
    save_overlay: bool = True,
    overlay_alpha: float = 0.55,
    save_contour_overlay: bool = False,
    contour_threshold: float = 0.5,
    contour_min_area: float = 25.0,
    contour_blur_kernel: int = 0,
    contour_morph_kernel: int = 3,
    contour_morph_iterations: int = 1,
    contour_line_thickness: int = 2,
    contour_draw_fill: bool = False,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(result.image_path).stem
    image_pil = load_rgb_image(result.image_path)
    image_rgb = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)

    anomaly_map_npy = output_dir / f"{image_stem}_anomaly_map.npy"
    anomaly_map_png = output_dir / f"{image_stem}_anomaly_map.png"
    patch_scores_npy = output_dir / f"{image_stem}_patch_scores.npy"
    summary_json = output_dir / f"{image_stem}_summary.json"

    save_numpy_map(result.anomaly_map, anomaly_map_npy)
    save_grayscale_png(result.anomaly_map, anomaly_map_png)
    save_numpy_map(result.patch_scores, patch_scores_npy)

    exported: dict[str, str] = {
        "anomaly_map_npy": str(anomaly_map_npy.resolve()),
        "anomaly_map_png": str(anomaly_map_png.resolve()),
        "patch_scores_npy": str(patch_scores_npy.resolve()),
    }

    if save_overlay:
        overlay_png = output_dir / f"{image_stem}_overlay.png"
        save_overlay_png(
            image_pil=image_pil,
            anomaly_map=result.anomaly_map,
            output_path=overlay_png,
            alpha=overlay_alpha,
        )
        exported["overlay_png"] = str(overlay_png.resolve())

    contour_summary: dict[str, Any] | None = None
    if save_contour_overlay:
        contours, _, binary_mask = extract_anomaly_contours(
            anomaly_map=result.anomaly_map,
            threshold=contour_threshold,
            min_area=contour_min_area,
            blur_kernel=contour_blur_kernel,
            morph_kernel=contour_morph_kernel,
            morph_iterations=contour_morph_iterations,
        )

        contour_overlay_rgb = draw_anomaly_contours(
            image_rgb=image_rgb,
            contours=contours,
            line_thickness=contour_line_thickness,
            draw_fill=contour_draw_fill,
        )
        heatmap_contour_rgb = draw_anomaly_contours_on_heatmap(
            anomaly_map=result.anomaly_map,
            contours=contours,
            line_thickness=contour_line_thickness,
        )

        contour_overlay_png = output_dir / f"{image_stem}_contour_overlay.png"
        contour_heatmap_png = output_dir / f"{image_stem}_contour_heatmap_overlay.png"
        contour_mask_png = output_dir / f"{image_stem}_contour_mask.png"
        contour_summary_json = output_dir / f"{image_stem}_contour_summary.json"

        save_rgb_png(contour_overlay_rgb, contour_overlay_png)
        save_rgb_png(heatmap_contour_rgb, contour_heatmap_png)
        save_grayscale_png(binary_mask.astype(np.float32) / 255.0, contour_mask_png)

        contour_summary = build_anomaly_shape_summary(
            anomaly_map=result.anomaly_map,
            contours=contours,
        )
        contour_summary["settings"] = {
            "threshold": float(contour_threshold),
            "min_area": float(contour_min_area),
            "blur_kernel": int(contour_blur_kernel),
            "morph_kernel": int(contour_morph_kernel),
            "morph_iterations": int(contour_morph_iterations),
            "line_thickness": int(contour_line_thickness),
            "draw_fill": bool(contour_draw_fill),
        }
        contour_summary_json.write_text(json.dumps(contour_summary, indent=2), encoding="utf-8")

        exported["contour_overlay_png"] = str(contour_overlay_png.resolve())
        exported["contour_heatmap_overlay_png"] = str(contour_heatmap_png.resolve())
        exported["contour_mask_png"] = str(contour_mask_png.resolve())
        exported["contour_summary_json"] = str(contour_summary_json.resolve())

    summary = {
        "image_path": result.image_path,
        "category": result.category,
        "image_score": result.image_score,
        "patch_grid_size_hw": list(result.patch_grid_size_hw),
        "image_size_hw": list(result.image_size_hw),
        "exports": exported,
    }
    if contour_summary is not None:
        summary["contour_regions"] = {
            "num_regions": contour_summary["num_regions"],
        }

    ensure_parent_dir(summary_json)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    exported["summary_json"] = str(summary_json.resolve())

    return exported