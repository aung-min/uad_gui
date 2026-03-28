from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.inference.ae_inference_engine import AEInferenceEngine, load_rgb_image
from src.inference.feature_inference_engine import FeatureInferenceEngine
from src.utils.anomaly_bboxes import (
    build_anomaly_shape_summary,
    draw_anomaly_contours,
    draw_anomaly_contours_on_heatmap,
    extract_anomaly_contours,
)


@dataclass(frozen=True)
class HybridInferenceResult:
    image_path: str
    category: str
    image_score: float
    image_size_hw: tuple[int, int]
    fused_anomaly_map: np.ndarray
    feature_anomaly_map: np.ndarray
    ae_anomaly_map: np.ndarray
    feature_score: float
    ae_score: float
    fusion_weight_feature: float
    fusion_weight_ae: float


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def minmax_normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
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


class HybridInferenceEngine:
    def __init__(
        self,
        feature_memory_bank_path: str | Path,
        ae_checkpoint_path: str | Path,
        image_size: int = 256,
        device: str = "auto",
        chunk_size: int = 4096,
        feature_weight: float = 0.5,
        ae_weight: float = 0.5,
        score_mode: str = "max",
    ) -> None:
        if feature_weight < 0 or ae_weight < 0:
            raise ValueError("feature_weight and ae_weight must be non-negative.")
        if feature_weight == 0 and ae_weight == 0:
            raise ValueError("At least one fusion weight must be > 0.")
        if score_mode not in {"max", "mean"}:
            raise ValueError("score_mode must be 'max' or 'mean'.")

        self.feature_engine = FeatureInferenceEngine(
            memory_bank_path=feature_memory_bank_path,
            image_size=image_size,
            device=device,
            chunk_size=chunk_size,
        )
        self.ae_engine = AEInferenceEngine(
            checkpoint_path=ae_checkpoint_path,
            image_size=image_size,
            device=device,
        )

        self.category = self.feature_engine.category
        self.image_size = int(image_size)
        self.feature_weight = float(feature_weight)
        self.ae_weight = float(ae_weight)
        self.score_mode = score_mode

    def _fuse_maps(
        self,
        feature_map: np.ndarray,
        ae_map: np.ndarray,
    ) -> np.ndarray:
        feature_norm = minmax_normalize_map(feature_map)
        ae_norm = minmax_normalize_map(ae_map)

        weight_sum = self.feature_weight + self.ae_weight
        fused = (
            self.feature_weight * feature_norm +
            self.ae_weight * ae_norm
        ) / weight_sum
        return fused.astype(np.float32)

    def _score_from_map(self, fused_map: np.ndarray) -> float:
        if self.score_mode == "max":
            return float(fused_map.max())
        return float(fused_map.mean())

    def predict_image(
        self,
        image_path: str | Path,
    ) -> HybridInferenceResult:
        feature_result = self.feature_engine.predict_image(image_path)
        ae_result = self.ae_engine.predict_image(image_path)

        if feature_result.image_size_hw != ae_result.image_size_hw:
            raise ValueError(
                f"Image size mismatch between feature and AE branches: "
                f"{feature_result.image_size_hw} vs {ae_result.image_size_hw}"
            )

        fused_map = self._fuse_maps(
            feature_map=feature_result.anomaly_map,
            ae_map=ae_result.anomaly_map,
        )
        image_score = self._score_from_map(fused_map)

        return HybridInferenceResult(
            image_path=str(image_path),
            category=self.category,
            image_score=image_score,
            image_size_hw=feature_result.image_size_hw,
            fused_anomaly_map=fused_map,
            feature_anomaly_map=feature_result.anomaly_map.astype(np.float32),
            ae_anomaly_map=ae_result.anomaly_map.astype(np.float32),
            feature_score=float(feature_result.image_score),
            ae_score=float(ae_result.image_score),
            fusion_weight_feature=self.feature_weight,
            fusion_weight_ae=self.ae_weight,
        )

    def predict_folder(
        self,
        image_dir: str | Path,
    ) -> list[HybridInferenceResult]:
        image_dir = Path(image_dir)
        image_paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        )
        return [self.predict_image(p) for p in image_paths]


def export_hybrid_inference_result(
    result: HybridInferenceResult,
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

    fused_npy = output_dir / f"{image_stem}_hybrid_anomaly_map.npy"
    fused_png = output_dir / f"{image_stem}_hybrid_anomaly_map.png"
    feature_npy = output_dir / f"{image_stem}_feature_anomaly_map.npy"
    feature_png = output_dir / f"{image_stem}_feature_anomaly_map.png"
    ae_npy = output_dir / f"{image_stem}_ae_anomaly_map.npy"
    ae_png = output_dir / f"{image_stem}_ae_anomaly_map.png"
    summary_json = output_dir / f"{image_stem}_summary.json"

    save_numpy_map(result.fused_anomaly_map, fused_npy)
    save_grayscale_png(result.fused_anomaly_map, fused_png)
    save_numpy_map(result.feature_anomaly_map, feature_npy)
    save_grayscale_png(result.feature_anomaly_map, feature_png)
    save_numpy_map(result.ae_anomaly_map, ae_npy)
    save_grayscale_png(result.ae_anomaly_map, ae_png)

    exported: dict[str, str] = {
        "hybrid_anomaly_map_npy": str(fused_npy.resolve()),
        "hybrid_anomaly_map_png": str(fused_png.resolve()),
        "feature_anomaly_map_npy": str(feature_npy.resolve()),
        "feature_anomaly_map_png": str(feature_png.resolve()),
        "ae_anomaly_map_npy": str(ae_npy.resolve()),
        "ae_anomaly_map_png": str(ae_png.resolve()),
    }

    if save_overlay:
        overlay_png = output_dir / f"{image_stem}_hybrid_overlay.png"
        save_overlay_png(
            image_pil=image_pil,
            anomaly_map=result.fused_anomaly_map,
            output_path=overlay_png,
            alpha=overlay_alpha,
        )
        exported["hybrid_overlay_png"] = str(overlay_png.resolve())

    contour_summary: dict[str, Any] | None = None
    if save_contour_overlay:
        contours, _, binary_mask = extract_anomaly_contours(
            anomaly_map=result.fused_anomaly_map,
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
            anomaly_map=result.fused_anomaly_map,
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
            anomaly_map=result.fused_anomaly_map,
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
        "feature_score": result.feature_score,
        "ae_score": result.ae_score,
        "image_size_hw": list(result.image_size_hw),
        "fusion_weight_feature": result.fusion_weight_feature,
        "fusion_weight_ae": result.fusion_weight_ae,
        "exports": exported,
    }
    if contour_summary is not None:
        summary["contour_regions"] = {
            "num_regions": contour_summary["num_regions"],
        }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    exported["summary_json"] = str(summary_json.resolve())

    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid anomaly inference by fusing feature and AE maps.")
    parser.add_argument(
        "--feature-memory-bank",
        type=str,
        required=True,
        help="Path to saved feature memory bank .pt file.",
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=str,
        required=True,
        help="Path to trained AE checkpoint .pt file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to one image for inference.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to a folder of images for inference.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize size used before inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size for feature nearest-neighbor computation.",
    )
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=0.5,
        help="Fusion weight for feature anomaly map.",
    )
    parser.add_argument(
        "--ae-weight",
        type=float,
        default=0.5,
        help="Fusion weight for AE anomaly map.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="How image_score is reduced from fused map.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hybrid_predictions",
        help="Directory for saving hybrid prediction outputs.",
    )
    parser.add_argument(
        "--save-contour-overlay",
        action="store_true",
        help="Also export contour-shaped anomaly overlays and summaries.",
    )
    parser.add_argument(
        "--contour-threshold",
        type=float,
        default=0.5,
        help="Normalized threshold for extracting anomaly contours.",
    )
    parser.add_argument(
        "--contour-min-area",
        type=float,
        default=25.0,
        help="Minimum contour area to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide either --image or --image-dir")

    engine = HybridInferenceEngine(
        feature_memory_bank_path=args.feature_memory_bank,
        ae_checkpoint_path=args.ae_checkpoint,
        image_size=args.image_size,
        device=args.device,
        chunk_size=args.chunk_size,
        feature_weight=args.feature_weight,
        ae_weight=args.ae_weight,
        score_mode=args.score_mode,
    )

    if args.image:
        result = engine.predict_image(args.image)
        image_output_dir = Path(args.output_dir) / engine.category
        exports = export_hybrid_inference_result(
            result=result,
            output_dir=image_output_dir,
            save_contour_overlay=args.save_contour_overlay,
            contour_threshold=args.contour_threshold,
            contour_min_area=args.contour_min_area,
        )

        output = {
            "mode": "single_image",
            "image_path": result.image_path,
            "category": result.category,
            "image_score": result.image_score,
            "feature_score": result.feature_score,
            "ae_score": result.ae_score,
            "image_size_hw": list(result.image_size_hw),
            "fusion_weight_feature": result.fusion_weight_feature,
            "fusion_weight_ae": result.fusion_weight_ae,
            "exports": exports,
        }
        print(json.dumps(output, indent=2))
        return

    results = engine.predict_folder(args.image_dir)
    folder_output_dir = Path(args.output_dir) / engine.category
    folder_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for result in results:
        exports = export_hybrid_inference_result(
            result=result,
            output_dir=folder_output_dir,
            save_contour_overlay=args.save_contour_overlay,
            contour_threshold=args.contour_threshold,
            contour_min_area=args.contour_min_area,
        )
        rows.append(
            {
                "image_path": result.image_path,
                "category": result.category,
                "image_score": result.image_score,
                "feature_score": result.feature_score,
                "ae_score": result.ae_score,
                "image_size_hw": list(result.image_size_hw),
                "fusion_weight_feature": result.fusion_weight_feature,
                "fusion_weight_ae": result.fusion_weight_ae,
                "exports": exports,
            }
        )

    rows = sorted(rows, key=lambda x: x["image_score"], reverse=True)

    summary_json = folder_output_dir / "folder_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    output = {
        "mode": "folder",
        "category": engine.category,
        "total_images": len(rows),
        "summary_json": str(summary_json.resolve()),
        "top_5_scores": rows[:5],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()