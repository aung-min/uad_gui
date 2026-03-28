from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_anomaly_map(anomaly_map_path: str | Path) -> np.ndarray:
    anomaly_map_path = Path(anomaly_map_path)
    if anomaly_map_path.suffix.lower() == ".npy":
        arr = np.load(str(anomaly_map_path)).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got shape {arr.shape}")
        return arr

    image = Image.open(anomaly_map_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def minmax_normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
    min_v = float(anomaly_map.min())
    max_v = float(anomaly_map.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - min_v) / (max_v - min_v)


def binary_mask_from_anomaly_map(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    blur_kernel: int = 0,
) -> np.ndarray:
    anomaly_map = minmax_normalize_map(anomaly_map)

    if blur_kernel and blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        anomaly_map = cv2.GaussianBlur(anomaly_map, (blur_kernel, blur_kernel), 0)

    mask = (anomaly_map >= float(threshold)).astype(np.uint8) * 255
    return mask


def clean_binary_mask(
    binary_mask: np.ndarray,
    morph_kernel: int = 3,
    morph_iterations: int = 1,
) -> np.ndarray:
    if morph_kernel <= 1 or morph_iterations <= 0:
        return binary_mask.copy()

    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    return cleaned


def contour_score_stats(
    contour: np.ndarray,
    anomaly_map_norm: np.ndarray,
) -> dict[str, float]:
    mask = np.zeros(anomaly_map_norm.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)

    values = anomaly_map_norm[mask > 0]
    if values.size == 0:
        return {
            "mean_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
        }

    return {
        "mean_score": float(values.mean()),
        "max_score": float(values.max()),
        "min_score": float(values.min()),
    }


def contour_to_summary(
    contour: np.ndarray,
    contour_index: int,
    anomaly_map_norm: np.ndarray,
) -> dict[str, Any]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))

    x, y, w, h = cv2.boundingRect(contour)
    stats = contour_score_stats(contour=contour, anomaly_map_norm=anomaly_map_norm)

    polygon = contour.reshape(-1, 2).tolist()

    return {
        "contour_index": int(contour_index),
        "area": area,
        "perimeter": perimeter,
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "polygon_points": polygon,
        "num_points": int(len(polygon)),
        "mean_score": stats["mean_score"],
        "max_score": stats["max_score"],
        "min_score": stats["min_score"],
    }


def extract_anomaly_contours(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    min_area: float = 25.0,
    blur_kernel: int = 0,
    morph_kernel: int = 3,
    morph_iterations: int = 1,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    anomaly_map_norm = minmax_normalize_map(anomaly_map)
    binary_mask = binary_mask_from_anomaly_map(
        anomaly_map=anomaly_map_norm,
        threshold=threshold,
        blur_kernel=blur_kernel,
    )
    binary_mask = clean_binary_mask(
        binary_mask=binary_mask,
        morph_kernel=morph_kernel,
        morph_iterations=morph_iterations,
    )

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept_contours: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area >= float(min_area):
            kept_contours.append(contour)

    kept_contours = sorted(kept_contours, key=cv2.contourArea, reverse=True)
    return kept_contours, anomaly_map_norm, binary_mask


def draw_anomaly_contours(
    image_rgb: np.ndarray,
    contours: list[np.ndarray],
    line_thickness: int = 2,
    draw_fill: bool = False,
    fill_alpha: float = 0.20,
    contour_color_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")

    output = image_rgb.copy()
    contour_color_bgr = (contour_color_rgb[2], contour_color_rgb[1], contour_color_rgb[0])

    if draw_fill and contours:
        overlay_bgr = cv2.cvtColor(output.copy(), cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, contourIdx=-1, color=contour_color_bgr, thickness=-1)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        output = np.clip(
            (1.0 - fill_alpha) * output.astype(np.float32) + fill_alpha * overlay_rgb.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if contours:
        cv2.drawContours(
            output_bgr,
            contours,
            contourIdx=-1,
            color=contour_color_bgr,
            thickness=int(line_thickness),
        )
    output = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    return output


def draw_anomaly_contours_on_heatmap(
    anomaly_map: np.ndarray,
    contours: list[np.ndarray],
    line_thickness: int = 2,
) -> np.ndarray:
    heat = (minmax_normalize_map(anomaly_map) * 255.0).clip(0, 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR)
    if contours:
        cv2.drawContours(heat_bgr, contours, contourIdx=-1, color=(0, 0, 255), thickness=int(line_thickness))
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def build_anomaly_shape_summary(
    anomaly_map: np.ndarray,
    contours: list[np.ndarray],
) -> dict[str, Any]:
    anomaly_map_norm = minmax_normalize_map(anomaly_map)

    regions = [
        contour_to_summary(
            contour=contour,
            contour_index=idx,
            anomaly_map_norm=anomaly_map_norm,
        )
        for idx, contour in enumerate(contours)
    ]

    return {
        "num_regions": int(len(regions)),
        "regions": regions,
    }


def save_rgb_image(array: np.ndarray, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    Image.fromarray(array, mode="RGB").save(output_path)


def save_binary_mask(binary_mask: np.ndarray, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    Image.fromarray(binary_mask, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and draw contour-shaped anomaly regions from a heat map.")
    parser.add_argument("--image", type=str, required=True, help="Path to original RGB image.")
    parser.add_argument("--anomaly-map", type=str, required=True, help="Path to anomaly map (.npy or grayscale image).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Normalized anomaly threshold.")
    parser.add_argument("--min-area", type=float, default=25.0, help="Minimum contour area to keep.")
    parser.add_argument("--blur-kernel", type=int, default=0, help="Optional Gaussian blur kernel size.")
    parser.add_argument("--morph-kernel", type=int, default=3, help="Morphology kernel size.")
    parser.add_argument("--morph-iterations", type=int, default=1, help="Morphology iterations.")
    parser.add_argument("--line-thickness", type=int, default=2, help="Contour line thickness.")
    parser.add_argument("--draw-fill", action="store_true", help="Fill contour shapes on the overlay.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/anomaly_shapes",
        help="Directory for saved contour-shape outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_rgb = load_rgb_image(args.image)
    anomaly_map = load_anomaly_map(args.anomaly_map)

    contours, anomaly_map_norm, binary_mask = extract_anomaly_contours(
        anomaly_map=anomaly_map,
        threshold=args.threshold,
        min_area=args.min_area,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
        morph_iterations=args.morph_iterations,
    )

    overlay_rgb = draw_anomaly_contours(
        image_rgb=image_rgb,
        contours=contours,
        line_thickness=args.line_thickness,
        draw_fill=args.draw_fill,
    )
    heatmap_overlay_rgb = draw_anomaly_contours_on_heatmap(
        anomaly_map=anomaly_map_norm,
        contours=contours,
        line_thickness=args.line_thickness,
    )
    summary = build_anomaly_shape_summary(
        anomaly_map=anomaly_map_norm,
        contours=contours,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = output_dir / "anomaly_shape_overlay.png"
    heatmap_overlay_path = output_dir / "anomaly_shape_heatmap_overlay.png"
    mask_path = output_dir / "anomaly_shape_mask.png"
    summary_path = output_dir / "anomaly_shape_summary.json"

    save_rgb_image(overlay_rgb, overlay_path)
    save_rgb_image(heatmap_overlay_rgb, heatmap_overlay_path)
    save_binary_mask(binary_mask, mask_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output = {
        "num_regions": summary["num_regions"],
        "overlay_png": str(overlay_path.resolve()),
        "heatmap_overlay_png": str(heatmap_overlay_path.resolve()),
        "mask_png": str(mask_path.resolve()),
        "summary_json": str(summary_path.resolve()),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()