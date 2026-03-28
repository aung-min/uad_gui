from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import build_image_transform
from src.models.conv_autoencoder import build_conv_autoencoder, resolve_device


@dataclass(frozen=True)
class AEInferenceResult:
    image_path: str
    category: str
    image_score: float
    image_size_hw: tuple[int, int]
    anomaly_map: np.ndarray
    reconstruction_rgb: np.ndarray


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


def save_reconstruction_png(
    reconstruction_rgb: np.ndarray,
    output_path: str | Path,
) -> None:
    ensure_parent_dir(output_path)
    Image.fromarray(reconstruction_rgb, mode="RGB").save(output_path)


class AEInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        image_size: int = 256,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.image_size = int(image_size)

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})

        self.category = str(metadata.get("category", "unknown"))
        base_channels = int(metadata.get("base_channels", 64))
        latent_channels = int(metadata.get("latent_channels", 512))

        self.model = build_conv_autoencoder(
            in_channels=3,
            base_channels=base_channels,
            latent_channels=latent_channels,
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.image_transform = build_image_transform(
            image_size=self.image_size,
            normalize_imagenet=False,
        )

    @torch.no_grad()
    def predict_image(
        self,
        image_path: str | Path,
    ) -> AEInferenceResult:
        image_path = Path(image_path)
        original_image = load_rgb_image(image_path)
        original_size_hw = (original_image.height, original_image.width)

        input_tensor = self.image_transform(original_image).unsqueeze(0).to(self.device)

        reconstruction = self.model.reconstruct(input_tensor)
        error_map = torch.mean(torch.abs(input_tensor - reconstruction), dim=1, keepdim=True)

        upsampled_map = F.interpolate(
            error_map,
            size=original_size_hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0].detach().cpu().numpy().astype(np.float32)

        image_score = float(error_map.flatten(start_dim=1).mean(dim=1).item())

        reconstruction_rgb = reconstruction[0].detach().cpu().clamp(0.0, 1.0)
        reconstruction_rgb = (reconstruction_rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        reconstruction_rgb = np.asarray(
            Image.fromarray(reconstruction_rgb, mode="RGB").resize(
                (original_image.width, original_image.height),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )

        return AEInferenceResult(
            image_path=str(image_path),
            category=self.category,
            image_score=image_score,
            image_size_hw=original_size_hw,
            anomaly_map=upsampled_map,
            reconstruction_rgb=reconstruction_rgb,
        )

    @torch.no_grad()
    def predict_folder(
        self,
        image_dir: str | Path,
    ) -> list[AEInferenceResult]:
        image_dir = Path(image_dir)
        image_paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        )
        return [self.predict_image(p) for p in image_paths]


def export_ae_inference_result(
    result: AEInferenceResult,
    output_dir: str | Path,
    save_overlay: bool = True,
    overlay_alpha: float = 0.55,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(result.image_path).stem
    image_pil = load_rgb_image(result.image_path)

    anomaly_map_npy = output_dir / f"{image_stem}_anomaly_map.npy"
    anomaly_map_png = output_dir / f"{image_stem}_anomaly_map.png"
    reconstruction_png = output_dir / f"{image_stem}_reconstruction.png"
    summary_json = output_dir / f"{image_stem}_summary.json"

    save_numpy_map(result.anomaly_map, anomaly_map_npy)
    save_grayscale_png(result.anomaly_map, anomaly_map_png)
    save_reconstruction_png(result.reconstruction_rgb, reconstruction_png)

    exported: dict[str, str] = {
        "anomaly_map_npy": str(anomaly_map_npy.resolve()),
        "anomaly_map_png": str(anomaly_map_png.resolve()),
        "reconstruction_png": str(reconstruction_png.resolve()),
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

    summary = {
        "image_path": result.image_path,
        "category": result.category,
        "image_score": result.image_score,
        "image_size_hw": list(result.image_size_hw),
        "exports": exported,
    }

    ensure_parent_dir(summary_json)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    exported["summary_json"] = str(summary_json.resolve())

    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE reconstruction anomaly inference.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained AE checkpoint (.pt).",
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
        help="Resize size used before AE inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ae_predictions",
        help="Directory for saving prediction outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide either --image or --image-dir")

    engine = AEInferenceEngine(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
    )

    if args.image:
        result = engine.predict_image(args.image)
        image_output_dir = Path(args.output_dir) / engine.category
        exports = export_ae_inference_result(result=result, output_dir=image_output_dir)

        output = {
            "mode": "single_image",
            "image_path": result.image_path,
            "category": result.category,
            "image_score": result.image_score,
            "image_size_hw": list(result.image_size_hw),
            "exports": exports,
        }
        print(json.dumps(output, indent=2))
        return

    results = engine.predict_folder(args.image_dir)
    folder_output_dir = Path(args.output_dir) / engine.category
    folder_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for result in results:
        exports = export_ae_inference_result(result=result, output_dir=folder_output_dir)
        rows.append(
            {
                "image_path": result.image_path,
                "category": result.category,
                "image_score": result.image_score,
                "image_size_hw": list(result.image_size_hw),
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