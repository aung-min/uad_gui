from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def ensure_grayscale(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def pil_mask_to_float_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.uint8)
    arr = (arr > 0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def normalize_tensor(
    image_tensor: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            f"Expected image tensor shape [3, H, W], got {tuple(image_tensor.shape)}"
        )

    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor


class Compose:
    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


@dataclass(frozen=True)
class ResizeImage:
    image_size: int
    resample: int = Image.BILINEAR

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ensure_rgb(image)
        return image.resize((self.image_size, self.image_size), self.resample)


@dataclass(frozen=True)
class ResizeMask:
    image_size: int
    resample: int = Image.NEAREST

    def __call__(self, mask: Image.Image) -> Image.Image:
        mask = ensure_grayscale(mask)
        return mask.resize((self.image_size, self.image_size), self.resample)


class ToFloatTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return pil_to_float_tensor(image)


class MaskToFloatTensor:
    def __call__(self, mask: Image.Image) -> torch.Tensor:
        return pil_mask_to_float_tensor(mask)


@dataclass(frozen=True)
class NormalizeImageNet:
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(image_tensor=image_tensor, mean=self.mean, std=self.std)


@dataclass(frozen=True)
class ResizeToTensor:
    image_size: int
    normalize_imagenet: bool = False

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pipeline: list[Callable[[Any], Any]] = [
            ResizeImage(image_size=self.image_size),
            ToFloatTensor(),
        ]
        if self.normalize_imagenet:
            pipeline.append(NormalizeImageNet())
        return Compose(pipeline)(image)


@dataclass(frozen=True)
class ResizeMaskToTensor:
    image_size: int

    def __call__(self, mask: Image.Image) -> torch.Tensor:
        return Compose(
            [
                ResizeMask(image_size=self.image_size),
                MaskToFloatTensor(),
            ]
        )(mask)


@dataclass(frozen=True)
class ResizePairToTensor:
    image_size: int
    normalize_imagenet: bool = False

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = ResizeToTensor(
            image_size=self.image_size,
            normalize_imagenet=self.normalize_imagenet,
        )(image)
        mask_tensor = ResizeMaskToTensor(image_size=self.image_size)(mask)
        return image_tensor, mask_tensor


def build_image_transform(
    image_size: int,
    normalize_imagenet: bool = False,
) -> Callable[[Image.Image], torch.Tensor]:
    return ResizeToTensor(
        image_size=image_size,
        normalize_imagenet=normalize_imagenet,
    )


def build_mask_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return ResizeMaskToTensor(image_size=image_size)


def build_paired_transforms(
    image_size: int,
    normalize_imagenet: bool = False,
) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
    return (
        build_image_transform(image_size=image_size, normalize_imagenet=normalize_imagenet),
        build_mask_transform(image_size=image_size),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect image and mask transforms.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to a binary mask.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize target.")
    parser.add_argument(
        "--normalize-imagenet",
        action="store_true",
        help="Apply ImageNet mean/std normalization to the image tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = Image.open(args.image).convert("RGB")
    image_transform = build_image_transform(
        image_size=args.image_size,
        normalize_imagenet=args.normalize_imagenet,
    )
    image_tensor = image_transform(image)

    result: dict[str, Any] = {
        "image_path": str(Path(args.image).resolve()),
        "image_shape": list(image_tensor.shape),
        "image_dtype": str(image_tensor.dtype),
        "image_min": float(image_tensor.min().item()),
        "image_max": float(image_tensor.max().item()),
        "normalize_imagenet": args.normalize_imagenet,
    }

    if args.mask:
        mask = Image.open(args.mask).convert("L")
        mask_transform = build_mask_transform(image_size=args.image_size)
        mask_tensor = mask_transform(mask)
        result["mask_path"] = str(Path(args.mask).resolve())
        result["mask_shape"] = list(mask_tensor.shape)
        result["mask_dtype"] = str(mask_tensor.dtype)
        result["mask_positive_pixels"] = int((mask_tensor > 0).sum().item())

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()