from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_MVTEC_AD_CATEGORIES = (
    "bottle",
    "transistor",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
)

VALID_SPLITS = ("train", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MVTecADSample:
    dataset_name: str
    category: str
    split: str
    label: int
    is_anomalous: bool
    defect_type: str
    image_path: str
    mask_path: str | None


def _sorted_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _pil_image_to_float_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _pil_mask_to_float_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.uint8)
    arr = (arr > 0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _load_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _load_mask_image(mask_path: Path | None, size_hw: tuple[int, int]) -> Image.Image:
    width, height = size_hw[1], size_hw[0]
    if mask_path is None:
        return Image.new("L", (width, height), color=0)
    return Image.open(mask_path).convert("L")


def _resolve_mask_path(category_dir: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None

    gt_dir = category_dir / "ground_truth" / defect_type
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found for category='{category_dir.name}', "
            f"defect_type='{defect_type}': {gt_dir}"
        )

    expected = gt_dir / f"{image_path.stem}_mask.png"
    if expected.exists():
        return expected

    candidates = sorted(gt_dir.glob(f"{image_path.stem}_mask.*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not resolve mask for image '{image_path}' inside '{gt_dir}'."
    )


def _validate_category(category: str) -> None:
    if category not in VALID_MVTEC_AD_CATEGORIES:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Valid categories: {list(VALID_MVTEC_AD_CATEGORIES)}"
        )


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Valid splits: {list(VALID_SPLITS)}")


def collect_mvtec_ad_samples(
    dataset_root: str | Path,
    category: str,
    split: str,
) -> list[MVTecADSample]:
    _validate_category(category)
    _validate_split(split)

    root = Path(dataset_root)
    category_dir = root / category
    split_dir = category_dir / split

    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    samples: list[MVTecADSample] = []

    if split == "train":
        good_dir = split_dir / "good"
        for image_path in _sorted_image_files(good_dir):
            samples.append(
                MVTecADSample(
                    dataset_name="mvtec_ad",
                    category=category,
                    split=split,
                    label=0,
                    is_anomalous=False,
                    defect_type="good",
                    image_path=str(image_path),
                    mask_path=None,
                )
            )
        return samples

    defect_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for defect_dir in defect_dirs:
        defect_type = defect_dir.name
        for image_path in _sorted_image_files(defect_dir):
            is_anomalous = defect_type != "good"
            mask_path = _resolve_mask_path(category_dir, defect_type, image_path) if is_anomalous else None

            samples.append(
                MVTecADSample(
                    dataset_name="mvtec_ad",
                    category=category,
                    split=split,
                    label=1 if is_anomalous else 0,
                    is_anomalous=is_anomalous,
                    defect_type=defect_type,
                    image_path=str(image_path),
                    mask_path=str(mask_path) if mask_path is not None else None,
                )
            )

    return samples


class MVTecADDataset(Dataset):
    """
    Clean reusable MVTec AD dataset loader for the selected categories only.

    Default behavior:
    - image is returned as float tensor [C, H, W] in [0, 1]
    - mask is returned as float tensor [1, H, W] with values {0, 1}
    - good samples always receive a zero mask tensor
    """

    def __init__(
        self,
        dataset_root: str | Path,
        category: str,
        split: str,
        image_transform: Callable[[Any], Any] | None = None,
        mask_transform: Callable[[Any], Any] | None = None,
        return_metadata: bool = True,
        load_mask: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.return_metadata = return_metadata
        self.load_mask = load_mask

        self.samples = collect_mvtec_ad_samples(
            dataset_root=self.dataset_root,
            category=self.category,
            split=self.split,
        )

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for category='{self.category}', split='{self.split}'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = Path(sample.image_path)
        mask_path = Path(sample.mask_path) if sample.mask_path is not None else None

        image_pil = _load_rgb_image(image_path)
        mask_pil = _load_mask_image(mask_path, size_hw=(image_pil.height, image_pil.width))

        image = (
            self.image_transform(image_pil)
            if self.image_transform is not None
            else _pil_image_to_float_tensor(image_pil)
        )

        item: dict[str, Any] = {
            "image": image,
            "label": sample.label,
            "is_anomalous": sample.is_anomalous,
            "category": sample.category,
            "defect_type": sample.defect_type,
            "image_path": sample.image_path,
        }

        if self.load_mask:
            mask = (
                self.mask_transform(mask_pil)
                if self.mask_transform is not None
                else _pil_mask_to_float_tensor(mask_pil)
            )
            item["mask"] = mask
            item["mask_path"] = sample.mask_path

        if self.return_metadata:
            item["dataset_name"] = sample.dataset_name
            item["split"] = sample.split
            item["sample_index"] = index
            item["original_size_hw"] = (image_pil.height, image_pil.width)

        return item

    def summary(self) -> dict[str, Any]:
        total = len(self.samples)
        normal = sum(1 for s in self.samples if not s.is_anomalous)
        anomalous = total - normal

        defect_type_counts: dict[str, int] = {}
        for s in self.samples:
            defect_type_counts[s.defect_type] = defect_type_counts.get(s.defect_type, 0) + 1

        return {
            "dataset_name": "mvtec_ad",
            "category": self.category,
            "split": self.split,
            "total_samples": total,
            "normal_samples": normal,
            "anomalous_samples": anomalous,
            "defect_type_counts": dict(sorted(defect_type_counts.items())),
        }


def discover_available_categories(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    available: list[str] = []
    for category in VALID_MVTEC_AD_CATEGORIES:
        if (root / category).exists():
            available.append(category)
    return available


def build_dataset_summary_table(dataset_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in discover_available_categories(dataset_root):
        for split in VALID_SPLITS:
            dataset = MVTecADDataset(
                dataset_root=dataset_root,
                category=category,
                split=split,
                return_metadata=False,
                load_mask=(split == "test"),
            )
            rows.append(dataset.summary())
    return rows


def _preview_sample(item: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {
        "label": item["label"],
        "is_anomalous": item["is_anomalous"],
        "category": item["category"],
        "defect_type": item["defect_type"],
        "image_path": item["image_path"],
    }

    image = item["image"]
    if isinstance(image, torch.Tensor):
        preview["image_shape"] = list(image.shape)
        preview["image_dtype"] = str(image.dtype)
    else:
        preview["image_type"] = type(image).__name__

    if "mask" in item:
        mask = item["mask"]
        preview["mask_path"] = item.get("mask_path")
        if isinstance(mask, torch.Tensor):
            preview["mask_shape"] = list(mask.shape)
            preview["mask_dtype"] = str(mask.dtype)
            preview["mask_positive_pixels"] = int((mask > 0).sum().item())
        else:
            preview["mask_type"] = type(mask).__name__

    if "original_size_hw" in item:
        preview["original_size_hw"] = item["original_size_hw"]

    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect selected MVTec AD categories.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to MVTec AD root folder, e.g. data/mvtec_ad",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        choices=VALID_MVTEC_AD_CATEGORIES,
        help="Category to inspect.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=VALID_SPLITS,
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of sample previews to print.",
    )
    parser.add_argument(
        "--summary-all",
        action="store_true",
        help="Print summary for all available selected categories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.summary_all:
        rows = build_dataset_summary_table(args.root)
        print(json.dumps(rows, indent=2))
        return

    dataset = MVTecADDataset(
        dataset_root=args.root,
        category=args.category,
        split=args.split,
        return_metadata=True,
        load_mask=(args.split == "test"),
    )

    print(json.dumps(dataset.summary(), indent=2))

    preview_count = min(args.limit, len(dataset))
    print(f"\nPreviewing {preview_count} sample(s):")
    for idx in range(preview_count):
        print(json.dumps(_preview_sample(dataset[idx]), indent=2))


if __name__ == "__main__":
    main()