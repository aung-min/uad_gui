from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def resolve_device(device: str = "auto") -> torch.device:
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class ConvAutoencoderConfig:
    in_channels: int = 3
    base_channels: int = 64
    latent_channels: int = 512
    image_size: int = 256


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvAutoencoder(nn.Module):
    """
    Clean convolutional autoencoder for normal-only reconstruction learning.
    Input  : [B, 3, H, W]
    Output : reconstruction [B, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 512,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c1),
            DownBlock(c1, c1),
            DownBlock(c1, c2),
            DownBlock(c2, c3),
            DownBlock(c3, latent_channels),
        )

        self.decoder = nn.Sequential(
            UpBlock(latent_channels, c3),
            UpBlock(c3, c2),
            UpBlock(c2, c1),
            UpBlock(c1, c1),
            nn.Conv2d(c1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
        }

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {x.shape[1]} channels")

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(x)["reconstruction"]
        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def reconstruction_error_map(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.reconstruct(x)
        error_map = torch.mean(torch.abs(x - reconstruction), dim=1, keepdim=True)
        return error_map.contiguous()

    @torch.no_grad()
    def image_scores(self, x: torch.Tensor) -> torch.Tensor:
        error_map = self.reconstruction_error_map(x)
        scores = error_map.flatten(start_dim=1).mean(dim=1)
        return scores.contiguous()


def build_conv_autoencoder(
    in_channels: int = 3,
    base_channels: int = 64,
    latent_channels: int = 512,
) -> ConvAutoencoder:
    return ConvAutoencoder(
        in_channels=in_channels,
        base_channels=base_channels,
        latent_channels=latent_channels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test the convolutional autoencoder.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size.")
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size.")
    parser.add_argument("--base-channels", type=int, default=64, help="Base channel width.")
    parser.add_argument("--latent-channels", type=int, default=512, help="Latent channel width.")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model = build_conv_autoencoder(
        in_channels=3,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)
    model.eval()

    x = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
    out = model(x)
    reconstruction = out["reconstruction"]
    latent = out["latent"]
    error_map = model.reconstruction_error_map(x)
    scores = model.image_scores(x)

    result: dict[str, Any] = {
        "device": str(device),
        "input_shape": list(x.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "latent_shape": list(latent.shape),
        "error_map_shape": list(error_map.shape),
        "image_scores_shape": list(scores.shape),
        "image_scores": [float(v) for v in scores.detach().cpu().tolist()],
        "base_channels": args.base_channels,
        "latent_channels": args.latent_channels,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()