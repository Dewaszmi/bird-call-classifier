import torch
import torch.nn as nn

POOLING_MODES = ("gap", "masked-gap")


def masked_global_avg_pool2d(x: torch.Tensor, valid_widths: torch.Tensor) -> torch.Tensor:
    """Average over height and valid time columns, ignoring padded regions."""
    _, _, height, _ = x.shape
    width = x.size(3)
    col_idx = torch.arange(width, device=x.device)
    mask = (col_idx.unsqueeze(0) < valid_widths.unsqueeze(1)).to(x.dtype)
    mask = mask.unsqueeze(1).unsqueeze(1)

    summed = (x * mask).sum(dim=(2, 3))
    counts = (valid_widths * height).clamp(min=1).unsqueeze(1).to(x.dtype)
    return summed / counts


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BirdVGG(nn.Module):
    """Small VGG-style CNN for log-mel spectrogram classification."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        pooling_mode: str = "gap",
    ):
        super().__init__()
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {POOLING_MODES}")

        self.pooling_mode = pooling_mode
        self.features = nn.Sequential(
            VGGBlock(in_channels, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            VGGBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def _forward_gap(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.gap(self.features(x)).flatten(1)
        return self.fc(self.dropout(pooled))

    def _forward_masked(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(x.size(0)):
            sample = x[i : i + 1, :, :, : int(lengths[i])]
            features = self.features(sample)
            valid_width = torch.tensor([features.size(3)], device=x.device)
            pooled = masked_global_avg_pool2d(features, valid_width)
            outputs.append(self.fc(self.dropout(pooled)))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pooling_mode == "masked-gap":
            if lengths is None:
                raise ValueError("lengths are required when pooling_mode='masked-gap'")
            return self._forward_masked(x, lengths)
        return self._forward_gap(x)
