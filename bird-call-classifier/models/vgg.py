import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_convs):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BirdVGG(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(in_channels, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            VGGBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
