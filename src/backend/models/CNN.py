from typing import Any, Iterable

import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 8,
        conv_channels: Iterable[int] | None = None,
        linear_dims: Iterable[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = (32, 64, 128)
        if linear_dims is None:
            linear_dims = (128,)

        conv_channels = tuple(int(c) for c in conv_channels)
        linear_dims = tuple(int(d) for d in linear_dims)

        feature_layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for idx, out_channels in enumerate(conv_channels):
            out_channels = int(out_channels)
            feature_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            # Downsample after each block to control spatial resolution.
            feature_layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*feature_layers)

        classifier_layers: list[nn.Module] = [nn.Flatten()]
        in_features = conv_channels[-1]
        for hidden_dim in linear_dims:
            classifier_layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = hidden_dim

        classifier_layers.append(nn.Linear(in_features, int(num_classes)))
        self.classifier = nn.Sequential(*classifier_layers)

        self.model_config = {
            "input_channels": int(input_channels),
            "num_classes": int(num_classes),
            "conv_channels": list(conv_channels),
            "linear_dims": list(linear_dims),
            "dropout": float(dropout),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def get_config(self) -> dict[str, Any]:
        """Return a shallow copy of the architecture configuration."""

        return dict(self.model_config)
