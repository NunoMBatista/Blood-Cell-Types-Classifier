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
        """
        Conv channels (32, 64, 128):
            
            Architecture: 
                
                1st Conv Block:
                Input: (3, 28, 28) -> Conv2d(3, 32, kernel_size=3, padding=1) -> Output (32, 28, 28) -> MaxPool2d(kernel_size=2) -> Output (32, 14, 14)
                
                2nd Conv Block:
                Input: (32, 14, 14) -> Conv2d(32, 64, kernel_size=3, padding=1) -> Output (64, 14, 14) -> MaxPool2d(kernel_size=2) -> Output (64, 7, 7)
                
                3rd Conv Block:
                Input: (64, 7, 7) -> Conv2d(64, 128, kernel_size=3, padding=1) -> Output (128, 7, 7) -> MaxPool2d(kernel_size=2) -> Output (128, 3, 3)
                
                AdaptiveAvgPool2d((1, 1)) -> Output (128, 1, 1)
            
            
            Parameter count (bias = False in Conv2d): 
                Number of parameters in a Conv2d layer = (kernel_h * kernel_w * in_channels) * out_channels
                1st Conv Block: (3*3*3)*32 = 864
                2nd Conv Block: (3*3*32)*64 = 18432
                3rd Conv Block: (3*3*64)*128 = 73728
                Total Conv Layer Parameters = 864 + 18432 + 73728 = 93024

                (Plus the linear layer parameters)
        
        """
        
        for idx, out_channels in enumerate(conv_channels):
            out_channels = int(out_channels)
            feature_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # performs the convolution operation
                    nn.BatchNorm2d(out_channels), # normalizes the output of the convolution layer 
                    nn.ReLU(inplace=True), # applies the ReLU activation function to each element
                ]
            )
            # downsample after each block to control spatial resolution.
            feature_layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1))) # adaptive pooling to get fixed size output
        self.features = nn.Sequential(*feature_layers)

        # we need to flatten the output from the feature extractor before passing it to the classifier
        classifier_layers: list[nn.Module] = [nn.Flatten()]
        in_features = conv_channels[-1] # since after AdaptiveAvgPool2d((1, 1)), the feature map size is (out_channels, 1, 1)
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
