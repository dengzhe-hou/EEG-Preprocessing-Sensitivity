"""
EEG decoder models: EEGNet and ShallowFBCSPNet.

These are standard baselines used across all experiments.
PGI is a training objective, not an architecture — it wraps any decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """EEGNet-v4 (Lawhern et al., 2018).

    Compact CNN for EEG decoding. Default config for 22-channel, 250 Hz data.
    """

    def __init__(self, n_channels=22, n_times=1000, n_classes=4,
                 F1=8, D=2, F2=16, kernel_length=64,
                 pool1=4, pool2=8, dropout=0.5):
        super().__init__()
        self.n_classes = n_classes

        # Block 1: Temporal + Spatial convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16),
                                         padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        # Compute flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self._forward_features(x)
            n_flat = x.shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(1)
        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) or (B, 1, C, T) EEG tensor

        Returns:
            logits: (B, n_classes) unnormalized scores
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        features = self._forward_features(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract features before classifier (for probing)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self._forward_features(x)


class ShallowNet(nn.Module):
    """Simplified ShallowFBCSPNet-style model."""

    def __init__(self, n_channels=22, n_times=1000, n_classes=4,
                 n_filters=40, pool_size=75, dropout=0.5):
        super().__init__()
        self.n_classes = n_classes

        self.temporal_conv = nn.Conv2d(1, n_filters, (1, 25), bias=False)
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_channels, 1),
                                      bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_size // 3))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self._forward_features(x)
            n_flat = x.shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x.pow(2)  # squaring nonlinearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # log nonlinearity
        x = self.drop(x)
        return x.flatten(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self._forward_features(x))

    def get_features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self._forward_features(x)


def get_model(name, n_channels, n_times, n_classes, **kwargs):
    """Factory function for models."""
    if name == "eegnet":
        return EEGNet(n_channels=n_channels, n_times=n_times,
                      n_classes=n_classes, **kwargs)
    elif name == "shallow":
        return ShallowNet(n_channels=n_channels, n_times=n_times,
                          n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
