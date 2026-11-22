"""InceptionTime classifier for multivariate headset windows.

This is the residual multi-scale CNN from Ismail Fawaz et al. (2020), not a
transformer. Parallel convolutions look at short, medium, and long motion, then
a global pool votes on whether the window contains exactly one label transition.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _odd_kernels(kernel_size):
    """Three odd kernel sizes: `kernel_size`, half, and a quarter."""
    sizes = [max(kernel_size // (2 ** i), 1) for i in range(3)]
    return [size if size % 2 else size - 1 for size in sizes]


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes, bottleneck_channels):
        super().__init__()
        use_bottleneck = in_channels > 1
        bottleneck_in = bottleneck_channels if use_bottleneck else in_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if use_bottleneck else nn.Identity()
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_in, n_filters, kernel_size=kernel, padding='same', bias=False)
            for kernel in kernel_sizes
        ])
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        branches = [conv(bottleneck) for conv in self.convs]
        branches.append(self.pool(x))
        return F.relu(self.batch_norm(torch.cat(branches, dim=1)))


class InceptionTime(nn.Module):
    """Single InceptionTime network. The paper ensembles five of these."""

    def __init__(self, in_channels=13, num_classes=2, n_filters=128, depth=12,
                 kernel_size=40, bottleneck_channels=32):
        super().__init__()
        kernel_sizes = _odd_kernels(kernel_size)
        blocks = []
        shortcuts = []
        channels = in_channels
        residual_channels = in_channels
        for index in range(depth):
            block = InceptionModule(channels, n_filters, kernel_sizes, bottleneck_channels)
            blocks.append(block)
            channels = block.out_channels
            # A residual link closes every third module, from that group's input.
            if index % 3 == 2:
                if residual_channels == channels:
                    shortcuts.append(nn.BatchNorm1d(residual_channels))
                else:
                    shortcuts.append(nn.Sequential(
                        nn.Conv1d(residual_channels, channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(channels),
                    ))
                residual_channels = channels
        self.blocks = nn.ModuleList(blocks)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        residual = x
        shortcut_index = 0
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index % 3 == 2:
                x = F.relu(x + self.shortcuts[shortcut_index](residual))
                shortcut_index += 1
                residual = x
        pooled = self.gap(x).flatten(1)
        return self.fc(pooled)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
