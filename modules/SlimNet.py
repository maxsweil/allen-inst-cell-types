import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np


class SlimNet(nn.Module):

    def __init__(self, nfl_units, n_classes=10):
        """
        Args:
            nfl_units (int): number of first layer units, assumed to be 8 for output size calculations
        """

        super(SlimNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, nfl_units, kernel_size=3, padding=1),  # Output = 8x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output = 8x16x16

            nn.Conv2d(nfl_units, 2 * nfl_units, kernel_size=3, padding=1),  # Output = 16x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output = 16x8x8

            nn.Conv2d(2 * nfl_units, 4 * nfl_units, kernel_size=3, padding=1),  # Output = 32x8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(4 * nfl_units, 4 * nfl_units, kernel_size=3, padding=1),  # Output = 32x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output = 32x4x4

            nn.Conv2d(4 * nfl_units, 8 * nfl_units, kernel_size=3, padding=1),  # Output = 64x4x4
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * nfl_units, 8 * nfl_units, kernel_size=3, padding=1),  # Output = 64x4x4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output = 64x2x2

            nn.Conv2d(8 * nfl_units, 8 * nfl_units, kernel_size=3, padding=1),  # Output = 64x2x2
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * nfl_units, 8 * nfl_units, kernel_size=3, padding=1),  # Output = 64x2x2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output = 64x1x1
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Output = 64x7x7
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 8 * nfl_units, 16 * nfl_units),  # Output = 1x128
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(16 * nfl_units, 16 * nfl_units),  # Output = 1x128
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(16 * nfl_units, n_classes)  # Output = n_classes
        )
        self._initialize_weights()

    def forward(self, input_tensor):

        x = self.features(input_tensor)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)