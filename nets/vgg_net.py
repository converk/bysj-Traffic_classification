import torch.nn.functional as F
import torch
from torch import nn


# VGG-16模型
class VGGNeT(nn.Module):
    def __init__(self):
        super(VGGNeT, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(128, 12)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        cfg=[16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)