import torch
from torchvision import models


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(
            *list(self.resnet.children())[:-2],
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x