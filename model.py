"""
Author: Pravesh Bawangade
File Name: model.py

"""
import torch.nn as nn
import torchvision.models as models


class ColorizationUpsampling(nn.Module):
    """
    Autoencoder architecture.
    """
    def __init__(self, input_size=128):
        super(ColorizationUpsampling, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## Encoder
        resnet = models.resnet18()
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        ## Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        ## Encoder
        x = self.midlevel_resnet(x)
        ## Decoder
        x = self.upsample(x)
        return x

