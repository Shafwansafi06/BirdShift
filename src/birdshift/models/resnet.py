import timm
import torch.nn as nn

class BirdResNet(nn.Module):
    def __init__(self, num_classes=200, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
                "resnet50",
                pretrained=pretrained,
                num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)
