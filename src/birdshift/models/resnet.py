import timm
import torch.nn as nn

class BirdResNet(nn.Module):
    def __init__(self, num_classes=200, pretrained=True, weights="resnet50"):
        # weights: timm name, e.g. "resnet50" (= resnet50.a1_in1k) or "resnet50.tv_in1k"
        super().__init__()

        self.backbone = timm.create_model(
                weights,
                pretrained=pretrained,
                num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)
