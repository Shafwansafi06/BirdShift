"""ResNet-50 classifier used by every experiment (He et al. 2016)."""
import timm
import torch.nn as nn


class BirdResNet(nn.Module):
    """ImageNet-pretrained ResNet-50 with its 1000-way head replaced by a `num_classes` head.

    weights: timm model name. "resnet50" loads timm's default a1_in1k weights;
    "resnet50.tv_in1k" loads the classic torchvision weights, which fine-tune 7.8 pp
    better on CUB (see README, "Training recipe"). pretrained=False builds the same
    architecture with random weights, used when a checkpoint is loaded afterwards.
    """

    def __init__(self, num_classes=200, pretrained=True, weights="resnet50"):
        super().__init__()
        # timm swaps the final fully connected layer for a new randomly initialised one.
        self.backbone = timm.create_model(weights, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        """x: (B, 3, H, W) normalised images -> (B, num_classes) logits."""
        return self.backbone(x)
