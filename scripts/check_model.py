"""Sanity check: build the ResNet-50 and run one random batch to confirm output shape."""
import torch

from birdshift.models.resnet import BirdResNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BirdResNet(
            num_classes=200,
            pretrained=True
    ).to(device)

    images = torch.randn(4, 3, 224, 224, device=device)

    model.eval()

    with torch.no_grid():
        outputs = model(images)

    print("Device:", device)
    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    assert outputs.shape == (4, 200)

if __name__ == "__main__":
    main()
