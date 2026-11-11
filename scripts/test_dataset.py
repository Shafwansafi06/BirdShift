"""Sanity check: load CUB splits, print their sizes and push one batch through the model."""
import torch
from torch.utils.data import DataLoader

from birdshift.datasets.cub import CUBDataset
from birdshift.models.resnet import BirdResNet


def main():
    root = "data/raw/CUB_200_2011"

    train = CUBDataset(root, split="train")
    val = CUBDataset(root, split="val")
    test = CUBDataset(root, split="test")

    print("Training images:", len(train))
    print("Validation images:", len(val))
    print("Testing images:", len(test))

    loader = DataLoader(
        train,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Label range:", labels.min().item(), labels.max().item())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BirdResNet(
        num_classes=200,
        pretrained=True
    ).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(images.to(device))

    print("Model output:", outputs.shape)

    assert outputs.shape == (8, 200)


if __name__ == "__main__":
    main()

