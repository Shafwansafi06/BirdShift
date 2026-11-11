"""Sanity check: show a few bird-crop training samples to verify box cropping."""
import matplotlib.pyplot as plt

from birdshift.datasets.cub import CUBDataset


root = "data/raw/CUB_200_2011"

dataset = CUBDataset(
    root,
    split="train",
    mode="crop"
)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for index, ax in enumerate(axes.flat):
    sample = dataset[index]
    image = sample["image"].permute(1, 2, 0)

    for channel in range(3):
        image[:, :, channel] = (
            image[:, :, channel] * std[channel] + mean[channel]
        )

    image = image.clamp(0, 1)

    ax.imshow(image)
    ax.set_title(f"Class {sample['label']}")
    ax.axis("off")

plt.tight_layout()
plt.show()
