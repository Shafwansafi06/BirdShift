"""Visualize the pre-augmentation BirdShift mixture on fixed CUB training images."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from birdshift.datasets.cub import CUBDataset
from birdshift.transforms.background import background_variant, bbox_to_pixels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw/CUB_200_2011")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--blur-radius", type=float, default=22)
    p.add_argument("--output", default="results/figures/birdshift_training")
    args = p.parse_args()
    dataset = CUBDataset(args.data_root, split="train", mode="full")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    indices = list(range(0, len(dataset), max(1, len(dataset)//max(1,args.samples))))[:args.samples]
    for idx in indices:
        row = dataset.data.iloc[idx]
        with Image.open(Path(args.data_root) / "images" / row["image_path"]) as im:
            original = im.convert("RGB")
        bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
        blur = background_variant(original, bbox, kind="blur", blur_radius=args.blur_radius)
        replace = background_variant(original, bbox, kind="replace")
        crop = original.crop(bbox_to_pixels(bbox, original.size))
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, img, title in zip(axes, [original, blur, replace, crop],
                                  ["Full image", "Blur outside bbox", "Neutral outside bbox", "Bird bbox crop"]):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"image_{int(row['id'])}.png", dpi=130)
        plt.close(fig)
    print(f"Saved {len(indices)} visualization(s) in {out.resolve()}")


if __name__ == "__main__":
    main()
