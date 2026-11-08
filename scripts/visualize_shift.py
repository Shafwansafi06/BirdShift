"""Save original, blurred and neutral-replaced backgrounds as image triplets."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from birdshift.datasets.cub import CUBDataset
from birdshift.transforms.background import background_variant, bbox_to_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/CUB_200_2011")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--blur-radius", type=float, default=22)
    parser.add_argument("--output", default="results/figures/background_shift")
    args = parser.parse_args()
    ds = CUBDataset(args.data_root, split=args.split, mode="full")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Sample evenly across the split for reproducibility.
    indices = list(range(0, len(ds), max(1, len(ds) // max(1, args.samples))))[:args.samples]
    for idx in indices:
        row = ds.data.iloc[idx]
        with Image.open(Path(args.data_root) / "images" / row["image_path"]) as im:
            original = im.convert("RGB")
        bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
        images = [background_variant(original, bbox, kind, blur_radius=args.blur_radius)
                  for kind in ("original", "blur", "replace")]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, image, title in zip(axes, images, ("Original", "Blurred outside bbox", "Replaced outside bbox")):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"CUB image ID {int(row['id'])} — class {int(row['label'])}")
        fig.tight_layout()
        fig.savefig(output / f"image_{int(row['id'])}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(indices)} comparison figures in {output.resolve()}")


if __name__ == "__main__":
    main()
