"""Background-only feature extraction for CUB-200-2011.

Covers each bird's bounding box (enlarged by a small margin) with neutral gray,
then extracts 2048-d penultimate-layer features with an ImageNet-pretrained
torchvision ResNet-50. Clusters built on these features describe the scene,
not the species.

Outputs (in --out-dir):
  background_features.npy       float32 array, shape (11788, 2048), rows in image_id order
  background_features_ids.csv   image_id, image_path, label (0-based), bg_fraction

Usage (from the repository root):
  python scripts/background/extract_features.py
  python scripts/background/extract_features.py --cub-root "C:/path/to/CUB_200_2011" --out-dir data/processed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

FILL_RGB = (124, 116, 104)  # ImageNet mean colour, neutral for the network


def load_metadata(cub_root: Path) -> pd.DataFrame:
    def read(name, cols):
        return pd.read_csv(cub_root / name, sep=" ", header=None, names=cols)

    meta = (
        read("images.txt", ["image_id", "image_path"])
        .merge(read("image_class_labels.txt", ["image_id", "class_id"]), on="image_id")
        .merge(read("bounding_boxes.txt", ["image_id", "x", "y", "w", "h"]), on="image_id")
    )
    meta["label"] = meta["class_id"] - 1
    meta = meta.sort_values("image_id").reset_index(drop=True)
    assert len(meta) == 11788 and meta["image_id"].is_unique, len(meta)
    assert meta["label"].between(0, 199).all()
    return meta


def mask_bird(img, x, y, w, h, margin=0.05, fill=FILL_RGB):
    """Fill the enlarged bounding box with gray. Returns (masked image, background fraction)."""
    W, H = img.size
    dx, dy = w * margin, h * margin
    x0, y0 = max(0, int(round(x - dx))), max(0, int(round(y - dy)))
    x1, y1 = min(W, int(round(x + w + dx))), min(H, int(round(y + h + dy)))
    out = img.copy()
    if x1 > x0 and y1 > y0:
        ImageDraw.Draw(out).rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
    return out, 1.0 - ((x1 - x0) * (y1 - y0)) / float(W * H)


class BackgroundDataset(Dataset):
    def __init__(self, meta, cub_root, tfm, margin):
        self.meta, self.root, self.tfm, self.margin = meta.reset_index(drop=True), cub_root, tfm, margin

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        img = Image.open(self.root / "images" / r["image_path"]).convert("RGB")
        masked, frac = mask_bird(img, r["x"], r["y"], r["w"], r["h"], margin=self.margin)
        return self.tfm(masked), int(r["image_id"]), float(frac)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cub-root", type=Path, default=Path("data/raw/CUB_200_2011"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.05, help="box enlargement per side, as a fraction of box size")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_metadata(args.cub_root)
    tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),  # whole image, no crop: keep all background
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(BackgroundDataset(meta, args.cub_root, tfm, args.margin),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)

    feats, ids, fracs = [], [], []
    with torch.inference_mode():
        for x, batch_ids, batch_fracs in tqdm(loader, desc=f"Extracting on {device}"):
            feats.append(model(x.to(device)).float().cpu().numpy())
            ids.append(batch_ids.numpy())
            fracs.append(batch_fracs.numpy())

    feats = np.concatenate(feats).astype(np.float32)
    assert (np.concatenate(ids) == meta["image_id"].values).all(), "row order mismatch"
    assert feats.shape == (len(meta), 2048) and np.isfinite(feats).all()

    out = meta[["image_id", "image_path", "label"]].copy()
    out["bg_fraction"] = np.concatenate(fracs).round(4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "background_features.npy", feats)
    out.to_csv(args.out_dir / "background_features_ids.csv", index=False)
    print(f"Saved {feats.shape} features to {args.out_dir}")


if __name__ == "__main__":
    main()
