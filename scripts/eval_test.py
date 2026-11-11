"""Top-1 and balanced (mean per-species) accuracy of full-image checkpoints.

Default: CUB official test split.
  PYTHONPATH=src python scripts/eval_test.py final_sgd_tv
With --manifest: the given splits of a manifest, e.g. the geography split.
  PYTHONPATH=src python scripts/eval_test.py geo_sgd_tv --manifest data/splits/geography/geography_clusters.csv
Uses checkpoints/<run>/best.pth.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from birdshift.datasets.cub import CUBDataset
from birdshift.datasets.manifest import ManifestDataset
from birdshift.models.resnet import BirdResNet


@torch.no_grad()
def predict(model, ds, dev):
    preds, labels = [], []
    with torch.autocast(dev.type, dtype=torch.float16):
        for b in DataLoader(ds, batch_size=64, num_workers=8):
            preds.append(model(b["image"].to(dev)).argmax(1).cpu())
            labels.append(b["label"])
    return torch.cat(preds), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--manifest-root", default="data/raw")
    ap.add_argument("--splits", nargs="+", default=["test"])
    a = ap.parse_args()

    dev = torch.device("cuda")
    for run in a.runs:
        ck = torch.load(f"checkpoints/{run}/best.pth", map_location=dev, weights_only=False)
        model = BirdResNet(200, pretrained=False).to(dev)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        size = ck.get("image_size", 224)
        for split in a.splits:
            ds = (ManifestDataset(a.manifest, a.manifest_root, split, size) if a.manifest else
                  CUBDataset("data/raw/CUB_200_2011", split=split, mode="full", image_size=size))
            p, y = predict(model, ds, dev)
            hit = (p == y).float()
            per_class = [hit[y == c].mean().item() for c in y.unique()]
            print(f"{run} [{split}, {len(ds)} images, {len(per_class)} species]: epoch {ck['epoch']} "
                  f"top-1 {hit.mean().item():.2%} ({int(hit.sum())}/{len(ds)}) "
                  f"balanced {sum(per_class) / len(per_class):.2%}", flush=True)


if __name__ == "__main__":
    main()
