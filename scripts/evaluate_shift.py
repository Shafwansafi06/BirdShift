"""Compare full-image and bird-crop checkpoints on controlled CUB background shifts.

Run from repository root with PYTHONPATH=src. Uses the exact val split from CUBDataset
and the evaluation transforms corresponding to the saved model types.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from birdshift.datasets.cub import CUBDataset
from birdshift.models.resnet import BirdResNet
from birdshift.transforms.background import background_variant, bbox_to_pixels, SquarePad


class ShiftedCUBDataset(Dataset):
    def __init__(self, root, split, mode, shift, blur_radius=22, image_size=224):
        if mode not in ("full", "crop"):
            raise ValueError("mode must be full or crop")
        if shift not in ("original", "blur", "replace"):
            raise ValueError("shift must be original, blur or replace")
        self.base = CUBDataset(root, split=split, mode=mode)
        self.root = Path(root)
        self.mode = mode
        self.shift = shift
        self.blur_radius = blur_radius
        norm = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        if mode == "crop":
            self.preprocess = transforms.Compose([
                SquarePad(), transforms.Resize((image_size, image_size)), transforms.ToTensor(), norm
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(image_size * 8 // 7), transforms.CenterCrop(image_size),
                transforms.ToTensor(), norm
            ])

    def __len__(self):
        return len(self.base.data)

    def __getitem__(self, index):
        row = self.base.data.iloc[index]
        with Image.open(self.root / "images" / row["image_path"]) as img:
            original = img.convert("RGB")
        bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
        # 'replace' uses a neutral background to avoid contamination by another bird.
        shifted = background_variant(original, bbox, self.shift, blur_radius=self.blur_radius)
        if self.mode == "crop":
            shifted = shifted.crop(bbox_to_pixels(bbox, shifted.size))
        tensor = self.preprocess(shifted)
        return {"image": tensor, "label": int(row["label"]), "image_id": int(row["id"])}


def load_checkpoint(path, mode, device):
    model = BirdResNet(num_classes=200, pretrained=False).to(device)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        # Older saved checkpoints or torch versions may not support weights_only.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dict with model_state_dict")
    saved_mode = checkpoint.get("mode")
    if saved_mode is not None and saved_mode != mode:
        raise ValueError(f"Checkpoint mode={saved_mode!r}, requested mode={mode!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.image_size = checkpoint.get("image_size", 224)
    return model


@torch.inference_mode()
def evaluate(model, loader, device):
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        top5 = logits.topk(5, dim=1).indices.cpu().tolist()
        for image_id, true_label, prediction in zip(
            batch["image_id"].tolist(), batch["label"].tolist(), top5
        ):
            rows.append({
                "image_id": image_id,
                "true_label": true_label,
                "predicted_label": prediction[0],
                "correct_top1": int(true_label == prediction[0]),
                "correct_top5": int(true_label in prediction),
            })
    top1 = sum(r["correct_top1"] for r in rows) / len(rows)
    top5 = sum(r["correct_top5"] for r in rows) / len(rows)
    f1 = f1_score(
        [r["true_label"] for r in rows],
        [r["predicted_label"] for r in rows],
        average="macro", labels=list(range(200)), zero_division=0,
    )
    return rows, {"samples": len(rows), "top1_accuracy": float(top1),
                  "top5_accuracy": float(top5), "macro_f1": float(f1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/CUB_200_2011")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--full-checkpoint", default="checkpoints/baseline/best.pth")
    parser.add_argument("--crop-checkpoint", default="checkpoints/crop/best.pth")
    parser.add_argument("--birdshift-checkpoint", type=str, default="checkpoints/birdshift/best.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blur-radius", type=float, default=22)
    parser.add_argument("--output", default="results/background_shift")
    args = parser.parse_args()
    models = [
        ("full", args.full_checkpoint, "full"),
        ("crop", args.crop_checkpoint, "crop"),
        ("birdshift", args.birdshift_checkpoint, "full"),  # trained with bg suppression, evaluated on full images
    ]
    for _, path, _ in models:
        if not Path(path).exists():
            parser.error(f"Checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, raw_predictions = [], {}
    for model_name, path, mode in models:
        model = load_checkpoint(path, model_name, device)  # model_name matches checkpoint's saved mode
        for shift in ("original", "blur", "replace"):
            data = ShiftedCUBDataset(args.data_root, args.split, mode, shift, args.blur_radius,
                                     image_size=model.image_size)
            loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=device.type == "cuda")
            predictions, metrics = evaluate(model, loader, device)
            with (output / f"{model_name}_{shift}_{args.split}_predictions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
            summaries.append({"model": model_name, "checkpoint": path, "shift": shift, **metrics})
            raw_predictions[(model_name, shift)] = predictions
            print(f"{model_name:9s} | {shift:8s} | top1={metrics['top1_accuracy']:.2%} "
                  f"top5={metrics['top5_accuracy']:.2%} f1={metrics['macro_f1']:.3f}")
    # Verify the exact same sample sequence was used in every condition.
    reference = [(r["image_id"], r["true_label"]) for r in raw_predictions[("full", "original")]]
    for key, preds in raw_predictions.items():
        assert [(r["image_id"], r["true_label"]) for r in preds] == reference, f"Sample mismatch: {key}"

    baselines = {r["model"]: r["top1_accuracy"] for r in summaries if r["shift"] == "original"}
    for r in summaries:
        r["accuracy_drop_pp"] = round((baselines[r["model"]] - r["top1_accuracy"]) * 100, 4)
    with (output / f"summary_{args.split}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / f"summary_{args.split}.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    crop_orig = raw_predictions[("crop", "original")]
    for shift in ("blur", "replace"):
        crop_shift = raw_predictions[("crop", shift)]
        exact_same = all(a["predicted_label"] == b["predicted_label"] for a,b in zip(crop_orig,crop_shift))
        print(f"Crop invariant to {shift}: {exact_same}")
        if not exact_same:
            print("WARNING: Crop predictions differ; verify crop bbox/preprocessing consistency.")
    print(f"Saved files to {output.resolve()}")


if __name__ == "__main__":
    main()
