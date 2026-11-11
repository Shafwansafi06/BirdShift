"""Plain top-1 / macro-F1 evaluation of one checkpoint on CUB val or test.

Kept as an independent second evaluator: scripts/audit_evaluators.py checks it agrees
image by image with evaluate_shift.py on unshifted images.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from birdshift.datasets.cub import CUBDataset
from birdshift.models.resnet import BirdResNet


@torch.no_grad()
def predict(model, loader, device):
    """Predicted and true labels for every image in the loader."""
    model.eval()
    records = []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        probabilities = logits.softmax(dim=1)
        top5_prob, top5_indices = probabilities.topk(k=5, dim=1)
        labels = batch["label"].tolist()
        image_ids = batch["image_id"].tolist()
        for i, image_id in enumerate(image_ids):
            indices = top5_indices[i].cpu().tolist()
            probs = top5_prob[i].cpu().tolist()
            records.append({
                "image_id": image_id,
                "true_label": labels[i],
                "predicted_label": indices[0],
                "top5_correct": int(labels[i] in indices),
                "top5_labels": " ".join(map(str, indices)),
                "top5_probabilities": " ".join(f"{p:.6f}" for p in probs),
            })
    return records


def main():
    """Evaluate one checkpoint and save metrics and per-image predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/CUB_200_2011")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["full", "crop"], default="full")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CUBDataset(args.data_root, split=args.split, mode=args.mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = BirdResNet(num_classes=200, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_mode = checkpoint.get("mode")
    if saved_mode is not None and saved_mode != args.mode:
        raise ValueError(f"Checkpoint trained as {saved_mode}, requested {args.mode}")
    model.load_state_dict(checkpoint["model_state_dict"])
    records = predict(model, loader, device)

    y_true = [row["true_label"] for row in records]
    y_pred = [row["predicted_label"] for row in records]
    metrics = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "split": args.split,
        "samples": len(records),
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": sum(row["top5_correct"] for row in records) / len(records),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(200)),
                                   average="macro", zero_division=0)),
    }
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"cub_{args.mode}_{args.split}"
    with (out_dir / f"{prefix}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(metrics, indent=2))
    print("Saved results to", out_dir)


if __name__ == "__main__":
    main()
