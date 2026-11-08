"""Compare standard CUB validation and background-shift 'original' inputs/predictions.
Run from the BirdShift repository root with PYTHONPATH=src.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_shift import ShiftedCUBDataset  # existing BirdShift script
from birdshift.datasets.cub import CUBDataset
from birdshift.models.resnet import BirdResNet


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BirdResNet(num_classes=200, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device.type == "cuda"))
    predictions = {}
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        classes = logits.argmax(dim=1).cpu().tolist()
        for image_id, label, pred in zip(batch["image_id"].tolist(),
                                         batch["label"].tolist(), classes):
            predictions[int(image_id)] = (int(label), int(pred))
    return predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/CUB_200_2011")
    p.add_argument("--checkpoint", default="checkpoints/baseline/best.pth")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard = CUBDataset(args.root, split=args.split, mode="full")
    shifted_original = ShiftedCUBDataset(args.root, args.split, "full", "original")

    print("Checkpoint SHA256:", file_hash(args.checkpoint))
    print("Dataset lengths:", len(standard), len(shifted_original))
    if len(standard) != len(shifted_original):
        raise RuntimeError("The two dataset lengths differ")

    tensor_mismatch = []
    max_diff = 0.0
    for i in range(len(standard)):
        a, b = standard[i], shifted_original[i]
        if a["image_id"] != b["image_id"] or a["label"] != b["label"]:
            raise RuntimeError(f"Different image ID or label at position {i}: "
                               f"{a['image_id']} vs {b['image_id']}")
        diff = (a["image"] - b["image"]).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            tensor_mismatch.append((int(a["image_id"]), round(diff, 6)))

    print("Image tensor mismatches:", len(tensor_mismatch))
    print("Maximum absolute tensor difference:", round(max_diff, 7))
    if tensor_mismatch:
        print("First 10 mismatches (image ID, max difference):", tensor_mismatch[:10])

    model, ckpt = load_model(args.checkpoint, device)
    print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))
    print("Checkpoint recorded val accuracy:", ckpt.get("val_accuracy", "unknown"))
    a = predict(model, standard, device, args.batch_size, args.workers)
    b = predict(model, shifted_original, device, args.batch_size, args.workers)
    if a.keys() != b.keys():
        raise RuntimeError("Image ID sets are different")

    prediction_mismatch = [k for k in a if a[k] != b[k]]
    acc_a = sum(label == pred for label, pred in a.values()) / len(a)
    acc_b = sum(label == pred for label, pred in b.values()) / len(b)
    print(f"Standard original Top-1: {acc_a:.4%}")
    print(f"Shifted original Top-1:  {acc_b:.4%}")
    print("Prediction mismatches:", len(prediction_mismatch))
    print("First 10 differing image IDs:", prediction_mismatch[:10])

    result = {
        "checkpoint_sha256": file_hash(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "sample_count": len(a),
        "tensor_mismatch_count": len(tensor_mismatch),
        "maximum_tensor_difference": max_diff,
        "prediction_mismatch_count": len(prediction_mismatch),
        "standard_top1": acc_a,
        "shift_original_top1": acc_b,
    }
    out = Path("results/metrics/evaluation_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", out)
    if not tensor_mismatch and not prediction_mismatch:
        print("PASS: pipelines agree with current checkpoint and dataset code")
        print("If older reported accuracies differ, compare old checkpoint hashes and logs.")
    else:
        print("CHECK: preprocessing or prediction paths are inconsistent.")


if __name__ == "__main__":
    main()
