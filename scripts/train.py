import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from birdshift.datasets.cub import CUBDataset
from birdshift.models.resnet import BirdResNet
from birdshift.training.trainer import train_one_epoch


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/CUB_200_2011")
    parser.add_argument("--mode", choices=["full", "crop", "birdshift"], default="full")
    parser.add_argument("--suppression-prob", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--imbalance-ratio", type=float, default=1.0,
                        help="Long-tail train split: max/min class size (1 = original data)")
    parser.add_argument("--balance", choices=["none", "sampler", "cb", "la"], default="none",
                        help="none=plain CE, sampler=WeightedRandomSampler, cb=class-balanced CE, "
                             "la=logit-adjusted CE (Menon et al. 2021)")
    parser.add_argument("--la-tau", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp", action="store_true", help="float16 mixed precision + channels_last")
    parser.add_argument("--cb-beta", type=float, default=0.999)
    parser.add_argument("--run-name", default="",
                        help="Checkpoint/log name; defaults to the mode's directory")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Device:", device)

    train_ds = CUBDataset(args.data_root, split="train", mode=args.mode,
                          suppression_prob=args.suppression_prob,
                          imbalance_ratio=args.imbalance_ratio,
                          image_size=args.image_size)
    val_ds = CUBDataset(args.data_root, split="val", mode=args.mode, image_size=args.image_size)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda")
    labels = torch.tensor(train_ds.data["label"].values)
    counts = torch.bincount(labels, minlength=200).float()
    print(f"Train images: {len(train_ds)} | per-class max={int(counts.max())} min={int(counts.min())}")
    sampler = None
    if args.balance == "sampler":
        # Each class drawn equally often in expectation; tail images repeat.
        sampler = WeightedRandomSampler((1.0 / counts)[labels].double(), len(labels), replacement=True)
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)

    model = BirdResNet(num_classes=200, pretrained=not bool(args.resume)).to(device)
    criterion = nn.CrossEntropyLoss()
    train_criterion = criterion
    if args.balance == "cb":
        # Cui et al. 2019: weight_c = (1 - beta) / (1 - beta^n_c), normalised to sum to C.
        weights = (1 - args.cb_beta) / (1 - args.cb_beta ** counts)
        weights = weights / weights.sum() * len(counts)
        train_criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    elif args.balance == "la":
        # Logit-adjusted loss: train on logits + tau*log(prior); predict with raw logits.
        log_prior = torch.log(counts / counts.sum()).to(device)
        train_criterion = lambda logits, y: criterion(logits + args.la_tau * log_prior, y)
    scaler = torch.amp.GradScaler(device.type) if args.amp else None
    if args.amp:
        model = model.to(memory_format=torch.channels_last)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    run_dir = Path("checkpoints") / {
        "full": "baseline",
        "crop": "crop",
        "birdshift": "birdshift"
    }[args.mode]
    if args.run_name:
        run_dir = Path("checkpoints") / args.run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("results/metrics") / f"cub_{args.run_name or args.mode}_training.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_accuracy = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_accuracy = float(checkpoint["val_accuracy"])
        previous_mode = checkpoint.get("mode")
        if previous_mode is not None and previous_mode != args.mode:
            raise ValueError(f"Checkpoint mode={previous_mode}, requested mode={args.mode}")
        print(f"Resuming from epoch {start_epoch}; best validation accuracy {best_accuracy:.2%}")

    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    append = bool(args.resume) and log_path.exists()
    with log_path.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not append:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            train_metrics = train_one_epoch(model, train_loader, train_criterion, optimizer, device, scaler)
            val_metrics = validate(model, val_loader, criterion, device)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {row['train_loss']:.4f} | "
                f"Train Acc: {row['train_accuracy']:.4f} | "
                f"Val Loss: {row['val_loss']:.4f} | "
                f"Val Acc: {row['val_accuracy']:.4f}"
            )

            improved = val_metrics["accuracy"] > best_accuracy
            if improved:
                best_accuracy = val_metrics["accuracy"]

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_accuracy,
                "mode": args.mode,
                "imbalance_ratio": args.imbalance_ratio,
                "balance": args.balance,
                "image_size": args.image_size,
            }
            torch.save(checkpoint, run_dir / "last.pth")
            if improved:
                torch.save(checkpoint, run_dir / "best.pth")
                print(f"Saved best model! Accuracy: {best_accuracy:.2%}")

    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print("Training log:", log_path)
    print("Checkpoints:", run_dir)


if __name__ == "__main__":
    main()


