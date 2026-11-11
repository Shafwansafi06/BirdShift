"""Compare class-imbalance methods trained on the long-tailed CUB split.

Validation set is the unchanged (balanced) val split. Classes are grouped into
head / medium / tail thirds by their image count in the long-tailed train split.
Run from repository root with PYTHONPATH=src.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from birdshift.datasets.cub import CUBDataset
from evaluate_shift import evaluate, load_checkpoint  # same script dir

# name -> (checkpoint, post-hoc logit adjustment tau or None)
RUNS = {
    "Standard CE": ("checkpoints/longtail_ce/best.pth", None),
    "Weighted sampling": ("checkpoints/longtail_sampler/best.pth", None),
    "Class-balanced CE": ("checkpoints/longtail_cb/best.pth", None),
    "Logit-adjusted CE": ("checkpoints/longtail_la/best.pth", None),
    "CE + post-hoc logit adj.": ("checkpoints/longtail_ce/best.pth", 1.0),
}


class PosthocLogitAdjust(torch.nn.Module):
    """Menon et al. 2021: subtract tau*log(class prior) from a trained model's logits."""

    def __init__(self, model, log_prior, tau):
        super().__init__()
        self.model, self.tau = model, tau
        self.register_buffer("log_prior", log_prior)

    def forward(self, x):
        # Removing the prior undoes the bias towards frequent classes learned by plain CE.
        return self.model(x) - self.tau * self.log_prior


def class_groups(root, ratio):
    """Split classes into head / medium / tail thirds by their long-tailed training count."""
    counts = CUBDataset(root, split="train", imbalance_ratio=ratio).data["label"].value_counts()
    ranked = counts.sort_values(ascending=False, kind="stable").index.tolist()
    third = len(ranked) // 3
    return {"head": set(ranked[:third]), "medium": set(ranked[third:-third]),
            "tail": set(ranked[-third:])}, counts


def group_accuracy(rows, classes):
    """Top-1 accuracy over the images whose true class is in `classes`."""
    hits = [r["correct_top1"] for r in rows if r["true_label"] in classes]
    return sum(hits) / len(hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/CUB_200_2011")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="results/longtail")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = CUBDataset(args.data_root, split=args.split, mode="full")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    summaries, ratios = [], set()
    for name, (path, tau) in RUNS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        ratio = ckpt["imbalance_ratio"]
        ratios.add(ratio)
        groups, counts = class_groups(args.data_root, ratio)
        model = load_checkpoint(path, "full", device)
        if tau is not None:
            prior = torch.tensor([counts.get(c, 0) for c in range(200)], dtype=torch.float)
            model = PosthocLogitAdjust(model, torch.log(prior / prior.sum()), tau).to(device)
        rows, metrics = evaluate(model, loader, device)
        summaries.append({
            "method": name, "checkpoint": path, "posthoc_tau": tau, "epoch": ckpt["epoch"],
            "imbalance_ratio": ratio, "train_images": int(counts.sum()),
            **metrics,
            **{f"{g}_accuracy": group_accuracy(rows, c) for g, c in groups.items()},
        })
        stem = Path(path).parent.name + ("_posthoc_la" if tau is not None else "")
        with (output / f"{stem}_{args.split}_predictions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    assert len(ratios) == 1, f"Runs used different imbalance ratios: {ratios}"

    with (output / f"summary_{args.split}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / f"summary_{args.split}.json").write_text(json.dumps(summaries, indent=2))

    print(f"| Method | Top-1 | Macro-F1 | Head acc | Medium acc | Tail acc |  (ratio {ratios.pop():g}, {args.split})")
    print("|---|---|---|---|---|---|")
    for s in summaries:
        print(f"| {s['method']} | {s['top1_accuracy']:.2%} | {s['macro_f1']:.3f} | "
              f"{s['head_accuracy']:.2%} | {s['medium_accuracy']:.2%} | {s['tail_accuracy']:.2%} |")
    print(f"Saved files to {output.resolve()}")


if __name__ == "__main__":
    main()
