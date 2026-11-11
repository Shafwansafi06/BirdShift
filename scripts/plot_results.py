"""Plot training curves and result summaries into results/figures/.

Run from repository root: python scripts/plot_results.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = Path("results/metrics")
OUT = Path("results/figures")
COLORS = {"Full-image": "#2563eb", "Bird-crop": "#f59e0b", "BirdShift": "#16a34a",
          "Standard CE": "#2563eb", "Weighted sampler": "#f59e0b", "Class-balanced CE": "#16a34a",
          "Logit-adjusted CE": "#dc2626", "CE + post-hoc LA": "#9333ea",
          "AdamW (old recipe)": "#94a3b8", "AdamW + fg crop + cosine": "#2563eb",
          "SGD, timm a1 weights": "#f59e0b", "SGD, torchvision weights": "#16a34a",
          "CUB model (not retrained)": "#94a3b8", "Val (training clusters)": "#2563eb",
          "Test (held-out cluster)": "#dc2626"}


def read_log(name):
    """Per-epoch training log of one run as {column: [values]}."""
    with (METRICS / f"cub_{name}_training.csv").open() as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def plot_curves(runs, title, path):
    """Loss and accuracy curves (solid = val, dashed = train) for {label: run_name}."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, log_name in runs.items():
        log = read_log(log_name)
        c = COLORS[label]
        ax_loss.plot(log["epoch"], log["train_loss"], color=c, ls="--", alpha=.7)
        ax_loss.plot(log["epoch"], log["val_loss"], color=c, label=label)
        ax_acc.plot(log["epoch"], [a * 100 for a in log["train_accuracy"]], color=c, ls="--", alpha=.7)
        ax_acc.plot(log["epoch"], [a * 100 for a in log["val_accuracy"]], color=c, label=label)
        best = max(range(len(log["epoch"])), key=lambda i: log["val_accuracy"][i])
        ax_acc.scatter(log["epoch"][best], log["val_accuracy"][best] * 100, color=c, zorder=3, s=30)
    ax_loss.set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    ax_acc.set(title="Top-1 accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=.3)
    ax_acc.legend(title="solid = val, dashed = train", loc="lower right")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_bars(groups, series, ylabel, title, path, ylim=None, legend_loc="upper right"):
    """Grouped bar chart: one bar per series in each group, values labelled on top."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series.items()):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width, label=label, color=COLORS[label])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set(ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Rebuild every figure in results/figures from the saved metrics."""
    OUT.mkdir(parents=True, exist_ok=True)
    plot_curves({"Full-image": "full", "Bird-crop": "crop", "BirdShift": "birdshift"},
                "Experiment 1: training curves", OUT / "curves_background.png")
    plot_curves({"Full-image": "baseline_448", "Bird-crop": "crop_448", "BirdShift": "birdshift_448"},
                "Experiment 1 at 448 px (mixed precision): training curves", OUT / "curves_background_448.png")
    plot_curves({"AdamW (old recipe)": "baseline_448", "AdamW + fg crop + cosine": "recipe_adamw",
                 "SGD, timm a1 weights": "recipe_sgd", "SGD, torchvision weights": "recipe_sgd_tv"},
                "Training recipe search, full-image model at 448 px", OUT / "curves_recipe.png")
    plot_curves({"Standard CE": "longtail_ce", "Weighted sampler": "longtail_sampler",
                 "Class-balanced CE": "longtail_cb", "Logit-adjusted CE": "longtail_la"},
                "Experiment 2: training curves (long-tailed split)", OUT / "curves_longtail.png")

    names = {"full": "Full-image", "crop": "Bird-crop", "birdshift": "BirdShift"}
    conds = ["original", "blur", "replace"]
    for folder, suffix, res, ylim in (("background_shift", "", "224 px", (74, 80.5)),
                                       ("background_shift_448", "_448", "448 px", (76, 83))):
        shift = json.loads(Path(f"results/{folder}/summary_val.json").read_text())
        grouped_bars(["Original", "Blurred BG", "Replaced BG"],
                     {names[m]: [next(r["top1_accuracy"] * 100 for r in shift
                                      if r["model"] == m and r["shift"] == c) for c in conds]
                      for m in names},
                     "Top-1 accuracy (%)", f"Background shift at {res}, CUB val (900 images)",
                     OUT / f"results_background{suffix}.png", ylim=ylim)

    lt = json.loads(Path("results/longtail/summary_val.json").read_text())
    keys = ["head_accuracy", "medium_accuracy", "tail_accuracy", "top1_accuracy"]
    grouped_bars(["Head", "Medium", "Tail", "Overall"],
                 {{"Weighted sampling": "Weighted sampler",
                   "CE + post-hoc logit adj.": "CE + post-hoc LA"}.get(r["method"], r["method"]):
                  [r[k] * 100 for k in keys] for r in lt},
                 "Top-1 accuracy (%)", "Long-tail methods by class frequency, CUB val",
                 OUT / "results_longtail.png", ylim=(0, 95))
    plot_curves({"Full-image": "geo_sgd_tv"},
                "Experiment 3: geography split (iNaturalist, 448 px, SGD recipe)", OUT / "curves_geography.png")
    plot_curves({"Full-image": "bg_full_sgd_tv", "Bird-crop": "bg_crop_sgd_tv", "BirdShift": "bg_birdshift_sgd_tv"},
                "Experiment 4: background-cluster split (448 px, SGD recipe)", OUT / "curves_background_split.png")

    # Experiments 3 and 4: val (training clusters) vs test (held-out cluster)
    with (METRICS / "split_shift_accuracy.csv").open() as f:
        rows = list(csv.DictReader(f))
    acc = {(r["experiment"], r["model"], r["split"]): float(r["top1"]) * 100 for r in rows}
    groups = [("geography", "CUB model (not retrained)", "Geography:\nCUB model, not retrained"),
              ("geography", "Full-image", "Geography:\nfull-image"),
              ("background", "Full-image", "Background:\nfull-image"),
              ("background", "Bird-crop", "Background:\nbird-crop"),
              ("background", "BirdShift", "Background:\nBirdShift")]
    grouped_bars([g[2] for g in groups],
                 {"Val (training clusters)": [acc[(e, m, "val")] for e, m, _ in groups],
                  "Test (held-out cluster)": [acc[(e, m, "test")] for e, m, _ in groups]},
                 "Top-1 accuracy (%)", "Held-out cluster tests: training clusters vs held-out cluster",
                 OUT / "results_split_shift.png", ylim=(40, 92), legend_loc="upper left")
    print("Saved:", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
