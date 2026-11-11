"""K-means on background-only features and held-out-background splits for CUB-200-2011.

Reads the output of extract_features.py, clusters the L2-normalised features with
K-means, names the clusters, holds out one or more clusters as test backgrounds and
writes a manifest that the existing training/evaluation code can use.

Outputs:
  <splits-dir>/background_clusters.csv   one row per CUB image:
        image_id,image_path,label,cluster_id,cluster_name,split,bg_fraction,species_dropped
  <splits-dir>/species_coverage.csv      per-species image counts in train / val / test
  <splits-dir>/split_summary.json        all numbers reported in the README
  <fig-dir>/cluster_<id>.png             16-image gallery per cluster
  <fig-dir>/test_species_imbalance.png   test images per species in the held-out cluster(s)
  optional: <fig-dir>/k_selection.png (--sweep), <fig-dir>/embedding.png (--embedding)

Usage (from the repository root):
  python scripts/background/cluster_backgrounds.py
  python scripts/background/cluster_backgrounds.py --cub-root "C:/path/to/CUB_200_2011" --features-dir "C:/path/to/work"
  add --sweep to redo the k = 4..12 silhouette sweep, --embedding for the t-SNE plot
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

SEED = 42
N_INIT = 10
VAL_FRAC = 0.15
MIN_TRAIN = 10  # species need at least this many images in the training clusters
# Chosen k (silhouette sweep + visual inspection, see data/splits/background/README.md)
K = 5
# Names given after looking at the galleries (cluster id -> name)
CLUSTER_NAMES = {0: 'ground / grass', 1: 'dense foliage', 2: 'man-made / mixed perches', 3: 'plain sky / blurred backdrop', 4: 'water'}
# Cluster ids used ONLY for testing (chosen from the species coverage comparison)
HELD_OUT = (0,)

MANIFEST_COLS = ["image_id", "image_path", "label", "cluster_id", "cluster_name", "split",
                 "bg_fraction", "species_dropped"]


# ----------------------------------------------------------------------------- inputs
def load_inputs(features_dir: Path):
    feats = np.load(features_dir / "background_features.npy")
    ids = pd.read_csv(features_dir / "background_features_ids.csv")
    assert len(feats) == len(ids), "features and ids do not match"
    assert ids["image_id"].is_unique and ids["image_id"].is_monotonic_increasing, "ids must be sorted"
    X = normalize(feats, norm="l2").astype(np.float32)
    ids["species"] = ids["image_path"].str.split("/").str[0].str.split(".", n=1).str[1]
    return X, ids


def fit_kmeans(X, k=K, seed=SEED):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X).labels_.astype(int)


def centroids_of(X, labels, k=K):
    return normalize(np.stack([X[labels == c].mean(0) for c in range(k)])).astype(np.float32)


def assign_clusters(X, centroids_path, refit=False):
    """Cluster ids that stay stable across machines.

    K-means numbers its clusters arbitrarily, and small feature differences (GPU, library
    versions) permute the ids, which would silently swap CLUSTER_NAMES and HELD_OUT. The
    committed centroids define the clusters: by default each image goes to the nearest saved
    centroid. With --refit, K-means is re-run and its ids are matched to the saved centroids.
    """
    if centroids_path.exists() and not refit:
        return (X @ np.load(centroids_path).T).argmax(1).astype(int)
    labels = fit_kmeans(X)
    if centroids_path.exists():
        old = np.load(centroids_path)
        _, new_to_old = linear_sum_assignment(-(centroids_of(X, labels) @ old.T))
        labels = new_to_old[labels]
    np.save(centroids_path, centroids_of(X, labels))
    return labels


# ----------------------------------------------------------------------------- splits
def build_splits(df, held_out, val_frac=VAL_FRAC, seed=SEED):
    """test = held-out clusters; val = random val_frac of training-cluster images per species."""
    df = df.sort_values("image_id").reset_index(drop=True).copy()
    df["split"] = np.where(df["cluster_id"].isin(held_out), "test", "train")
    rng = np.random.default_rng(seed)
    for label in sorted(df["label"].unique()):
        idx = df.index[(df["label"] == label) & (df["split"] == "train")].to_numpy()
        n_val = int(round(val_frac * len(idx)))
        if n_val:
            df.loc[rng.choice(idx, size=n_val, replace=False), "split"] = "val"
    return df


def coverage_table(df, min_train=MIN_TRAIN):
    counts = pd.crosstab(df["label"], df["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    cov = df.groupby("label")["species"].first().to_frame()
    cov["n_train"], cov["n_val"], cov["n_test"] = counts["train"], counts["val"], counts["test"]
    cov["n_train_clusters"] = cov["n_train"] + cov["n_val"]
    cov["kept"] = cov["n_train_clusters"] >= min_train
    return cov.reset_index()


def validate(man, held_out, cub_root=None):
    assert man["image_id"].is_unique, "duplicate image ids"
    assert man["split"].isin(["train", "val", "test"]).all(), "bad split value"
    assert ((man["split"] == "test") == man["cluster_id"].isin(held_out)).all(), "test must equal held-out clusters"
    assert man["label"].between(0, 199).all(), "labels must be 0-based"
    if cub_root is not None and (cub_root / "images.txt").exists():
        ref = pd.read_csv(cub_root / "images.txt", sep=" ", header=None, names=["image_id", "image_path"])
        lab = pd.read_csv(cub_root / "image_class_labels.txt", sep=" ", header=None, names=["image_id", "c"])
        ref = ref.merge(lab, on="image_id").sort_values("image_id").reset_index(drop=True)
        m = man.sort_values("image_id").reset_index(drop=True)
        assert len(m) == len(ref) and (m["image_id"].values == ref["image_id"].values).all(), "not every CUB image once"
        assert (m["image_path"].values == ref["image_path"].values).all(), "image_path differs from images.txt"
        assert (m["label"].values == ref["c"].values - 1).all(), "label is not CUB label - 1"


def summarize(df, cov, names, held_out):
    kept = cov[cov["kept"]]
    t = kept["n_test"]
    kept_df = df[df["species_dropped"] == 0]

    def split_counts(d):
        return {s: {"images": int((d["split"] == s).sum()),
                    "species": int(d.loc[d["split"] == s, "label"].nunique())} for s in ["train", "val", "test"]}

    clusters = []
    for c, g in df.groupby("cluster_id"):
        clusters.append({"id": int(c), "name": names[c], "images": int(len(g)),
                         "species": int(g["label"].nunique()),
                         "mean_bg_fraction": round(float(g["bg_fraction"].mean()), 3),
                         "role": "test (held out)" if c in held_out else "train / val"})
    order = kept.sort_values("n_test", ascending=False)
    return {
        "k": K, "seed": SEED, "n_init": N_INIT, "val_fraction": VAL_FRAC, "min_train_images": MIN_TRAIN,
        "held_out_clusters": [int(c) for c in held_out],
        "clusters": clusters,
        "splits_all_species": split_counts(df),
        "splits_kept_species": split_counts(kept_df),
        "species_kept": int(len(kept)),
        "species_dropped": cov.loc[~cov["kept"], "species"].tolist(),
        "test_imbalance": {
            "species_with_0_test": int((t == 0).sum()),
            "species_with_1_to_4": int(t.between(1, 4).sum()),
            "species_with_5_to_9": int(t.between(5, 9).sum()),
            "species_with_10_plus": int((t >= 10).sum()),
            "min": int(t.min()), "median": float(t.median()), "max": int(t.max()),
            "most": [[s, int(n)] for s, n in zip(order["species"].head(5), order["n_test"].head(5))],
            "least": [[s, int(n)] for s, n in zip(order["species"].tail(5), order["n_test"].tail(5))],
        },
    }


# ----------------------------------------------------------------------------- figures
def save_galleries(df, cub_root, names, held_out, fig_dir, n=16, seed=SEED):
    for old in fig_dir.glob("cluster_*.png"):
        old.unlink()
    for c in sorted(df["cluster_id"].unique()):
        g = df[df["cluster_id"] == c]
        pick = g.sample(min(n, len(g)), random_state=seed)
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax, (_, r) in zip(axes.flat, pick.iterrows()):
            img = Image.open(cub_root / "images" / r["image_path"]).convert("RGB")
            img.thumbnail((300, 300))
            ax.imshow(img)
        for ax in axes.flat:
            ax.axis("off")
        tag = "  [held-out test]" if c in held_out else ""
        fig.suptitle(f"Cluster {c}: {names[c]}  ({len(g):,} images){tag}", fontsize=16)
        plt.tight_layout()
        fig.savefig(fig_dir / f"cluster_{c}.png", dpi=120)
        plt.close(fig)


def plot_test_imbalance(cov, names, held_out, fig_dir):
    kept = cov[cov["kept"]].sort_values("n_test", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(kept)), kept["n_test"].values, width=1.0)
    med = kept["n_test"].median()
    ax.axhline(med, color="red", ls="--", lw=1, label=f"median = {med:.0f}")
    ax.set_xlabel(f"species, sorted ({len(kept)} kept species)")
    ax.set_ylabel("test images")
    ax.set_title("Test images per species in held-out background: " + ", ".join(names[c] for c in held_out))
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "test_species_imbalance.png", dpi=150)
    plt.close(fig)


def run_sweep(X, fig_dir, splits_dir, k_range=range(4, 13), sample=6000, seed=SEED):
    rows = []
    for k in k_range:
        lab = fit_kmeans(X, k, seed)
        sil = silhouette_score(X, lab, sample_size=min(sample, len(X)), random_state=seed)
        sizes = np.bincount(lab)
        rows.append({"k": k, "silhouette": round(float(sil), 4), "smallest_cluster": int(sizes.min()),
                     "largest_cluster": int(sizes.max())})
        print(f"k={k}: silhouette={sil:.4f}")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(splits_dir / "k_selection.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(sweep["k"], sweep["silhouette"], "o-")
    ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.grid(alpha=.3)
    ax.set_title("Silhouette score vs k")
    plt.tight_layout(); fig.savefig(fig_dir / "k_selection.png", dpi=150); plt.close(fig)


def plot_embedding(X, labels, names, fig_dir, seed=SEED):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Z = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(Z)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    k = len(names)
    fig = plt.figure(figsize=(9, 8))
    for c in range(k):
        m = labels == c
        plt.scatter(emb[m, 0], emb[m, 1], s=3, alpha=.6, color=colors[(2 * c) % 20 if k <= 10 else c],
                    label=f"{c}: {names[c]} ({m.sum():,})")
    plt.legend(markerscale=5, fontsize=9); plt.xticks([]); plt.yticks([])
    plt.title(f"Background features (t-SNE), k = {k}")
    plt.tight_layout(); fig.savefig(fig_dir / "embedding.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cub-root", type=Path, default=Path("data/raw/CUB_200_2011"))
    p.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits/background"))
    p.add_argument("--fig-dir", type=Path, default=Path("results/figures/background_clusters"))
    p.add_argument("--held-out", type=int, nargs="+", default=list(HELD_OUT))
    p.add_argument("--no-galleries", action="store_true", help="skip gallery figures (faster)")
    p.add_argument("--sweep", action="store_true", help="redo the k = 4..12 silhouette sweep")
    p.add_argument("--embedding", action="store_true", help="also make the t-SNE plot")
    p.add_argument("--refit", action="store_true",
                   help="re-run K-means instead of using the saved centroids (ids are aligned to them)")
    args = p.parse_args()

    held_out = tuple(sorted(set(args.held_out)))
    assert set(CLUSTER_NAMES) == set(range(K)), "CLUSTER_NAMES must have one name per cluster"
    assert set(held_out) <= set(range(K)), "held-out ids must be valid cluster ids"
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    X, df = load_inputs(args.features_dir)
    if args.sweep:
        run_sweep(X, args.fig_dir, args.splits_dir)

    centroids_path = args.splits_dir / "cluster_centroids.npy"
    mode = "K-means refit" if args.refit or not centroids_path.exists() else f"nearest centroid ({centroids_path})"
    print(f"Clusters: {mode}, k={K}, on {X.shape[0]} images ...")
    df["cluster_id"] = assign_clusters(X, centroids_path, args.refit)
    df["cluster_name"] = df["cluster_id"].map(CLUSTER_NAMES)
    df = build_splits(df, held_out)
    cov = coverage_table(df)
    df["species_dropped"] = (~df["label"].map(cov.set_index("label")["kept"])).astype(int)

    man = df[MANIFEST_COLS]
    validate(man, held_out, args.cub_root)
    man.to_csv(args.splits_dir / "background_clusters.csv", index=False)
    cov.to_csv(args.splits_dir / "species_coverage.csv", index=False)
    summary = summarize(df, cov, CLUSTER_NAMES, held_out)
    with open(args.splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_galleries:
        save_galleries(df, args.cub_root, CLUSTER_NAMES, held_out, args.fig_dir)
    plot_test_imbalance(cov, CLUSTER_NAMES, held_out, args.fig_dir)
    if args.embedding:
        plot_embedding(X, df["cluster_id"].values, CLUSTER_NAMES, args.fig_dir)

    s = summary["splits_kept_species"]
    print("Held-out test clusters:", ", ".join(f"{c} ({CLUSTER_NAMES[c]})" for c in held_out))
    print(f"Species kept: {summary['species_kept']}, dropped: {len(summary['species_dropped'])}")
    for name in ["train", "val", "test"]:
        print(f"  {name:5s}: {s[name]['images']:6d} images, {s[name]['species']:3d} species")
    print("Wrote", args.splits_dir / "background_clusters.csv")


if __name__ == "__main__":
    main()
