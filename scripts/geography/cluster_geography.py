#!/usr/bin/env python3
"""Stage 2: cluster candidates by geography, choose held-out regions, build disjoint splits.
Writes geography_clusters.csv, heldout_search.csv, README.md (in --out-dir) and the map figure."""
import argparse
import datetime
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def nearest_km(q, ref):
    """Distance in km from each row of q to its nearest row of ref (both (n,2) arrays in radians)."""
    d, _ = BallTree(ref, metric="haversine").query(q, k=1)
    return d[:, 0] * EARTH_KM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/raw/inat_geo_candidates.csv.gz")
    ap.add_argument("--species-map", default="data/splits/geography/species_map.csv")
    ap.add_argument("--out-dir", default="data/splits/geography")
    ap.add_argument("--fig", default="results/figures/geography/clusters_map.png")
    ap.add_argument("--snapshot", default="unknown")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-held", type=int, default=2)
    ap.add_argument("--held-out", default="", help="override automatic choice, e.g. '0,1'")
    ap.add_argument("--buffer-km", type=float, default=100.0)
    ap.add_argument("--min-train", type=int, default=10)
    ap.add_argument("--min-test", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    min_pool = int(np.ceil(a.min_train / (1 - a.val_frac)))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.candidates, dtype={"photo_id": str, "observation_uuid": str})
    sm = pd.read_csv(a.species_map)
    df["species"] = df.label.map(dict(zip(sm.cub_label, sm.cub_name)))
    n_candidates = len(df)

    # ---- 1. cluster on the sphere (3-D unit vectors: no dateline wrap, no polar distortion)
    km = KMeans(a.k, n_init=20, random_state=a.seed).fit(xyz(df.latitude.values, df.longitude.values))
    cen = km.cluster_centers_
    order = np.argsort(np.degrees(np.arctan2(cen[:, 1], cen[:, 0])))  # number clusters west -> east
    df["cluster_id"] = pd.Series(km.labels_).map({old: new for new, old in enumerate(order)}).values
    cen = cen[order]
    cen_lat = np.degrees(np.arcsin(cen[:, 2] / np.linalg.norm(cen, axis=1)))
    cen_lon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))
    rad = np.radians(df[["latitude", "longitude"]].values)
    allc = df[["cluster_id", "latitude", "longitude"]].copy()

    def pools(held):
        is_test = df.cluster_id.isin(held).values
        idx = np.where(~is_test)[0]
        far = nearest_km(rad[idx], rad[is_test]) >= a.buffer_km
        is_train = np.zeros(len(df), bool)
        is_train[idx[far]] = True
        return is_test, is_train

    def survivors(is_test, is_train):
        te = df[is_test].label.value_counts()
        tr = df[is_train].label.value_counts()
        return [l for l in te.index if te[l] >= a.min_test and tr.get(l, 0) >= min_pool]

    # ---- 2. choose held-out clusters by species coverage only (never by model accuracy)
    rows = []
    for r in range(1, a.max_held + 1):
        for held in itertools.combinations(range(a.k), r):
            t, tr = pools(held)
            ok = survivors(t, tr)
            m = df.label.isin(ok).values
            nt, ntr = int((t & m).sum()), int((tr & m).sum())
            rows.append(dict(held_out=",".join(map(str, held)), species=len(ok), test_images=nt,
                             train_pool_images=ntr, test_share=round(nt / max(nt + ntr, 1), 3)))
    search = pd.DataFrame(rows).sort_values(["species", "test_images"], ascending=False)
    search.to_csv(out_dir / "heldout_search.csv", index=False)
    print("\nHELD-OUT SEARCH (clusters numbered west to east)")
    print(search.to_string(index=False))
    if a.held_out:
        held = tuple(int(x) for x in a.held_out.split(","))
    else:
        el = search[search.test_share.between(0.15, 0.40)]
        el = el if len(el) else search
        best = el.assign(gap=(el.test_share - 0.25).abs()).sort_values(["species", "gap"], ascending=[False, True]).iloc[0]
        held = tuple(int(x) for x in best.held_out.split(","))
    print("\nHELD-OUT CLUSTERS:", held)

    # ---- 3. disjoint splits
    is_test, is_train = pools(held)
    ok = set(survivors(is_test, is_train))
    df["split"] = np.where(is_test, "test", np.where(is_train, "train", "buffer"))
    buffer_df = df[df.split == "buffer"]
    df = df[(df.split != "buffer") & df.label.isin(ok)].copy()
    # same observer + day + ~10 km cell = one outing; an outing never straddles train and val
    df["group"] = (df.observer_id.astype(str) + "|" + df.observed_on.astype(str) + "|"
                   + df.latitude.round(1).astype(str) + "|" + df.longitude.round(1).astype(str))
    rng = np.random.RandomState(a.seed)
    val_idx = []
    for lab, g in df[df.split == "train"].groupby("label"):
        sizes = g.group.value_counts()
        gids = sizes.index.to_numpy().copy()
        rng.shuffle(gids)
        target, n, chosen = int(round(a.val_frac * len(g))), 0, set()
        for gid in gids:
            if n >= target:
                break
            chosen.add(gid)
            n += sizes[gid]
        val_idx += list(g.index[g.group.isin(chosen)])
    df.loc[val_idx, "split"] = "val"
    ntr = df[df.split == "train"].label.value_counts()
    nte = df[df.split == "test"].label.value_counts()
    final = [l for l in ok if ntr.get(l, 0) >= a.min_train and nte.get(l, 0) >= a.min_test]
    df = df[df.label.isin(final)].copy()

    df["image_path"] = ["inat_geo/%04d/%s.%s" % (l + 1, p, e) for l, p, e in zip(df.label, df.photo_id, df.extension)]
    df["observation_id"] = df.observation_uuid
    df["license"] = df.license.str.lower()
    out = df[["image_path", "label", "species", "latitude", "longitude", "cluster_id", "split", "observation_id",
              "license", "photo_id", "extension", "photographer", "observer_id", "observed_on",
              "positional_accuracy"]].sort_values(["split", "label", "photo_id"])
    out.to_csv(out_dir / "geography_clusters.csv", index=False)

    # ---- 4. map
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("tab10")
    ax.scatter(buffer_df.longitude, buffer_df.latitude, s=2, c="lightgray", label=f"buffer ({a.buffer_km:.0f} km, dropped)")
    for c in range(a.k):
        s = out[out.cluster_id == c]
        h = c in held
        ax.scatter(s.longitude, s.latitude, s=9 if h else 4, color=cmap(c), marker="^" if h else "o", alpha=0.6,
                   label=f"cluster {c}" + (" - HELD OUT (test only)" if h else ""))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Geographic clusters (k-means on the sphere); triangles = held-out test regions")
    ax.grid(alpha=0.3)
    ax.legend(markerscale=3, loc="lower left")
    fig.savefig(a.fig, dpi=200, bbox_inches="tight")

    # ---- 5. validation checks
    tr = out[out.split.isin(["train", "val"])]
    te = out[out.split == "test"]
    assert set(out.split) <= {"train", "val", "test"}
    assert out.observation_id.is_unique, "an observation appears more than once"
    assert set(tr.cluster_id).isdisjoint(set(te.cluster_id)), "cluster in both train/val and test"
    per_test_species = out[out.split == "train"].label.value_counts().reindex(sorted(set(te.label)))
    assert (per_test_species >= a.min_train).all(), "a test species has < min-train train images"
    gap = nearest_km(np.radians(te[["latitude", "longitude"]].values), np.radians(tr[["latitude", "longitude"]].values)).min()
    print("\nVALIDATION")
    print("OK split values are only train/val/test")
    print("OK zero observation_id overlap between splits (every observation appears once)")
    print(f"OK every test species has >= {a.min_train} train images (min = {int(per_test_species.min())})")
    print(f"OK train/val and test use disjoint clusters; closest train/val-to-test pair is {gap:.0f} km apart")
    print(f"species surviving: {len(final)} of 200")
    print(out.groupby("split").agg(images=("label", "size"), species=("label", "nunique")).to_string())

    # ---- 6. README
    tbl = "\n".join(
        f"| {c} | {'**held out (test only)**' if c in held else 'train/val'} | {int((allc.cluster_id == c).sum())} | "
        f"{cen_lat[c]:.1f}, {cen_lon[c]:.1f} | {g.latitude.quantile(.02):.0f} to {g.latitude.quantile(.98):.0f} | "
        f"{g.longitude.quantile(.02):.0f} to {g.longitude.quantile(.98):.0f} |"
        for c, g in allc.groupby("cluster_id"))
    cnt = "\n".join(f"| {s} | {len(g)} | {g.label.nunique()} |" for s, g in out.groupby("split"))
    readme = f"""# Geographic held-out split (iNaturalist)

**Source.** iNaturalist Open Data metadata snapshot `{a.snapshot}` (observations, photos, observers, taxa) and
images from the public `inaturalist-open-data` bucket (`medium` size). Split built on {datetime.date.today().isoformat()}.
Images are not committed; `scripts/geography/download_inat.py` rebuilds `data/raw/inat_geo/` from the manifest.

**Species.** CUB class names were mapped to iNaturalist taxa in `species_map.csv` (11 manual overrides in
`species_overrides.csv`). Observations of subspecies are counted under their parent species.

**Filters.** research grade; first photo of each observation; photo licence CC0, CC-BY or CC-BY-NC (kept in the
`license` column, with the photographer's login in `photographer`); coordinates present with positional accuracy
<= 10 km (this removes obscured/fuzzed locations); at most 400 random observations per species before the photo
join. `observation_id` is the Open Data observation UUID (Open Data has no numeric IDs); `photo_id` is the key the
download script uses.

**Clustering.** k-means with k={a.k} on 3-D unit vectors of (lat, lon), so distances are chord distances on the
sphere (no dateline wrap, no polar distortion). Clusters are numbered west to east. Held-out clusters were chosen
by species coverage only (the number of species with at least {a.min_test} test images and enough train images;
ties broken by a test share near 25%), never by model accuracy. All candidates are in `heldout_search.csv`.
A {a.buffer_km:.0f} km buffer around the held-out points is removed from train/val so the regions are cleanly
disjoint (drawn in grey on the map).

| cluster | role | candidate images | centroid (lat, lon) | lat range (2-98%) | lon range (2-98%) |
|---|---|---|---|---|---|
{tbl}

**Splits.** test = held-out clusters only. train/val = the other clusters; val is ~{a.val_frac:.0%} of each species,
chosen by whole groups of (observer, day, ~10 km cell) so one outing never lands in both. Only species with at
least {a.min_train} train and {a.min_test} test images are kept: **{len(final)} of 200** survive. Label indices are
the CUB labels, so the 200-way head is unchanged.

| split | images | species |
|---|---|---|
{cnt}

Candidate images before filtering to surviving species and removing the buffer: {n_candidates}.
Do not use the test split for checkpoint selection or tuning.
"""
    (out_dir / "README.md").write_text(readme)
    print(f"\nWROTE {out_dir}/geography_clusters.csv, README.md, heldout_search.csv and {a.fig}")


if __name__ == "__main__":
    main()
