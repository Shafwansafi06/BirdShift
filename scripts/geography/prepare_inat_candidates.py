#!/usr/bin/env python3
"""Stage 1: filter iNaturalist Open Data metadata to candidate photos of the CUB species.

Inputs (tab-separated): observations.csv, photos.csv, observers.csv, taxa.csv from the metadata
snapshot, plus data/splits/geography/species_map.csv.  Output: data/raw/inat_geo_candidates.csv.gz

Filters: research grade; taxon is a mapped species or an infraspecific taxon below it; coordinates
present with positional_accuracy <= --max-accuracy metres (drops obscured/fuzzed locations);
observed_on present; first photo only; licence CC0 / CC-BY / CC-BY-NC; jpg/jpeg/png.
At most --cap randomly chosen observations per species are kept before the photo join.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

LICENSES = {"CC0", "CC-BY", "CC-BY-NC"}
EXTS = {"jpg", "jpeg", "png"}
CHUNK = 5_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--species-map", default="data/splits/geography/species_map.csv")
    ap.add_argument("--out", default="data/raw/inat_geo_candidates.csv.gz")
    ap.add_argument("--max-accuracy", type=float, default=10000.0)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = Path(a.meta)

    sm = pd.read_csv(a.species_map).dropna(subset=["inat_taxon_id"])
    sp2label = {int(t): int(l) for t, l in zip(sm.inat_taxon_id, sm.cub_label)}
    assert len(sp2label) == len(sm), "two CUB classes share a taxon id"

    taxa = pd.read_csv(meta / "taxa.csv", sep="\t", usecols=["taxon_id", "ancestry", "rank_level"],
                       dtype={"taxon_id": "int64", "ancestry": "string"})
    sub = taxa[(taxa.rank_level < 10) & taxa.ancestry.notna()]
    parts = sub.ancestry.str.split("/")
    p1 = pd.to_numeric(parts.str[-1], errors="coerce")
    p2 = pd.to_numeric(parts.str[-2], errors="coerce")
    tax2label = dict(sp2label)
    for tid, x, y in zip(sub.taxon_id, p1, p2):
        if x in sp2label:
            tax2label[int(tid)] = sp2label[x]
        elif y in sp2label:
            tax2label[int(tid)] = sp2label[y]
    print(f"{len(sp2label)} species + {len(tax2label) - len(sp2label)} infraspecific taxa mapped", flush=True)

    keys = list(tax2label)
    cols = ["observation_uuid", "observer_id", "latitude", "longitude", "positional_accuracy",
            "taxon_id", "quality_grade", "observed_on"]
    kept, seen = [], 0
    reader = pd.read_csv(meta / "observations.csv", sep="\t", usecols=cols, chunksize=CHUNK,
                         dtype={"observation_uuid": str, "quality_grade": str, "observed_on": str})
    for i, ch in enumerate(reader):
        seen += len(ch)
        ch = ch[ch.taxon_id.isin(keys) & (ch.quality_grade == "research")]
        ch = ch[ch.latitude.notna() & ch.longitude.notna() & ch.observed_on.notna()
                & (ch.positional_accuracy <= a.max_accuracy)]
        kept.append(ch)
        if i % 5 == 0:
            print(f"observations: {seen / 1e6:.0f}M rows scanned, {sum(map(len, kept))} kept", flush=True)
    obs = pd.concat(kept, ignore_index=True)
    obs["label"] = obs.taxon_id.astype("int64").map(tax2label)
    print(f"observations kept before cap: {len(obs)} over {obs.label.nunique()} species", flush=True)

    obs["_r"] = np.random.RandomState(a.seed).random_sample(len(obs))
    obs = obs.sort_values("_r").groupby("label").head(a.cap).drop(columns="_r")
    print(f"after cap {a.cap}/species: {len(obs)} observations", flush=True)

    want = set(obs.observation_uuid)
    pcols = ["photo_id", "observation_uuid", "extension", "license", "position"]
    got, seen = [], 0
    for i, ch in enumerate(pd.read_csv(meta / "photos.csv", sep="\t", usecols=pcols, chunksize=CHUNK, dtype=str)):
        seen += len(ch)
        ch = ch[ch.observation_uuid.isin(want)]
        ch = ch[(ch.position == "0") & ch.license.isin(LICENSES) & ch.extension.str.lower().isin(EXTS)]
        got.append(ch)
        if i % 5 == 0:
            print(f"photos: {seen / 1e6:.0f}M rows scanned, {sum(map(len, got))} kept", flush=True)
    ph = pd.concat(got).drop_duplicates("observation_uuid")

    obsr = pd.read_csv(meta / "observers.csv", sep="\t", usecols=["observer_id", "login"], dtype={"login": str})
    df = (obs.merge(ph[["observation_uuid", "photo_id", "extension", "license"]], on="observation_uuid")
             .merge(obsr, on="observer_id", how="left")
             .rename(columns={"login": "photographer"}))
    df["extension"] = df.extension.str.lower()
    df = df[["observation_uuid", "observer_id", "photographer", "latitude", "longitude", "positional_accuracy",
             "observed_on", "label", "photo_id", "extension", "license"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    per = df.label.value_counts()
    print(f"WROTE {a.out}: {len(df)} images, {len(per)} species, per-species min/median/max = "
          f"{per.min()}/{int(per.median())}/{per.max()}")


if __name__ == "__main__":
    main()
