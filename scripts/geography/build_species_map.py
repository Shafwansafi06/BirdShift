#!/usr/bin/env python3
"""Map the 200 CUB classes to iNaturalist taxon IDs -> data/splits/geography/species_map.csv
Overrides: data/splits/geography/species_overrides.csv (cub_label,inat_taxon_id) always win."""
import argparse, csv, difflib, json, re, sys, time
from pathlib import Path
import requests

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "BirdShift-research/0.1 (CUB to iNaturalist species mapping)"}
FUZZY_CUTOFF = 0.88

def variants(name):
    s = name.replace("_", " ").lower()
    keep = re.sub(r"[^a-z0-9]", "", s)
    strip = re.sub(r"[^a-z0-9]", "", re.sub(r"['\u2019]s\b", "", s))
    return {keep, strip}

class Api:
    def __init__(self, cache_path, delay=1.1):
        self.cache_path, self.delay = cache_path, delay
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def get(self, path, **params):
        key = path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        if key in self.cache:
            return self.cache[key]
        for attempt in range(5):
            r = requests.get(API + path, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status(); data = r.json(); break
        else:
            raise RuntimeError(f"API kept failing for {key}")
        self.cache[key] = data
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
        time.sleep(self.delay)
        return data
    def search_species(self, q):
        return self.get("/taxa", q=q, rank="species", taxon_id=3, is_active="true",
                        all_names="true", per_page=30).get("results", [])
    def taxon(self, tid):
        res = self.get(f"/taxa/{tid}").get("results", [])
        return res[0] if res else None

def english_names(t):
    names = []
    if t.get("preferred_common_name"): names.append(t["preferred_common_name"])
    if t.get("english_common_name"): names.append(t["english_common_name"])
    for n in t.get("names", []) or []:
        if n.get("locale") in ("en", "en-US", "en-GB") and n.get("name"): names.append(n["name"])
    return list(dict.fromkeys(names))

def match_one(api, cub_name):
    words = cub_name.replace("_", " ").split()
    queries = [" ".join(words)]
    if len(words) > 2: queries.append(" ".join(words[-2:]))
    if len(words) > 1: queries.append(words[-1])
    target, pool = variants(cub_name), {}
    for q in queries:
        for t in api.search_species(q): pool[t["id"]] = t
        exact = [t for t in pool.values() if any(variants(n) & target for n in english_names(t))]
        if exact:
            exact.sort(key=lambda t: -(t.get("observations_count") or 0))
            return exact[0], "exact", ("" if len(exact) == 1 else f"{len(exact)} exact candidates, took most observed")
    best, best_score = None, 0.0
    for t in pool.values():
        for n in english_names(t):
            for a in variants(n):
                for b in target:
                    sc = difflib.SequenceMatcher(None, a, b).ratio()
                    if sc > best_score: best, best_score = t, sc
    if best is not None and best_score >= FUZZY_CUTOFF:
        return best, "fuzzy", f"similarity {best_score:.2f}"
    return None, "none", "no candidate"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="data/raw/CUB_200_2011/classes.txt")
    ap.add_argument("--out", default="data/splits/geography/species_map.csv")
    ap.add_argument("--overrides", default="data/splits/geography/species_overrides.csv")
    ap.add_argument("--cache", default="data/raw/inat_cache/taxa_cache.json")
    args = ap.parse_args()
    classes = []
    for line in Path(args.classes).read_text().splitlines():
        if line.strip():
            idx, name = line.split(maxsplit=1)
            classes.append((int(idx) - 1, name.split(".", 1)[1]))
    assert len(classes) == 200
    overrides = {}
    if Path(args.overrides).exists():
        with open(args.overrides) as f:
            for row in csv.DictReader(f): overrides[int(row["cub_label"])] = int(row["inat_taxon_id"])
    api, rows = Api(Path(args.cache)), []
    for label, name in classes:
        if label in overrides:
            t, mtype, note = api.taxon(overrides[label]), "manual", "override"
        else:
            t, mtype, note = match_one(api, name)
        if t is None:
            row = dict(cub_label=label, cub_name=name, inat_taxon_id="", inat_name="",
                       match_type="none", inat_common_name="", review="yes")
        else:
            row = dict(cub_label=label, cub_name=name, inat_taxon_id=t["id"], inat_name=t["name"],
                       match_type=mtype, inat_common_name=t.get("preferred_common_name", ""),
                       review="" if (mtype in ("exact", "manual") and "took most" not in note) else "yes")
        rows.append(row)
        print(f"{label:3d} {name:35s} -> {row['inat_name'] or '-':30s} [{mtype}] {note}", file=sys.stderr)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cub_label", "cub_name", "inat_taxon_id", "inat_name", "match_type", "inat_common_name", "review"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    ids = [r["inat_taxon_id"] for r in rows if r["inat_taxon_id"] != ""]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    n = {m: sum(r["match_type"] == m for r in rows) for m in ("exact", "fuzzy", "manual", "none")}
    print(f"\nCOUNTS {n}", file=sys.stderr)
    if dupes: print(f"DUPLICATE TAXON IDS: {dupes}", file=sys.stderr)
    print("\nNEEDS REVIEW:", file=sys.stderr)
    for r in rows:
        if r["review"]: print(f"  {r['cub_label']:3d} {r['cub_name']} ({r['match_type']}: {r['inat_name']})", file=sys.stderr)

if __name__ == "__main__":
    main()
