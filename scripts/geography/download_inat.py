#!/usr/bin/env python3
"""Rebuild data/raw/inat_geo/ from the manifest by fetching each photo from the public
iNaturalist Open Data bucket. Skips files that already exist; exits non-zero if any image is missing."""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/{size}.{ext}"


def fetch(sess, url, dest, retries=5):
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(r.content)
                tmp.rename(dest)
                return "ok"
            if r.status_code in (403, 404):
                return f"http{r.status_code}"
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/splits/geography/geography_clusters.csv")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, dtype={"photo_id": str})
    root = Path(a.root)
    sess = requests.Session()
    sess.headers["User-Agent"] = "BirdShift-research/0.1"
    jobs = [(URL.format(pid=p, size=a.size, ext=Path(ip).suffix[1:]), root / ip) for p, ip in zip(m.photo_id, m.image_path)]
    stats = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda j: fetch(sess, *j), jobs), 1):
            stats[res] = stats.get(res, 0) + 1
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)} {stats}", flush=True)
    missing = sum(not (root / ip).exists() for ip in m.image_path)
    print(f"done: {stats}; {len(m) - missing}/{len(m)} images present")
    if missing:
        sys.exit(f"{missing} images missing; re-run to retry")


if __name__ == "__main__":
    main()
