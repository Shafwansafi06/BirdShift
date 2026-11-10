#!/usr/bin/env python3
"""Drop candidate photos that are no longer in the iNaturalist bucket (HTTP 404 at download time)."""
import pandas as pd
from pathlib import Path
man = pd.read_csv("data/splits/geography/geography_clusters.csv", dtype={"photo_id": str})
f = Path("data/splits/geography/unavailable_photos.txt")
bad = set(f.read_text().split()) if f.exists() else set()
bad |= {p for p, ip in zip(man.photo_id, man.image_path) if not (Path("data/raw") / ip).exists()}
f.write_text("\n".join(sorted(bad)) + "\n")
c = pd.read_csv("data/raw/inat_geo_candidates.csv.gz", dtype={"photo_id": str})
c = c[~c.photo_id.isin(bad)]
c.to_csv("data/raw/inat_geo_candidates_available.csv.gz", index=False)
print(len(bad), "unavailable photos;", len(c), "candidates kept")
