# Geographic held-out split (iNaturalist)

**Source.** iNaturalist Open Data metadata snapshot `inaturalist-open-data-20261027` (observations, photos, observers, taxa) and
images from the public `inaturalist-open-data` bucket (`medium` size). Split built on 2026-11-10.
Images are not committed; `scripts/geography/download_inat.py` rebuilds `data/raw/inat_geo/` from the manifest.

**Species.** CUB class names were mapped to iNaturalist taxa in `species_map.csv` (11 manual overrides in
`species_overrides.csv`). Observations of subspecies are counted under their parent species.

**Filters.** research grade; first photo of each observation; photo licence CC0, CC-BY or CC-BY-NC (kept in the
`license` column, with the photographer's login in `photographer`); coordinates present with positional accuracy
<= 10 km (this removes obscured/fuzzed locations); at most 400 random observations per species before the photo
join. `observation_id` is the Open Data observation UUID (Open Data has no numeric IDs); `photo_id` is the key the
download script uses.

**Clustering.** k-means with k=5 on 3-D unit vectors of (lat, lon), so distances are chord distances on the
sphere (no dateline wrap, no polar distortion). Clusters are numbered west to east. Held-out cluster(s) 0 were set by hand with `--held-out 0`. The automatic rule (most species with at least 5 test images and enough train images, test share 15 to 40%, ties broken by a share near 25%) would pick 0,3.
Held-out clusters were never chosen by model accuracy. All candidates are in `heldout_search.csv`.
A 100 km buffer around the held-out points is removed from train/val so the regions are cleanly
disjoint (drawn in grey on the map).

| cluster | role | candidate images | centroid (lat, lon) | lat range (2-98%) | lon range (2-98%) |
|---|---|---|---|---|---|
| 0 | **held out (test only)** | 21606 | 42.0, -118.5 | 27 to 64 | -170 to -101 |
| 1 | train/val | 42512 | 34.6, -85.2 | 9 to 47 | -102 to -65 |
| 2 | train/val | 1843 | -33.6, -6.3 | -43 to 3 | -72 to 37 |
| 3 | train/val | 3907 | 57.2, 25.2 | 28 to 79 | -29 to 85 |
| 4 | train/val | 2048 | 27.4, 132.7 | -38 to 59 | -177 to 175 |

**Splits.** test = held-out clusters only. train/val = the other clusters; val is ~15% of each species,
chosen by whole groups of (observer, day, ~10 km cell) so one outing never lands in both. Only species with at
least 10 train and 5 test images are kept: **156 of 200** survive. Label indices are
the CUB labels, so the 200-way head is unchanged.

| split | images | species |
|---|---|---|
| test | 18953 | 156 |
| train | 32083 | 156 |
| val | 5664 | 156 |

Candidate images before filtering to surviving species and removing the buffer: 71916.
Do not use the test split for checkpoint selection or tuning.
