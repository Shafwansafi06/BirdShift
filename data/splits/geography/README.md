# Geographic held-out split (iNaturalist)

**Source.** iNaturalist Open Data metadata snapshot `inaturalist-open-data-20261027` (observations, photos, observers, taxa) and
images from the public `inaturalist-open-data` bucket (`medium` size). Split built on 2026-11-09.
Images are not committed; `scripts/geography/download_inat.py` rebuilds `data/raw/inat_geo/` from the manifest.

**Species.** CUB class names were mapped to iNaturalist taxa in `species_map.csv` (11 manual overrides in
`species_overrides.csv`). Observations of subspecies are counted under their parent species.

**Filters.** research grade; first photo of each observation; photo licence CC0, CC-BY or CC-BY-NC (kept in the
`license` column, with the photographer's login in `photographer`); coordinates present with positional accuracy
<= 10 km (this removes obscured/fuzzed locations); at most 400 random observations per species before the photo
join. `observation_id` is the Open Data observation UUID (Open Data has no numeric IDs); `photo_id` is the key the
download script uses.

**Clustering.** k-means with k=5 on 3-D unit vectors of (lat, lon), so distances are chord distances on the
sphere (no dateline wrap, no polar distortion). Clusters are numbered west to east. Held-out clusters were chosen
by species coverage only (the number of species with at least 5 test images and enough train images;
ties broken by a test share near 25%), never by model accuracy. All candidates are in `heldout_search.csv`.
A 100 km buffer around the held-out points is removed from train/val so the regions are cleanly
disjoint (drawn in grey on the map).

| cluster | role | candidate images | centroid (lat, lon) | lat range (2-98%) | lon range (2-98%) |
|---|---|---|---|---|---|
| 0 | **held out (test only)** | 22086 | 42.0, -118.5 | 27 to 64 | -170 to -100 |
| 1 | train/val | 42695 | 34.5, -85.2 | 8 to 47 | -102 to -65 |
| 2 | train/val | 1749 | -33.9, -3.2 | -45 to 1 | -72 to 38 |
| 3 | train/val | 3950 | 57.1, 26.0 | 28 to 79 | -23 to 85 |
| 4 | train/val | 1960 | 26.1, 133.1 | -37 to 59 | -177 to 175 |

**Splits.** test = held-out clusters only. train/val = the other clusters; val is ~15% of each species,
chosen by whole groups of (observer, day, ~10 km cell) so one outing never lands in both. Only species with at
least 10 train and 5 test images are kept: **154 of 200** survive. Label indices are
the CUB labels, so the 200-way head is unchanged.

| split | images | species |
|---|---|---|
| test | 19395 | 154 |
| train | 31166 | 154 |
| val | 5510 | 154 |

Candidate images before filtering to surviving species and removing the buffer: 72440.
Do not use the test split for checkpoint selection or tuning.

## Reproducing and deviations from the issue
Run order (from the repo root): `build_species_map.py`, `prepare_inat_candidates.py --meta <metadata dir>`,
`cluster_geography.py --held-out 0`, `download_inat.py`, `drop_unavailable.py`, then
`cluster_geography.py --held-out 0 --candidates data/raw/inat_geo_candidates_available.csv.gz` and `download_inat.py` again.

- Cap is 400 observations per species before splitting (the issue suggests 30 to 100). A geographic split discards many images,
  so a lower cap would leave far fewer species with at least 10 train and 5 test images.
- 260 photos were deleted from iNaturalist after the metadata snapshot; they are listed in `unavailable_photos.txt` and excluded.
- `species_map.csv` rows with `review=yes` need checking. White_necked_Raven (107) and Great_Grey_Shrike (111) matched
  Old World taxa and Winter_Wren (198) matched a taxon with no observations (iNaturalist now uses `Nannus hiemalis` / `Nannus pacificus`),
  so these three are not in the benchmark.
- 5 photos appear under two observations by the same observer on the same day; both copies are always in the same split.
