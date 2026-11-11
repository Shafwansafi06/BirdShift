# Geography split: run order and notes
Run from the repo root.
1. `python scripts/geography/build_species_map.py`
2. `python scripts/geography/prepare_inat_candidates.py --meta <metadata dir>` (reads species_map.csv and taxon_restrictions.csv, writes data/raw/inat_geo_candidates.csv.gz)
3. `python scripts/geography/drop_unavailable.py`
4. `python scripts/geography/cluster_geography.py --snapshot <name> --held-out 0 --candidates data/raw/inat_geo_candidates_available.csv.gz`
5. `python scripts/geography/download_inat.py`
6. Repeat steps 3 to 5 once so photos deleted since the snapshot (HTTP 404) are excluded; the last download must report every image present.

## Notes
- Cap of 400 observations per species before splitting (the issue suggests 30 to 100), so that enough species keep at least 10 train and 5 test images.
- The cap is a seeded random draw, but it depends on the filtered pool, so changing a taxon mapping redraws most images.
- taxon_restrictions.csv: Myrtle_Warbler (170) keeps only Setophaga coronata coronata observations (Audubon's and species-level IDs are excluded). Sayornis (102) is excluded because the CUB class is a genus with mixed phoebe species.
- species_map.csv rows with review=yes need checking: 107, 111 and 198 matched Old World or empty taxa and drop out.
- unavailable_photos.txt lists photos that returned HTTP 404 at download time (deleted since the metadata snapshot); they are excluded.
- Subspecies audit: among classes with at least 15% of observations at subspecies rank, only Myrtle_Warbler (100%), Gray_crowned_Rosy_Finch (38%) and Pied_Kingfisher (33%) appear. The audit sees only forms that observers recorded; a regional-form class with no iNaturalist subspecies would not show up.
- Held-out region: cluster 0 (western North America and Alaska) is set by hand with `--held-out 0`. The automatic rule now picks 0,3, which would add cluster 3 (Europe and northern Asia) to the test set. We keep cluster 0 alone so the test set is one region and the comparison stays east vs west North America.
