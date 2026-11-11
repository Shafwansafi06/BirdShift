"""ManifestDataset reads one split, skips dropped species, returns CUB-style samples."""
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from birdshift.datasets.manifest import ManifestDataset

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "imgs").mkdir()
    for i in range(4):
        Image.new("RGB", (60, 40), (i * 50, 0, 0)).save(d / "imgs" / f"{i}.jpg")
    pd.DataFrame({
        "image_path": [f"imgs/{i}.jpg" for i in range(4)],
        "label": [3, 7, 7, 199],
        "split": ["train", "train", "test", "train"],
        "species_dropped": [0, 0, 0, 1],
    }).to_csv(d / "m.csv", index=False)

    tr = ManifestDataset(d / "m.csv", d, "train", image_size=32)
    te = ManifestDataset(d / "m.csv", d, "test", image_size=32)
    assert len(tr) == 2 and list(tr.data.label) == [3, 7], "dropped species must be skipped"
    assert len(te) == 1 and te[0]["label"] == 7
    assert tuple(tr[0]["image"].shape) == (3, 32, 32)
    try:
        ManifestDataset(d / "m.csv", d, "val")
        raise AssertionError("missing split must raise")
    except ValueError:
        pass
print("ok")

# CUB-based manifest (background split): CUBDataset follows its splits in every mode.
bg = Path("data/splits/background/background_clusters.csv")
if Path("data/raw/CUB_200_2011").exists() and bg.exists():
    from birdshift.datasets.cub import CUBDataset
    m = pd.read_csv(bg)
    for mode in ("full", "crop", "birdshift"):
        for split in ("train", "val", "test"):
            ds = CUBDataset("data/raw/CUB_200_2011", split=split, mode=mode, image_size=32, manifest=bg)
            assert set(ds.data["id"]) == set(m[m["split"] == split]["image_id"]), (mode, split)
    assert tuple(ds[0]["image"].shape) == (3, 32, 32)
    print("background manifest ok")
