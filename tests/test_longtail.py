"""Run: PYTHONPATH=src python tests/test_longtail.py (needs data/raw/CUB_200_2011)."""
from birdshift.datasets.cub import CUBDataset

ROOT = "data/raw/CUB_200_2011"


def test_longtail_split():
    base = CUBDataset(ROOT, split="train")
    lt = CUBDataset(ROOT, split="train", imbalance_ratio=10)
    counts = lt.data["label"].value_counts()
    assert counts.index.nunique() == 200, "every class must keep at least one image"
    assert counts.max() / counts.min() >= 8, counts.describe()
    assert set(lt.data["id"]) <= set(base.data["id"]), "long-tail must be a subset of train"
    # Deterministic, and val is never touched.
    assert lt.data["id"].tolist() == CUBDataset(ROOT, split="train", imbalance_ratio=10).data["id"].tolist()
    assert (CUBDataset(ROOT, split="val", imbalance_ratio=10).data["id"].tolist()
            == CUBDataset(ROOT, split="val").data["id"].tolist())
    print(f"OK: {len(base.data)} -> {len(lt.data)} train images, "
          f"per-class {counts.max()}..{counts.min()}")


if __name__ == "__main__":
    test_longtail_split()
