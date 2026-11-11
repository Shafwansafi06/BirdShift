from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from birdshift.datasets.cub import full_image_transform


class ManifestDataset(Dataset):
    """One split of a manifest CSV (data/splits/geography or data/splits/background).

    The manifest needs image_path, label (0-199, CUB indices) and split columns. Rows with
    species_dropped == 1 are skipped. image_path is relative to `root`: data/raw for the
    geography split, data/raw/CUB_200_2011/images for the background split.
    Full-image pipeline only; manifests from iNaturalist have no bounding boxes.
    """

    def __init__(self, manifest, root, split, image_size=224, train_aug="rrc"):
        df = pd.read_csv(manifest)
        if split not in set(df["split"]):
            raise ValueError(f"split {split!r} not in {manifest}")
        if "species_dropped" in df:
            df = df[df["species_dropped"] == 0]
        self.data = df[df["split"] == split][["image_path", "label"]].reset_index(drop=True)
        self.root = Path(root)
        self.split = split
        self.transform = full_image_transform(split == "train", image_size, train_aug)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(self.root / row["image_path"]) as file:
            image = file.convert("RGB")
        return {"image": self.transform(image), "label": int(row["label"])}
