"""CUB-200-2011 dataset with the three input variants (full image, bird crop, BirdShift).

CUB-200-2011 (Wah et al. 2011): 11,788 photos of 200 bird species, each with one
bounding box. The official split has 5,994 train and 5,794 test images; we hold out a
stratified 15% of train (900 images) as validation for all model selection.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from birdshift.transforms.background import SquarePad, background_variant, bbox_to_pixels

# ImageNet channel means and standard deviations, matching the pretrained backbone.
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def full_image_transform(train, image_size=224, train_aug="rrc"):
    """Full-image pipeline shared by CUBDataset ('full' / 'birdshift') and ManifestDataset.

    train: rrc = ImageNet-style RandomResizedCrop (can keep only 8% of the image);
           fg  = fine-grained style Resize(size*8/7) + RandomCrop(size), bird stays in frame.
    eval:  Resize(size*8/7) + CenterCrop(size), e.g. 256 -> 224 or 512 -> 448.
    """
    if not train:
        return transforms.Compose([
            transforms.Resize(image_size * 8 // 7), transforms.CenterCrop(image_size),
            transforms.ToTensor(), NORMALIZE,
        ])
    if train_aug not in {"rrc", "fg"}:
        raise ValueError("train_aug must be rrc or fg")
    crop = (transforms.RandomResizedCrop(image_size) if train_aug == "rrc" else
            transforms.Compose([transforms.Resize(image_size * 8 // 7), transforms.RandomCrop(image_size)]))
    return transforms.Compose([crop, transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORMALIZE])


def make_long_tailed(df, ratio, seed=42):
    """Subsample a split to an exponential long-tail profile (Experiment 2).

    Class at rank r keeps n_max * ratio**(-r/(C-1)) images (at least 1), so the
    most frequent class has `ratio` times more images than the rarest. Class
    order is a seeded random permutation because CUB labels are taxonomically
    ordered and we do not want "tail" to mean "one bird family".
    """
    labels = np.random.default_rng(seed).permutation(sorted(df["label"].unique()))
    n_max = df["label"].value_counts().max()
    parts = []
    for rank, label in enumerate(labels):
        n = max(1, int(round(n_max * ratio ** (-rank / (len(labels) - 1)))))
        cls = df[df["label"] == label]
        parts.append(cls.sample(n=min(n, len(cls)), random_state=seed))
    return pd.concat(parts)


class CUBDataset(Dataset):
    """One split of CUB-200-2011, returning {"image", "label", "image_id"} samples.

    root: folder with images/, images.txt, image_class_labels.txt,
          train_test_split.txt and bounding_boxes.txt.
    split: train, val, test, or trainval (all 5,994 official training images).
    mode:
      full      - whole photo.
      crop      - bounding-box crop, square-padded; needs the box at test time too.
      birdshift - whole photo; during training, with probability `suppression_prob`
                  the background outside the box is blurred (or replaced) so the
                  habitat stops being a reliable cue. Evaluation is the plain photo.
    imbalance_ratio: > 1 subsamples the train split to a long tail (Experiment 2).
    train_aug: rrc or fg, see full_image_transform.
    manifest: CSV with image_id (CUB id), label and split columns, e.g. the background-
              cluster split of Experiment 4. Replaces the official train/val/test split.
    """

    def __init__(self, root, split="train", mode="full", image_size=224, suppression_prob=0.5,
                 suppression_kind="blur", blur_radius=22, imbalance_ratio=1.0, train_aug="rrc",
                 manifest=None):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        if mode not in {"full", "crop", "birdshift"}:
            raise ValueError("mode must be full, crop, or birdshift")
        if not 0 <= suppression_prob <= 1:
            raise ValueError("suppression_prob must be in [0, 1]")
        if suppression_kind not in {"blur", "replace"}:
            raise ValueError("suppression_kind must be blur or replace")
        self.suppression_prob = suppression_prob
        self.suppression_kind = suppression_kind
        self.blur_radius = blur_radius

        def load(name, columns):
            """Read one of CUB's space-separated metadata files."""
            return pd.read_csv(self.root / name, sep=r"\s+", header=None, names=columns)

        # One row per image: id, path, label, official split flag and bounding box.
        df = load("images.txt", ["id", "image_path"])
        df = df.merge(load("image_class_labels.txt", ["id", "label"]), on="id")
        df = df.merge(load("train_test_split.txt", ["id", "is_train"]), on="id")
        df = df.merge(load("bounding_boxes.txt", ["id", "x", "y", "w", "h"]), on="id")
        df["label"] = df["label"] - 1  # CUB labels are 1-200; PyTorch expects 0-199

        # Fixed validation split: 15% of the official train set, stratified by species.
        train_pool = df[df["is_train"] == 1]
        test_df = df[df["is_train"] == 0]
        train_df, val_df = train_test_split(
            train_pool, test_size=0.15, stratify=train_pool["label"], random_state=42)
        choices = {"train": train_df, "val": val_df, "test": test_df, "trainval": train_pool}
        if manifest:
            # Use the manifest's split instead of the official one; keep CUB's boxes.
            m = pd.read_csv(manifest)
            if "species_dropped" in m:
                m = m[m["species_dropped"] == 0]
            m = df.merge(m[["image_id", "label", "split"]], left_on="id", right_on="image_id",
                         suffixes=("", "_manifest"))
            assert (m["label"] == m["label_manifest"]).all(), "manifest labels disagree with CUB"
            choices = {s: m[m["split"] == s] for s in ("train", "val", "test")}
        if split not in choices:
            raise ValueError(f"split must be one of {sorted(choices)}")
        data = choices[split]
        if split == "trainval":
            split = self.split = "train"  # same augmentation as train
        if split == "train" and imbalance_ratio > 1:
            data = make_long_tailed(data, imbalance_ratio)
        self.data = data.reset_index(drop=True)

        if mode == "crop":
            # Pad to square before resizing so the bird keeps its shape (README, problem 1).
            if split == "train":
                self.transform = transforms.Compose([
                    SquarePad(), transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
                    transforms.ToTensor(), NORMALIZE,
                ])
            else:
                self.transform = transforms.Compose([
                    SquarePad(), transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(), NORMALIZE,
                ])
        else:
            # 'birdshift' and 'full' share the pipeline; BirdShift only changes the
            # pixels before it, in __getitem__.
            self.transform = full_image_transform(split == "train", image_size, train_aug)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(self.root / "images" / row["image_path"]) as file:
            image = file.convert("RGB")
        bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))

        if self.mode == "crop":
            image = image.crop(bbox_to_pixels(bbox, image.size))
        elif self.mode == "birdshift" and self.split == "train":
            # BirdShift augmentation: suppress the background for a random half of
            # training images, so the same bird is seen with and without its habitat.
            if random.random() < self.suppression_prob:
                image = background_variant(image, bbox, kind=self.suppression_kind,
                                           blur_radius=self.blur_radius)

        return {"image": self.transform(image), "label": int(row["label"]), "image_id": int(row["id"])}
