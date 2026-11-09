from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from birdshift.transforms.background import background_variant, SquarePad, bbox_to_pixels
import random

import numpy as np


def make_long_tailed(df, ratio, seed=42):
    """Subsample a split to an exponential long-tail profile.

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
    def __init__(self, root, split="train", mode="full", image_size=224, suppression_prob=0.5, suppression_kind="blur", blur_radius=22, imbalance_ratio=1.0, train_aug="rrc"):
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
            return pd.read_csv(
                self.root / name, sep=r"\s+", header=None, names=columns
            )

        df = load("images.txt", ["id", "image_path"])
        df = df.merge(load("image_class_labels.txt", ["id", "label"]), on="id")
        df = df.merge(load("train_test_split.txt", ["id", "is_train"]), on="id")
        df = df.merge(
            load("bounding_boxes.txt", ["id", "x", "y", "w", "h"]), on="id"
        )
        df["label"] = df["label"] - 1

        train_pool = df[df["is_train"] == 1]
        test_df = df[df["is_train"] == 0]
        train_df, val_df = train_test_split(
            train_pool,
            test_size=0.15,
            stratify=train_pool["label"],
            random_state=42,
        )
        # trainval = all 5,994 official training images (final runs, no held-out val)
        choices = {"train": train_df, "val": val_df, "test": test_df, "trainval": train_pool}
        if split not in choices:
            raise ValueError("split must be train, val, test, or trainval")
        data = choices[split]
        if split == "trainval":
            split = self.split = "train"  # same augmentation as train
        if split == "train" and imbalance_ratio > 1:
            data = make_long_tailed(data, imbalance_ratio)
        self.data = data.reset_index(drop=True)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        if mode == "crop":
            # Preserve aspect ratio and augment, matching the successful crop run.
            if split == "train":
                self.transform = transforms.Compose([
                    SquarePad(), transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
                    transforms.ToTensor(), normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    SquarePad(), transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(), normalize,
                ])
        elif split == "train":
            # 'birdshift' and 'full' use the identical post-perturbation pipeline.
            # rrc: ImageNet-style RandomResizedCrop (can keep only 8% of the image).
            # fg: fine-grained style Resize(size*8/7) + RandomCrop(size), bird stays in frame.
            if train_aug not in {"rrc", "fg"}:
                raise ValueError("train_aug must be rrc or fg")
            crop = (transforms.RandomResizedCrop(image_size) if train_aug == "rrc" else
                    transforms.Compose([transforms.Resize(image_size * 8 // 7),
                                        transforms.RandomCrop(image_size)]))
            self.transform = transforms.Compose([
                crop,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize,
            ])
        else:
            # Exact same evaluation pipeline as the full-image baseline.
            self.transform = transforms.Compose([
                transforms.Resize(image_size * 8 // 7), transforms.CenterCrop(image_size),
                transforms.ToTensor(), normalize,
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        path = self.root / "images" / row["image_path"]
        with Image.open(path) as file:
            image = file.convert("RGB")

        if self.mode == "crop":
            bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
            image = image.crop(bbox_to_pixels(bbox, image.size))
        elif self.mode == "birdshift" and self.split == "train":
            if random.random() < self.suppression_prob:
                bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
                image = background_variant(
                    image, bbox, kind=self.suppression_kind,
                    blur_radius=self.blur_radius
                )

        image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "image_id": int(row["id"]),
        }
