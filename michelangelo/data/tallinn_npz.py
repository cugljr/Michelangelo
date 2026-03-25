# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from michelangelo.data.transforms import build_transforms


class TallinnNPZDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        transform: Optional[Callable] = None,
        samples_subdir: str = "samples",
        splits_subdir: str = "splits",
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples_dir = self.data_root / samples_subdir
        self.split_file = self.data_root / splits_subdir / f"{split}.txt"
        self.sample_ids = self._load_split()

    def _load_split(self) -> List[str]:
        if not self.split_file.exists():
            raise FileNotFoundError(f"split file not found: {self.split_file}")

        with self.split_file.open("r", encoding="utf-8") as f:
            sample_ids = [line.strip() for line in f if line.strip()]

        missing = [sample_id for sample_id in sample_ids if not (self.samples_dir / f"{sample_id}.npz").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples listed in {self.split_file} are missing under {self.samples_dir}. "
                f"First missing sample: {missing[0]}"
            )

        if not sample_ids:
            raise RuntimeError(f"no samples found in split file: {self.split_file}")

        return sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id = self.sample_ids[index]
        sample_path = self.samples_dir / f"{sample_id}.npz"
        with np.load(sample_path) as data:
            sample = {
                "surface": data["surface"],
                "vol_points": data["vol_points"],
                "vol_label": data["vol_label"],
                "near_points": data["near_points"],
                "near_label": data["near_label"],
                "loc": data["loc"],
                "scale": data["scale"],
                "sample_id": sample_id,
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TallinnNPZDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last_train: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.drop_last_train = drop_last_train

        self.train_transform = build_transforms(train_transform)
        self.val_transform = build_transforms(val_transform)
        self.test_transform = build_transforms(test_transform)

        self.train_dataset: Optional[TallinnNPZDataset] = None
        self.val_dataset: Optional[TallinnNPZDataset] = None
        self.test_dataset: Optional[TallinnNPZDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TallinnNPZDataset(
                data_root=self.data_root,
                split="train",
                transform=self.train_transform,
            )
            self.val_dataset = TallinnNPZDataset(
                data_root=self.data_root,
                split="val",
                transform=self.val_transform,
            )

        if stage in (None, "test"):
            self.test_dataset = TallinnNPZDataset(
                data_root=self.data_root,
                split="test",
                transform=self.test_transform,
            )

    def _make_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=self.drop_last_train)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return self._make_loader(self.test_dataset, shuffle=False, drop_last=False)
