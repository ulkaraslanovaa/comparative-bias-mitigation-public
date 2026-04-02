## dataset_loader.py
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse the dataset definitions and transforms from the Disent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISENT_DATA_DIR = PROJECT_ROOT / "methods" / "disent" / "data"
if str(DISENT_DATA_DIR) not in sys.path:
    sys.path.append(str(DISENT_DATA_DIR))

from util import get_dataset  # type: ignore


class DatasetLoader:
    """Builds train/val/test loaders for Waterbirds and Fairface.

    The loaders mirror the preprocessing used in JTT/Disent
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        training_cfg = config.get("training", {})
        self.batch_size: int = int(training_cfg.get("batch_size", 64))
        self.num_workers: int = int(config.get("num_workers", 4))
        self.pin_memory: bool = torch.cuda.is_available()
        self.seed: int = int(config.get("seed", 42))

        # Dataset selection
        self.dataset_name: str = str(config.get("dataset_name", "waterbirds")).lower()
        # Resolve dataset roots
        data_root = Path(config.get("data_root", PROJECT_ROOT / "data"))
        self.default_paths: Dict[str, Path] = {
            "waterbirds": data_root,
            "fairface": data_root,
        }

    def resolve_dataset_root(self) -> Path:
        if self.dataset_name not in self.default_paths:
            raise ValueError(f"Unsupported dataset '{self.dataset_name}'.")
        return self.default_paths[self.dataset_name]

    def seed_worker(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))

    def build_dataset(self, split: str):
        dataset_root = self.resolve_dataset_root()
        if not dataset_root.exists():
            raise RuntimeError(f"Dataset root '{dataset_root}' not found.")

        split_key = "valid" if split == "val" else split
        return get_dataset(
            dataset=self.dataset_name,
            data_dir=str(dataset_root),
            dataset_split=split_key,
            transform_split=split_key,
            use_preprocess=True,
        )

    def load_data(self) -> Dict[str, DataLoader]:
        datasets = {split: self.build_dataset(split) for split in ["train", "val", "test"]}
        
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_loader = DataLoader(
            datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.seed_worker,
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.seed_worker,
        )

        test_loader = DataLoader(
            datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.seed_worker,
        )

        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }