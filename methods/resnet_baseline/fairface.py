import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd

class FairfaceDataset(Dataset):
    """
    Baseline Fairface dataset.
    Returns:
        img: Tensor
        attr: Tensor([gender, ethnicity])
        path: str
    """

    def __init__(self, root, split, transform=None):
        split_map = {"train": 0, "valid": 1, "test": 2}

        df = pd.read_csv(os.path.join(root, "metadata.csv"))
        df = df[df["split"] == split_map[split]].reset_index(drop=True)

        self.paths = [
            os.path.join(root, "data", fn)
            for fn in df["file"]
        ]
        self.labels = df["gender"].astype(int).to_numpy()
        self.places = df["ethnicity"].astype(int).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        attr = torch.tensor(
            [int(self.labels[idx]), int(self.places[idx])],
            dtype=torch.long
        )

        return img, attr, self.paths[idx]
