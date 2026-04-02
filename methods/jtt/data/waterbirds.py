import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from data.confounder_dataset import ConfounderDataset
from models import model_attributes

class WaterbirdsDataset(ConfounderDataset):
    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        augment_data=False,
        model_type=None,
        metadata_csv_name="metadata.csv"
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        # Define where the data lives - images are in data/ subdirectory
        self.data_dir = os.path.join(self.root_dir, "data")
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"{self.data_dir} does not exist yet. Please generate or copy dataset first.")

        # Read in metadata
        print(f"Reading '{os.path.join(self.root_dir, metadata_csv_name)}'")
        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, metadata_csv_name))

        # Get the y values
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 1

        self.img_paths = [
            os.path.join(self.data_dir, fn)
            for fn in self.metadata_df["unique_img_filename"]
        ]
        
        # Map to groups
        self.n_groups = pow(2, 2)
        assert self.n_groups == 4, "check the code if you are running otherwise"
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["unique_img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.train_transform = get_transform_cub(self.model_type,
                                                    train=True,
                                                    augment_data=augment_data)
        self.eval_transform = get_transform_cub(self.model_type,
                                                train=False,
                                                augment_data=augment_data)

def get_transform_cub(model_type, train=False, augment_data=False):
    target_resolution = model_attributes[model_type]["target_resolution"]
    assert target_resolution is not None

    mean = [0.4198, 0.4343, 0.3832]
    std  = [0.2092, 0.2052, 0.2123]

    if train and augment_data:
        # This changes the bias structure
        transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # Bias-preserving (default)
        transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform