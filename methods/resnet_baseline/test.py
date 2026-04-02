import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import pandas as pd
import random
import numpy as np

from methods.resnet_baseline.waterbirds import WaterbirdsDataset
from methods.resnet_baseline.fairface import FairfaceDataset
from methods.resnet_baseline.train import (
    WATERBIRDS_TRANSFORM,
    IMAGENET_TRANSFORM,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_test_dataset(dataset, data_root):
    if dataset == "waterbirds":
        return WaterbirdsDataset(
            root=data_root,
            split="test",
            transform=WATERBIRDS_TRANSFORM,
        )

    elif dataset == "fairface":
        return FairfaceDataset(
            root=data_root,
            split="test",
            transform=IMAGENET_TRANSFORM,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def test(
    data_root,
    pretrained_path,
    output_csv,
    batch_size,
    dataset,
    worst_group,
    seed,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset --------
    test_ds = build_test_dataset(dataset, data_root)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # -------- model --------
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if worst_group:
        print("Evaluating worst group performance")
        model.load_state_dict(torch.load(os.path.join(pretrained_path, "best_worst_group_model.pth"), map_location=device))
    else:
        print("Evaluating average performance across all groups")
        model.load_state_dict(torch.load(os.path.join(pretrained_path, "best_model.pth"), map_location=device))
    model.to(device)
    model.eval()

    rows = []

    # -------- inference --------
    with torch.no_grad():
        for imgs, attr, paths in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)

            y_true = attr[:, 0].to(device)
            attr_true = attr[:, 1].to(device)

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)

            y_pred = probs.argmax(dim=1)
            p_0 = probs[:, 0]
            p_1 = probs[:, 1]
            p_max = probs.max(dim=1).values
            correct = (y_pred == y_true)

            for i in range(len(y_true)):
                rows.append({
                    "img_path": paths[i],
                    "y_true": int(y_true[i].item()),
                    "attr_true": int(attr_true[i].item()),
                    "y_pred": int(y_pred[i].item()),
                    "p_0": float(p_0[i].item()),
                    "p_1": float(p_1[i].item()),
                    "p_max": float(p_max[i].item()),
                    "correct": int(correct[i].item()),
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved baseline evaluation CSV to: {output_csv}")
