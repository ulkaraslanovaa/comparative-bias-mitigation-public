import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import wandb
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Transforms
WATERBIRDS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4198, 0.4343, 0.3832],
        std=[0.2092, 0.2052, 0.2123],
    ),
])

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def build_datasets(dataset, data_root):
    if dataset == "waterbirds":
        from methods.resnet_baseline.waterbirds import WaterbirdsDataset

        train_ds = WaterbirdsDataset(
            root=data_root,
            split="train",
            transform=WATERBIRDS_TRANSFORM,
        )
        val_ds = WaterbirdsDataset(
            root=data_root,
            split="valid",
            transform=WATERBIRDS_TRANSFORM,
        )
        num_groups = 4

    elif dataset == "fairface":
        from methods.resnet_baseline.fairface import FairfaceDataset

        train_ds = FairfaceDataset(
            root=data_root,
            split="train",
            transform=IMAGENET_TRANSFORM,
        )
        val_ds = FairfaceDataset(
            root=data_root,
            split="valid",
            transform=IMAGENET_TRANSFORM,
        )
        num_groups = 4

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_ds, val_ds, num_groups


def eval_group_accuracy(model, loader, device):
    model.eval()

    correct = torch.zeros(4, device=device)
    total = torch.zeros(4, device=device)

    with torch.no_grad():
        for imgs, attr, _ in loader:
            imgs = imgs.to(device)
            y = attr[:, 0].to(device)   # target
            c = attr[:, 1].to(device)   # spurious attribute

            group = y * 2 + c
            preds = model(imgs).argmax(dim=1)

            for g in range(4):
                mask = group == g
                if mask.any():
                    correct[g] += (preds[mask] == y[mask]).sum()
                    total[g] += mask.sum()

    group_acc = correct / total.clamp(min=1)
    worst_acc = group_acc.min().item()
    avg_acc = (correct.sum() / total.sum()).item()

    return group_acc.cpu().tolist(), worst_acc, avg_acc


def train(
    data_root,
    dataset,
    results_dir,
    epochs,
    batch_size,
    lr,
    weight_decay,
    seed
):
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    wandb.init(
        project=f"resnet18_baseline_{dataset}",
        name=f"lr{lr}_wd{weight_decay}",
        config={
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay
        },
    )
    train_ds, val_ds, _ = build_datasets(dataset, data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_worst_group_acc = 0.0

    log_path = os.path.join(results_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_avg_acc",
            "val_worst_acc",
            "acc_g0",
            "acc_g1",
            "acc_g2",
            "acc_g3",
        ])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, attr, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            labels = attr[:, 0].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        group_acc, worst_acc, avg_acc = eval_group_accuracy(
            model, val_loader, device
        )

        if worst_acc > best_worst_group_acc:
            best_worst_group_acc = worst_acc
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, "best_worst_group_model.pth"),
            )
            # write latest worst-group result (overwrite)
            os.makedirs(results_dir, exist_ok=True)
            val_worst_path = os.path.join(results_dir, "val_worst_result.txt")
            with open(val_worst_path, "w") as wf:
                wf.write(f"Epoch: {epoch+1}\tAccuracy: {worst_acc:.4f}\n")
            
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, "best_model.pth"),
            )
            # write latest validation result (overwrite)
            os.makedirs(results_dir, exist_ok=True)
            val_result_path = os.path.join(results_dir, "val_result.txt")
            with open(val_result_path, "w") as vf:
                vf.write(f"Epoch: {epoch+1}\tAccuracy: {avg_acc:.4f}\n")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_avg_acc": avg_acc,
            "val_worst_acc": worst_acc,
            "group_acc_0": group_acc[0],
            "group_acc_1": group_acc[1],
            "group_acc_2": group_acc[2],
            "group_acc_3": group_acc[3],
        })

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                avg_acc,
                worst_acc,
                *group_acc,
            ])

        print(
            f"Epoch {epoch+1}: "
            f"loss={train_loss:.4f}, "
            f"avg_acc={avg_acc:.3f}, "
            f"worst_acc={worst_acc:.3f}"
        )

    torch.save(
        model.state_dict(),
        os.path.join(results_dir, "final_model.pth"),
    )

    wandb.finish()

    print(f"Training finished. Best val acc = {best_val_acc:.3f}, Best worst group acc = {best_worst_group_acc:.3f}")
