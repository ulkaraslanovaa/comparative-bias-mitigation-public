from omegaconf import DictConfig
from methods.resnet_baseline.train import train
from methods.resnet_baseline.test import test
import os

def main(cfg: DictConfig):
    print(f"Running baseline for dataset: {cfg.dataset.name}")

    dataset = cfg.dataset.name
    data_root = cfg.dataset.root_dir

    results_dir = cfg.method.results_dir
    exp_name = cfg.method.exp_name
    lr = cfg.method.optimizer.lr
    weight_decay = cfg.method.optimizer.weight_decay  
    batch_size = cfg.method.training.batch_size

    output_dir = f"{results_dir}/{exp_name}"
    seed = cfg.seed

    if cfg.stage == "train":
        epochs = cfg.method.training.epochs
        
        train(
            data_root=data_root,
            dataset=dataset,
            results_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        
    else:
        pretrained_path = cfg.method.testing.pretrained_path
        output_csv = cfg.method.testing.output_csv
        worst_group = cfg.method.testing.worst_group
        
        test(
            data_root=data_root,
            pretrained_path=pretrained_path,
            output_csv=output_csv,
            batch_size=batch_size,
            dataset=dataset,
            worst_group=worst_group,
            seed=seed,
        )
