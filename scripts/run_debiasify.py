import os
import subprocess
import yaml
from pathlib import Path
from omegaconf import DictConfig

def build_config(cfg: DictConfig) -> dict:
    dataset = cfg.dataset.name
    root_dir = cfg.dataset.root_dir
    method_cfg = cfg.method
    training_cfg = method_cfg.training
    optimizer_cfg = method_cfg.optimizer

    # Clustering gamma lookup with fallback. Accept scalar or mapping.
    gamma_cfg = getattr(method_cfg, "clustering", {}).get("gamma", None) if hasattr(method_cfg, "clustering") else None
    
    if gamma_cfg is None:
        gamma_value = 0.02
    else:
        gamma_value = float(gamma_cfg)

    config = {
        "dataset_name": dataset,
        "data_root": root_dir,
        "seed": cfg.seed,
        "exp_name": str(method_cfg.exp_name),
        "checkpoint_dir": os.path.join(method_cfg.results_dir, dataset, method_cfg.exp_name),
        "training": {
            "learning_rate": float(optimizer_cfg.lr),
            "batch_size": int(training_cfg.batch_size),
            "epochs": int(training_cfg.epochs),
            "weight_decay": float(optimizer_cfg.weight_decay),
            "alpha": float(getattr(training_cfg, "alpha", 0.1)),
            "warmup_epochs": int(getattr(training_cfg, "warmup_epochs", 5)),
            "wandb": bool(getattr(training_cfg, "wandb", False)),
        },
        "wandb": {
            "entity": "debias-medimg",
            "project": f"debiasify_{dataset}",
            "name": str(method_cfg.exp_name),
        },
        "model": {
            "backbone": getattr(method_cfg, "model", {}).get("backbone", "ResNet18"),
            "pretrained": bool(getattr(method_cfg, "model", {}).get("pretrained", True)),
        },
        "clustering": {
            # store scalar gamma in generated config
            "gamma": gamma_value,
            "method": "adaptive K-means",
        }
    }

    return config


def main(cfg: DictConfig) -> None:
    stage = cfg.stage
    method_dir = Path(cfg.method.method_dir)
    config_path = method_dir / "config.autogen.yaml"

    config = build_config(cfg)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    if stage == "train":
        cmd = [
            "python",
            str(method_dir / "main.py"),
            "--config",
            str(config_path),
        ]
        print("\nRunning Debiasify training")
        print("Executing:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("\nDebiasify training (with val/test eval) complete.")
    else:
        # Test-only: load checkpoint and export CSV
        testing = cfg.method.testing
        ckpt_path = testing.pretrained_path
        out_csv = testing.output_csv
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        cmd = [
            "python",
            str(method_dir / "test.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(ckpt_path),
            "--output_csv",
            str(out_csv),
            "--split",
            "test",
        ]

        print("\nRunning Debiasify test")
        print("Executing:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("\nDebiasify test complete.")
