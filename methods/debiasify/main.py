#!/usr/bin/env python3
"""
main.py

Entry point for the Debiasify project. This script loads configuration parameters,
instantiates the DatasetLoader, Model, Trainer, and Evaluation classes, and then
orchestrates the training and evaluation process for Debiasify.
"""

import os
import sys
import yaml
import argparse
from typing import Any, Dict

# Import project modules
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the configuration parameters from a YAML file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config_data: Dict[str, Any] = yaml.safe_load(f)
    return config_data


def main() -> None:
    """
    Main function that coordinates data loading, model creation, training, and evaluation.
    """
    parser = argparse.ArgumentParser(description="Run Debiasify training/eval")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # 1. Load Configuration Parameters
    config_path: str = args.config
    config: Dict[str, Any] = load_config(config_path)
    print(f"Configuration loaded successfully from '{config_path}'.")

    # 2. Instantiate the DatasetLoader and load data
    dataset_loader: DatasetLoader = DatasetLoader(config)
    data_loaders: Dict[str, Any] = dataset_loader.load_data()

    dataset_name: str = dataset_loader.dataset_name
    num_train: int = len(data_loaders.get("train").dataset)
    num_val: int = len(data_loaders.get("val").dataset)
    num_test: int = len(data_loaders.get("test").dataset)
    print(f"Using dataset '{dataset_name}' | train: {num_train} | val: {num_val} | test: {num_test}")

    # 3. Instantiate the Model using the provided configuration.
    # Default number of classes is set to 2.
    model: Model = Model(config, num_classes=2)
    print("Model instantiated successfully.")

    # 4. Instantiate the Trainer with the model, selected data, config, and dataset name.
    trainer: Trainer = Trainer(model, data_loaders, config, dataset_name)
    print("Starting training process...")
    trainer.train()
    print("Training completed successfully.")

    # 5. Instantiate the Evaluation module and perform evaluation.
    evaluator: Evaluation = Evaluation(model, data_loaders, config)
    print("Evaluating the model on the validation dataset...")
    val_results: Dict[str, Any] = evaluator.evaluate(split="val")

    print("Evaluating the model on the test dataset...")
    test_results: Dict[str, Any] = evaluator.evaluate(split="test")

    # 6. Report Evaluation Results
    print("\n=== Validation Results ===")
    for metric_name, metric_value in val_results.items():
        print(f"{metric_name}: {metric_value}")

    print("\n=== Test Results ===")
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value}")

    # Optional: Uncomment the following line to visualize t-SNE embeddings of the deep features.
    # evaluator.visualize_tsne(feature_type="deep", perplexity=30.0, n_samples=1000)


if __name__ == "__main__":
    main()


### 
# train and save the best avg acc and worst group acc models
# evaluate saved checkpoint models on test set (just need to save similar file as we had for other methods)
# dataloaders should be same as in other methods, either reuse or integrate, seed 42, check the dataloading process
# wandb saving
# check paper and hyperparameters, tune hyperparams


# gamma: search log-scale e.g. [1e-4, 1e-3, 1e-2, 1e-1]; finer around values that give sensible cluster counts.
# clustering_update_frequency: try [1, 2, 5] (every epoch, every 2, every 5).
# warmup_epochs: try [0, 2, 5, 10]; if model needs stable features, increase.