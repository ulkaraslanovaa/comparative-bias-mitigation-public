## trainer.py
import os
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch import nn, Tensor
from tqdm import tqdm

from losses import Losses
from clustering import Clustering
from evaluation import Evaluation


class Trainer:
    """Trainer class for Debiasify.

    This class encapsulates the training loop including a warm-up phase,
    periodic clustering updates using shallow features, and loss computation
    that integrates classification loss, KL divergence loss, and MMD loss.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Dict[str, Any],
        config: Dict[str, Any],
        dataset_name: str = "CelebA"
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): An instance of the Model class.
            data (Dict[str, Any]): Dictionary containing DataLoader objects. Expected to have a "train" key.
            config (Dict[str, Any]): Configuration parameters loaded from config.yaml.
            dataset_name (str, optional): Name of the dataset to determine clustering gamma. Default is "CelebA".
        """
        self.model: nn.Module = model
        self.data: Dict[str, Any] = data
        self.config: Dict[str, Any] = config
        self.exp_name: str = str(self.config.get("exp_name", "debiasify"))
        self.dataset_name: str = dataset_name
        self.wandb = None

        # Device configuration.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training hyperparameters.
        training_config: Dict[str, Any] = self.config.get("training", {})
        self.learning_rate: float = float(training_config.get("learning_rate", 1e-4))
        self.batch_size: int = int(training_config.get("batch_size", 100))
        self.epochs: int = int(training_config.get("epochs", 50))
        self.weight_decay: float = float(training_config.get("weight_decay", 0.01))
        self.alpha: float = float(training_config.get("alpha", 0.1))
        self.use_wandb: bool = bool(training_config.get("wandb", False))

        # Warm-up phase duration. Default to 5 epochs if not specified.
        self.warmup_epochs: int = int(training_config.get("warmup_epochs", 5))

        # Determine clustering gamma from configuration.
        # Accept either a scalar (float/str) or a dict mapping dataset names to values.
        clustering_config: Dict[str, Any] = self.config.get("clustering", {})
        gamma_data: Any = clustering_config.get("gamma", None)

        if gamma_data is None:
            gamma_value = 0.01
        else:
            # scalar (could be float or numeric string)
            gamma_value = gamma_data

        try:
            self.gamma = float(gamma_value)
        except Exception:
            raise ValueError(f"Invalid gamma value: {gamma_value}. Must be a float or numeric string.")

        # Instantiate the Clustering object.
        # Here, we set use_pca to True and choose a default number of PCA components (e.g., 50).
        self.clustering: Clustering = Clustering(gamma=self.gamma, use_pca=True, pca_components=50)
        # Clustering update frequency after warm-up (update every epoch by default).
        self.clustering_update_frequency: int = int(training_config.get("clustering_update_frequency", 1))

        # Create the optimizer.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Set up checkpointing directory.
        self.checkpoint_dir: str = self.config.get("checkpoint_dir", "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Evaluation helper and tracking for best checkpoints
        self.evaluator = Evaluation(self.model, self.data, self.config)
        self.best_overall: float = -1.0
        self.best_worst: float = -1.0
        self.best_overall_path = os.path.join(self.checkpoint_dir, "best_overall.pth")
        self.best_worst_path = os.path.join(self.checkpoint_dir, "best_worst_group.pth")

        if self.use_wandb:
            self.init_wandb()

    def init_wandb(self) -> None:
        """Initialize W&B logging when enabled in config."""
        wandb_cfg: Dict[str, Any] = self.config.get("wandb", {})
        project: str = str(
            wandb_cfg.get(
                "project",
                f"Debiasify_{self.dataset_name}"
            )
        )
        entity: str = str(wandb_cfg.get("entity", "debias-medimg"))
        run_name: str = str(wandb_cfg.get("name", self.exp_name))

        try:
            import wandb as wandb_module
            self.wandb = wandb_module
            self.wandb.init(
                entity=entity,
                project=project,
                id=None,
                resume=False,
                force=True,
                name=run_name,
                config={
                    "dataset": self.dataset_name,
                    "exp_name": self.exp_name,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "weight_decay": self.weight_decay,
                    "alpha": self.alpha,
                    "warmup_epochs": self.warmup_epochs,
                    "clustering_update_frequency": self.clustering_update_frequency,
                    "gamma": self.gamma,
                    "use_pca": self.clustering.use_pca,
                    "pca_components": self.clustering.pca_components,
                },
            )
        except Exception as exc:
            self.wandb = None
            print(f"[WARN] W&B init failed. Continuing without W&B logging. Reason: {exc}")

    def log_wandb(self, payload: Dict[str, Any]) -> None:
        if self.wandb is not None:
            self.wandb.log(payload)

    def unpack_batch(self, batch):
        if not isinstance(batch, (list, tuple)):
            raise ValueError("Batch data must be a tuple or list.")

        # Expect the shared format: (image, attr, path)
        if len(batch) < 2:
            raise ValueError("Batch must contain image and attr.")

        inputs = batch[0]
        attr = batch[1]

        if not (torch.is_tensor(attr) and attr.dim() >= 2 and attr.size(1) >= 2):
            raise ValueError("Attr tensor must provide label and bias (shape: [N, 2]).")

        targets = attr[:, 0]
        biases = attr[:, 1]

        return inputs, targets, biases

    def update_clustering(self) -> Dict[str, float]:
        """Update clustering assignments using shallow features from the training set."""
        self.model.eval()
        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []
        train_loader = self.data.get("train", None)
        if train_loader is None:
            raise ValueError("Training DataLoader not found in provided data.")

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Clustering Update", leave=False):
                inputs, labels, _ = self.unpack_batch(batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Extract shallow features using the dedicated method.
                features = self.model.get_shallow_features(inputs)
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        # Concatenate features and labels from all batches.
        features_tensor: Tensor = torch.cat(all_features, dim=0)
        labels_tensor: Tensor = torch.cat(all_labels, dim=0)

        # Update clusters using the Clustering object.
        clustering_info = self.clustering.update_clusters(features_tensor, labels_tensor)
        cluster_counts: List[int] = [
            int(info["model"].n_clusters) for info in clustering_info.values()
        ]

        summary: Dict[str, float] = {
            "num_classes_clustered": float(len(cluster_counts)),
            "avg_clusters_per_class": float(sum(cluster_counts) / len(cluster_counts)) if cluster_counts else 0.0,
            "min_clusters_per_class": float(min(cluster_counts)) if cluster_counts else 0.0,
            "max_clusters_per_class": float(max(cluster_counts)) if cluster_counts else 0.0,
        }
        # (Optional) The returned clustering_info can be used for further analysis.
        self.model.train()
        return summary

    def train(self) -> None:
        """Runs the training loop for the model with Debiasify's self-distillation strategy."""
        train_loader = self.data.get("train", None)
        if train_loader is None:
            raise ValueError("Training DataLoader not found in provided data.")

        for epoch in range(self.epochs):
            epoch_loss_total: float = 0.0
            epoch_loss_ce: float = 0.0
            epoch_loss_kl: float = 0.0
            epoch_loss_akd: float = 0.0
            epoch_valid_cluster_samples: int = 0
            epoch_total_cluster_samples: int = 0
            num_batches: int = 0

            clustering_summary: Optional[Dict[str, float]] = None
            # If beyond warm-up, update clustering assignments periodically.
            if epoch >= self.warmup_epochs and ((epoch - self.warmup_epochs) % self.clustering_update_frequency == 0):
                print(f"Epoch {epoch + 1}: Updating clustering assignments.")
                clustering_summary = self.update_clustering()

            self.model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{self.epochs}]", leave=True)

            for batch in progress_bar:
                inputs, targets, _ = self.unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass through the model.
                outputs: Dict[str, Tensor] = self.model.forward(inputs)
                logits_shallow: Tensor = outputs.get("logits_shallow")
                logits_deep: Tensor = outputs.get("logits_deep")
                shallow_features: Tensor = outputs.get("shallow_features")
                deep_features: Tensor = outputs.get("deep_features")

                # Compute classification loss (averaged cross-entropy).
                loss_ce: Tensor = Losses.classification_loss(logits_shallow, logits_deep, targets)

                if epoch < self.warmup_epochs:
                    # Warm-up phase: use only classification loss.
                    total_loss: Tensor = loss_ce
                    loss_kl_value: float = 0.0
                    loss_akd_value: float = 0.0
                    valid_cluster_ratio: float = 0.0
                else:
                    # Full training phase: compute additional loss components.
                    loss_kl: Tensor = Losses.kl_divergence_loss(logits_shallow, logits_deep)
                    cluster_ids: Tensor = self.clustering.predict_clusters_by_label(
                        shallow_features.detach(),
                        targets.detach(),
                    )
                    valid_mask = cluster_ids >= 0
                    valid_cluster_count = int(valid_mask.sum().item())
                    total_cluster_count = int(cluster_ids.numel())
                    epoch_valid_cluster_samples += valid_cluster_count
                    epoch_total_cluster_samples += total_cluster_count
                    valid_cluster_ratio = (
                        float(valid_cluster_count) / float(total_cluster_count)
                        if total_cluster_count > 0 else 0.0
                    )
                    loss_akd: Tensor = Losses.attribute_kd_mmd_loss(
                        shallow_features,
                        deep_features,
                        targets,
                        cluster_ids,
                    )
                    total_loss = loss_ce + loss_akd + self.alpha * loss_kl
                    loss_kl_value = loss_kl.item()
                    loss_akd_value = loss_akd.item()

                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses for reporting.
                batch_loss_ce: float = loss_ce.item()
                batch_loss_total: float = total_loss.item()
                epoch_loss_ce += batch_loss_ce
                epoch_loss_total += batch_loss_total
                epoch_loss_kl += loss_kl_value
                epoch_loss_akd += loss_akd_value
                num_batches += 1

                progress_bar.set_postfix({
                    "L_CE": f"{batch_loss_ce:.4f}",
                    "L_KL": f"{loss_kl_value:.4f}",
                    "L_AKD": f"{loss_akd_value:.4f}",
                    "ValidC": f"{valid_cluster_ratio:.2%}",
                    "L_Total": f"{batch_loss_total:.4f}"
                })

            avg_loss_ce: float = epoch_loss_ce / num_batches
            avg_loss_kl: float = epoch_loss_kl / num_batches if num_batches > 0 else 0.0
            avg_loss_akd: float = epoch_loss_akd / num_batches if num_batches > 0 else 0.0
            avg_loss_total: float = epoch_loss_total / num_batches
            epoch_valid_cluster_ratio: float = (
                float(epoch_valid_cluster_samples) / float(epoch_total_cluster_samples)
                if epoch_total_cluster_samples > 0 else 0.0
            )

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] - "
                f"Avg L_CE: {avg_loss_ce:.4f}, Avg L_KL: {avg_loss_kl:.4f}, "
                f"Avg L_AKD: {avg_loss_akd:.4f}, Avg L_Total: {avg_loss_total:.4f}, "
                f"Valid Cluster IDs: {epoch_valid_cluster_ratio:.2%}"
            )

            # Validation metrics for model selection
            val_metrics = self.evaluator.evaluate(split="val")
            val_overall = float(val_metrics.get("overall_accuracy", 0.0))
            val_worst = float(val_metrics.get("worst_group_accuracy", 0.0))
            print(f"Validation - Overall Accuracy: {val_overall:.4f}, Worst Group Accuracy: {val_worst:.4f}")

            log_payload: Dict[str, Any] = {
                "epoch": epoch + 1,
                "loss_ce_train": avg_loss_ce,
                "loss_kl_train": avg_loss_kl,
                "loss_akd_train": avg_loss_akd,
                "loss_total_train": avg_loss_total,
                "cluster_valid_ratio_train": epoch_valid_cluster_ratio,
                "acc_overall_val": val_overall,
                "acc_worst_group_val": val_worst,
                "best_acc_overall_val": self.best_overall,
                "best_acc_worst_group_val": self.best_worst,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "warmup_epochs": self.warmup_epochs,
                "clustering_update_frequency": self.clustering_update_frequency,
            }

            if clustering_summary is not None:
                log_payload.update({
                    "clusters/num_classes_clustered": clustering_summary["num_classes_clustered"],
                    "clusters/avg_clusters_per_class": clustering_summary["avg_clusters_per_class"],
                    "clusters/min_clusters_per_class": clustering_summary["min_clusters_per_class"],
                    "clusters/max_clusters_per_class": clustering_summary["max_clusters_per_class"],
                })
            # Save best overall accuracy model
            if val_overall > self.best_overall:
                self.best_overall = val_overall
                best_payload = {
                    "epoch": epoch + 1,
                    "val_overall_accuracy": val_overall,
                    "val_worst_group_accuracy": val_worst,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "clustering_state": self.clustering.kmeans_models,
                }
                torch.save(best_payload, self.best_overall_path)
                # Save human-readable summary for quick inspection
                best_overall_txt = os.path.join(self.checkpoint_dir, "best_overall.txt")
                with open(best_overall_txt, "w") as f:
                    f.write(f"epoch: {epoch + 1}\n")
                    f.write(f"val_avg_accuracy: {val_overall:.6f}\n")
                    f.write(f"val_worst_group_accuracy: {val_worst:.6f}\n")
                print(f"Best average val accuracy improved to {val_overall:.4f}. Saved to {self.best_overall_path} and {best_overall_txt}")

            # Save best worst-group accuracy model
            if val_worst > self.best_worst:
                self.best_worst = val_worst
                best_payload = {
                    "epoch": epoch + 1,
                    "val_avg_accuracy": val_overall,
                    "val_worst_group_accuracy": val_worst,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "clustering_state": self.clustering.kmeans_models,
                }
                torch.save(best_payload, self.best_worst_path)
                best_worst_txt = os.path.join(self.checkpoint_dir, "best_worst.txt")
                with open(best_worst_txt, "w") as f:
                    f.write(f"epoch: {epoch + 1}\n")
                    f.write(f"val_avg_accuracy: {val_overall:.6f}\n")
                    f.write(f"val_worst_group_accuracy: {val_worst:.6f}\n")
                print(f"Best worst-group val accuracy improved to {val_worst:.4f}. Saved to {self.best_worst_path} and {best_worst_txt}")

            log_payload["best_acc_overall_val"] = self.best_overall
            log_payload["best_acc_worst_group_val"] = self.best_worst
            self.log_wandb(log_payload)

        print("Training complete.")
        if self.wandb is not None:
            self.wandb.finish()
