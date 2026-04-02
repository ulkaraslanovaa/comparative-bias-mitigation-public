import subprocess
import os
from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:
    """
    Run Disentangled Feature Augmentation workflow.
    """
    stage = cfg.stage
    dataset = cfg.dataset.name
    root_dir = cfg.dataset.root_dir
    exp_name = cfg.method.exp_name
    disent_dir = cfg.method.method_dir
    results_dir = cfg.method.results_dir

    if stage == "train":
        epochs = cfg.method.training.epochs
        lr = str(cfg.method.optimizer.lr)
        weight_decay = str(cfg.method.optimizer.weight_decay)

        lambda_dis_align = cfg.method.loss.lambda_dis_align
        lambda_swap_align = cfg.method.loss.lambda_swap_align
        lambda_swap = cfg.method.loss.lambda_swap

        curr_epoch = cfg.method.training.curr_epoch
        train_ours = cfg.method.training.train_ours
        wandb = cfg.method.training.wandb

        print(wandb)
        print("\nRunning Disentangled Training")

        cmd = [
            "python",
            f"{disent_dir}/train.py",
            "--dataset",
            str(dataset),
            "--exp",
            str(exp_name),
            "--data_dir",
            str(root_dir),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--weight_decay",
            str(weight_decay),
            "--lambda_dis_align",
            str(lambda_dis_align),
            "--lambda_swap_align",
            str(lambda_swap_align),
            "--lambda_swap",
            str(lambda_swap),
            "--curr_epoch",
            str(curr_epoch),
            "--log_dir",
            str(results_dir),
        ]

        if train_ours:
            cmd.append("--train_ours")

        if wandb:
            cmd.append("--wandb")

        print("\nExecuting:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)

        print("\nDisentangled Feature Augmentation training complete.")

    else:
        print("\nRunning Disentangled Evaluation")
        pretrained_path = cfg.method.testing.pretrained_path
        out_csv = cfg.method.testing.output_csv
        worst_group = cfg.method.testing.worst_group

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        cmd = [
            "python",
            f"{disent_dir}/test.py",
            "--dataset",
            str(dataset),
            "--exp",
            str(exp_name),
            "--data_dir",
            str(root_dir),
            "--pretrained_path",
            str(pretrained_path),
            "--out_csv",
            str(out_csv),
        ]

        if worst_group:
            cmd.append("--worst_group")

        print("\nExecuting:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("\nDisentangled Feature Augmentation evaluation complete.")
