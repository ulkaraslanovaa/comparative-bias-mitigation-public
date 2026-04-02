import os
import subprocess
from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:
    """
    Run JTT workflow.
    """
    stage = cfg.stage
    jtt_dir = cfg.method.method_dir
    exp_name = cfg.method.exp_name
    dataset = cfg.dataset.name
    results_dir = cfg.method.results_dir

    if stage == "train":
        root_dir = cfg.dataset.root_dir
        epochs = cfg.method.training.epochs
        final_epoch = cfg.method.training.final_epoch
        lr = str(cfg.method.optimizer.lr)
        weight_decay = str(cfg.method.optimizer.weight_decay)

        print("\nERM (baseline)")
        cmd = [
            "python",
            f"{jtt_dir}/generate_downstream.py",
            "--exp_name",
            str(exp_name),
            "--dataset",
            str(dataset),
            "--n_epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--results_dir",
            str(results_dir),
            "--weight_decay",
            str(weight_decay),
            "--root_dir",
            str(root_dir),
            "--method",
            "ERM",
            "--cwd",
            str(jtt_dir),
            "--deploy",
        ]
        print("\nExecuting:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)

        print("\nJTT training")
        cmd = [
            "python",
            f"{jtt_dir}/process_training.py",
            "--exp_name",
            str(exp_name),
            "--dataset",
            str(dataset),
            "--n_epochs",
            str(epochs),
            "--folder_name",
            f"ERM_upweight_0_epochs_{epochs}_lr_{lr}_weight_decay_{weight_decay}",
            "--lr",
            str(lr),
            "--weight_decay",
            str(weight_decay),
            "--final_epoch",
            str(final_epoch),
            "--results_dir",
            str(results_dir),
            "--root_dir",
            str(root_dir),
            "--cwd",
            str(jtt_dir),
            "--deploy",
        ]
        print("\nExecuting:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)

    else: 
        print("\nJTT analysis")
        pretrained_path = cfg.method.testing.pretrained_path
        output_csv = cfg.method.testing.output_csv 
        worst_group = cfg.method.testing.worst_group
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        cmd = [
            "python",
            f"{jtt_dir}/test.py",
            "--pretrained_path",
            pretrained_path,
            "--dataset",
            dataset,
            "--output_csv",
            output_csv,
            "--root_dir",
            cfg.dataset.root_dir,
        ]

        if worst_group:
            cmd.append("--worst_group")

        subprocess.run(cmd, check=True)
