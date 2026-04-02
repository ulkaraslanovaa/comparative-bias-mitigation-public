import argparse
import os
import yaml
import torch
import pandas as pd
from typing import Any, Dict

from model import Model
from dataset_loader import DatasetLoader


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_to_csv(model: torch.nn.Module, loader, device, out_csv: str) -> None:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs, attr, paths = batch
            inputs = inputs.to(device)
            labels = attr[:, 0].to(device)
            biases = attr[:, 1].to(device)

            outputs = model(inputs)
            logits = outputs["logits_deep"]
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            p0 = probs[:, 0]
            p1 = probs[:, 1]
            pmax = probs.max(dim=1).values
            correct = (preds == labels)

            for i in range(len(labels)):
                rows.append(
                    {
                        "img_path": str(paths[i]),
                        "y_true": int(labels[i]),
                        "attr_true": int(biases[i]),
                        "y_pred": int(preds[i]),
                        "p_0": float(p0[i]),
                        "p_1": float(p1[i]),
                        "p_max": float(pmax[i]),
                        "correct": int(correct[i]),
                    }
                )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved evaluation CSV to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Debiasify checkpoint and export CSV")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to Debiasify config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., best_overall.pth)")
    parser.add_argument("--output_csv", type=str, default="./evaluation/debiasify_test.csv", help="Where to write CSV")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DatasetLoader(config).load_data()[args.split]

    model = Model(config, num_classes=2).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    evaluate_to_csv(model, loader, device, args.output_csv)


if __name__ == "__main__":
    main()
