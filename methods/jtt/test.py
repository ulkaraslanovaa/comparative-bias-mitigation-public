from data.data import prepare_data
from data import dro_dataset
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import torchvision
import torch.nn as nn

join = os.path.join

@torch.no_grad()
def test_jtt(
    test_loader,
    test_data,
    pretrained_path,
    worst_group,
    output_csv,
    device,
):
    """
    Test a pretrained JTT downstream model and save predictions
    with the standard evaluation schema.
    """

    if worst_group:
        print("Evaluating on worst group")
        pretrained_path = os.path.join(pretrained_path, "best_worst_group_model.pth")
    else:
        print("Evaluating on avg model")
        pretrained_path = os.path.join(pretrained_path, "best_model.pth")
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    model = torch.load(pretrained_path, map_location=device)
    model.to(device)
    model.eval()

    rows = []

    for batch in tqdm(test_loader, desc="JTT testing"):
        x, y, g, idx = batch
        x = x.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        y_pred = probs.argmax(dim=1)
        p_max = probs.max(dim=1).values

        for i in range(len(idx)):
            data_idx = idx[i].item()

            base_dataset = test_data.dataset.dataset
            row = {
                "img_path": base_dataset.img_paths[data_idx],
                "y_true": y[i].item(),
                "attr_true": base_dataset.confounder_array[data_idx],
                "y_pred": y_pred[i].item(),
                "p_0": probs[i, 0].item(),
                "p_1": probs[i, 1].item(),
                "p_max": p_max[i].item(),
                "correct": int(y_pred[i].item() == y[i].item()),
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="a name for the dataset directory",
    )
    parser.add_argument("--dataset",
                        type=str,
                        default="waterbirds",
                        help="waterbirds, or fairface")
    
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--worst_group", action="store_true", default=False)

    args = parser.parse_args()
    
    args.shift_type = "confounder"

    if args.dataset == "waterbirds":
        args.target_name = "y"
        args.confounder_names = "place"
        args.n_classes = 2
    elif args.dataset == "fairface":
        args.target_name = "gender"
        args.confounder_names = "ethnicity"
        args.n_classes = 2
    else:
        raise NotImplementedError

    
    args.augment_data = False
    args.metadata_csv_name = "metadata.csv"
    args.model = "resnet18"
    args.fraction = 1.0
    args.batch_size = 128

    test_data = prepare_data(
            args,
            train=False,
        )[0]
    
    test_loader = dro_dataset.get_loader(test_data, train=False, reweight_groups=None)

    test_jtt(test_loader,
             test_data,
             args.pretrained_path,
             worst_group=args.worst_group,
             output_csv=args.output_csv,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             )
