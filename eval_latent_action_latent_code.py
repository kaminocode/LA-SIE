import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.latent_action_eval import (
    LatentActionFeatureAdapter,
    RelativePosePairDataset,
    auto_prediction_kinds,
    build_eval_transform,
    default_device,
    flatten_prediction_component,
    load_experiment_args,
)


parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", type=Path, required=True)
parser.add_argument("--dataset-root", type=Path, default=None)
parser.add_argument("--train-images-file", type=Path, default=Path("./data/train_images.npy"))
parser.add_argument("--train-labels-file", type=Path, default=Path("./data/train_labels.npy"))
parser.add_argument("--val-images-file", type=Path, default=Path("./data/val_images.npy"))
parser.add_argument("--val-labels-file", type=Path, default=Path("./data/val_labels.npy"))
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--hidden-dim", type=int, default=256)
parser.add_argument("--target-space", type=str, choices=["quat", "euler"], default="quat")
parser.add_argument(
    "--input-kind",
    type=str,
    choices=["auto", "code", "coefficients", "raw_matrix", "operator"],
    default="auto",
)
parser.add_argument("--pairs-per-object", type=int, default=16)
parser.add_argument("--pair-seed", type=int, default=0)
parser.add_argument("--size-dataset", type=int, default=-1)
parser.add_argument("--output-json", type=Path, default=None)


class RegressionProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        if input_dim > 4096:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        return self.net(x)


def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    if float(ss_tot) == 0.0:
        return torch.zeros((), device=output.device)
    return 1 - ss_res / ss_tot


def quaternion_angle_error_deg(output, target):
    output = F.normalize(output, dim=1)
    target = F.normalize(target, dim=1)
    cosine = torch.sum(output * target, dim=1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(cosine)).mean()


def standardize(train_x, val_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return (train_x - mean) / std, (val_x - mean) / std


@torch.no_grad()
def evaluate_probe(model, features, targets, target_space):
    model.eval()
    predictions = model(features)
    metrics = {
        "mse": float(F.mse_loss(predictions, targets).cpu()),
        "r2": float(r2_score(predictions, targets).cpu()),
    }
    if target_space == "quat":
        metrics["angle_error_deg"] = float(
            quaternion_angle_error_deg(predictions, targets).cpu()
        )
    return metrics


def train_probe(train_x, train_y, val_x, val_y, cli_args):
    device = default_device(cli_args.device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    train_x, val_x = standardize(train_x, val_x)
    model = RegressionProbe(train_x.shape[1], train_y.shape[1], cli_args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.wd)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cli_args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_metrics = None
    for epoch in range(cli_args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            loss.backward()
            optimizer.step()

        current_metrics = evaluate_probe(model, val_x, val_y, cli_args.target_space)
        if best_metrics is None or current_metrics["mse"] < best_metrics["mse"]:
            best_metrics = dict(current_metrics)
            best_metrics["epoch"] = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train": evaluate_probe(model, train_x, train_y, cli_args.target_space),
        "val": evaluate_probe(model, val_x, val_y, cli_args.target_space),
        "best_val": best_metrics,
    }


@torch.no_grad()
def extract_split_features(adapter, loader, requested_input_kind):
    feature_store = None
    targets = []

    for x, y, z, _ in loader:
        pair_features = adapter.pair_features(x, y)
        prediction = pair_features["prediction"]
        active_kinds = (
            auto_prediction_kinds(prediction)
            if requested_input_kind == "auto"
            else [requested_input_kind]
        )
        if feature_store is None:
            feature_store = {kind: [] for kind in active_kinds}
        for kind in active_kinds:
            feature_store[kind].append(
                flatten_prediction_component(prediction, kind).detach().cpu()
            )
        targets.append(z.detach().cpu())

    if feature_store is None:
        raise RuntimeError("No batches were extracted for the latent-code evaluation.")

    features = {
        kind: torch.cat(chunks, dim=0)
        for kind, chunks in feature_store.items()
    }
    return features, torch.cat(targets, dim=0)


def main():
    cli_args = parser.parse_args()
    train_args = load_experiment_args(cli_args.exp_dir)
    dataset_root = cli_args.dataset_root or train_args.dataset_root
    if dataset_root is None:
        raise ValueError("dataset_root could not be resolved from the checkpoint or CLI.")

    device = default_device(cli_args.device)
    adapter = LatentActionFeatureAdapter(cli_args.exp_dir, device=device)
    transform = build_eval_transform(train_args)

    train_dataset = RelativePosePairDataset(
        dataset_root,
        cli_args.train_images_file,
        cli_args.train_labels_file,
        experience=cli_args.target_space,
        size_dataset=cli_args.size_dataset,
        transform=transform,
        pairs_per_object=cli_args.pairs_per_object,
        pair_seed=cli_args.pair_seed,
    )
    val_dataset = RelativePosePairDataset(
        dataset_root,
        cli_args.val_images_file,
        cli_args.val_labels_file,
        experience=cli_args.target_space,
        size_dataset=cli_args.size_dataset,
        transform=transform,
        pairs_per_object=cli_args.pairs_per_object,
        pair_seed=cli_args.pair_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
    )

    train_features, train_targets = extract_split_features(
        adapter, train_loader, cli_args.input_kind
    )
    val_features, val_targets = extract_split_features(
        adapter, val_loader, cli_args.input_kind
    )

    results = {
        "experience": adapter.args.experience,
        "target_space": cli_args.target_space,
        "pairs_per_object": cli_args.pairs_per_object,
        "size_dataset": cli_args.size_dataset,
        "input_kinds": {},
    }
    for kind in train_features:
        metrics = train_probe(
            train_features[kind],
            train_targets,
            val_features[kind],
            val_targets,
            cli_args,
        )
        metrics["feature_dim"] = int(train_features[kind].shape[1])
        results["input_kinds"][kind] = metrics
        print(
            f"{kind}: val_mse={metrics['val']['mse']:.6f} "
            f"val_r2={metrics['val']['r2']:.6f}"
        )
        if "angle_error_deg" in metrics["val"]:
            print(f"{kind}: val_angle_error_deg={metrics['val']['angle_error_deg']:.6f}")

    output_json = cli_args.output_json or (
        cli_args.exp_dir / "latent_action_eval" / "latent_code_reuse.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved latent-code reuse metrics to {output_json}")


if __name__ == "__main__":
    main()
