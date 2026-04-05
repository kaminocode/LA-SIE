import argparse
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

import src.experience_registry as exp_registry
from src.latent_action_eval import (
    build_training_dataset,
    default_device,
    load_checkpoint_model,
    load_experiment_args,
)


parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", type=Path, required=True)
parser.add_argument("--dataset-root", type=Path, default=None)
parser.add_argument("--images-file", type=Path, default=None)
parser.add_argument("--labels-file", type=Path, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--num-batches", type=int, default=10)
parser.add_argument("--size-dataset", type=int, default=-1)
parser.add_argument("--output-json", type=Path, default=None)

def _init_dist(device):
    backend = "nccl" if str(device).startswith("cuda") else "gloo"
    if dist.is_initialized():
        return
    with tempfile.NamedTemporaryFile() as tmp:
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{tmp.name}",
            world_size=1,
            rank=0,
        )

def _to_device(batch, device):
    return [item.to(device) if torch.is_tensor(item) else item for item in batch]


def main():
    cli_args = parser.parse_args()
    args = load_experiment_args(cli_args.exp_dir)
    args.dataset_root = cli_args.dataset_root or args.dataset_root
    args.images_file = cli_args.images_file or args.images_file
    args.labels_file = cli_args.labels_file or args.labels_file
    args.size_dataset = cli_args.size_dataset if cli_args.size_dataset > 0 else args.size_dataset
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    device = default_device(cli_args.device)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    _init_dist(device)

    if not exp_registry.is_latent_action_experience(args.experience):
        raise ValueError(f"{args.experience} is not a latent-action experiment.")

    dataset = build_training_dataset(args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
    )

    _, model, msg = load_checkpoint_model(
        cli_args.exp_dir,
        device=device,
        require_latent_action=True,
        strict=False,
    )
    print(f"Loaded checkpoint with msg: {msg}")

    metrics = {}
    count = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            batch = _to_device(batch, device)
            if exp_registry.is_triplet_experience(args.experience):
                _, _, stats, _ = model.forward(*batch)
            else:
                _, _, stats, _ = model.forward(*batch)
            for key, value in stats.items():
                if key.startswith("Latent/") or key.startswith("LatentEval/"):
                    metrics[key] = metrics.get(key, 0.0) + float(value.detach().cpu())
            count += 1
            if batch_id + 1 >= cli_args.num_batches:
                break

    if count == 0:
        raise RuntimeError("No batches were evaluated.")

    metrics = {key: value / float(count) for key, value in metrics.items()}
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6f}")

    output_json = cli_args.output_json or (cli_args.exp_dir / "latent_action_eval" / "diagnostics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Saved diagnostics to {output_json}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
