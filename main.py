# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from torchvision import transforms

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

#import augmentations as aug

import src.dataset as ds
import src.experience_registry as exp_registry
from src.logging_utils import ScalarLogger, args_to_serializable_dict
import src.models as m

parser = argparse.ArgumentParser()

# Model
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--equi", type=int, default=256)
parser.add_argument("--experience", type=str, choices=exp_registry.ALL_EXPERIENCES, default="SIE")
parser.add_argument("--hypernetwork", type=str, choices=["linear","deep"],default="linear")
# Only for when using an expander
parser.add_argument("--mlp", default="2048-2048-2048")
#Predictor architecture, in format "intermediate1-intermediate2-..."
parser.add_argument("--predictor", default="")
parser.add_argument("--pred-size-in",type=int, default=10)
parser.add_argument("--predictor-relu",  action="store_true")

# Predictor
parser.add_argument("--predictor-type",type=str,choices=["hypernetwork","mlp"],default="hypernetwork")
parser.add_argument("--bias-pred", action="store_true")
parser.add_argument("--bias-hypernet", action="store_true")
parser.add_argument("--simclr-temp",type=float,default=0.1)
parser.add_argument("--ec-weight",type=float,default=1)
parser.add_argument("--tf-num-layers",type=int,default=1)
parser.add_argument("--latent-action-dim", type=int, default=8)
parser.add_argument("--num-generators", type=int, default=8)
parser.add_argument("--latent-operator-hidden-dim", type=int, default=None)
parser.add_argument("--latent-align-weight", type=float, default=None)
parser.add_argument("--latent-identity-weight", type=float, default=1.0)
parser.add_argument("--latent-inverse-weight", type=float, default=1.0)
parser.add_argument("--latent-composition-weight", type=float, default=1.0)
parser.add_argument("--latent-enable-identity", action="store_true")
parser.add_argument("--latent-enable-inverse", action="store_true")
parser.add_argument("--latent-enable-composition", action="store_true")
parser.add_argument("--latent-enable-pred-std", action="store_true")
parser.add_argument("--latent-online-eval", action="store_true")
parser.add_argument("--latent-online-eval-samples", type=int, default=16)



# Optim
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--base-lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-6)

parser.add_argument("--warmup-start",type=int, default=0)
parser.add_argument("--warmup-length",type=int, default=0)


# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
parser.add_argument("--images-file", type=Path, default="./data/train_images.npy", required=True)
parser.add_argument("--labels-file", type=Path, default="./data/val_images.npy", required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--size-dataset", type=int, default=-1)

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--eval-freq", type=int, default=10)
parser.add_argument("--log-freq-time", type=int, default=30)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", ""))
parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY", ""))
parser.add_argument("--wandb-name", type=str, default="")
parser.add_argument("--wandb-dir", type=Path, default=None)

# Loss
parser.add_argument("--sim-coeff", type=float, default=10.0)
parser.add_argument("--equi-factor", type=float, default=4.5)
parser.add_argument("--std-coeff", type=float, default=10.0)
parser.add_argument("--cov-coeff", type=float, default=1.0)

# Running
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--port", type=int, default=52472)



def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        host_name = os.environ["MASTER_ADDR"]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)



def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.root_log_dir.mkdir(parents=True, exist_ok=True)

    # Config dump
    if args.rank == 0:
        dict_args = args_to_serializable_dict(args)
        with open(args.exp_dir / "params.json", 'w') as f:
            json.dump(dict_args, f)

    # Tensorboard setup
    if args.rank == 0:
        if str(args.exp_dir)[-1] == "/":
            exp_name = str(args.exp_dir)[:-1].split("/")[-1]
        else:
            exp_name = str(args.exp_dir).split("/")[-1]
        logdir = args.root_log_dir / exp_name
        writer = SummaryWriter(log_dir=logdir)
        logger = ScalarLogger(
            writer,
            use_wandb=args.wandb,
            project=args.wandb_project or None,
            entity=args.wandb_entity or None,
            name=args.wandb_name or exp_name,
            run_dir=args.wandb_dir if args.wandb_dir is not None else args.exp_dir,
            config=dict_args,
        )

    if args.rank == 0:
        print(" ".join(sys.argv))
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        normalize,
    ])
    if exp_registry.is_triplet_experience(args.experience):
        dataset = ds.Dataset3DIEBenchTriplet(
            args.dataset_root,
            args.images_file,
            args.labels_file,
            size_dataset=args.size_dataset,
            transform=transform,
        )
    elif exp_registry.uses_rotcolor_dataset(args.experience):
        dataset = ds.Dataset3DIEBenchRotColor(
            args.dataset_root,
            args.images_file,
            args.labels_file,
            size_dataset=args.size_dataset,
            transform=transform,
        )
    else:
        dataset = ds.Dataset3DIEBench(
            args.dataset_root,
            args.images_file,
            args.labels_file,
            size_dataset=args.size_dataset,
            transform=transform,
        )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("per_device_batch_size",per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = m.__dict__[args.experience](args).cuda(gpu)
    if args.experience in ["SimCLR","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
        model.gpu = gpu
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=False)

    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.wd
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader, start=epoch * len(loader)):
            if exp_registry.is_triplet_experience(args.experience):
                x0, x1, x2, z01, z12, z02, labels = batch
                x0 = x0.cuda(gpu, non_blocking=True)
                x1 = x1.cuda(gpu, non_blocking=True)
                x2 = x2.cuda(gpu, non_blocking=True)
                z01 = z01.cuda(gpu, non_blocking=True)
                z12 = z12.cuda(gpu, non_blocking=True)
                z02 = z02.cuda(gpu, non_blocking=True)
                labels = labels.cuda(gpu, non_blocking=True)
            else:
                x, y, z, labels = batch
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)
                z = z.cuda(gpu, non_blocking=True)
                labels = labels.cuda(gpu, non_blocking=True)

            
            optimizer.zero_grad()

            # MAIN TRAINING PART
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                if exp_registry.is_triplet_experience(args.experience):
                    loss, classif_loss, stats, stats_eval = model.forward(
                        x0, x1, x2, z01, z12, z02, labels
                    )
                else:
                    loss, classif_loss, stats, stats_eval = model.forward(x, y, z, labels)
                total_loss = loss + classif_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                # General logs
                logger.add_scalar('General/epoch', epoch, step)
                logger.add_scalar('General/time_elapsed', int(current_time - start_time), step)
                #logger.add_scalar('General/lr', lr, step)
                logger.add_scalar('General/lr', args.base_lr, step)
                logger.add_scalar('General/Current GPU memory', torch.cuda.memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)
                logger.add_scalar('General/Max GPU memory', torch.cuda.max_memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)

                # Loss related logs
                logger.add_scalar('Loss/Total loss', stats["loss"].item(), step)
                if args.experience in ["SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor"]:
                    logger.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                if not args.experience in ["SimCLR","SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
                    logger.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                    logger.add_scalar('Loss/Std loss', stats["std_loss"].item(), step)
                    logger.add_scalar('Loss/Covariance loss', stats["cov_loss"].item(), step)
                if not args.experience in ["VICReg","VICRegNoCov","VICRegCos","VICRegL1","VICRegL1repr","FullEqui","VICRegPartInv","SimCLR","VICRegPartInv2Exps","SimCLROnlyEqui","SIERotColor","SimCLROnlyEquiRotColor"] :
                    logger.add_scalar('Loss/Equivariance loss', stats["repr_loss_equi"].item(), step)
                if args.experience in ["SIEOnlyEqui","SIE","SIEAll","SIERotColor"]:
                    logger.add_scalar('Loss/Pred Std loss', stats["pred_std_loss"].item(), step)
                if exp_registry.is_latent_action_experience(args.experience) and args.latent_enable_pred_std:
                    logger.add_scalar('Loss/Pred Std loss', stats["pred_std_loss"].item(), step)
                # Representations/embeddings stats
                logger.add_scalar('Stats/Corr. representations view1', stats["coremb_view1"].item(), step)
                logger.add_scalar('Stats/Corr. representations view2', stats["coremb_view2"].item(), step)
                logger.add_scalar('Stats/Std representations view1', stats["stdemb_view1"].item(), step)
                logger.add_scalar('Stats/Std representations view2', stats["stdemb_view2"].item(), step)
                logger.add_scalar('Stats/Corr. embeddings view1', stats["corhead_view1"].item(), step)
                logger.add_scalar('Stats/Corr. embeddings view2', stats["corhead_view2"].item(), step)
                logger.add_scalar('Stats/Std embeddings view1', stats["stdhead_view1"].item(), step)
                logger.add_scalar('Stats/Std embeddings view2', stats["stdhead_view2"].item(), step)
                if "stdemb_pred" in stats.keys():
                    logger.add_scalar('Stats/Corr. predictor output', stats["coremb_pred"].item(), step)
                    logger.add_scalar('Stats/Std predictor output', stats["stdemb_pred"].item(), step)

                
                # Online evaluation logs
                for key,value in stats_eval.items():
                    if "representations" in key:
                        logger.add_scalar(f'Online eval reprs/{key}', value, step)
                    elif "embeddings" in key:
                        logger.add_scalar(f'Online eval embs/{key}', value, step)
                for key,value in stats.items():
                    if key.startswith("Latent/") or key.startswith("LatentEval/"):
                        logger.add_scalar(key, value, step)
                logger.flush()
                print("Logged, step :",step)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        logger.close()
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "final_weights.pth")


def exclude_bias_and_norm(p):
    return p.ndim == 1

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    main()
