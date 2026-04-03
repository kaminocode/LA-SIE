#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --job-name=sie_4gpu
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/sie_%j.out
#SBATCH --error=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/sie_%j.err
#SBATCH --signal=USR1@60
#SBATCH --open-mode=append

set -euo pipefail

WORK="/cluster/work/projects/nn8106k/$USER/github/LA-SIE"
RUN="${RUN:-$WORK/runs/la_sie_$(date +%Y%m%d_%H%M%S)}"
PY_PKGS="$WORK/.py_pkgs"
CONTAINER="/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif"
DATASET_ROOT="/cluster/work/projects/nn8106k/$USER/sie/datasets/3DIEBench"
IMAGES_FILE="$WORK/data/train_images.npy"
LABELS_FILE="$WORK/data/train_labels.npy"

mkdir -p "$RUN" "$RUN/SIE" "$RUN/logs" "$WORK/.wandb_cache" "$PY_PKGS"

export MASTER_ADDR=127.0.0.1

# W&B
source ~/.secrets/wandb.sh
export WANDB_MODE=online
export WANDB_PROJECT="test"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="$RUN"
export WANDB_CACHE_DIR="$WORK/.wandb_cache"
export WANDB_CONFIG_DIR="/cluster/home/$USER/.config/wandb"

echo "RUN=$RUN"
echo "Using package path: $PY_PKGS"
echo "Using container: $CONTAINER"
echo "Using dataset root: $DATASET_ROOT"

MAIN_ARGS=(
  --experience SIE
  --exp-dir "$RUN/SIE"
  --root-log-dir "$RUN/logs"
  --epochs 10
  --arch resnet18
  --equi 256
  --batch-size 256
  --base-lr 1e-3
  --dataset-root "$DATASET_ROOT"
  --images-file "$IMAGES_FILE"
  --labels-file "$LABELS_FILE"
  --sim-coeff 10
  --std-coeff 10
  --cov-coeff 1
  --mlp 2048-2048-2048
  --equi-factor 0.45
  --hypernetwork linear
)

apptainer exec --nv \
  --bind "$WORK":"$WORK" \
  --bind "$PY_PKGS":/ext_pkgs_root \
  --bind "$DATASET_ROOT":"$DATASET_ROOT" \
  "$CONTAINER" \
  env PYTHONPATH=/ext_pkgs_root/wandb_env:${PYTHONPATH:-} \
  python "$WORK/main.py" "${MAIN_ARGS[@]}"
