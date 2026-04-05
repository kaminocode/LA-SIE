#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --job-name=e41_1000steps
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/e41_1000steps_%j.out
#SBATCH --error=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/e41_1000steps_%j.err
#SBATCH --signal=USR1@60
#SBATCH --open-mode=append

set -euo pipefail

WORK="/cluster/work/projects/nn8106k/$USER/github/LA-SIE"
RUN="${RUN:-$WORK/runs/e41_1000steps_$(date +%Y%m%d_%H%M%S)}"
PY_PKGS="$WORK/.py_pkgs"
CONTAINER="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
DATASET_ROOT="${DATASET_ROOT:-/cluster/work/projects/nn8106k/$USER/sie/datasets/3DIEBench}"
IMAGES_FILE="${IMAGES_FILE:-$WORK/data/train_images.npy}"
LABELS_FILE="${LABELS_FILE:-$WORK/data/train_labels.npy}"

# Defaults chosen so that on 1 GPU:
# steps_per_epoch = 800 / 32 = 25
# total_steps = 40 * 25 = 1000
SIZE_DATASET="${SIZE_DATASET:-800}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESOLUTION="${RESOLUTION:-256}"
BASE_LR="${BASE_LR:-3e-4}"
EQUI_DIMS="${EQUI_DIMS:-256}"
MLP="${MLP:-2048-2048-2048}"
SIM_COEFF="${SIM_COEFF:-10}"
STD_COEFF="${STD_COEFF:-10}"
COV_COEFF="${COV_COEFF:-1}"
EQUI_FACTOR="${EQUI_FACTOR:-0.45}"
LATENT_ALIGN_WEIGHT="${LATENT_ALIGN_WEIGHT:-4.5}"
LATENT_ACTION_DIM="${LATENT_ACTION_DIM:-8}"
NUM_GENERATORS="${NUM_GENERATORS:-8}"
LATENT_IDENTITY_WEIGHT="${LATENT_IDENTITY_WEIGHT:-1.0}"
LATENT_INVERSE_WEIGHT="${LATENT_INVERSE_WEIGHT:-1.0}"
LATENT_COMPOSITION_WEIGHT="${LATENT_COMPOSITION_WEIGHT:-1.0}"
LOG_FREQ_TIME="${LOG_FREQ_TIME:-15}"
NO_AMP_DEFAULT="${NO_AMP:-1}"

mkdir -p "$RUN" "$RUN/E41_sharedgen_learned_3v_comp" "$RUN/logs" "$WORK/.wandb_cache" "$PY_PKGS" "$WORK/slurm"

export MASTER_ADDR=127.0.0.1

if [ -f "$HOME/.secrets/wandb.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.secrets/wandb.sh"
  ENABLE_WANDB="${ENABLE_WANDB:-1}"
else
  ENABLE_WANDB="${ENABLE_WANDB:-0}"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-e41_debug}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="$RUN"
export WANDB_CACHE_DIR="$WORK/.wandb_cache"
export WANDB_CONFIG_DIR="/cluster/home/$USER/.config/wandb"

STEPS_PER_EPOCH=$(( (SIZE_DATASET + BATCH_SIZE - 1) / BATCH_SIZE ))
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))

echo "RUN=$RUN"
echo "Using package path: $PY_PKGS"
echo "Using container: $CONTAINER"
echo "Using dataset root: $DATASET_ROOT"
echo "Configured size_dataset=$SIZE_DATASET batch_size=$BATCH_SIZE epochs=$EPOCHS"
echo "Configured base_lr=$BASE_LR equi_factor=$EQUI_FACTOR latent_align_weight=$LATENT_ALIGN_WEIGHT"
echo "Configured no_amp=$NO_AMP_DEFAULT"
echo "Expected steps_per_epoch=$STEPS_PER_EPOCH total_steps=$TOTAL_STEPS"

MAIN_ARGS=(
  --experience sharedgen_learned_3v_comp
  --exp-dir "$RUN/E41_sharedgen_learned_3v_comp"
  --root-log-dir "$RUN/logs"
  --epochs "$EPOCHS"
  --arch resnet18
  --equi "$EQUI_DIMS"
  --batch-size "$BATCH_SIZE"
  --base-lr "$BASE_LR"
  --dataset-root "$DATASET_ROOT"
  --images-file "$IMAGES_FILE"
  --labels-file "$LABELS_FILE"
  --resolution "$RESOLUTION"
  --size-dataset "$SIZE_DATASET"
  --num-workers "$NUM_WORKERS"
  --sim-coeff "$SIM_COEFF"
  --std-coeff "$STD_COEFF"
  --cov-coeff "$COV_COEFF"
  --mlp "$MLP"
  --equi-factor "$EQUI_FACTOR"
  --latent-align-weight "$LATENT_ALIGN_WEIGHT"
  --hypernetwork linear
  --latent-action-dim "$LATENT_ACTION_DIM"
  --num-generators "$NUM_GENERATORS"
  --latent-identity-weight "$LATENT_IDENTITY_WEIGHT"
  --latent-inverse-weight "$LATENT_INVERSE_WEIGHT"
  --latent-composition-weight "$LATENT_COMPOSITION_WEIGHT"
  --log-freq-time "$LOG_FREQ_TIME"
)

if [ "$ENABLE_WANDB" = "1" ]; then
  MAIN_ARGS+=(--wandb --wandb-name "E41_1000steps")
fi

if [ "$NO_AMP_DEFAULT" = "1" ]; then
  MAIN_ARGS+=(--no-amp)
fi

if [ "${ENABLE_IDENTITY:-0}" = "1" ]; then
  MAIN_ARGS+=(--latent-enable-identity)
fi

apptainer exec --nv \
  --bind "$WORK":"$WORK" \
  --bind "$PY_PKGS":/ext_pkgs_root \
  --bind "$DATASET_ROOT":"$DATASET_ROOT" \
  "$CONTAINER" \
  env PYTHONPATH=/ext_pkgs_root/wandb_env:${PYTHONPATH:-} \
  python "$WORK/main.py" "${MAIN_ARGS[@]}"
