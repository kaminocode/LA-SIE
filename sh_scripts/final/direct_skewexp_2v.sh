#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --job-name=direct_skewexp_2v_final
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/direct_skewexp_2v_final_%j.out
#SBATCH --error=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/direct_skewexp_2v_final_%j.err
#SBATCH --signal=USR1@60
#SBATCH --open-mode=append

set -euo pipefail

WORK="${WORK:-/cluster/work/projects/nn8106k/$USER/github/LA-SIE}"
RUN_ROOT="${RUN_ROOT:-$WORK/runs}"
RUN="${RUN:-$RUN_ROOT/direct_skewexp_2v_final_$(date +%Y%m%d_%H%M%S)}"
PY_PKGS="${PY_PKGS:-$WORK/.py_pkgs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_LOCAL_PYTHON="${USE_LOCAL_PYTHON:-0}"
CONTAINER="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
DATASET_ROOT="${DATASET_ROOT:-/cluster/work/projects/nn8106k/$USER/sie/datasets/3DIEBench}"
IMAGES_FILE="${IMAGES_FILE:-$WORK/data/train_images.npy}"
LABELS_FILE="${LABELS_FILE:-$WORK/data/train_labels.npy}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SIZE_DATASET="${SIZE_DATASET:--1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RESOLUTION="${RESOLUTION:-256}"
BASE_LR="${BASE_LR:-2e-3}"
ARCH="${ARCH:-resnet18}"
EQUI_DIMS="${EQUI_DIMS:-256}"
MLP="${MLP:-2048-2048-2048}"
SIM_COEFF="${SIM_COEFF:-10}"
STD_COEFF="${STD_COEFF:-10}"
COV_COEFF="${COV_COEFF:-1}"
EQUI_FACTOR="${EQUI_FACTOR:-4.5}"
HYPERNETWORK="${HYPERNETWORK:-linear}"
LATENT_ACTION_DIM="${LATENT_ACTION_DIM:-8}"
NUM_GENERATORS="${NUM_GENERATORS:-8}"
LATENT_OPERATOR_HIDDEN_DIM="${LATENT_OPERATOR_HIDDEN_DIM:-1024}"
LATENT_IDENTITY_WEIGHT="${LATENT_IDENTITY_WEIGHT:-1.0}"
LATENT_INVERSE_WEIGHT="${LATENT_INVERSE_WEIGHT:-1.0}"
LATENT_COMPOSITION_WEIGHT="${LATENT_COMPOSITION_WEIGHT:-1.0}"
LOG_FREQ_TIME="${LOG_FREQ_TIME:-30}"
NO_AMP_DEFAULT="${NO_AMP:-1}"
LATENT_ONLINE_EVAL="${LATENT_ONLINE_EVAL:-0}"
LATENT_ONLINE_EVAL_SAMPLES="${LATENT_ONLINE_EVAL_SAMPLES:-16}"

# Inverse consistency is logged either way; this toggle only decides
# whether it contributes to the training objective.
ENABLE_PRED_STD="${ENABLE_PRED_STD:-1}"
ENABLE_IDENTITY="${ENABLE_IDENTITY:-1}"
ENABLE_INVERSE="${ENABLE_INVERSE:-1}"

mkdir -p "$RUN" "$RUN/direct_skewexp_2v_final" "$RUN/logs" "$WORK/.wandb_cache" "$PY_PKGS" "$WORK/slurm"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [ -f "$HOME/.secrets/wandb.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.secrets/wandb.sh"
  ENABLE_WANDB="${ENABLE_WANDB:-1}"
else
  ENABLE_WANDB="${ENABLE_WANDB:-0}"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-direct_skewexp_2v_final}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="${WANDB_DIR:-$RUN}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WORK/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/cluster/home/$USER/.config/wandb}"

echo "RUN=$RUN"
echo "Using container: $CONTAINER"
echo "Using dataset root: $DATASET_ROOT"
echo "Configured size_dataset=$SIZE_DATASET batch_size=$BATCH_SIZE epochs=$EPOCHS"
echo "Configured base_lr=$BASE_LR equi_factor=$EQUI_FACTOR"
echo "Configured latent_operator_hidden_dim=$LATENT_OPERATOR_HIDDEN_DIM"
echo "Configured toggles: pred_std=$ENABLE_PRED_STD identity=$ENABLE_IDENTITY inverse=$ENABLE_INVERSE"
echo "Configured latent_online_eval=$LATENT_ONLINE_EVAL samples=$LATENT_ONLINE_EVAL_SAMPLES"
echo "Configured no_amp=$NO_AMP_DEFAULT"

MAIN_ARGS=(
  --experience direct_skewexp_2v
  --exp-dir "$RUN/direct_skewexp_2v_final"
  --root-log-dir "$RUN/logs"
  --arch "$ARCH"
  --equi "$EQUI_DIMS"
  --batch-size "$BATCH_SIZE"
  --base-lr "$BASE_LR"
  --epochs "$EPOCHS"
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
  --hypernetwork "$HYPERNETWORK"
  --latent-action-dim "$LATENT_ACTION_DIM"
  --num-generators "$NUM_GENERATORS"
  --latent-operator-hidden-dim "$LATENT_OPERATOR_HIDDEN_DIM"
  --latent-identity-weight "$LATENT_IDENTITY_WEIGHT"
  --latent-inverse-weight "$LATENT_INVERSE_WEIGHT"
  --latent-composition-weight "$LATENT_COMPOSITION_WEIGHT"
  --log-freq-time "$LOG_FREQ_TIME"
)

if [ "$ENABLE_WANDB" = "1" ]; then
  MAIN_ARGS+=(--wandb --wandb-name "direct_skewexp_2v_final")
fi

if [ "$NO_AMP_DEFAULT" = "1" ]; then
  MAIN_ARGS+=(--no-amp)
fi

if [ "$ENABLE_PRED_STD" = "1" ]; then
  MAIN_ARGS+=(--latent-enable-pred-std)
fi

if [ "$ENABLE_IDENTITY" = "1" ]; then
  MAIN_ARGS+=(--latent-enable-identity)
fi

if [ "$ENABLE_INVERSE" = "1" ]; then
  MAIN_ARGS+=(--latent-enable-inverse)
fi

if [ "$LATENT_ONLINE_EVAL" = "1" ]; then
  MAIN_ARGS+=(--latent-online-eval --latent-online-eval-samples "$LATENT_ONLINE_EVAL_SAMPLES")
fi

if [ "$USE_LOCAL_PYTHON" = "1" ]; then
  (
    cd "$WORK"
    PYTHONPATH="$WORK:${PYTHONPATH:-}" "$PYTHON_BIN" "$WORK/main.py" "${MAIN_ARGS[@]}"
  )
else
  apptainer exec --nv \
    --bind "$WORK":"$WORK" \
    --bind "$PY_PKGS":/ext_pkgs_root \
    --bind "$DATASET_ROOT":"$DATASET_ROOT" \
    "$CONTAINER" \
    env PYTHONPATH=/ext_pkgs_root/wandb_env:${PYTHONPATH:-} \
    "$PYTHON_BIN" "$WORK/main.py" "${MAIN_ARGS[@]}"
fi
