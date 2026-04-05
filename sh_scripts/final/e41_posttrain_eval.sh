#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --job-name=e41_posttrain_eval
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/e41_posttrain_eval_%j.out
#SBATCH --error=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/e41_posttrain_eval_%j.err
#SBATCH --signal=USR1@60
#SBATCH --open-mode=append

set -euo pipefail

WORK="/cluster/work/projects/nn8106k/$USER/github/LA-SIE"
PY_PKGS="$WORK/.py_pkgs"
USE_LOCAL_PYTHON="${USE_LOCAL_PYTHON:-0}"
LOCAL_PYTHON_BIN="${LOCAL_PYTHON_BIN:-python}"
CONTAINER_PYTHON_BIN="${CONTAINER_PYTHON_BIN:-python}"
CONTAINER="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
DEFAULT_DATASET_ROOT="/cluster/work/projects/nn8106k/$USER/sie/datasets/3DIEBench"
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

PRETRAIN_RUN="${PRETRAIN_RUN:-}"
E41_EXP_SUBDIR="${E41_EXP_SUBDIR:-E41_sharedgen_learned_3v_comp}"
EXP_DIR="/cluster/work/projects/nn8106k/rabab/github/LA-SIE/runs/e41_full_dataset_20260404_094204/E41_sharedgen_learned_3v_comp"

TRAIN_IMAGES_FILE="${TRAIN_IMAGES_FILE:-$WORK/data/train_images.npy}"
TRAIN_LABELS_FILE="${TRAIN_LABELS_FILE:-$WORK/data/train_labels.npy}"
VAL_IMAGES_FILE="${VAL_IMAGES_FILE:-$WORK/data/val_images.npy}"
VAL_LABELS_FILE="${VAL_LABELS_FILE:-$WORK/data/val_labels.npy}"

EVAL_SIZE_DATASET="${EVAL_SIZE_DATASET:--1}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
PROBE_NUM_WORKERS="${PROBE_NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda:0}"

CLASSIF_EPOCHS="${CLASSIF_EPOCHS:-50}"
CLASSIF_LR="${CLASSIF_LR:-1e-3}"
CLASSIF_WD="${CLASSIF_WD:-0.0}"

ANGLE_EPOCHS="${ANGLE_EPOCHS:-50}"
ANGLE_LR="${ANGLE_LR:-1e-3}"
ANGLE_WD="${ANGLE_WD:-0.0}"

COLOR_EPOCHS="${COLOR_EPOCHS:-50}"
COLOR_LR="${COLOR_LR:-1e-3}"
COLOR_WD="${COLOR_WD:-0.0}"

LATENT_DIAG_BATCH_SIZE="${LATENT_DIAG_BATCH_SIZE:-16}"
LATENT_DIAG_NUM_WORKERS="${LATENT_DIAG_NUM_WORKERS:-0}"
LATENT_DIAG_NUM_BATCHES="${LATENT_DIAG_NUM_BATCHES:-20}"

LATENT_CODE_BATCH_SIZE="${LATENT_CODE_BATCH_SIZE:-128}"
LATENT_CODE_NUM_WORKERS="${LATENT_CODE_NUM_WORKERS:-0}"
LATENT_CODE_EPOCHS="${LATENT_CODE_EPOCHS:-50}"
LATENT_CODE_LR="${LATENT_CODE_LR:-1e-3}"
LATENT_CODE_WD="${LATENT_CODE_WD:-0.0}"
LATENT_CODE_HIDDEN_DIM="${LATENT_CODE_HIDDEN_DIM:-256}"
LATENT_CODE_PAIRS_PER_OBJECT="${LATENT_CODE_PAIRS_PER_OBJECT:-16}"
LATENT_CODE_PAIR_SEED="${LATENT_CODE_PAIR_SEED:-0}"
LATENT_TARGET_SPACE="${LATENT_TARGET_SPACE:-quat}"

if [ -n "$PRETRAIN_RUN" ] && [ -z "$EXP_DIR" ]; then
  EXP_DIR="$PRETRAIN_RUN/$E41_EXP_SUBDIR"
fi

if [ -z "$EXP_DIR" ]; then
  echo "Set EXP_DIR to the trained E41 experiment directory, or PRETRAIN_RUN to its parent run directory"
  exit 1
fi

if [ ! -f "$EXP_DIR/params.json" ]; then
  echo "Could not find params.json under $EXP_DIR"
  exit 1
fi

if [ ! -f "$EXP_DIR/model.pth" ]; then
  echo "Could not find model.pth under $EXP_DIR"
  exit 1
fi

mkdir -p "$EXP_DIR/eval" "$EXP_DIR/eval/logs" "$EXP_DIR/latent_action_eval" "$WORK/.wandb_cache" "$PY_PKGS" "$WORK/slurm"

export MASTER_ADDR=127.0.0.1

if [ -f "$HOME/.secrets/wandb.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.secrets/wandb.sh"
  ENABLE_WANDB="${ENABLE_WANDB:-1}"
else
  ENABLE_WANDB="${ENABLE_WANDB:-0}"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-e41_posttrain_eval}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="$EXP_DIR"
export WANDB_CACHE_DIR="$WORK/.wandb_cache"
export WANDB_CONFIG_DIR="/cluster/home/$USER/.config/wandb"

run_project_python() {
  local script_path="$1"
  shift

  if [ "$USE_LOCAL_PYTHON" = "1" ]; then
    (
      cd "$WORK"
      PYTHONPATH="$WORK:${PYTHONPATH:-}" "$LOCAL_PYTHON_BIN" "$WORK/$script_path" "$@"
    )
    return
  fi

  apptainer exec --nv \
    --bind "$WORK":"$WORK" \
    --bind "$PY_PKGS":/ext_pkgs_root \
    --bind "$DATASET_ROOT":"$DATASET_ROOT" \
    "$CONTAINER" \
    env PYTHONPATH=/ext_pkgs_root/wandb_env:${PYTHONPATH:-} \
    "$CONTAINER_PYTHON_BIN" "$WORK/$script_path" "$@"
}

run_python_snippet() {
  local code="$1"
  shift

  if [ "$USE_LOCAL_PYTHON" = "1" ]; then
    (
      cd "$WORK"
      PYTHONPATH="$WORK:${PYTHONPATH:-}" "$LOCAL_PYTHON_BIN" -c "$code" "$@"
    )
    return
  fi

  apptainer exec --nv \
    --bind "$WORK":"$WORK" \
    --bind "$PY_PKGS":/ext_pkgs_root \
    --bind "$DATASET_ROOT":"$DATASET_ROOT" \
    "$CONTAINER" \
    env PYTHONPATH=/ext_pkgs_root/wandb_env:${PYTHONPATH:-} \
    "$CONTAINER_PYTHON_BIN" -c "$code" "$@"
}

json_value() {
  local key="$1"
  run_python_snippet \
    "import json,sys; from pathlib import Path; data=json.load(open(Path(sys.argv[1])/'params.json')); print(data.get(sys.argv[2], ''))" \
    "$EXP_DIR" "$key"
}

EXPERIENCE="$(json_value experience)"
if [ "$EXPERIENCE" != "sharedgen_learned_3v_comp" ]; then
  echo "This script expects sharedgen_learned_3v_comp, but params.json reports $EXPERIENCE"
  exit 1
fi

CHECKPOINT_DATASET_ROOT="$(json_value dataset_root)"
if [ "$DATASET_ROOT" = "$DEFAULT_DATASET_ROOT" ] && [ -n "$CHECKPOINT_DATASET_ROOT" ]; then
  DATASET_ROOT="$CHECKPOINT_DATASET_ROOT"
fi

ARCH="${ARCH:-$(json_value arch)}"
EQUI_DIMS="${EQUI_DIMS:-$(json_value equi)}"
FULL_REPR_DIMS="${FULL_REPR_DIMS:-$(
  run_python_snippet \
    "import sys; import src.resnet as resnet; _, repr_size = resnet.__dict__[sys.argv[1]](zero_init_residual=True); print(repr_size)" \
    "$ARCH"
)}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-$(
  run_python_snippet \
    "from pathlib import Path; import sys; from src.latent_action_eval import resolve_backbone_weights_file; print(resolve_backbone_weights_file(Path(sys.argv[1])))" \
    "$EXP_DIR"
)}"

EVAL_ROOT="${EVAL_ROOT:-$EXP_DIR/eval}"
ROOT_LOG_DIR="${ROOT_LOG_DIR:-$EVAL_ROOT/logs}"
LATENT_EVAL_ROOT="${LATENT_EVAL_ROOT:-$EXP_DIR/latent_action_eval}"

echo "EXP_DIR=$EXP_DIR"
echo "Using package path: $PY_PKGS"
echo "Using container: $CONTAINER"
echo "Using dataset root: $DATASET_ROOT"
echo "Using backbone weights: $BACKBONE_WEIGHTS"
echo "Configured arch=$ARCH equi_dims=$EQUI_DIMS full_repr_dims=$FULL_REPR_DIMS"
echo "Configured eval_size_dataset=$EVAL_SIZE_DATASET probe_batch_size=$PROBE_BATCH_SIZE device=$DEVICE"
echo "Configured classif_epochs=$CLASSIF_EPOCHS angle_epochs=$ANGLE_EPOCHS color_epochs=$COLOR_EPOCHS latent_code_epochs=$LATENT_CODE_EPOCHS"

COMMON_PROBE_ARGS=(
  --arch "$ARCH"
  --dataset-root "$DATASET_ROOT"
  --root-log-dir "$ROOT_LOG_DIR"
  --batch-size "$PROBE_BATCH_SIZE"
  --num-workers "$PROBE_NUM_WORKERS"
  --device "$DEVICE"
  --size-dataset "$EVAL_SIZE_DATASET"
)

if [ "$ENABLE_WANDB" = "1" ]; then
  COMMON_PROBE_ARGS+=(
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-entity "$WANDB_ENTITY"
    --wandb-dir "$EXP_DIR"
  )
fi

CLASSIFICATION_FULL_ARGS=(
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/classification_full"
  --epochs "$CLASSIF_EPOCHS"
  --lr "$CLASSIF_LR"
  --wd "$CLASSIF_WD"
  --train-images-file "$TRAIN_IMAGES_FILE"
  --train-labels-file "$TRAIN_LABELS_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --val-labels-file "$VAL_LABELS_FILE"
  --equi-dims "$FULL_REPR_DIMS"
)

CLASSIFICATION_EQUI_ARGS=(
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/classification_equi"
  --epochs "$CLASSIF_EPOCHS"
  --lr "$CLASSIF_LR"
  --wd "$CLASSIF_WD"
  --train-images-file "$TRAIN_IMAGES_FILE"
  --train-labels-file "$TRAIN_LABELS_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --val-labels-file "$VAL_LABELS_FILE"
  --equi-dims "$EQUI_DIMS"
)

CLASSIFICATION_INV_ARGS=(
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/classification_inv"
  --epochs "$CLASSIF_EPOCHS"
  --lr "$CLASSIF_LR"
  --wd "$CLASSIF_WD"
  --train-images-file "$TRAIN_IMAGES_FILE"
  --train-labels-file "$TRAIN_LABELS_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --val-labels-file "$VAL_LABELS_FILE"
  --equi-dims "$EQUI_DIMS"
  --inv-part
)

ANGLE_FULL_ARGS=(
  --experience quat
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/angle_full"
  --epochs "$ANGLE_EPOCHS"
  --lr "$ANGLE_LR"
  --wd "$ANGLE_WD"
  --deep-end
  --train-images-file "$TRAIN_IMAGES_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --equi-dims "$FULL_REPR_DIMS"
)

ANGLE_EQUI_ARGS=(
  --experience quat
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/angle_equi"
  --epochs "$ANGLE_EPOCHS"
  --lr "$ANGLE_LR"
  --wd "$ANGLE_WD"
  --deep-end
  --train-images-file "$TRAIN_IMAGES_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --equi-dims "$EQUI_DIMS"
)

ANGLE_INV_ARGS=(
  --experience quat
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/angle_inv"
  --epochs "$ANGLE_EPOCHS"
  --lr "$ANGLE_LR"
  --wd "$ANGLE_WD"
  --deep-end
  --train-images-file "$TRAIN_IMAGES_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --equi-dims "$EQUI_DIMS"
  --inv-part
)

COLOR_FULL_ARGS=(
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/color_full"
  --epochs "$COLOR_EPOCHS"
  --lr "$COLOR_LR"
  --wd "$COLOR_WD"
  --deep-end
  --train-images-file "$TRAIN_IMAGES_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --equi-dims "$FULL_REPR_DIMS"
)

COLOR_EQUI_ARGS=(
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/color_equi"
  --epochs "$COLOR_EPOCHS"
  --lr "$COLOR_LR"
  --wd "$COLOR_WD"
  --deep-end
  --train-images-file "$TRAIN_IMAGES_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --equi-dims "$EQUI_DIMS"
)

COLOR_INV_ARGS=(
  --weights-file "$BACKBONE_WEIGHTS"
  --exp-dir "$EVAL_ROOT/color_inv"
  --epochs "$COLOR_EPOCHS"
  --lr "$COLOR_LR"
  --wd "$COLOR_WD"
  --deep-end
  --train-images-file "$TRAIN_IMAGES_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --equi-dims "$EQUI_DIMS"
  --inv-part
)

if [ "$ENABLE_WANDB" = "1" ]; then
  CLASSIFICATION_FULL_ARGS+=(--wandb-name "classification_full_E41")
  CLASSIFICATION_EQUI_ARGS+=(--wandb-name "classification_equi_E41")
  CLASSIFICATION_INV_ARGS+=(--wandb-name "classification_inv_E41")
  ANGLE_FULL_ARGS+=(--wandb-name "angle_full_E41")
  ANGLE_EQUI_ARGS+=(--wandb-name "angle_equi_E41")
  ANGLE_INV_ARGS+=(--wandb-name "angle_inv_E41")
  COLOR_FULL_ARGS+=(--wandb-name "color_full_E41")
  COLOR_EQUI_ARGS+=(--wandb-name "color_equi_E41")
  COLOR_INV_ARGS+=(--wandb-name "color_inv_E41")
fi

run_project_python "eval_classification.py" "${CLASSIFICATION_FULL_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"
run_project_python "eval_classification.py" "${CLASSIFICATION_EQUI_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"
run_project_python "eval_classification.py" "${CLASSIFICATION_INV_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"

run_project_python "eval_angle_prediction.py" "${ANGLE_FULL_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"
run_project_python "eval_angle_prediction.py" "${ANGLE_EQUI_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"
run_project_python "eval_angle_prediction.py" "${ANGLE_INV_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"

run_project_python "eval_color_prediction.py" "${COLOR_FULL_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"
run_project_python "eval_color_prediction.py" "${COLOR_EQUI_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"
run_project_python "eval_color_prediction.py" "${COLOR_INV_ARGS[@]}" "${COMMON_PROBE_ARGS[@]}"

LATENT_DIAGNOSTICS_ARGS=(
  --exp-dir "$EXP_DIR"
  --dataset-root "$DATASET_ROOT"
  --images-file "$VAL_IMAGES_FILE"
  --labels-file "$VAL_LABELS_FILE"
  --device "$DEVICE"
  --batch-size "$LATENT_DIAG_BATCH_SIZE"
  --num-workers "$LATENT_DIAG_NUM_WORKERS"
  --num-batches "$LATENT_DIAG_NUM_BATCHES"
  --size-dataset "$EVAL_SIZE_DATASET"
  --output-json "$LATENT_EVAL_ROOT/diagnostics.json"
)

LATENT_CODE_ARGS=(
  --exp-dir "$EXP_DIR"
  --dataset-root "$DATASET_ROOT"
  --train-images-file "$TRAIN_IMAGES_FILE"
  --train-labels-file "$TRAIN_LABELS_FILE"
  --val-images-file "$VAL_IMAGES_FILE"
  --val-labels-file "$VAL_LABELS_FILE"
  --device "$DEVICE"
  --batch-size "$LATENT_CODE_BATCH_SIZE"
  --num-workers "$LATENT_CODE_NUM_WORKERS"
  --epochs "$LATENT_CODE_EPOCHS"
  --lr "$LATENT_CODE_LR"
  --wd "$LATENT_CODE_WD"
  --hidden-dim "$LATENT_CODE_HIDDEN_DIM"
  --target-space "$LATENT_TARGET_SPACE"
  --input-kind auto
  --pairs-per-object "$LATENT_CODE_PAIRS_PER_OBJECT"
  --pair-seed "$LATENT_CODE_PAIR_SEED"
  --size-dataset "$EVAL_SIZE_DATASET"
  --output-json "$LATENT_EVAL_ROOT/latent_code_reuse.json"
)

run_project_python "eval_latent_action_diagnostics.py" "${LATENT_DIAGNOSTICS_ARGS[@]}"
run_project_python "eval_latent_action_latent_code.py" "${LATENT_CODE_ARGS[@]}"
