#!/bin/bash
set -euo pipefail

prepare_latent_action_env() {
  WORK="${WORK:-/cluster/work/projects/nn8106k/$USER/github/LA-SIE}"
  RUN_ROOT="${RUN_ROOT:-$WORK/runs}"
  RUN="${RUN:-$RUN_ROOT/la_sie_$(date +%Y%m%d_%H%M%S)}"
  PY_PKGS="${PY_PKGS:-$WORK/.py_pkgs}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  USE_LOCAL_PYTHON="${USE_LOCAL_PYTHON:-0}"
  CONTAINER="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
  DATASET_ROOT="${DATASET_ROOT:-/cluster/work/projects/nn8106k/$USER/sie/datasets/3DIEBench}"
  IMAGES_FILE="${IMAGES_FILE:-$WORK/data/train_images.npy}"
  LABELS_FILE="${LABELS_FILE:-$WORK/data/train_labels.npy}"
  export WORK RUN PY_PKGS PYTHON_BIN USE_LOCAL_PYTHON CONTAINER DATASET_ROOT IMAGES_FILE LABELS_FILE

  mkdir -p "$RUN" "$RUN/logs" "$WORK/.wandb_cache" "$PY_PKGS" "$WORK/slurm"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export WANDB_DIR="${WANDB_DIR:-$RUN}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WORK/.wandb_cache}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/cluster/home/$USER/.config/wandb}"
  export WANDB_PROJECT="${WANDB_PROJECT:-latent_action}"
  export WANDB_ENTITY="${WANDB_ENTITY:-}"

  if [ -f "$HOME/.secrets/wandb.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.secrets/wandb.sh"
    export ENABLE_WANDB="${ENABLE_WANDB:-1}"
  else
    export ENABLE_WANDB="${ENABLE_WANDB:-0}"
  fi

  echo "RUN=$RUN"
  echo "Using dataset root: $DATASET_ROOT"
  echo "Using container: $CONTAINER"
}

run_project_python() {
  local script_path="$1"
  shift

  if [ "$USE_LOCAL_PYTHON" = "1" ]; then
    (
      cd "$WORK"
      PYTHONPATH="$WORK:${PYTHONPATH:-}" "$PYTHON_BIN" "$WORK/$script_path" "$@"
    )
    return
  fi

  apptainer exec --nv \
    --bind "$WORK":"$WORK" \
    --bind "$PY_PKGS":/ext_pkgs_root \
    --bind "$DATASET_ROOT":"$DATASET_ROOT" \
    "$CONTAINER" \
    env PYTHONPATH=/ext_pkgs_root/wandb_env:${PYTHONPATH:-} \
    "$PYTHON_BIN" "$WORK/$script_path" "$@"
}


run_latent_action_experiment() {
  local experience="$1"
  local exp_name="$2"
  shift 2

  local exp_dir="$RUN/$exp_name"
  mkdir -p "$exp_dir"

  local -a main_args=(
    --experience "$experience"
    --exp-dir "$exp_dir"
    --root-log-dir "$RUN/logs"
    --arch "${ARCH:-resnet18}"
    --equi "${EQUI_DIMS:-256}"
    --batch-size "${BATCH_SIZE:-64}"
    --base-lr "${BASE_LR:-1e-3}"
    --epochs "${EPOCHS:-1}"
    --dataset-root "$DATASET_ROOT"
    --images-file "$IMAGES_FILE"
    --labels-file "$LABELS_FILE"
    --sim-coeff "${SIM_COEFF:-10}"
    --std-coeff "${STD_COEFF:-10}"
    --cov-coeff "${COV_COEFF:-1}"
    --mlp "${MLP:-2048-2048-2048}"
    --equi-factor "${EQUI_FACTOR:-0.45}"
    --hypernetwork "${HYPERNETWORK:-linear}"
    --latent-action-dim "${LATENT_ACTION_DIM:-8}"
    --num-generators "${NUM_GENERATORS:-8}"
    --latent-identity-weight "${LATENT_IDENTITY_WEIGHT:-1.0}"
    --latent-inverse-weight "${LATENT_INVERSE_WEIGHT:-1.0}"
    --latent-composition-weight "${LATENT_COMPOSITION_WEIGHT:-1.0}"
    --size-dataset "${SIZE_DATASET:-64}"
    --num-workers "${NUM_WORKERS:-2}"
    --no-amp
  )

  if [ "$ENABLE_WANDB" = "1" ]; then
    main_args+=(--wandb --wandb-name "$exp_name")
  fi

  main_args+=("$@")
  run_project_python "main.py" "${main_args[@]}"
}
