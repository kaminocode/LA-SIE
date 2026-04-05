#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --job-name=sharedgen_3v_comp_final
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/sharedgen_3v_comp_final_%j.out
#SBATCH --error=/cluster/work/projects/nn8106k/%u/github/LA-SIE/slurm/sharedgen_3v_comp_final_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../common_latent_action.sh
source "$SCRIPT_DIR/../common_latent_action.sh"

prepare_latent_action_env
export EPOCHS="${EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export SIZE_DATASET="${SIZE_DATASET:--1}"
run_latent_action_experiment "sharedgen_learned_3v_comp" "sharedgen_learned_3v_comp_final"
