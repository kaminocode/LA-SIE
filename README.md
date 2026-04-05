# Latent-Action Experiments on 3DIEBench

This repository contains a label-free latent-action experiment family for learning relative operators between views of the same object in 3DIEBench.

The current implementation covers pair and triplet training, online monitoring during pretraining, and post-train evaluation scripts for both generic representation probes and latent-action-specific diagnostics.

## What Is In This Repo

- Pair latent-action experiments that learn an operator from two views.
- Triplet latent-action experiments that add inverse and composition structure.
- Shared-generator and direct-operator variants.
- Built-in online evaluation during training.
- Post-train evaluation scripts for classification, relative rotation, relative color, latent diagnostics, and latent-code reuse.
- SLURM-ready scripts for smoke, debug, and longer runs.

The latent-action training losses do not use the known group element `z`. Relative pose targets are used only for online evaluation, diagnostics, logging, and sanity checks.

## Experiment Ladder

Implemented experiments:

| Code | `--experience` | Views | Idea | Extra structure |
| --- | --- | --- | --- | --- |
| `E00` | `sie_oracle` | 2 | Baseline alias kept for regression | none |
| `E10` | `direct_full_matrix_2v` | 2 | Predict a full pair-specific operator directly | none |
| `E11` | `direct_skewexp_2v` | 2 | Predict a skew-symmetric matrix and exponentiate it | invertible-style operator |
| `E12` | `latentcode_to_full_matrix_2v` | 2 | Predict a latent code, then decode a full matrix | bottlenecked operator |
| `E20` | `sharedgen_fixed_2v` | 2 | Predict coefficients over fixed shared generators | shared basis |
| `E21` | `sharedgen_learned_2v` | 2 | Predict coefficients over learned shared generators | shared basis |
| `E30` | `sharedgen_learned_2v_identity` | 2 | Learned shared generators | identity loss enabled by default |
| `E31` | `sharedgen_learned_2v_inverse` | 2 | Learned shared generators | inverse loss enabled by default |
| `E40` | `sharedgen_learned_3v_no_comp` | 3 | Triplet training with shared learned generators | inverse loss enabled by default |
| `E41` | `sharedgen_learned_3v_comp` | 3 | Triplet training with shared learned generators | inverse + composition enabled by default |

Planned but not yet implemented as named experiments:

- `E50`: action input ablations
- `E51`: dedicated generator-count ablation names
- `E52`: generator-type ablations beyond the current skew-symmetric setup
- `E53`: bridge loss ablation

## Data Contract

Training expects 3DIEBench-style object folders with images and per-view latent files:

```text
DATASET_ROOT/
  object_x/
    image_0.jpg
    ...
    image_49.jpg
    latent_0.npy
    ...
    latent_49.npy
```

Index files in [`data/`](data) point to the train and validation objects:

- `data/train_images.npy`
- `data/train_labels.npy`
- `data/val_images.npy`
- `data/val_labels.npy`

Pair experiments sample:

```text
(x, y, z, label)
```

Triplet experiments sample:

```text
(x0, x1, x2, z01, z12, z02, label)
```

For latent-action experiments, `z`, `z01`, `z12`, and `z02` are evaluation-only targets. They are not fed into the latent training losses.

`--size-dataset` is useful for smoke and debug runs:

- positive integer: truncate the dataset
- `-1`: use the full dataset

## Training Objective

All latent-action models keep the split representation structure:

- invariant branch
- equivariant branch

Latent operators act only on the equivariant branch.

### Pair Objective

For pair experiments, the total latent loss is:

```text
L_pair =
  sim_coeff * L_inv_repr
  + align_weight * L_align
  + std_coeff * L_std
  + cov_coeff * L_cov
  + [pred-std] * std_coeff * L_pred_std
  + [identity] * latent_identity_weight * L_identity
  + [inverse] * latent_inverse_weight * L_inverse
```

Where:

- `L_inv_repr`: MSE between projected invariant views
- `L_align`: MSE between the transported source equivariant features and the target equivariant features
- `L_std`: per-dimension variance regularizer on projected views
- `L_cov`: covariance regularizer on projected views
- `L_pred_std`: optional variance regularizer on transported equivariant predictions
- `L_identity`: identity consistency for a self-pair operator
- `L_inverse`: inverse consistency between forward and backward operators

### Triplet Objective

For triplet experiments, the loss extends the pair setup with three-view alignment and optional composition:

```text
L_triplet =
  sim_coeff * L_inv_repr
  + align_weight * L_align
  + std_coeff * L_std
  + cov_coeff * L_cov
  + [pred-std] * std_coeff * L_pred_std
  + [identity] * latent_identity_weight * L_identity
  + [inverse] * latent_inverse_weight * L_inverse
  + [composition] * latent_composition_weight * L_comp
```

Where:

- `L_align` averages alignment over `0->1`, `1->2`, and `0->2`
- `L_identity` averages self-operator identity errors
- `L_inverse` averages forward/backward inverse errors
- `L_comp` matches the direct `0->2` operator against the composition of `0->1` and `1->2`

### Operator Families

- `direct_full_matrix_2v`: predicts a dense operator directly
- `direct_skewexp_2v`: predicts a raw matrix, projects it to skew-symmetric form, then uses `matrix_exp`
- `latentcode_to_full_matrix_2v`: predicts a compact latent code before decoding an operator
- `sharedgen_*`: predicts coefficients over a shared generator bank; the current generator family is skew-symmetric

## Online Evaluation During Training

Training logs three groups of metrics:

- `Loss/*`: core optimization terms
- `Latent/*`: latent-action losses and operator diagnostics
- `Online eval reprs/*` and `Online eval embs/*`: detached probe metrics

The built-in online evaluator trains lightweight heads on detached features and logs:

- object classification cross-entropy, top-1, and top-5
- relative rotation prediction MSE and `R2`
- both representation-space and projected-embedding-space versions
- invariant-only classification when an invariant branch exists
- equivariant-only rotation prediction when an equivariant branch exists

Triplet experiments currently reuse the online evaluator on `(x0, x1, z01)`.

### Latent-Specific Online Metrics

Latent-action training also logs:

- `Latent/alignment_loss`
- `Latent/identity_loss`
- `Latent/inverse_loss`
- `Latent/composition_loss`
- `Latent/operator_norm`
- `Latent/code_norm`
- `Latent/coeff_norm`
- `Latent/generator_norm`
- `Latent/pred_std_loss` when enabled

Optional cross-image transfer monitoring is available with:

- `--latent-online-eval`
- `--latent-online-eval-samples`

When enabled, training logs:

- `LatentEval/transfer_same_class_mse`
- `LatentEval/transfer_diff_class_mse`

These metrics reuse operators from donor pairs and apply them to matched recipient pairs with similar or dissimilar action targets.

### Logging Note

For latent-action runs, `inverse` and `composition` are logged under `Latent/*`. They are not currently mirrored under `Loss/*`.

## Checkpoints

Each training run writes:

- `model.pth`: full training checkpoint
- `final_weights.pth`: backbone-only weights for post-train probe scripts
- `params.json`: serialized CLI arguments for the run

Use `final_weights.pth` for classification, angle, and color probes. Use the run directory itself for latent diagnostics and latent-code evaluation.

## Quick Start

### Requirements

The code expects:

- Python 3
- PyTorch
- torchvision
- NumPy
- SciPy
- Pillow
- TensorBoard
- W&B optionally
- Apptainer optionally for the provided cluster scripts

GPU training is expected. `main.py` launches one process per visible GPU.

For a single-GPU local run, set:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Pair Training Example

This is a small local run for `direct_skewexp_2v`:

```bash
python main.py \
  --experience direct_skewexp_2v \
  --exp-dir runs/direct_skewexp_2v_smoke \
  --root-log-dir runs/logs \
  --dataset-root /path/to/3DIEBench \
  --images-file ./data/train_images.npy \
  --labels-file ./data/train_labels.npy \
  --arch resnet18 \
  --equi 256 \
  --batch-size 32 \
  --epochs 5 \
  --size-dataset 256 \
  --base-lr 3e-4 \
  --sim-coeff 10 \
  --std-coeff 10 \
  --cov-coeff 1 \
  --mlp 2048-2048-2048 \
  --equi-factor 4.5 \
  --latent-operator-hidden-dim 1024 \
  --latent-enable-pred-std \
  --latent-enable-identity \
  --latent-enable-inverse \
  --latent-online-eval \
  --latent-online-eval-samples 16 \
  --no-amp
```

### Triplet Training Example

This is a small local run for `sharedgen_learned_3v_comp`:

```bash
python main.py \
  --experience sharedgen_learned_3v_comp \
  --exp-dir runs/e41_smoke \
  --root-log-dir runs/logs \
  --dataset-root /path/to/3DIEBench \
  --images-file ./data/train_images.npy \
  --labels-file ./data/train_labels.npy \
  --arch resnet18 \
  --equi 256 \
  --batch-size 32 \
  --epochs 5 \
  --size-dataset 256 \
  --base-lr 3e-4 \
  --sim-coeff 10 \
  --std-coeff 10 \
  --cov-coeff 1 \
  --mlp 2048-2048-2048 \
  --equi-factor 4.5 \
  --latent-align-weight 4.5 \
  --latent-action-dim 8 \
  --num-generators 8 \
  --latent-inverse-weight 1.0 \
  --latent-composition-weight 1.0 \
  --latent-online-eval \
  --latent-online-eval-samples 16 \
  --no-amp
```

`sharedgen_learned_3v_comp` is currently more stable with `--no-amp`, especially on the ARM cluster setup used by the provided scripts.

### Useful Latent Flags

- `--latent-action-dim`: latent code size used by bottlenecked and shared-generator operators
- `--num-generators`: number of shared generators for `sharedgen_*`
- `--latent-operator-hidden-dim`: hidden size for direct operator heads
- `--latent-align-weight`: explicit alignment weight override; if omitted, alignment defaults to `equi_factor * sim_coeff`
- `--latent-identity-weight`: weight for identity consistency
- `--latent-inverse-weight`: weight for inverse consistency
- `--latent-composition-weight`: weight for composition consistency
- `--latent-enable-identity`: add identity loss for experiments that do not enable it by default
- `--latent-enable-inverse`: add inverse loss for experiments that do not enable it by default
- `--latent-enable-composition`: add composition loss for experiments that do not enable it by default
- `--latent-enable-pred-std`: add variance regularization on transported predictions
- `--latent-online-eval`: enable cross-image transfer metrics
- `--latent-online-eval-samples`: donor count used by cross-image transfer monitoring
- `--size-dataset`: truncate the dataset for debug runs
- `--no-amp`: disable AMP; recommended for E41 debug and cluster runs

Some named experiments already enable losses by construction:

- `sharedgen_learned_2v_identity`: identity
- `sharedgen_learned_2v_inverse`: inverse
- `sharedgen_learned_3v_no_comp`: inverse
- `sharedgen_learned_3v_comp`: inverse and composition

## Cluster Scripts

Ready-made launchers live in [`sh_scripts/`](sh_scripts):

- [`sh_scripts/e41_1000steps.sh`](sh_scripts/e41_1000steps.sh): small debug run for `sharedgen_learned_3v_comp`
- [`sh_scripts/e41_full_dataset.sh`](sh_scripts/e41_full_dataset.sh): longer E41 run on the full dataset
- [`sh_scripts/final/direct_skewexp_2v.sh`](sh_scripts/final/direct_skewexp_2v.sh): longer pair run
- [`sh_scripts/final/sharedgen_learned_3v_comp.sh`](sh_scripts/final/sharedgen_learned_3v_comp.sh): longer triplet run
- [`sh_scripts/final/e41_posttrain_eval.sh`](sh_scripts/final/e41_posttrain_eval.sh): bundled post-train evaluation for an E41 checkpoint

Example SLURM submissions:

```bash
sbatch sh_scripts/e41_1000steps.sh
sbatch sh_scripts/final/direct_skewexp_2v.sh
sbatch sh_scripts/final/sharedgen_learned_3v_comp.sh
```

## Post-Train Evaluation

### 1. Classification Probe

Train a linear or MLP classifier on frozen backbone features.

Full representation:

```bash
python eval_classification.py \
  --weights-file runs/e41_smoke/final_weights.pth \
  --dataset-root /path/to/3DIEBench \
  --train-images-file ./data/train_images.npy \
  --train-labels-file ./data/train_labels.npy \
  --val-images-file ./data/val_images.npy \
  --val-labels-file ./data/val_labels.npy \
  --exp-dir runs/e41_smoke/eval/classification_full \
  --root-log-dir runs/e41_smoke/eval/logs \
  --arch resnet18 \
  --equi-dims 512 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --wd 0.0 \
  --device cuda:0
```

Equivariant-only part: set `--equi-dims` to the equivariant width and omit `--inv-part`.

Invariant-only part: keep the same `--equi-dims` and add `--inv-part`.

### 2. Relative Rotation Probe

Predict relative pose between two views from frozen features.

```bash
python eval_angle_prediction.py \
  --experience quat \
  --weights-file runs/e41_smoke/final_weights.pth \
  --dataset-root /path/to/3DIEBench \
  --train-images-file ./data/train_images.npy \
  --val-images-file ./data/val_images.npy \
  --exp-dir runs/e41_smoke/eval/angle_full \
  --root-log-dir runs/e41_smoke/eval/logs \
  --arch resnet18 \
  --equi-dims 512 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --wd 0.0 \
  --device cuda:0 \
  --deep-end
```

Use `--inv-part` to probe the invariant slice instead, or set `--equi-dims` to the equivariant width to probe only that branch.

### 3. Relative Color Probe

Predict relative color change between two views from frozen features.

```bash
python eval_color_prediction.py \
  --weights-file runs/e41_smoke/final_weights.pth \
  --dataset-root /path/to/3DIEBench \
  --train-images-file ./data/train_images.npy \
  --val-images-file ./data/val_images.npy \
  --exp-dir runs/e41_smoke/eval/color_full \
  --root-log-dir runs/e41_smoke/eval/logs \
  --arch resnet18 \
  --equi-dims 512 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --wd 0.0 \
  --device cuda:0 \
  --deep-end
```

### 4. Latent Diagnostics

Re-run a trained latent-action checkpoint over a few batches and average the latent metrics written during training:

```bash
python eval_latent_action_diagnostics.py \
  --exp-dir runs/e41_smoke \
  --dataset-root /path/to/3DIEBench \
  --images-file ./data/val_images.npy \
  --labels-file ./data/val_labels.npy \
  --device cuda:0 \
  --batch-size 16 \
  --num-workers 0 \
  --num-batches 20 \
  --output-json runs/e41_smoke/latent_action_eval/diagnostics.json
```

This writes averaged `Latent/*` and `LatentEval/*` metrics to JSON.

### 5. Latent-Code Reuse Probe

Train a small regression probe from the learned operator internals back to relative pose.

```bash
python eval_latent_action_latent_code.py \
  --exp-dir runs/e41_smoke \
  --dataset-root /path/to/3DIEBench \
  --train-images-file ./data/train_images.npy \
  --train-labels-file ./data/train_labels.npy \
  --val-images-file ./data/val_images.npy \
  --val-labels-file ./data/val_labels.npy \
  --device cuda:0 \
  --batch-size 128 \
  --epochs 50 \
  --lr 1e-3 \
  --wd 0.0 \
  --hidden-dim 256 \
  --target-space quat \
  --input-kind auto \
  --pairs-per-object 16 \
  --pair-seed 0 \
  --output-json runs/e41_smoke/latent_action_eval/latent_code_reuse.json
```

`--input-kind auto` uses the most natural internal representation for the checkpoint:

- `coefficients` when available
- otherwise `code`
- otherwise `raw_matrix`
- otherwise `operator`

The output JSON includes:

- `mse`
- `r2`
- `angle_error_deg` for quaternion targets
- selected epoch and feature dimensionality

### Bundled E41 Evaluation

To run the full E41 post-train bundle:

```bash
sbatch --export=ALL,EXP_DIR=/path/to/run/E41_sharedgen_learned_3v_comp \
  sh_scripts/final/e41_posttrain_eval.sh
```

That script runs:

- classification probes on full, equivariant, and invariant features
- relative rotation probes on full, equivariant, and invariant features
- relative color probes on full, equivariant, and invariant features
- latent diagnostics
- latent-code reuse evaluation

Outputs are written under:

- `EXP_DIR/eval/`
- `EXP_DIR/latent_action_eval/`

## Code Map

Main latent-action entry points:

- [`main.py`](main.py): CLI, training loop, logging, dataset routing
- [`src/experience_registry.py`](src/experience_registry.py): canonical experiment list
- [`src/dataset.py`](src/dataset.py): pair and triplet datasets
- [`src/latent_action.py`](src/latent_action.py): operator modules and algebraic helpers
- [`src/latent_action_models.py`](src/latent_action_models.py): pair/triplet model definitions and losses
- [`src/latent_action_online_eval.py`](src/latent_action_online_eval.py): cross-image transfer metrics
- [`src/latent_action_eval.py`](src/latent_action_eval.py): checkpoint loading and evaluation helpers
- [`eval_latent_action_diagnostics.py`](eval_latent_action_diagnostics.py): post-train latent diagnostics
- [`eval_latent_action_latent_code.py`](eval_latent_action_latent_code.py): latent-code reuse probe

## Notes

- The README above focuses on the latent-action workflow in this repository.
- Legacy retrieval and prediction-error scripts remain in the tree, but they are not part of the current latent-action evaluation path documented here.
- Keep an eye on `Latent/*` metrics in TensorBoard or W&B for inverse and composition behavior; those terms are more informative there than under `Loss/*`.

## License

Code in this repository is released under GPL v3.0. See [LICENSE](LICENSE).

The dataset assets under [`data/`](data) carry their own license notes; see [`data/LICENSE`](data/LICENSE).
