# 🧠 Olivia Cluster Quick Commands

Minimal cheat sheet for training, monitoring, and logging.

> Replace `<JOB_ID>` and `<run>` accordingly.

---

## 🚀 Submit Training

```bash
sbatch train.sh
```

---

## ⏳ Check Job Status

```bash
squeue --me
```

Detailed job info:
```bash
scontrol show job <JOB_ID>
```

Live resource usage:
```bash
sstat -j <JOB_ID>
```

Finished job stats:
```bash
sacct -j <JOB_ID>
```

---

## 🧪 Debug / Interactive Run

```bash
salloc --nodes=1 --time=00:30:00 --qos=devel \
       --partition=accel --account=nn8106k \
       --gpus=1 --cpus-per-task=32 --mem=80G
```

---

## ❌ Cancel Job

```bash
scancel <JOB_ID>
```

---

## 📦 Dataset Management

Copy from NIRD → work:
```bash
rsync -avh /nird/.../dataset/ /cluster/work/projects/nn8106k/$USER/sie/datasets/
```

Copy results back:
```bash
rsync -avh /cluster/work/projects/nn8106k/$USER/sie/runs/<run>/ /nird/.../archive/
```

---

## 📊 TensorBoard

```bash
tensorboard --logdir /cluster/work/projects/nn8106k/$USER/sie/logs --port 6006
```

SSH tunnel:
```bash
ssh -L 6006:localhost:6006 <username>@olivia.sigma2.no
```

---

## 📈 Weights & Biases (Offline → Sync)

Run (inside job):
```bash
export WANDB_MODE=offline
```

After training (on login node):
```bash
wandb sync /cluster/work/projects/nn8106k/$USER/sie/runs/368655/wandb/offline-run-20260329_151332-wek6axq6
```

---

## 🐳 Run Inside Container (Manual)

```bash
salloc --nodes=1 --time=00:30:00 --qos=devel --partition=accel        --account=nn8106k --mem=40G --cpus-per-task=4 --gpus=1
apptainer exec --nv /cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif \
python script.py
```

---

## 🧹 Clean Old Runs

```bash
rm -rf /cluster/work/projects/nn8106k/$USER/sie/runs/<run>
```

---

## 🏷️ Run Naming Convention

```bash
runs/<JOB_ID>/
runs/<EXP_NAME>_<DATE>/
```

---

## ⚡ Tips

- Always run training on `/cluster/work/projects/...` (not `$HOME`)
- Move important results to NIRD (work storage gets deleted)
- Use `--qos=devel` for quick debugging
- Batch size must be divisible by number of GPUs
