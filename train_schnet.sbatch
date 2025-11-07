#!/bin/bash
#SBATCH -J schnet_logS
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --array=0-4

set -euo pipefail

# ---- user config ----
ENV_ACTIVATE="source .venv/bin/activate"
PY=python
SEEDS=(1337 2025 31415 424242 8675309)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Model hyperparameters
CUT=5.0
FRAMES=5
BS=16
EPOCHS=50
LR=5e-4
HID=128
BLOCKS=5
RBF=64
SPLITS_DIR="$SLURM_SUBMIT_DIR/splits"

# ---- setup ----
mkdir -p "$SLURM_SUBMIT_DIR/logs"
echo "Job $SLURM_JOB_ID on host $(hostname)"
nvidia-smi || true

# Activate env
eval "$ENV_ACTIVATE"

# Make node-local scratch staging dir
SCRATCH="${SLURM_TMPDIR:-/tmp}/schnet_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"

# Stage data to local SSD (fast I/O)
cp -v "$SLURM_SUBMIT_DIR/combined_filtered_structures_with_energy.xyz" "$SCRATCH/"
if [ -f "$SLURM_SUBMIT_DIR/xyz_index.csv" ]; then
  cp -v "$SLURM_SUBMIT_DIR/xyz_index.csv" "$SCRATCH/"
fi

# Always keep small CSVs on submit dir (fine), they're read once
PAIR="$SLURM_SUBMIT_DIR/pair_map.csv"
LABS="$SLURM_SUBMIT_DIR/labels_by_pair.csv"

# Output dir tagged by job id/params/seed
OUTDIR="$SLURM_SUBMIT_DIR/runs/cluster_fpp${FRAMES}_b${BLOCKS}_r${RBF}_cut${CUT//./}_seed${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

# --- optional: one-epoch probe to gauge time ---
$PY train_schnet.py \
  --xyz "$SCRATCH/combined_filtered_structures_with_energy.xyz" \
  --index_csv "$SCRATCH/xyz_index.csv" \
  --pair_map_csv "$PAIR" \
  --labels_csv "$LABS" \
  --frames_per_pair $FRAMES \
  --batch_size $BS \
  --epochs 1 \
  --lr $LR \
  --seed $SEED \
  --hidden $HID --blocks $BLOCKS --rbf $RBF \
  --cutoff $CUT \
  --outdir "$OUTDIR/_probe" \
  --load_splits "$SPLITS_DIR" || true

# --- main run ---
$PY train_schnet.py \
  --xyz "$SCRATCH/combined_filtered_structures_with_energy.xyz" \
  --index_csv "$SCRATCH/xyz_index.csv" \
  --pair_map_csv "$PAIR" \
  --labels_csv "$LABS" \
  --frames_per_pair $FRAMES \
  --batch_size $BS \
  --epochs $EPOCHS \
  --lr $LR \
  --seed $SEED \
  --hidden $HID --blocks $BLOCKS --rbf $RBF \
  --cutoff $CUT \
  --outdir "$OUTDIR" \
  --load_splits "$SPLITS_DIR"

# Copy any artifacts you care about back to submit dir (already in $OUTDIR)
echo "Done. Artifacts in $OUTDIR"
