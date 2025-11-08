#!/usr/bin/env bash
set -euo pipefail

# ======= user paths =======
PYTHON="${PYTHON:-python}"               
DATA_DIR="data"
XYZ="${DATA_DIR}/combined_filtered_structures_with_energy.xyz"
INDEX="${DATA_DIR}/xyz_index.csv"
PAIR_MAP="${DATA_DIR}/pair_map.csv"
LABELS="${DATA_DIR}/labels_by_pair.csv"

RUNS="schnet_runs"
SPLITS_DIR="${RUNS}/probe/splits"
NUM_WORKERS="${NUM_WORKERS:-8}"          # tune if the cluster prefers fewer

mkdir -p "${RUNS}"

# sanity: CUDA?
${PYTHON} - << 'PY' || true
import torch, sys
print("[env] torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available(): print("[env] gpu:", torch.cuda.get_device_name(0))
PY

# ======= 0) make / reuse frozen split =======
if [[ ! -f "${SPLITS_DIR}/train_pair_ids.txt" ]]; then
  echo "[probe] creating frozen split at ${SPLITS_DIR}"
  ${PYTHON} train_schnet.py \
    --xyz "${XYZ}" \
    --index_csv "${INDEX}" \
    --pair_map_csv "${PAIR_MAP}" \
    --labels_csv "${LABELS}" \
    --frames_per_pair 1 \
    --batch_size 4 \
    --epochs 1 \
    --lr 1e-3 \
    --cutoff 5.0 \
    --hidden 64 --blocks 2 --rbf 16 \
    --num_workers 4 \
    --outdir "${RUNS}/probe" \
    --save_splits "${SPLITS_DIR}"
else
  echo "[probe] using existing frozen split at ${SPLITS_DIR}"
fi

# helper to run one job
run_job () {
  local OUT="$1"; shift
  echo "=== running → ${OUT} ==="
  ${PYTHON} train_schnet.py \
    --xyz "${XYZ}" \
    --index_csv "${INDEX}" \
    --pair_map_csv "${PAIR_MAP}" \
    --labels_csv "${LABELS}" \
    --num_workers "${NUM_WORKERS}" \
    --outdir "${OUT}" \
    --load_splits "${SPLITS_DIR}" \
    "$@"
}

# ======= 1) cutoff sweep @ k=3, 128×3×32 =======
run_job "${RUNS}/abl_cut40_fpp3_b3_r32" \
  --frames_per_pair 3 --batch_size 8 --epochs 20 --lr 1e-3 \
  --hidden 128 --blocks 3 --rbf 32 --cutoff 4.0

run_job "${RUNS}/abl_cut50_fpp3_b3_r32" \
  --frames_per_pair 3 --batch_size 8 --epochs 20 --lr 1e-3 \
  --hidden 128 --blocks 3 --rbf 32 --cutoff 5.0

run_job "${RUNS}/abl_cut60_fpp3_b3_r32" \
  --frames_per_pair 3 --batch_size 8 --epochs 20 --lr 1e-3 \
  --hidden 128 --blocks 3 --rbf 32 --cutoff 6.0

# ======= 2) more conformers (k=5) =======
run_job "${RUNS}/fpp5_b3_r32_cut60" \
  --frames_per_pair 5 --batch_size 8 --epochs 30 --lr 1e-3 \
  --hidden 128 --blocks 3 --rbf 32 --cutoff 6.0

# ======= 3) capacity bump (blocks=4, rbf=48) =======
# tip: if you OOM, drop --batch_size 8 → 6 or 4
run_job "${RUNS}/fpp3_b4_r48_cut60" \
  --frames_per_pair 3 --batch_size 8 --epochs 20 --lr 1e-3 \
  --hidden 128 --blocks 4 --rbf 48 --cutoff 6.0

# ======= 4) longer + lower LR, bigger model (k=5, b=5, rbf=64) =======
run_job "${RUNS}/fpp5_b5_r64_cut60_lr5e4" \
  --frames_per_pair 5 --batch_size 8 --epochs 50 --lr 5e-4 \
  --hidden 128 --blocks 5 --rbf 64 --cutoff 6.0

# ======= 5) print summary =======
echo
echo "=== summary (rmse/mae/r2) ==="
${PYTHON} - << 'PY'
import json, glob, os
def safe(v): return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)
rows=[]
for mpath in glob.glob("schnet_runs/*/test_metrics.json"):
    outdir=os.path.dirname(mpath)
    try:
        metr=json.load(open(mpath))
        rows.append((outdir, metr.get("rmse"), metr.get("mae"), metr.get("r2")))
    except Exception:
        pass
rows.sort(key=lambda x: (x[2] is None, x[2]), reverse=True)  # rough sort by mae desc? we'll show r2 anyway
# pretty print
w=max(len(r[0]) for r in rows) if rows else 20
print(f"{'run'.ljust(w)} | rmse   | mae    | r2")
print("-"*(w+26))
for r in rows:
    print(f"{r[0].ljust(w)} | {safe(r[1]).rjust(6)} | {safe(r[2]).rjust(6)} | {safe(r[3]).rjust(6)}")
PY
echo "done."
