import os
import sys
import argparse
import json
import math
import random
import time
import subprocess
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import torch

# ---- Compatibility shims: nullcontext + AMP (Torch 2.2.x and older) ----
# nullcontext (Python <3.7 fallback)
try:
    from contextlib import nullcontext
except Exception:
    class nullcontext:  # type: ignore
        def __init__(self, enter_result=None): self.enter_result = enter_result
        def __enter__(self): return self.enter_result
        def __exit__(self, *exc): return False

# AMP: prefer torch.cuda.amp; fall back to no-op on CPU/very old builds
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except Exception:
    AMP_AVAILABLE = False
    autocast = None
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = enabled
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
# ------------------------------------------------------------------------

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from schnet_data import (
    index_xyz,
    ExplicitSolvDataset,
    PairGroupedExplicitSolvDataset,
    make_pair_group_collate,
)
from schnet_model import SchNet, batch_collate

def set_seed(seed: int = 17, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Helper
def _read_list(path):
    with open(path, "r") as f:
        ids = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            # normalize: accept "1.0", "1", or "ABC"
            try:
                s = str(int(float(s)))  # "1.0" -> "1"
            except ValueError:
                pass
            ids.append(s)
        return ids

def _write_list(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(f"{str(x)}\n")

def _normalize_pair_id(x):
    """Return a canonical string representation for pair ids (handles float-like inputs)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return str(int(float(s)))
        except ValueError:
            return s
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)

def _normalize_set(values):
    out = set()
    for v in values:
        norm = _normalize_pair_id(v)
        if norm is not None:
            out.add(norm)
    return out

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except (TypeError, ValueError):
        return None

def _get_git_commit_hash() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            out = res.stdout.strip()
            return out if out else None
    except Exception:
        return None
    return None

def _runtime_metadata(device: torch.device, amp_enabled: bool) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda_available),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": cuda_device_name,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_version": _safe_int(torch.backends.cudnn.version()),
        "amp_available": bool(AMP_AVAILABLE),
        "amp_enabled": bool(amp_enabled),
        "device": str(device),
    }

def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.device):
        return str(obj)
    return obj

def _write_json(path: str, payload: Any):
    with open(path, "w") as f:
        json.dump(_jsonify(payload), f, indent=2)

def split_by_pair(labels_df: pd.DataFrame, seed: int = 17, val_frac: float=0.1, test_frac: float=0.1, 
                  load_splits: str = None, save_splits: str = None):
    """Split by pair_id with option to save/load splits for reproducibility."""
    if load_splits:
        print(f"Loading splits from {load_splits}")
        train_pairs = set(_read_list(os.path.join(load_splits, "train_pair_ids.txt")))
        val_pairs = set(_read_list(os.path.join(load_splits, "val_pair_ids.txt")))
        test_pairs = set(_read_list(os.path.join(load_splits, "test_pair_ids.txt")))
    else:
        rng = np.random.default_rng(seed)
        pairs = labels_df["pair_id"].drop_duplicates().values
        rng.shuffle(pairs)
        n = len(pairs)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test_pairs = set(pairs[:n_test])
        val_pairs = set(pairs[n_test:n_test+n_val])
        train_pairs = set(pairs[n_test+n_val:])
        
        if save_splits:
            print(f"Saving splits to {save_splits}")
            os.makedirs(save_splits, exist_ok=True)
            _write_list(os.path.join(save_splits, "train_pair_ids.txt"), sorted(train_pairs))
            _write_list(os.path.join(save_splits, "val_pair_ids.txt"), sorted(val_pairs))
            _write_list(os.path.join(save_splits, "test_pair_ids.txt"), sorted(test_pairs))
    
    return train_pairs, val_pairs, test_pairs

def select_frames(index_df: pd.DataFrame, labels_df: pd.DataFrame,
                  frames_per_pair: int = 5) -> pd.DataFrame:
    """
    Merge XYZ index (with energies) to labels (per pair), then select top-k lowest-energy frames per pair.
    Returns a dataframe with at least: [pair_id, frame_id, byte_offset, Temperature_K, LogS]
    """
    df = index_df.merge(labels_df[["pair_id","Temperature_K","LogS"]].drop_duplicates("pair_id"),
                        on="pair_id", how="inner")
    # per-pair sort by energy, take k
    df["rank"] = df.groupby("pair_id")["energy_kcal_per_mol"].rank(method="first", ascending=True)
    df = df[df["rank"] <= frames_per_pair].copy()
    return df[["pair_id","frame_id","byte_offset","Temperature_K","LogS"]].reset_index(drop=True)

def make_loader(xyz_path: str, index_csv: str, frames_df: pd.DataFrame,
                batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False,
                seed: int = 17) -> DataLoader:
    ds = ExplicitSolvDataset(xyz_path, index_csv, frames_df)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: batch_collate(b),
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(num_workers > 0),
                      worker_init_fn=_seed_worker,
                      generator=generator)

def make_pair_loader(xyz_path: str, index_csv: str, frames_df: pd.DataFrame,
                     batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False,
                     seed: int = 17) -> DataLoader:
    ds = PairGroupedExplicitSolvDataset(xyz_path, index_csv, frames_df)
    generator = torch.Generator()
    generator.manual_seed(seed)
    collate_fn = make_pair_group_collate(xyz_path=xyz_path, center=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(num_workers > 0),
                      worker_init_fn=_seed_worker,
                      generator=generator)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

def train_loop(model, loader, device, optimizer, scaler, grad_clip: float = 0.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        z = batch["z"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).squeeze(-1)
        T = batch["T"].to(device, non_blocking=True)
        b = batch["batch"].to(device, non_blocking=True)
        Tn = (T - 298.15) / 100.0

        optimizer.zero_grad(set_to_none=True)
        ctx = autocast() if (device.type == "cuda" and AMP_AVAILABLE) else nullcontext()
        with ctx:
            pred = model(z, pos, b, Tn)
            loss = torch.nn.functional.mse_loss(pred, y)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            if device.type == "cuda" and AMP_AVAILABLE:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item()) * y.shape[0]
        n += y.shape[0]
    return total / max(n,1)

def train_loop_attention(model, loader, device, optimizer, scaler, grad_clip: float = 0.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        z = batch["z"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)
        y_pair = batch["y"].to(device, non_blocking=True).squeeze(-1)
        T = batch["T"].to(device, non_blocking=True)  # [n_frames_total, 1]
        atom_batch = batch["batch"].to(device, non_blocking=True)
        frame_to_pair = batch["frame_to_pair"].to(device, non_blocking=True)
        Tn = (T - 298.15) / 100.0

        optimizer.zero_grad(set_to_none=True)
        ctx = autocast() if (device.type == "cuda" and AMP_AVAILABLE) else nullcontext()
        with ctx:
            pred_pair, _, _ = model.forward_attention(z, pos, atom_batch, Tn, frame_to_pair)
            loss = torch.nn.functional.mse_loss(pred_pair, y_pair)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            if device.type == "cuda" and AMP_AVAILABLE:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item()) * y_pair.shape[0]
        n += y_pair.shape[0]
    return total / max(n, 1)

@torch.no_grad()
def eval_loop(model, loader, device, y_mu, y_sd):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        y = batch["y"].to(device).squeeze(-1)  # [B]
        T = batch["T"].to(device)
        b = batch["batch"].to(device)
        Tn = (T - 298.15) / 100.0
        pred = model(z, pos, b, Tn)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    if len(ys) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_true = y_true * y_sd + y_mu
    y_pred = y_pred * y_sd + y_mu
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}

@torch.no_grad()
def eval_loop_attention(model, loader, device, y_mu, y_sd):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        y_pair = batch["y"].to(device).squeeze(-1)
        T = batch["T"].to(device)
        atom_batch = batch["batch"].to(device)
        frame_to_pair = batch["frame_to_pair"].to(device)
        Tn = (T - 298.15) / 100.0
        pred_pair, _, _ = model.forward_attention(z, pos, atom_batch, Tn, frame_to_pair)
        ys.append(y_pair.detach().cpu().numpy())
        ps.append(pred_pair.detach().cpu().numpy())
    if len(ys) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_true = y_true * y_sd + y_mu
    y_pred = y_pred * y_sd + y_mu
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}

def _build_scheduler(name: str, optimizer, epochs: int):
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=5)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return None

def _save_checkpoint(path: str, model, optimizer, scheduler, scaler, args, epoch: int,
                     best_val_rmse: float, y_mu: float, y_sd: float, val_metrics: Dict[str, float]):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
        "epoch": int(epoch),
        "best_val_rmse": _safe_float(best_val_rmse),
        "val_metrics": val_metrics,
        "args": vars(args),
        "y_mu": float(y_mu),
        "y_sd": float(y_sd),
    }
    torch.save(payload, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", type=str, required=True, help="Path to combined_filtered_structures_with_energy.xyz")
    ap.add_argument("--index_csv", type=str, default="xyz_index.csv")
    ap.add_argument("--pair_map_csv", type=str, default="pair_map.csv")
    ap.add_argument("--labels_csv", type=str, default="labels_by_pair.csv")
    ap.add_argument("--frames_per_pair", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4, help="systems per batch")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--rbf", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers per process")
    ap.add_argument("--run_dir", type=str, default=None, help="Directory for run outputs.")
    ap.add_argument("--outdir", type=str, default="schnet_runs", help=argparse.SUPPRESS)
    ap.add_argument("--save_splits", type=str, default=None,
                    help="Directory to save the train/val/test pair_id lists.")
    ap.add_argument("--load_splits", type=str, default=None,
                    help="Directory to load existing train/val/test pair_id lists.")
    ap.add_argument("--seed", type=int, default=17,
                    help="Random seed for reproducibility")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience on val RMSE.")
    ap.add_argument("--scheduler", type=str, default="plateau",
                    choices=["plateau", "cosine", "none"],
                    help="LR scheduler type.")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm (0 disables).")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "attention"],
                    help="Frame-to-pair aggregation mode.")
    args = ap.parse_args()

    if args.run_dir is None:
        args.run_dir = args.outdir
    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(args.seed, deterministic=True)

    # 1) Index XYZ (skip if exists)
    if not os.path.exists(args.index_csv):
        print(f"Indexing XYZ at {args.xyz} ...")
        idx_df = index_xyz(args.xyz, args.index_csv)
    else:
        print(f"Using existing index at {args.index_csv}")
        idx_df = pd.read_csv(args.index_csv)

    # 2) Load labels
    if not os.path.exists(args.labels_csv):
        raise SystemExit(f"Missing labels CSV at {args.labels_csv}. Create it first.")
    labels_df = pd.read_csv(args.labels_csv)

    # Normalize column names
    rename_map = {
        "LogS(mol/L)": "LogS",
        "SMILES_Solute": "Solute_SMILES",
        "SMILES_Solvent": "Solvent_SMILES"
    }
    labels_df = labels_df.rename(columns={k:v for k,v in rename_map.items() if k in labels_df.columns})

    # If pair_id is missing, try to map it from pair_map_csv (SMILES -> pair_id)
    if "pair_id" not in labels_df.columns:
        if not os.path.exists(args.pair_map_csv):
            raise SystemExit("labels_csv lacks 'pair_id' and pair_map_csv is missing. "
                            "Provide pair_map.csv to map SMILES↔pair_id.")
        pm = pd.read_csv(args.pair_map_csv)
        # expect columns: pair_id, Solute_SMILES, Solvent_SMILES
        labels_df = labels_df.merge(pm[["pair_id","Solute_SMILES","Solvent_SMILES"]],
                                    on=["Solute_SMILES","Solvent_SMILES"], how="inner")

    labels_df = labels_df.dropna(subset=["LogS", "pair_id"]).copy()

    # 3) Split by pair
    split_save_dir = args.save_splits if args.save_splits else os.path.join(args.run_dir, "splits")
    train_pairs, val_pairs, test_pairs = split_by_pair(
        labels_df, seed=args.seed, val_frac=0.1, test_frac=0.1,
        load_splits=args.load_splits,
        save_splits=split_save_dir if not args.load_splits else None
    )
    print(f"Pairs -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")
    # 4) Select frames
    frames_df = select_frames(idx_df, labels_df, frames_per_pair=args.frames_per_pair)
    # Normalize pair_id to string format
    frames_df["pair_id"] = frames_df["pair_id"].apply(_normalize_pair_id)
    train_pairs = _normalize_set(train_pairs)
    val_pairs = _normalize_set(val_pairs)
    test_pairs = _normalize_set(test_pairs)
    
    # Filter by split
    train_frames = frames_df[frames_df["pair_id"].isin(train_pairs)].reset_index(drop=True)
    val_frames = frames_df[frames_df["pair_id"].isin(val_pairs)].reset_index(drop=True)
    test_frames = frames_df[frames_df["pair_id"].isin(test_pairs)].reset_index(drop=True)


    # 5) Runtime + initial run config (saved before scaler stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(device.type == "cuda" and AMP_AVAILABLE)
    print(f"Using device: {device} | AMP enabled: {amp_enabled}")

    run_config_path = os.path.join(args.run_dir, "run_config.json")
    scaler_path = os.path.join(args.run_dir, "scaler.json")
    history_path = os.path.join(args.run_dir, "history.json")
    git_commit = _get_git_commit_hash()
    run_config = {
        "run_dir": str(args.run_dir),
        "agg_mode": args.agg,
        "batch_size_semantics": "pairs_per_batch" if args.agg == "attention" else "frames_per_batch",
        "args": vars(args),
        "git_commit": git_commit,
        "deterministic": {
            "seed": int(args.seed),
            "pythonhashseed": str(args.seed),
            "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "worker_seeded": True,
        },
        "runtime": _runtime_metadata(device, amp_enabled),
        "target_scaler": {
            "fitted_on": "train_split_only",
            "mean": None,
            "std": None,
        },
    }
    _write_json(run_config_path, run_config)

    # Compute train-only mean/std for target standardization
    y_mu = train_frames["LogS"].mean()
    y_sd = train_frames["LogS"].std() if train_frames["LogS"].std() > 0 else 1.0

    # Attach standardized target columns the Dataset will read (named LogS_z)
    for df in (train_frames, val_frames, test_frames):
        df["LogS_z"] = (df["LogS"] - y_mu) / y_sd

    # Update run config after scaler stats are known
    run_config["target_scaler"]["mean"] = float(y_mu)
    run_config["target_scaler"]["std"] = float(y_sd)
    _write_json(scaler_path, {"y_mu": float(y_mu), "y_sd": float(y_sd), "fitted_on": "train_split_only"})
    _write_json(run_config_path, run_config)

    # -- when creating loaders (replace make_loader calls)
    effective_workers = max(0, args.num_workers)
    use_pin = bool(device.type == "cuda" and effective_workers > 0)
    if args.agg == "mean":
        train_loader = make_loader(args.xyz, args.index_csv, train_frames,
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=effective_workers, pin_memory=use_pin,
                                seed=args.seed + 11)
        val_loader   = make_loader(args.xyz, args.index_csv, val_frames,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=effective_workers, pin_memory=use_pin,
                                seed=args.seed + 23)
        test_loader  = make_loader(args.xyz, args.index_csv, test_frames,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=effective_workers, pin_memory=use_pin,
                                seed=args.seed + 37)
    else:
        train_loader = make_pair_loader(args.xyz, args.index_csv, train_frames,
                                     batch_size=args.batch_size, shuffle=True,
                                     num_workers=effective_workers, pin_memory=use_pin,
                                     seed=args.seed + 11)
        val_loader   = make_pair_loader(args.xyz, args.index_csv, val_frames,
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=effective_workers, pin_memory=use_pin,
                                     seed=args.seed + 23)
        test_loader  = make_pair_loader(args.xyz, args.index_csv, test_frames,
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=effective_workers, pin_memory=use_pin,
                                     seed=args.seed + 37)

    model = SchNet(n_atom_types=100, hidden_dim=args.hidden, n_blocks=args.blocks,
                   n_rbf=args.rbf, cutoff=args.cutoff).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _build_scheduler(args.scheduler, opt, args.epochs)

    # 7) Train
    best_rmse = float("inf")
    best_ckpt_path = os.path.join(args.run_dir, "best.ckpt")
    last_ckpt_path = os.path.join(args.run_dir, "last.ckpt")
    best_model_path = os.path.join(args.run_dir, "best_model.pt")
    history = []
    _write_json(history_path, history)
    scaler = GradScaler(enabled=amp_enabled)
    bad_epochs = 0
    best_exists = False
    best_epoch = None

    for epoch in range(1, args.epochs+1):
        t0 = time.perf_counter()
        if args.agg == "mean":
            tr_loss = train_loop(model, train_loader, device, opt, scaler, grad_clip=args.grad_clip)
            val_metrics = eval_loop(model, val_loader, device, y_mu, y_sd)
        else:
            tr_loss = train_loop_attention(model, train_loader, device, opt, scaler, grad_clip=args.grad_clip)
            val_metrics = eval_loop_attention(model, val_loader, device, y_mu, y_sd)
        if sched is not None:
            if args.scheduler == "plateau":
                sched.step(val_metrics["rmse"])
            elif args.scheduler == "cosine":
                sched.step()

        epoch_time_sec = float(time.perf_counter() - t0)
        current_lr = float(opt.param_groups[0]["lr"])
        history.append({
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_r2": float(val_metrics["r2"]),
            "lr": float(current_lr),
            "epoch_time_sec": float(epoch_time_sec),
        })
        _write_json(history_path, history)

        _save_checkpoint(last_ckpt_path, model, opt, sched, scaler, args, epoch, best_rmse, y_mu, y_sd, val_metrics)

        improved = np.isfinite(val_metrics["rmse"]) and (val_metrics["rmse"] < best_rmse - 1e-5)
        if improved:
            best_rmse = val_metrics["rmse"]
            bad_epochs = 0
            best_epoch = int(epoch)
            best_exists = True
            _save_checkpoint(best_ckpt_path, model, opt, sched, scaler, args, epoch, best_rmse, y_mu, y_sd, val_metrics)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "y_mu": float(y_mu), "y_sd": float(y_sd)},
                best_model_path
            )
        elif not best_exists:
            best_exists = True
            best_epoch = int(epoch)
            _save_checkpoint(best_ckpt_path, model, opt, sched, scaler, args, epoch, best_rmse, y_mu, y_sd, val_metrics)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "y_mu": float(y_mu), "y_sd": float(y_sd)},
                best_model_path
            )
            bad_epochs += 1
        else:
            bad_epochs += 1

        print(
            "Epoch {0:03d} | train_loss {1:.6f} | val_rmse {2:.6f} | val_mae {3:.6f} | "
            "val_r2 {4:.6f} | lr {5:.6e} | time {6:.2f}s".format(
                epoch, tr_loss, val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"], current_lr, epoch_time_sec
            )
        )

        if args.patience > 0 and bad_epochs >= args.patience:
            run_config["early_stopping"] = {
                "enabled": True,
                "patience": int(args.patience),
                "stopped_early": True,
                "stop_epoch": int(epoch),
                "best_epoch": _safe_int(best_epoch),
                "best_val_rmse": _safe_float(best_rmse),
            }
            _write_json(run_config_path, run_config)
            _write_json(history_path, history)
            print(f"Early stopping at epoch {epoch}")
            break

    if "early_stopping" not in run_config:
        run_config["early_stopping"] = {
            "enabled": bool(args.patience > 0),
            "patience": int(args.patience),
            "stopped_early": False,
            "stop_epoch": None,
            "best_epoch": _safe_int(best_epoch),
            "best_val_rmse": _safe_float(best_rmse),
        }
    run_config["history_file"] = os.path.basename(history_path)
    _write_json(run_config_path, run_config)

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
    elif os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
    else:
        raise SystemExit("No checkpoint found (best.ckpt/last.ckpt). Training did not produce checkpoints.")
    model.load_state_dict(ckpt["model"])

    # 8) Test
    if args.agg == "mean":
        test_metrics = eval_loop(model, test_loader, device, y_mu, y_sd)
    else:
        test_metrics = eval_loop_attention(model, test_loader, device, y_mu, y_sd)
    with open(os.path.join(args.run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test:", test_metrics)

    # 9) Save predictions per pair (same schema for both agg modes)
    @torch.no_grad()
    def predict_frames_mean(frames_df: pd.DataFrame) -> pd.DataFrame:
        ds = ExplicitSolvDataset(args.xyz, args.index_csv, frames_df)
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=lambda b: batch_collate(b))
        out = []
        for batch in loader:
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            T = batch["T"].to(device)
            b = batch["batch"].to(device)
            Tn = (T - 298.15) / 100.0
            pred = model(z, pos, b, Tn).detach().cpu().numpy()
            pred = pred * y_sd + y_mu
            out.extend(pred.tolist())
        df = frames_df.copy()
        df["pred_LogS"] = np.array(out)
        agg = df.groupby("pair_id").agg(
            y_true=("LogS","mean"),
            y_pred=("pred_LogS","mean"),
            n_frames=("pred_LogS","size")
        ).reset_index()
        return agg

    @torch.no_grad()
    def predict_pairs_attention(frames_df: pd.DataFrame) -> pd.DataFrame:
        loader = make_pair_loader(
            args.xyz, args.index_csv, frames_df,
            batch_size=args.batch_size, shuffle=False,
            num_workers=effective_workers, pin_memory=use_pin,
            seed=args.seed + 101
        )
        rows = []
        for batch in loader:
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            T = batch["T"].to(device)
            atom_batch = batch["batch"].to(device)
            frame_to_pair = batch["frame_to_pair"].to(device)
            y_pair = batch["y"].to(device).squeeze(-1)

            Tn = (T - 298.15) / 100.0
            pred_pair, _, _ = model.forward_attention(z, pos, atom_batch, Tn, frame_to_pair)
            pred_pair = pred_pair.detach().cpu().numpy() * y_sd + y_mu
            y_true = y_pair.detach().cpu().numpy() * y_sd + y_mu
            pair_ids = batch["pair_id"].squeeze(-1).cpu().numpy()
            n_frames = batch["n_frames"].squeeze(-1).cpu().numpy()
            for i in range(len(pair_ids)):
                rows.append({
                    "pair_id": int(pair_ids[i]),
                    "y_true": float(y_true[i]),
                    "y_pred": float(pred_pair[i]),
                    "n_frames": int(n_frames[i]),
                })
        out_df = pd.DataFrame(rows, columns=["pair_id", "y_true", "y_pred", "n_frames"])
        if len(out_df) > 0:
            out_df = out_df.sort_values("pair_id").reset_index(drop=True)
        return out_df

    if args.agg == "mean":
        pred_val = predict_frames_mean(val_frames)
        pred_test = predict_frames_mean(test_frames)
    else:
        pred_val = predict_pairs_attention(val_frames)
        pred_test = predict_pairs_attention(test_frames)
    pred_val.to_csv(os.path.join(args.run_dir, "pred_val_by_pair.csv"), index=False)
    pred_test.to_csv(os.path.join(args.run_dir, "pred_test_by_pair.csv"), index=False)

    run_config["final_metrics"] = test_metrics
    run_config["prediction_files"] = ["pred_val_by_pair.csv", "pred_test_by_pair.csv"]
    _write_json(run_config_path, run_config)

    if not os.path.exists(best_model_path):
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "y_mu": float(y_mu), "y_sd": float(y_sd)},
            best_model_path
        )

if __name__ == "__main__":
    main()
