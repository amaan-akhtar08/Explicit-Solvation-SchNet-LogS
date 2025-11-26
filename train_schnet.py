import os
import argparse
import json
import math
import random
from typing import Tuple
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
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from schnet_data import index_xyz, ExplicitSolvDataset
from schnet_model import SchNet, batch_collate

def set_seed(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
                batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    ds = ExplicitSolvDataset(xyz_path, index_csv, frames_df)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: batch_collate(b),
                      num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

def train_loop(model, loader, device, optimizer, scaler):
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
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item()) * y.shape[0]
        n += y.shape[0]
    return total / max(n,1)

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
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--rbf", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers per process")
    ap.add_argument("--outdir", type=str, default="schnet_runs")
    ap.add_argument("--save_splits", type=str, default=None,
                    help="Directory to save the train/val/test pair_id lists.")
    ap.add_argument("--load_splits", type=str, default=None,
                    help="Directory to load existing train/val/test pair_id lists.")
    ap.add_argument("--seed", type=int, default=17,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

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
    split_save_dir = args.save_splits if args.save_splits else os.path.join(args.outdir, "splits")
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


    # Compute train-only mean/std for target standardization
    y_mu = train_frames["LogS"].mean()
    y_sd = train_frames["LogS"].std() if train_frames["LogS"].std() > 0 else 1.0

    # Attach standardized target columns the Dataset will read (named LogS_z)
    for df in (train_frames, val_frames, test_frames):
        df["LogS_z"] = (df["LogS"] - y_mu) / y_sd


    # 5) DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- when creating loaders (replace make_loader calls)
    effective_workers = max(0, args.num_workers)
    use_pin = bool(device.type == "cuda" and effective_workers > 0)
    train_loader = make_loader(args.xyz, args.index_csv, train_frames,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=effective_workers, pin_memory=use_pin)
    val_loader   = make_loader(args.xyz, args.index_csv, val_frames,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=effective_workers, pin_memory=use_pin)
    test_loader  = make_loader(args.xyz, args.index_csv, test_frames,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=effective_workers, pin_memory=use_pin)
    

    model = SchNet(n_atom_types=100, hidden_dim=args.hidden, n_blocks=args.blocks,
                   n_rbf=args.rbf, cutoff=args.cutoff).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=5)

    # 7) Train
    best_rmse = float("inf")
    best_path = os.path.join(args.outdir, "best_model.pt")
    history = {"epoch":[], "train_rmse":[], "val_rmse":[], "val_mae":[], "val_r2":[]}
    scaler = GradScaler(enabled=(device.type == "cuda" and AMP_AVAILABLE))
    patience, bad_epochs = 10, 0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_loop(model, train_loader, device, opt, scaler)
        val_metrics = eval_loop(model, val_loader, device, y_mu, y_sd)
        history["epoch"].append(epoch)
        history["train_rmse"].append(math.sqrt(tr_loss))  
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])

        sched.step(val_metrics["rmse"])
        

        improved = val_metrics["rmse"] < best_rmse - 1e-5
        if improved:
            best_rmse = val_metrics["rmse"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "y_mu": float(y_mu),
                    "y_sd": float(y_sd),
                },
                best_path,
            )
            # torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 8) Test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    y_mu = ckpt.get("y_mu", y_mu)
    y_sd = ckpt.get("y_sd", y_sd)
    test_metrics = eval_loop(model, test_loader, device, y_mu, y_sd)
    with open(os.path.join(args.outdir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test:", test_metrics)

    # 9) Save predictions per pair (averaged across frames)
    @torch.no_grad()
    def predict_frames(frames_df: pd.DataFrame) -> pd.DataFrame:
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
        # average per pair
        agg = df.groupby("pair_id").agg(
            y_true=("LogS","mean"),  # same label for all frames
            y_pred=("pred_LogS","mean"),
            n_frames=("pred_LogS","size")
        ).reset_index()
        return agg

    pred_val = predict_frames(val_frames)
    pred_test = predict_frames(test_frames)
    pred_val.to_csv(os.path.join(args.outdir, "pred_val_by_pair.csv"), index=False)
    pred_test.to_csv(os.path.join(args.outdir, "pred_test_by_pair.csv"), index=False)

if __name__ == "__main__":
    main()
