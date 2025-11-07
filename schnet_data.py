
import os
import re
import io
from typing import List, Tuple, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# --------------------- Periodic table mapping ---------------------
# Basic set is enough for typical organic systems (C, H, O, N, S, Cl, Br, I, F, P, etc.)
_SYMBOL_TO_Z = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,
    "Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,
    "Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54
}

def symbol_to_z(sym: str) -> int:
    s = sym.strip()
    if s not in _SYMBOL_TO_Z:
        raise ValueError(f"Unknown element symbol in XYZ: '{s}'")
    return _SYMBOL_TO_Z[s]

# --------------------- XYZ indexing ---------------------
_XYZ_TITLE_RE = re.compile(r"mol_(?P<pair>\d+)_(?P<frame>\d+)")
_EN_RE = re.compile(r"Energy:\s*([+-]?\d+(?:\.\d*)?)\s*kcal/mol", re.IGNORECASE)

def index_xyz(xyz_path: str, out_csv: str, max_frames: Optional[int]=None) -> pd.DataFrame:
    """
    Build a byte-offset index for a *multi-frame* XYZ file where each frame is:
        line 0: natoms
        line 1: comment, expected to contain something like 'mol_<pair>_<frame> | Energy: ...'
        line 2..(natoms+1): 'El x y z'
    Stores: byte_offset, natoms, frame_name, pair_id, frame_id, energy_kcal_per_mol
    """
    rows = []
    with open(xyz_path, "rb") as f:
        frame = 0
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip().split()[0])
            except Exception:
                # If the file has blank lines or EOF junk, try to continue
                break
            comment = f.readline().decode("utf-8", errors="replace").strip()
            m = _XYZ_TITLE_RE.search(comment)
            pair_id = int(m.group("pair")) if m else None
            frame_id = int(m.group("frame")) if m else None
            mE = _EN_RE.search(comment)
            energy = float(mE.group(1)) if mE else None

            # Skip coordinates quickly by reading natoms lines
            skipped = 0
            while skipped < natoms:
                l = f.readline()
                if not l:
                    break
                skipped += 1

            rows.append({
                "byte_offset": start,
                "natoms": natoms,
                "comment": comment,
                "frame_name": f"mol_{pair_id}_{frame_id}" if (pair_id is not None and frame_id is not None) else comment,
                "pair_id": pair_id,
                "frame_id": frame_id,
                "energy_kcal_per_mol": energy,
            })
            frame += 1
            if max_frames is not None and frame >= max_frames:
                break

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def _read_frame_at(fh, byte_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read one frame (atoms, positions) from a binary-opened XYZ using a known byte offset."""
    fh.seek(byte_offset)
    natoms = int(fh.readline().strip().split()[0])
    _ = fh.readline()  # comment
    symbols = []
    coords = np.zeros((natoms, 3), dtype=np.float32)
    for i in range(natoms):
        parts = fh.readline().decode("utf-8", errors="replace").strip().split()
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        symbols.append(sym)
        coords[i, 0] = x
        coords[i, 1] = y
        coords[i, 2] = z
    Z = np.array([symbol_to_z(s) for s in symbols], dtype=np.int64)
    return Z, coords

class ExplicitSolvDataset(Dataset):
    """
    Lazy dataset: loads coordinates directly from the big XYZ using byte offsets
    stored in an index CSV, and pairs them with LogS labels and Temperature.
    One sample = one explicit (solute+solvent) conformation.
    """
    def __init__(
        self,
        xyz_path: str,
        index_csv: str,
        frames_csv: pd.DataFrame,
        center: bool = True,
    ):
        """
        Args:
            xyz_path: path to the multi-frame XYZ file
            index_csv: CSV with columns [byte_offset, natoms, pair_id, frame_id, ...]
            frames_csv: DataFrame subset with columns [pair_id, frame_id, byte_offset, Temperature_K, LogS]
                        typically produced by a frame selection step (e.g., k lowest-energy frames / pair)
            center: subtract centroid from positions
        """
        self.xyz_path = xyz_path
        self.idx = pd.read_csv(index_csv) if isinstance(index_csv, str) else index_csv.copy()
        self.frames = frames_csv.reset_index(drop=True).copy()
        self.center = center

        # keep only frames that exist in the index (robust join on byte_offset)
        if "byte_offset" in self.frames.columns and "byte_offset" in self.idx.columns:
            self.frames = self.frames.merge(
                self.idx[["byte_offset", "natoms"]],
                on="byte_offset",
                how="inner"
            )
        # sanity
        assert {"Temperature_K","LogS","pair_id","byte_offset"}.issubset(self.frames.columns), \
            "frames_csv must include ['Temperature_K','LogS','pair_id','byte_offset']."

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.frames.iloc[i]
        with open(self.xyz_path, "rb") as fh:
            Z_np, pos_np = _read_frame_at(fh, int(row["byte_offset"]))
        pos = torch.from_numpy(pos_np)  # [N,3]
        if self.center:
            pos = pos - pos.mean(dim=0, keepdim=True)
        z = torch.from_numpy(Z_np.astype(np.int64))  # [N]
        # schnet_data.py __getitem__
        y_col = "LogS_z" if "LogS_z" in self.frames.columns else "LogS"
        y = torch.tensor([float(row[y_col])], dtype=torch.float32)  # [1]
        T = torch.tensor([float(row["Temperature_K"])], dtype=torch.float32)  # [1]
        pid = torch.tensor([int(row["pair_id"])], dtype=torch.long)  # [1]
        return {"z": z, "pos": pos, "y": y, "T": T, "pair_id": pid}
