import os, json, glob, math, csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---- helpers ----

def _safe_read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] can't read json: {path} -> {e}")
        return None

def _safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] can't read csv: {path} -> {e}")
        return None

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _annot(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

def _try_save(fig, out):
    try:
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        print("[ok] wrote", out)
    except Exception as e:
        print("[warn] save failed:", out, "->", e)
    plt.close(fig)

def _gather_runs(root="schnet_runs"):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        mpath = os.path.join(d, "test_metrics.json")
        hpath = os.path.join(d, "history.json")
        ppath = os.path.join(d, "pred_test_by_pair.csv")
        m = _safe_read_json(mpath) if os.path.exists(mpath) else None
        runs.append({
            "run": d,
            "metrics": m,
            "history_path": hpath if os.path.exists(hpath) else None,
            "pred_by_pair_path": ppath if os.path.exists(ppath) else None,
        })
    return runs

def _pretty_name(path):
    return os.path.basename(path.rstrip("/"))

# ---- main plotting ----

def plot_training_curves(runs, outdir="docs/figs"):
    """Plot RMSE curves for all runs that have history.json"""
    outdir = _ensure_dir(outdir)
    n_plots = 0
    for r in runs:
        if not r["history_path"]:
            continue
        hist = _safe_read_json(r["history_path"])
        if not hist or "val_rmse" not in hist:
            continue
        epochs = hist.get("epoch", list(range(1, len(hist.get("val_rmse", []))+1)))
        val_rmse = hist.get("val_rmse", [])
        tr_rmse = hist.get("train_rmse", [])
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        if tr_rmse:
            ax.plot(epochs, tr_rmse, label="train RMSE")
        if val_rmse:
            ax.plot(epochs, val_rmse, label="val RMSE")
        ax.legend()
        _annot(ax, f"RMSE vs epoch — { _pretty_name(r['run']) }", "epoch", "RMSE")
        _try_save(fig, os.path.join(outdir, f"curve_{_pretty_name(r['run'])}.png"))
        n_plots += 1
    if n_plots == 0:
        print("[info] no history.json found; skipping training curve plots.")

def plot_parity_and_residuals(best_run_dir, outdir="docs/figs"):
    """Parity plot (y_true vs y_pred) and residual histogram from pred_test_by_pair.csv"""
    outdir = _ensure_dir(outdir)
    ppath = os.path.join(best_run_dir, "pred_test_by_pair.csv")
    df = _safe_read_csv(ppath)
    if df is None or not {"y_true","y_pred"}.issubset(df.columns):
        print("[info] missing pred_test_by_pair.csv or columns; skipping parity/residual plots.")
        return

    # Parity scatter
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.scatter(df["y_true"], df["y_pred"], s=10, alpha=0.7)
    # y=x line
    lo = float(np.nanmin([df["y_true"].min(), df["y_pred"].min()]))
    hi = float(np.nanmax([df["y_true"].max(), df["y_pred"].max()]))
    ax.plot([lo,hi], [lo,hi])
    _annot(ax, f"Parity plot — { _pretty_name(best_run_dir) }", "True LogS", "Pred LogS")
    _try_save(fig, os.path.join(outdir, f"parity_{_pretty_name(best_run_dir)}.png"))

    # Residual histogram
    err = (df["y_pred"] - df["y_true"]).values
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.hist(err, bins=40, alpha=0.9)
    _annot(ax, f"Residuals (y_pred - y_true) — { _pretty_name(best_run_dir) }", "error", "count")
    _try_save(fig, os.path.join(outdir, f"residuals_{_pretty_name(best_run_dir)}.png"))

def plot_ablation_bars(runs, outdir="docs/figs"):
    """Aggregate all test_metrics.json and plot RMSE/MAE/R2 bars."""
    outdir = _ensure_dir(outdir)
    rows = []
    for r in runs:
        m = r["metrics"]
        if not isinstance(m, dict):
            continue
        rmse = m.get("rmse")
        mae  = m.get("mae")
        r2   = m.get("r2")
        if rmse is None or mae is None or r2 is None:
            continue
        rows.append((_pretty_name(r["run"]), rmse, mae, r2))
    if not rows:
        print("[info] no test_metrics.json found; skipping ablation bar charts.")
        return
    rows = sorted(rows, key=lambda x: x[1])  # sort by RMSE asc
    names = [r[0] for r in rows]
    rmse  = [r[1] for r in rows]
    mae   = [r[2] for r in rows]
    r2    = [r[3] for r in rows]

    # RMSE bar
    fig = plt.figure(figsize=(max(6, 0.22*len(names)+2), 4))
    ax = plt.gca()
    ax.bar(range(len(names)), rmse)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    _annot(ax, "Ablation — Test RMSE by run", "run", "RMSE")
    _try_save(fig, os.path.join(outdir, "ablation_rmse.png"))

    # MAE bar
    fig = plt.figure(figsize=(max(6, 0.22*len(names)+2), 4))
    ax = plt.gca()
    ax.bar(range(len(names)), mae)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    _annot(ax, "Ablation — Test MAE by run", "run", "MAE")
    _try_save(fig, os.path.join(outdir, "ablation_mae.png"))

    # R2 bar
    fig = plt.figure(figsize=(max(6, 0.22*len(names)+2), 4))
    ax = plt.gca()
    ax.bar(range(len(names)), r2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    _annot(ax, "Ablation — Test R² by run", "run", "R²")
    _try_save(fig, os.path.join(outdir, "ablation_r2.png"))

def plot_seed_stability(root="schnet_runs", stem="fpp5_b5_r64_cut60_lr5e4", outdir="docs/figs"):
    """If you trained multiple seeds like stem_seed1337/stem_seed2025... make a boxplot."""
    outdir = _ensure_dir(outdir)
    pattern = os.path.join(root, f"{stem}_seed*")
    paths = sorted(glob.glob(pattern))
    vals = {"rmse": [], "mae": [], "r2": []}
    labels = []
    for p in paths:
        m = _safe_read_json(os.path.join(p, "test_metrics.json"))
        if not m:
            continue
        labels.append(os.path.basename(p))
        for k in vals:
            if k in m:
                vals[k].append(m[k])
    if not labels:
        print("[info] no seed directories found matching", pattern, "; skipping seed stability plot.")
        return
    # Boxplots
    for k in ["rmse","mae","r2"]:
        if not vals[k]:  # Skip if no data for this metric
            continue
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        ax.boxplot([vals[k]], labels=[k.upper()], showmeans=True)  # Plot single boxplot for all seeds
        _annot(ax, f"Seed stability — {k.upper()}", "metric", k.upper())
        _try_save(fig, os.path.join(outdir, f"seed_stability_{k}.png"))

def main():
    outdir = _ensure_dir("docs/figs")
    runs = _gather_runs("schnet_runs")
    if not runs:
        print("[error] no runs found under schnet_runs/. Generate runs first, then re-run this script.")
        return

    # 1) Training curves (if history.json present)
    plot_training_curves(runs, outdir=outdir)

    # 2) Pick "best" run as the one with the smallest rmse in test_metrics.json
    best = None
    for r in runs:
        m = r["metrics"]
        if not isinstance(m, dict) or "rmse" not in m:
            continue
        if (best is None) or (m["rmse"] < best["metrics"]["rmse"]):
            best = r
    if best:
        print("[info] best run (by test RMSE):", best["run"], "->", best["metrics"])
        plot_parity_and_residuals(best["run"], outdir=outdir)
    else:
        print("[info] no runs with test_metrics.json found; skipping parity/residual plots.")

    # 3) Ablation bars (aggregates across runs)
    plot_ablation_bars(runs, outdir=outdir)

    # 4) Seed stability boxplots (optional; adjust 'stem' if you used a different naming)
    plot_seed_stability(root="schnet_runs", stem="fpp5_b5_r64_cut60_lr5e4", outdir=outdir)

if __name__ == "__main__":
    main()
