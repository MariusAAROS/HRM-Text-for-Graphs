"""Summarize the local full-backprop vs truncated runs (scripts/run_graphqa_fullbp_local.sh).

Reads ckpts/<run>/eval_old_<split>_ep<E>.json, writes a tidy CSV and two figures in the style of
report/paper_results.ipynb:
  ablation_fullbp_grid   test accuracy heatmaps over (H, L): truncated | full backprop | delta
  ablation_fullbp_lines  test accuracy vs L, one panel per H, both regimes (+-1 binomial SE),
                         with the Jean Zay truncated grid (data/results/ablation_data.csv) as reference

Usage:
    python scripts/summarize_graphqa_fullbp.py [--root /work/dfm/marius-ortega/graphqa_fullbp] [--split test]
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

NAME_RE = re.compile(r"graphqa_H(?P<H>\d+)_L(?P<L>\d+)_(?P<regime>fullbp|trunc)_local_s(?P<seed>\d+)$")
FIG_DIR = Path("report/figures")
COL_W = 3.35
ACCENT, MUTED = "#c0392b", "#7f8c8d"  # same roles as the notebook: truncated = accent, full BP = muted
TRUNC_BP, FULL_BP = "Truncated BPTT (5 blocks)", "Full backprop"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
    "axes.titleweight": "bold", "axes.edgecolor": "0.3", "axes.linewidth": 0.8,
    "grid.linewidth": 0.5, "legend.frameon": False,
})


def load(root: str, split: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "ckpts", "*", f"eval_old_{split}_ep*.json"))):
        run = os.path.basename(os.path.dirname(path))
        m = NAME_RE.match(run)
        if not m:
            continue
        with open(path) as f:
            report = json.load(f)
        rows.append(dict(run=run, H=int(m["H"]), L=int(m["L"]), seed=int(m["seed"]),
                         bp=FULL_BP if m["regime"] == "fullbp" else TRUNC_BP, split=split,
                         accuracy=report["accuracy"], correct=report["correct"], n=report["n"]))
    return pd.DataFrame(rows)


def plot_grid(df: pd.DataFrame, fname: str):
    mean = df.groupby(["bp", "H", "L"])["accuracy"].mean()
    H_vals, L_vals = sorted(df["H"].unique()), sorted(df["L"].unique())
    grids = {bp: mean[bp].unstack("L").reindex(index=H_vals, columns=L_vals) if bp in mean.index.levels[0]
             else pd.DataFrame(np.nan, index=H_vals, columns=L_vals) for bp in (TRUNC_BP, FULL_BP)}
    delta = grids[FULL_BP] - grids[TRUNC_BP]

    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2.1, 2.3), gridspec_kw=dict(wspace=0.35))
    panels = [(TRUNC_BP, grids[TRUNC_BP], "rocket_r", 0, 0.6),
              (FULL_BP, grids[FULL_BP], "rocket_r", 0, 0.6),
              ("Full $-$ truncated", delta, "vlag", -0.15, 0.15)]
    for ax, (title, grid, cmap, vmin, vmax) in zip(axes, panels):
        sns.heatmap(grid, ax=ax, mask=grid.isna(), annot=True, fmt="+.3f" if cmap == "vlag" else ".3f",
                    cmap=cmap, vmin=vmin, vmax=vmax, center=0 if cmap == "vlag" else None,
                    linewidths=0.4, linecolor="white", annot_kws={"fontsize": 6.5}, cbar=False)
        ax.set_facecolor("0.90")
        ax.grid(False)
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(L_vals, rotation=0, fontsize=7.5)
        ax.set_yticklabels(H_vals, rotation=0, fontsize=7.5)
        ax.set_xlabel("Low-level cycles $L$", fontsize=8)
        ax.set_ylabel("High-level cycles $H$" if ax is axes[0] else "", fontsize=8)
    save(fig, fname)


def plot_lines(df: pd.DataFrame, ref: pd.DataFrame, fname: str):
    H_vals = sorted(df["H"].unique())
    fig, axes = plt.subplots(1, len(H_vals), figsize=(COL_W * 2.1, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, H in zip(axes, H_vals):
        for bp, color, style in [(TRUNC_BP, ACCENT, "-"), (FULL_BP, MUTED, "--")]:
            sub = df[(df["H"] == H) & (df["bp"] == bp)]
            if sub.empty:
                continue
            g = sub.groupby("L").agg(acc=("accuracy", "mean"), correct=("correct", "sum"), n=("n", "sum")).reset_index()
            se = np.sqrt(g["acc"] * (1 - g["acc"]) / g["n"])  # binomial SE over all pooled eval samples
            ax.fill_between(g["L"], g["acc"] - se, g["acc"] + se, color=color, alpha=0.15, linewidth=0)
            ax.plot(g["L"], g["acc"], style, marker="o", markersize=5, linewidth=1.6, color=color, label=bp, zorder=3)
            if sub["seed"].nunique() > 1:  # individual seeds, so the spread is visible
                ax.scatter(sub["L"], sub["accuracy"], s=10, color=color, alpha=0.5, zorder=2, linewidth=0)
        r = ref[ref["H"] == H]
        ax.scatter(r["L"], r["accuracy"], s=22, facecolors="none", edgecolors="0.35", linewidth=0.9,
                   label="Truncated (Jean Zay grid)", zorder=4)
        ax.set_xscale("log", base=2)
        L_ticks = sorted(set(df.loc[df["H"] == H, "L"]))
        ax.set_xticks(L_ticks)
        ax.set_xticklabels(L_ticks, fontsize=7.5)
        ax.minorticks_off()
        ax.set_title(f"$H = {H}$", fontsize=9)
        ax.set_xlabel("Low-level cycles $L$", fontsize=8)
        ax.set_ylim(0, 0.65)  # H >= 4 cells collapse to ~0
        sns.despine(ax=ax)
    axes[0].set_ylabel("Test accuracy", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    save(fig, fname)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {FIG_DIR / name}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/dfm/marius-ortega/graphqa_fullbp")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--csv", default="data/results/graphqa-fullbp-local.csv")
    args = ap.parse_args()

    df = load(args.root, args.split)
    if df.empty:
        raise SystemExit(f"no eval reports under {args.root}/ckpts")
    df.sort_values(["H", "L", "bp", "seed"]).to_csv(args.csv, index=False)
    print(f"wrote {args.csv} ({len(df)} rows)")

    ref = pd.read_csv("data/results/ablation_data.csv")
    ref = ref[~ref["Name"].str.contains("_trm_")].rename(columns={"H_cycles": "H", "L_cycles": "L", "eval/accuracy": "accuracy"})
    ref = ref.merge(df[["H", "L"]].drop_duplicates(), on=["H", "L"])

    table = df.pivot_table(index=["H", "L"], columns="bp", values="accuracy", aggfunc="mean")
    table["delta"] = table.get(FULL_BP) - table.get(TRUNC_BP)
    table = table.join(ref.set_index(["H", "L"])["accuracy"].rename("truncated (JZ)"))
    print(table.round(3).to_string())

    plot_grid(df, "ablation_fullbp_grid")
    plot_lines(df, ref, "ablation_fullbp_lines")


if __name__ == "__main__":
    main()
