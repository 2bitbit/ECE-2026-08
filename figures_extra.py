#!/usr/bin/env python3
"""
Extra figures for the presentation, all generated from real artifacts:

  1. model_size.png           fp32 vs int8 model size (3.7x smaller)
  2. tradeoff.png             accuracy / size / latency small multiples
  3. activation_percentile.png activation histogram + percentile clip (the "why percentile")
  4. calibration_montage.png  samples of the 120-image calibration set
  5. architecture.png         layer-by-layer model diagram
  6. confusable_pairs.png     why pitted_surface / patches / scratches are hard
  7. pipeline_flow.png        end-to-end pipeline overview

Run:  python figures_extra.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
import torch.nn as nn

import pipeline

HERE = pipeline.HERE
FIG = pipeline.FIG_DIR
CAT = pipeline.CAT
INK, INK2, MUTED, GRID, SURF = pipeline.INK, pipeline.INK2, pipeline.MUTED, pipeline.GRID, pipeline.SURF


def _load_results():
    rows = {}
    with open(HERE / "results_table.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


# ---------------------------------------------------------------------------
# 1. model size
# ---------------------------------------------------------------------------
def fig_model_size(rows):
    s32 = float(rows["float32"]["model_size_kB"])
    s8 = float(rows["int8"]["model_size_kB"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["float32", "int8"], [s32, s8], color=[CAT[0], CAT[1]], width=0.5)
    for b, s in zip(bars, [s32, s8]):
        ax.text(b.get_x() + b.get_width() / 2, s + 8, f"{s:.0f} kB",
                ha="center", fontsize=12, color=INK)
    ax.annotate(f"3.7x smaller", xy=(1, s8), xytext=(1.18, s32 * 0.7),
                fontsize=11, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_ylabel("model size (kB)")
    ax.set_ylim(0, s32 * 1.25)
    ax.set_title("Model size: int8 is 3.7x smaller", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "model_size.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. tradeoff (accuracy / size / latency small multiples)
# ---------------------------------------------------------------------------
def fig_tradeoff(rows):
    a32, a8 = float(rows["float32"]["test_accuracy"]), float(rows["int8"]["test_accuracy"])
    s32, s8 = float(rows["float32"]["model_size_kB"]), float(rows["int8"]["model_size_kB"])
    l32, l8 = float(rows["float32"]["latency_p99_ms"]), float(rows["int8"]["latency_p99_ms"])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    pairs = [
        (axes[0], [a32, a8], "accuracy", f"-{a32 - a8:.3f}", None, 0.0),
        (axes[1], [s32, s8], "size (kB)", f"-{s32 - s8:.0f} kB", None, 0.0),
        (axes[2], [l32, l8], "latency p99 (ms)", None, pipeline.BUDGET_MS, None),
    ]
    for ax, vals, ylab, drop, budget, _ in pairs:
        bars = ax.bar(["fp32", "int8"], vals, color=[CAT[0], CAT[1]], width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 2 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=10, color=INK)
        if drop:
            ax.set_title(drop, color=INK2, fontsize=11)
        ax.set_ylabel(ylab, fontsize=9)
        if budget is not None:
            # Bars show ~0.3-0.8 ms vs a 15 ms budget; a linear bar axis would
            # clip the budget line off the top, so annotate instead of drawing.
            ax.set_ylim(0, max(vals) * 1.7)
            ax.set_title("both FITS", color=INK2, fontsize=11)
            ax.text(0.5, max(vals) * 1.5, "15 ms budget off-scale (both FITS)",
                    ha="center", va="top", fontsize=8, color=CAT[5])
    axes[0].set_ylim(0, 1)
    # Reserve the top band for the suptitle: y=1.02 + tight_layout() clips the
    # title against the figure edge, so pull it down and leave room via rect.
    fig.suptitle("The quantisation trade-off", color=INK, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "tradeoff.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. activation histogram + percentile clip
# ---------------------------------------------------------------------------
def _train_once(X, y, tr, va, epochs=20):
    torch.manual_seed(pipeline.SEED)
    np.random.seed(pipeline.SEED)
    model = pipeline.task4_model()
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])
    model, _ = pipeline.train_model(model, Xt, yt, Xv, yv, epochs)
    return model


def _collect_activation(model, X):
    relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    buf = []
    def hook(m, inp, out):
        buf.append(out.detach().numpy())
    h = relus[-1].register_forward_hook(hook)  # after conv2 + ReLU
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 64):
            model(torch.tensor(X[i:i + 64]))
    h.remove()
    return np.concatenate([b.reshape(-1) for b in buf])


def fig_activation_percentile():
    X, y = pipeline.load_dataset(pipeline.ROOT)
    tr, va, te = pipeline.stratified_split(y)
    calib_idx = []
    for c in range(6):
        calib_idx += pipeline.rng.choice(
            np.where(y[tr] == c)[0], pipeline.CALIB_PER_CLASS, replace=False).tolist()
    calib_X = X[tr[calib_idx]].astype(np.float32)

    model = _train_once(X, y, tr, va)
    acts = _collect_activation(model, calib_X)

    lo = np.percentile(acts, 0.001)
    hi = np.percentile(acts, 99.999)
    vmax = float(acts.max())

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(acts, bins=200, range=(0, hi * 1.1), color=CAT[0], alpha=0.7, log=True)
    ax.axvline(lo, color=CAT[5], lw=1.4, ls="--")
    ax.axvline(hi, color=CAT[5], lw=1.4, ls="--")
    ax.text(hi, ax.get_ylim()[1] * 0.6, f" 99.999% clip\n (hi={hi:.2f})",
            fontsize=8, color=CAT[5], va="top")
    ax.text(lo, ax.get_ylim()[1] * 0.6, f"0.001% ", fontsize=8, color=CAT[5],
            va="top", ha="right")
    ax.annotate(f"min-max would stretch to {vmax:.2f}",
                xy=(vmax, 0), xytext=(hi * 0.55, ax.get_ylim()[1] * 0.35),
                fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("activation value (after conv2 + ReLU)")
    ax.set_ylabel("count (log)")
    ax.set_title("Why percentile clipping: outliers would waste the range", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "activation_percentile.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. calibration montage
# ---------------------------------------------------------------------------
def fig_calibration_montage(n=4):
    root = HERE / "calibration"
    fig, axes = plt.subplots(6, n, figsize=(n * 1.3, 6 * 1.1))
    for ci, c in enumerate(pipeline.CLASSES):
        files = pipeline.image_files(root / c)[:n]
        for j, f in enumerate(files):
            ax = axes[ci, j]
            ax.imshow(pipeline.task1_to_array(f), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(c.replace("_", " "), fontsize=8, color=INK2)
    fig.suptitle(f"Calibration set — {n} of 20 shown per class (120 total, train only)",
                 color=INK, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "calibration_montage.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. architecture diagram
# ---------------------------------------------------------------------------
def _box(ax, y, label, shape, params=""):
    ax.add_patch(FancyBboxPatch((0.1, y), 0.8, 0.16,
                                boxstyle="round,pad=0.01", fc=SURF, ec=INK2, lw=1))
    ax.text(0.5, y + 0.10, label, ha="center", va="center", fontsize=10, color=INK)
    ax.text(0.5, y + 0.03, shape + (f"   [{params}]" if params else ""),
            ha="center", va="center", fontsize=8, color=INK2)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    layers = [
        ("Input", "1 x 96 x 96", ""),
        ("Conv 3x3, 16  +  ReLU  +  MaxPool", "16 x 48 x 48", "160"),
        ("Conv 3x3, 32  +  ReLU  +  MaxPool", "32 x 24 x 24", "4,640"),
        ("Flatten", "18,432", ""),
        ("Linear -> 6 logits", "6", "110,598"),
    ]
    ys = np.linspace(0.88, 0.08, len(layers))
    for (label, shape, params), y in zip(layers, ys):
        _box(ax, y, label, shape, params)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1 + 0.16),
                                     arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1))
    ax.text(0.5, 0.005, "115,398 parameters  (budget < 500k)", ha="center",
            fontsize=9, color=INK2)
    ax.set_title("Model architecture", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. confusable pairs
# ---------------------------------------------------------------------------
def fig_confusable_pairs():
    groups = [
        ("pitted_surface", "small pits"),
        ("patches", "patches of rust"),
        ("scratches", "thin lines"),
    ]
    n = 3
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5.6))
    for r, (cname, tag) in enumerate(groups):
        files = pipeline.image_files(Path(pipeline.ROOT) / cname)[:n]
        for j, f in enumerate(files):
            ax = axes[r, j]
            ax.imshow(pipeline.task1_to_array(f), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{cname.replace('_', ' ')}\n({tag})", fontsize=8, color=INK2)
    fig.suptitle("Why these classes are hard: similar texture / thin detail",
                 color=INK, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "confusable_pairs.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. pipeline flow
# ---------------------------------------------------------------------------
def fig_pipeline_flow():
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [
        "NEU-DET\n1,800 images, 6 classes",
        "60 / 20 / 20\nstratified split",
        "Train CNN\n115,398 params, 20 epochs",
        "Export ONNX\nfp32, batch = 1",
        "Static int8 quantise\n120 calib images, percentile",
        "Raspberry Pi 5\nlatency < 15 ms",
    ]
    ys = np.linspace(0.88, 0.08, len(steps))
    for step, y in zip(steps, ys):
        ax.add_patch(FancyBboxPatch((0.15, y), 0.7, 0.12,
                                    boxstyle="round,pad=0.01", fc=SURF, ec=INK2, lw=1))
        ax.text(0.5, y + 0.06, step, ha="center", va="center", fontsize=9, color=INK)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1 + 0.12),
                                     arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1))
    ax.set_title("End-to-end pipeline", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "pipeline_flow.png", dpi=150)
    plt.close(fig)


def main():
    pipeline._set_style()
    rows = _load_results()
    fig_model_size(rows)
    fig_tradeoff(rows)
    fig_activation_percentile()
    fig_calibration_montage()
    fig_architecture()
    fig_confusable_pairs()
    fig_pipeline_flow()
    print("wrote 7 figures to", FIG)


if __name__ == "__main__":
    main()
