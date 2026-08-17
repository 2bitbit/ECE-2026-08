#!/usr/bin/env python3
"""
Capacity A/B figure: narrow (115k) vs wide (240k) model.

Same-machine, same-config A/B measured on THIS machine (see capacity_measure.py
and capacity_measure.json): seed=0, 20 epochs, per-tensor QInt8 + QOperator,
ORT_ENABLE_ALL. The only variable is conv channels (16->32, 32->64).

IMPORTANT (honesty): the committed numbers from machine A (i7-10xxx,
commit 8a9a5de) showed accuracy DROPPING (0.867->0.825). Reproducing on this
machine (i5-11xxx) the accuracy actually RISES (0.872->0.892). The only stable
effects across machines are latency (~3x) and model size (~2x). So the figure
shows this machine's measured numbers and annotates the cross-machine flip —
the honest takeaway is "capacity does NOT reliably help; it reliably costs
latency and size".

Run:  python fig_capacity_ab.py   ->   figures/capacity_ab.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline

HERE = Path(__file__).resolve().parent
FIG = pipeline.FIG_DIR
CAT = pipeline.CAT
INK, INK2, MUTED = pipeline.INK, pipeline.INK2, pipeline.MUTED


def _load():
    """Read the same-machine measurement (capacity_measure.json), fall back to
    the committed machine-A numbers only if it is missing."""
    p = HERE / "capacity_measure.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d["narrow"], d["wide"], "this machine (i5-11xxx, per-tensor + QOperator)")
    return (
        {"fp32_acc": 0.8667, "int8_acc": 0.8306, "fp32_median_ms": 0.30,
         "int8_median_ms": 0.68, "fp32_kB": 460.2, "int8_kB": 118.5},
        {"fp32_acc": 0.8250, "int8_acc": 0.8139, "fp32_median_ms": 0.92,
         "int8_median_ms": 1.05, "fp32_kB": 947.0, "int8_kB": 240.3},
        "machine A (i7-10xxx, per-tensor + QOperator) — fallback",
    )


def _grouped(ax, narrow, wide, fmt, ypad):
    x = np.arange(2)
    w = 0.38
    bars1 = ax.bar(x - w / 2, [narrow[0], wide[0]], w, color=CAT[0], label="fp32")
    bars2 = ax.bar(x + w / 2, [narrow[1], wide[1]], w, color=CAT[1], label="int8")
    for b, v in zip(list(bars1) + list(bars2), narrow + wide):
        ax.text(b.get_x() + b.get_width() / 2, v + ypad, fmt(v),
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["narrow\n115k", "wide\n240k"], fontsize=9, color=INK2)
    ax.tick_params(axis="x", length=0)
    return bars1, bars2


def main():
    pipeline._set_style()
    narrow, wide, source = _load()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))

    acc_n = [narrow["fp32_acc"], narrow["int8_acc"]]
    acc_w = [wide["fp32_acc"], wide["int8_acc"]]
    _grouped(axes[0], acc_n, acc_w, lambda v: f"{v:.3f}", 0.02)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("test accuracy", fontsize=9)
    axes[0].set_title(f"accuracy  (fp32 {acc_n[0] - acc_w[0]:+.3f})",
                      color=INK, fontsize=10)

    ms_n = [narrow["fp32_median_ms"], narrow["int8_median_ms"]]
    ms_w = [wide["fp32_median_ms"], wide["int8_median_ms"]]
    _grouped(axes[1], ms_n, ms_w, lambda v: f"{v:.2f}", 0.03)
    axes[1].set_ylim(0, 1.25)
    axes[1].set_ylabel("latency median (ms)", fontsize=9)
    axes[1].set_title("latency  (all FITS <= 15 ms)", color=INK, fontsize=10)

    kb_n = [narrow["fp32_kB"], narrow["int8_kB"]]
    kb_w = [wide["fp32_kB"], wide["int8_kB"]]
    _grouped(axes[2], kb_n, kb_w, lambda v: f"{v:.0f}", 8)
    axes[2].set_ylim(0, 1150)
    axes[2].set_ylabel("model size (kB)", fontsize=9)
    axes[2].set_title("model size  (~2x)", color=INK, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)

    fig.suptitle("More parameters does not reliably help (capacity A/B)",
                 color=INK, y=0.99)
    fig.text(0.5, 0.01,
             f"{source}; same seed=0, 20 epochs, only conv channels differ. "
             "On machine A (i7-10xxx) accuracy instead DROPPED 0.867->0.825 — "
             "direction is not reproducible; the stable costs are ~3x latency, ~2x size",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FIG / "capacity_ab.png", dpi=150)
    plt.close(fig)
    print("saved", FIG / "capacity_ab.png")
    print(f"narrow {acc_n}  wide {acc_w}  (source: {source})")


if __name__ == "__main__":
    main()