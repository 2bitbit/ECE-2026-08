#!/usr/bin/env python3
"""
Capacity A/B figure: narrow (115k) vs wide (240k) model.

Three small multiples — test accuracy, latency median, model size — each
comparing the delivered narrow model against the wider variant we tried.
The only variable is conv channels (16→32, 32→64); seed, epochs, and the
quantisation config are identical, so any difference is capacity, not noise.

Numbers are the two runs from this session (see pipeline_v2修改经历.txt).

Run:  python fig_capacity_ab.py   ->   figures/capacity_ab.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline

FIG = pipeline.FIG_DIR
CAT = pipeline.CAT
INK, INK2, MUTED = pipeline.INK, pipeline.INK2, pipeline.MUTED

# narrow = the delivered 115k model; wide = the 240k A/B variant.
# Both: seed=0, 20 epochs, same quantisation (per-tensor + QOperator).
# Each list is [fp32, int8].
NARROW = {"acc": [0.8667, 0.8306], "ms": [0.30, 0.68], "kb": [460.2, 118.5]}
WIDE = {"acc": [0.8250, 0.8139], "ms": [0.92, 1.05], "kb": [947.0, 240.3]}


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
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))

    # Panel 1: accuracy. Full 0-1 range so the drop is not visually exaggerated;
    # the value labels + title carry the exact magnitude.
    _grouped(axes[0], NARROW["acc"], WIDE["acc"], lambda v: f"{v:.3f}", 0.02)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("test accuracy", fontsize=9)
    axes[0].set_title("test accuracy   (fp32 −4.2 pt)", color=INK, fontsize=10)

    # Panel 2: latency (all far below the 15 ms budget).
    _grouped(axes[1], NARROW["ms"], WIDE["ms"], lambda v: f"{v:.2f}", 0.03)
    axes[1].set_ylim(0, 1.25)
    axes[1].set_ylabel("latency median (ms)", fontsize=9)
    axes[1].set_title("latency   (all FITS ≤ 15 ms)", color=INK, fontsize=10)

    # Panel 3: model size.
    _grouped(axes[2], NARROW["kb"], WIDE["kb"], lambda v: f"{v:.0f}", 8)
    axes[2].set_ylim(0, 1150)
    axes[2].set_ylabel("model size (kB)", fontsize=9)
    axes[2].set_title("model size   (≈ 2×)", color=INK, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)

    fig.suptitle("More parameters ≠ better — wide (240k) vs narrow (115k)",
                 color=INK, y=0.99)
    fig.text(0.5, 0.01,
             "same seed=0, 20 epochs, same quantization (per-tensor + QOperator); "
             "only conv channels differ (16→32, 32→64)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(FIG / "capacity_ab.png", dpi=150)
    plt.close(fig)
    print("saved", FIG / "capacity_ab.png")


if __name__ == "__main__":
    main()
