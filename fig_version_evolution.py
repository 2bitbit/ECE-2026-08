#!/usr/bin/env python3
"""V1-V5 integrated comparison figure (cross-machine averages)."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline
from pipeline import CAT, INK, INK2, MUTED, FIG_DIR

NAMES = ["V1\nbaseline", "V2\nwide", "V3\nTTA", "V4\nGAP", "V5\nultimate"]
FP32 = [0.8704, 0.8583, 0.8709, 0.8834, 0.9287]
INT8 = [0.8472, 0.8473, 0.8639, 0.8528, 0.9250]
P99F = [0.50, 1.07, 7.89, 0.69, 15.84]
P99I = [1.35, 1.15, 7.50, 1.12, 11.23]
SIZE = [124.0, 240.4, 408.4, 70.0, 246.7]
BUDGET = 15.0

pipeline._set_style()
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
x = np.arange(len(NAMES))
w = 0.38

b1 = axes[0].bar(x - w / 2, FP32, w, color=CAT[0], label="fp32")
b2 = axes[0].bar(x + w / 2, INT8, w, color=CAT[1], label="int8")
for b, v in zip(list(b1) + list(b2), FP32 + INT8):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                 ha="center", fontsize=8, color=INK)
axes[0].set_ylim(0.80, 0.98)
axes[0].set_ylabel("test accuracy", fontsize=9)
axes[0].set_title("Accuracy (cross-machine mean)", color=INK, fontsize=10)
axes[0].text(0.03, 0.955, "V5: +5.9pt fp32 / +7.8pt int8 vs V1", fontsize=8, color=CAT[1])

b1 = axes[1].bar(x - w / 2, P99F, w, color=CAT[0], label="fp32")
b2 = axes[1].bar(x + w / 2, P99I, w, color=CAT[1], label="int8")
for b, v in zip(list(b1) + list(b2), P99F + P99I):
    axes[1].text(b.get_x() + b.get_width() / 2, max(v, 0.4) * 1.25, f"{v:.2f}",
                 ha="center", fontsize=8, color=INK)
axes[1].set_yscale("log")
axes[1].set_ylim(0.3, 60)
axes[1].axhline(BUDGET, color="red", ls="--", lw=1.5)
axes[1].text(4.45, BUDGET * 1.08, "15 ms", fontsize=8, color="red", ha="right")
axes[1].set_ylabel("latency p99 (ms, log)", fontsize=9)
axes[1].set_title("Latency p99 (int8 only one safe for V5)", color=INK, fontsize=10)

b1 = axes[2].bar(x, SIZE, w, color=CAT[1])
for b, v in zip(b1, SIZE):
    axes[2].text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=8, color=INK)
axes[2].set_ylabel("int8 model size (kB)", fontsize=9)
axes[2].set_title("Int8 size (all fit 500 kB target)", color=INK, fontsize=10)
axes[2].axhline(500, color=MUTED, ls=":", lw=1)
axes[2].text(4.45, 505, "500 kB", fontsize=7, color=MUTED, ha="right")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, fontsize=9, color=INK2)
    ax.tick_params(axis="x", length=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("V1-V5 evolution summary (cross-machine mean, 2-3 machines)", color=INK, y=0.99)
fig.tight_layout(rect=[0, 0.08, 1, 0.94])
fig.savefig(FIG_DIR / "version_evolution.png", dpi=150)
plt.close(fig)
print("saved", FIG_DIR / "version_evolution.png")