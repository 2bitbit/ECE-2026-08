#!/usr/bin/env python3
"""Cross-version comparison figure: baseline vs V3 vs V4 vs V5 (this-machine rerun)."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline
from pipeline import CAT, INK, INK2, MUTED

# this-machine rerun values (v*_output/results_table.csv after patch+rerun)
V = {
    "baseline": dict(fp32=0.8722, int8=0.8583, p99_fp=0.44, p99_i8=1.48, kb_fp=460.8, kb_i8=124.2),
    "V3":       dict(fp32=0.8778, int8=0.8806, p99_fp=8.23, p99_i8=6.73, kb_fp=1619.5, kb_i8=408.5),
    "V4":       dict(fp32=0.8806, int8=0.8750, p99_fp=0.78, p99_i8=1.36, kb_fp=254.9, kb_i8=70.2),
    "V5":       dict(fp32=0.9306, int8=0.9306, p99_fp=16.83, p99_i8=10.33, kb_fp=959.7, kb_i8=246.9),
}
NAMES = list(V)
BUDGET = 15.0

pipeline._set_style()
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
x = np.arange(len(NAMES))
w = 0.38

# 1. accuracy
acc_f = [V[n]["fp32"] for n in NAMES]
acc_i = [V[n]["int8"] for n in NAMES]
b1 = axes[0].bar(x - w / 2, acc_f, w, color=CAT[0], label="fp32")
b2 = axes[0].bar(x + w / 2, acc_i, w, color=CAT[1], label="int8")
for b, v in zip(list(b1) + list(b2), acc_f + acc_i):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                 ha="center", fontsize=8, color=INK)
axes[0].set_ylim(0.8, 1.0)
axes[0].set_ylabel("test accuracy", fontsize=9)
axes[0].set_title("Accuracy (this-machine rerun)", color=INK, fontsize=10)
axes[0].axhline(0.8722, color=MUTED, ls=":", lw=1)
axes[0].text(3.4, 0.8735, "baseline fp32", fontsize=7, color=MUTED)

# 2. latency p99 (log)
lat_f = [V[n]["p99_fp"] for n in NAMES]
lat_i = [V[n]["p99_i8"] for n in NAMES]
b1 = axes[1].bar(x - w / 2, lat_f, w, color=CAT[0], label="fp32")
b2 = axes[1].bar(x + w / 2, lat_i, w, color=CAT[1], label="int8")
for b, v in zip(list(b1) + list(b2), lat_f + lat_i):
    axes[1].text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.2f}",
                 ha="center", fontsize=8, color=INK)
axes[1].set_yscale("log")
axes[1].set_ylim(0.2, 60)
axes[1].axhline(BUDGET, color="red", ls="--", lw=1.5)
axes[1].text(3.42, BUDGET * 1.05, "15 ms budget", fontsize=8, color="red", ha="right")
axes[1].set_ylabel("latency p99 (ms, log)", fontsize=9)
axes[1].set_title("Latency p99 (TOO SLOW if > 15 ms)", color=INK, fontsize=10)

# 3. int8 size
kb = [V[n]["kb_i8"] for n in NAMES]
b1 = axes[2].bar(x, kb, w, color=CAT[1])
for b, v in zip(b1, kb):
    axes[2].text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
axes[2].set_ylabel("int8 model size (kB)", fontsize=9)
axes[2].set_title("Int8 size (V4 smallest)", color=INK, fontsize=10)
axes[2].axhline(500, color=MUTED, ls=":", lw=1)
axes[2].text(3.42, 505, "500 kB ref", fontsize=7, color=MUTED, ha="right")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, fontsize=9, color=INK2)
    ax.tick_params(axis="x", length=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("Cross-version comparison (V3/V4/V5 rerun on this machine, same seed)", color=INK, y=0.99)
fig.tight_layout(rect=[0, 0.08, 1, 0.94])
fig.savefig(pipeline.FIG_DIR / "cross_version_comparison.png", dpi=150)
plt.close(fig)
print("saved", pipeline.FIG_DIR / "cross_version_comparison.png")
print(f"fp32: {dict(zip(NAMES, acc_f))}")
print(f"int8: {dict(zip(NAMES, acc_i))}")
print(f"p99:  {dict(zip(NAMES, lat_f))} / {dict(zip(NAMES, lat_i))}")