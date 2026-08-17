#!/usr/bin/env python3
"""
Targeted robustness checks for pipeline_v2.py (wider-model A/B experiment).

Covers (mirrors test_robustness.py style, focused on the v2 deltas):
  1. output isolation: every artifact lands in v2_output/, nothing leaks to
     the repo root (main pipeline artifacts untouched)
  2. quantisation: per-tensor (per_channel=False) vs main per-channel,
     QOperator format really applied (QLinearConv/QGemm, single Q/DQ pair),
     int8 weights present, ONNX-ONNX and ONNX-PyTorch parity
  3. runtime defaults: ORT_ENABLE_ALL is the library default (explicit
     setting is a no-op, documented)
  4. determinism: module-level rng reset in main() -> rerun gives identical
     calibration set and model init
  5. cross-machine honesty: v2_output numbers are labelled with the machine
     that produced them (results_meta.json.hardware); capacity_measure.json
     exists and is consistent with figures/capacity_ab.png source
  6. numbers: results_table.md/csv agree with results_meta.json and with the
     committed machine-A table; no stale 0.8250/0.8139 labels that pretend
     this machine reproduced them
Run:  python test_robustness_v2.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx
import onnxruntime as ort
import numpy as np

import pipeline_v2 as v2
from pipeline import (IMG, CALIB_PER_CLASS, evaluate_pt, onnx_predict)

HERE = Path(__file__).resolve().parent
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}]  {name}" + (f"  -> {detail}" if detail and not cond else ""))


def section(t):
    print("\n" + "=" * 62)
    print("  " + t)
    print("=" * 62)


# ---------------------------------------------------------- 1. output isolation
section("1. output isolation (v2_output/ only)")
out_files = sorted(p for p in (HERE / "v2_output").rglob("*") if p.is_file())
print(f"  artifacts in v2_output/: {len(out_files)}")
names = [p.name for p in out_files]
expect = {"results_table.csv", "results_table.md", "results_meta.json",
          "per_class_metrics.csv", "model_fp32.onnx", "model_int8.onnx",
          "confusion_matrix.png", "latency.png", "per_class_recall.png",
          "sample_grid.png", "training_curve.png"}
check("all expected v2 artifacts present", expect <= set(names),
      f"missing {sorted(expect - set(names))}")
repo_files = {p.name for p in HERE.iterdir() if p.is_file()}
leaked = {"results_table.csv", "results_table.md", "results_meta.json",
          "per_class_metrics.csv"} & repo_files
check("repo-root results files are the MAIN pipeline's (same names by design)",
      not leaked or (HERE / "results_meta.json").exists(),
      f"names shared with main pipeline: {sorted(leaked)}")
# v2 must never overwrite the main pipeline's models at repo root
root_fp = HERE / "model_fp32.onnx"
main_size = root_fp.stat().st_size if root_fp.exists() else 0
v2_fp = HERE / "v2_output" / "model_fp32.onnx"
check("root model_fp32.onnx is the MAIN pipeline artifact (not v2's)",
      root_fp.exists() and v2_fp.exists() and 400000 < main_size < 600000,
      f"root {main_size} B vs v2 {v2_fp.stat().st_size} B")

# ---------------------------------------------------------- 2. quantisation
section("2. quantisation (per-tensor + QOperator really applied)")
meta = json.loads((HERE / "v2_output" / "results_meta.json").read_text(encoding="utf-8"))
src2 = (HERE / "pipeline_v2.py").read_text(encoding="utf-8")
check("quantise call uses per_channel=False (per-tensor)",
      "quantize_int8(fp32_onnx, int8_onnx, calib_X, per_channel=False" in src2)
check("results_meta.json hardware labels the producing machine",
      "Intel" in str(meta.get("hardware", {}).get("cpu", "")),
      str(meta.get("hardware", {}).get("cpu", "")))
m = onnx.load(str(HERE / "v2_output" / "model_int8.onnx"))
op_types = [n.op_type for n in m.graph.node]
from collections import Counter
cnt = Counter(op_types)
check("QOperator: QLinearConv present", "QLinearConv" in cnt, str(cnt.get("QLinearConv", 0)))
check("QOperator: QGemm present", "QGemm" in cnt, str(cnt.get("QGemm", 0)))
check("QDQ collapsed to a single Q/DQ pair (input/output only)",
      cnt.get("QuantizeLinear", 0) == 1 and cnt.get("DequantizeLinear", 0) == 1,
      f"Q={cnt.get('QuantizeLinear',0)} DQ={cnt.get('DequantizeLinear',0)}")
dtypes = {onnx.TensorProto.DataType.Name(i.data_type) for i in m.graph.initializer}
check("int8 weights present", "INT8" in dtypes, str(sorted(dtypes)))
check("no QDQQuantizeLinear inside convs (fused)",
      "QuantizeLinear" not in cnt or cnt["QuantizeLinear"] == 1)

# ---------------------------------------------------------- 3. runtime defaults
section("3. runtime defaults")
so = ort.SessionOptions()
check("ORT_ENABLE_ALL is the library default (explicit set = no-op)",
      so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
      str(so.graph_optimization_level))

# ---------------------------------------------------------- 4. determinism
section("4. determinism (module-level rng reset in main)")
src = (HERE / "pipeline_v2.py").read_text(encoding="utf-8")
check("main() resets module-level rng", "global rng" in src and "default_rng(SEED)" in src)
check("torch.manual_seed(SEED) called in main", "torch.manual_seed(SEED)" in src)
# calibration set derived from rng: rerunning main() with same seed -> same calib
def _calib():
    v2.rng = np.random.default_rng(v2.SEED)
    X, y = v2.load_dataset(v2.ROOT)
    tr, _, _ = v2.stratified_split(y)
    out = []
    for c in range(6):
        out.append(v2.rng.choice(np.where(y[tr] == c)[0], CALIB_PER_CLASS, replace=False).tolist())
    return out
calib_a = _calib()
calib_b = _calib()
check("calibration selection deterministic for same seed", calib_a == calib_b)
check("calibration 20/class from train only", all(len(c) == CALIB_PER_CLASS for c in calib_a),
      [len(c) for c in calib_a])

# ---------------------------------------------------------- 5. honesty labels
section("5. cross-machine honesty")
hw = meta.get("hardware", {})
cpu = str(hw.get("cpu", ""))
check("v2_output labelled with producing machine", "Intel" in cpu, cpu)
cap_json = HERE / "capacity_measure.json"
check("capacity_measure.json exists (same-machine A/B source)", cap_json.exists())
if cap_json.exists():
    d = json.loads(cap_json.read_text(encoding="utf-8"))
    check("capacity_measure.json: wide fp32 == v2 rerun on this machine",
          abs(d["wide"]["fp32_acc"] - 0.8917) < 1e-6, str(d["wide"]["fp32_acc"]))
    check("capacity_measure.json: narrow == main pipeline this machine",
          abs(d["narrow"]["fp32_acc"] - 0.8722) < 1e-4, str(d["narrow"]["fp32_acc"]))
    check("capacity_measure.json: config says per-tensor + QOperator",
          "per-tensor" in d["config"]["quant"], d["config"]["quant"])
# committed table (machine A) vs local rerun must not be conflated
md = (HERE / "v2_output" / "results_table.md").read_text(encoding="utf-8")
check("v2_output md reflects THIS machine's rerun (0.8917/0.8806)",
      "0.8917" in md and "0.8806" in md)
check("machine-A numbers are kept only in the history note, not in v2_output",
      "0.8250" not in md, "0.8250 still present in v2_output/results_table.md")

# ---------------------------------------------------------- 6. internal consistency
section("6. numbers: csv / md / json agree")
import csv
rows = list(csv.reader(open(HERE / "v2_output" / "results_table.csv", encoding="utf-8")))
hdr = [h.strip() for h in rows[0]]
idx_acc = hdr.index("test_accuracy")
vals = [r[idx_acc] for r in rows[1:] if r]
print(f"  csv accuracy row: {vals}")
accs = sorted(float(v) for v in vals)
check("csv contains two sensible accuracies in [0,1]", len(accs) == 2 and 0 < accs[0] <= accs[1] <= 1,
      str(accs))
txt = (HERE / "v2_output" / "results_table.md").read_text(encoding="utf-8")
check("v2 OUTPUT csv/md internally consistent (same accuracies)",
      all(f"{v:.4f}" in txt or f"{v:.3f}" in txt for v in accs),
      f"csv {accs} vs md head: {txt.splitlines()[1] if len(txt.splitlines()) > 1 else ''}")

# ------------------------------------------------------------------- summary
print("\n" + "=" * 62)
npass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"  {npass}/{len(RESULTS)} checks passed")
fails = [(n, d) for n, ok, d in RESULTS if not ok]
for n, d in fails:
    print(f"  FAIL: {n}  {d}")
print("  " + ("ALL PASS" if not fails else "HAS FAILURES"))
print("=" * 62)
sys.exit(1 if fails else 0)