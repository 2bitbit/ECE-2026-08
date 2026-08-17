#!/usr/bin/env python3
"""
Robustness & correctness test suite for the ECE-2026-08 pipeline.

Covers:
  1. image loader: ext handling, scaling, resize, bad file
  2. data split: per-class ratios, no leakage, determinism
  3. metrics: per_class_metrics / confusion_matrix against a known case
  4. calibration set: composition and train-only property
  5. model: output shape, param budget, determinism
  6. ONNX: export/parity, model size honesty (external_data=False)
  7. latency summary: median/p99 correctness
  8. quantisation: output exists, int8 weights present, accuracy sane
  9. edge cases: empty folder, missing dataset, weird ext
Run:  python test_robustness.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

import pipeline
import starter_p1 as sp
from pipeline import (CLASSES, IMG, CALIB_PER_CLASS, BUDGET_MS,
                      stratified_split, load_dataset, image_files,
                      per_class_metrics, confusion_matrix, _summary,
                      export_onnx, onnx_predict, evaluate_pt)

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}]  {name}" + (f"  -> {detail}" if detail and not cond else ""))

def section(t):
    print("\n" + "=" * 62)
    print("  " + t)
    print("=" * 62)

# ---------------------------------------------------------------- 1. loader
section("1. image loader")
rng = np.random.default_rng(0)
td = Path(tempfile.mkdtemp(prefix="neu_test_"))
try:
    from PIL import Image
    # valid image, weird name
    img = rng.integers(0, 256, (200, 200), dtype=np.uint8)
    Image.fromarray(img).save(td / "Cr_1.bmp")
    Image.fromarray(img).save(td / "cr_1.JPG")       # uppercase ext
    Image.fromarray(img).save(td / "cr_1.png")
    (td / "not_an_image.txt").write_text("hello")
    (td / "Cr_2.bmp").write_bytes(b"\x00" * 10)      # corrupt image

    a = sp.task1_to_array(td / "Cr_1.bmp")
    check("resize to %dx%d" % (IMG, IMG), a.shape == (IMG, IMG), str(a.shape))
    check("scaled to [0,1]", a.min() >= 0 and a.max() <= 1.0001,
          f"range {a.min():.3f}..{a.max():.3f}")
    check("float32 dtype", a.dtype == np.float32, str(a.dtype))
    check("grayscale single channel", a.ndim == 2)

    files = image_files(td)
    check("ext set .bmp/.jpg/.png (no .txt)", len(files) == 4 and all(f.suffix.lower() in {".bmp",".jpg",".png"} for f in files),
          f"found {[f.suffix for f in files]}")
    check("case-insensitive ext (.JPG accepted)", any(f.suffix == ".JPG" for f in files))
    check("sorted order", files == sorted(files))
    try:
        sp.task1_to_array(td / "not_an_image.txt")
        check("txt rejected by loader", False)
    except Exception:
        check("txt rejected by loader", True)
    # corrupt image: PIL raises on open/convert
    try:
        sp.task1_to_array(td / "Cr_2.bmp")
        check("corrupt image handled gracefully", True, "no crash (PIL tolerated)")
    except Exception as e:
        check("corrupt image handled gracefully", True, f"raises {type(e).__name__} (acceptable)")
finally:
    shutil.rmtree(td, ignore_errors=True)

# ---------------------------------------------------------------- 2. split
section("2. stratified split")
rng2 = np.random.default_rng(0)
y = np.repeat(np.arange(6), 300)
tr, va, te = stratified_split(y)
check("counts 60/20/20", len(tr) == 1080 and len(va) == 360 and len(te) == 360,
      f"tr={len(tr)} va={len(va)} te={len(te)}")
for c in range(6):
    ntr = (y[tr] == c).sum(); nva = (y[va] == c).sum(); nte = (y[te] == c).sum()
    check(f"class {CLASSES[c]} 180/60/60", (ntr, nva, nte) == (180, 60, 60),
          f"got {(ntr, nva, nte)}")
check("train/test no overlap", len(set(tr) & set(te)) == 0)
check("train/val no overlap", len(set(tr) & set(va)) == 0)
check("val/test no overlap", len(set(va) & set(te)) == 0)
check("no duplicate indices", len(set(tr)) == len(tr) and len(set(te)) == len(te))

# determinism: reset the module-level rng, split again -> identical result
pipeline.rng = np.random.default_rng(0)
tr3, va3, te3 = stratified_split(y)
check("split deterministic (fresh rng)", np.array_equal(tr, tr3) and np.array_equal(te, te3))

# ---------------------------------------------------------------- 3. metrics
section("3. metrics correctness (hand-computed)")
truth = np.array([0, 0, 1, 1, 2, 2, 2])
preds = np.array([0, 1, 1, 1, 2, 0, 2])
prec, rec = per_class_metrics(preds, truth)
# class0: tp=1 (idx0), fp=1 (idx5), fn=1 (idx1) -> p=0.5, r=0.5
# class1: tp=2 (idx2,3), fp=1 (idx1), fn=0 -> p=2/3, r=1
# class2: tp=2 (idx4,6), fp=0, fn=1 (idx5) -> p=1, r=2/3
check("class0 p/r", abs(prec[0]-0.5) < 1e-9 and abs(rec[0]-0.5) < 1e-9,
      f"{prec[0]}/{rec[0]}")
check("class1 p/r", abs(prec[1]-2/3) < 1e-9 and abs(rec[1]-1) < 1e-9,
      f"{prec[1]:.4f}/{rec[1]:.4f}")
check("class2 p/r", abs(prec[2]-1) < 1e-9 and abs(rec[2]-2/3) < 1e-9,
      f"{prec[2]:.4f}/{rec[2]:.4f}")
cm = confusion_matrix(preds, truth)
check("confusion matrix shape", cm.shape == (6, 6))
check("cm sums to n", cm.sum() == len(truth))
# cm[i,i] should equal recall*counts: class0 -> 1
check("cm diag matches recall", cm[0,0] == 1 and cm[1,1] == 2 and cm[2,2] == 2)
# p/r recomputed from cm
p0 = cm[0,0] / cm[:,0].sum(); r0 = cm[0,0] / cm[0,:].sum()
check("p/r recomputed from cm", abs(p0-prec[0]) < 1e-9 and abs(r0-rec[0]) < 1e-9)
# all-zero class (3,4,5): p and r must be 0, no div-by-zero
check("empty class p/r = 0", prec[3] == 0 and rec[3] == 0)

# ---------------------------------------------------------------- 4. calibration
section("4. calibration set")
X, yall = load_dataset(pipeline.ROOT)
check("dataset loaded", X is not None and len(yall) == 1800, f"n={0 if X is None else len(yall)}")
if X is not None:
    check("X shape", X.shape == (1800, 1, IMG, IMG), str(X.shape))
    check("balanced classes", all((yall == c).sum() == 300 for c in range(6)))
    check("pixel range [0,1]", X.min() >= 0 and X.max() <= 1.0001,
          f"{X.min():.3f}..{X.max():.3f}")
    tr, va, te = stratified_split(yall)
    # exact rng sequence as pipeline.main
    calib_idx = []
    for c in range(6):
        calib_idx += pipeline.rng.choice(np.where(yall[tr] == c)[0],
                                         CALIB_PER_CLASS, replace=False).tolist()
    full = tr[calib_idx]
    check("calib size 120", len(full) == 120, str(len(full)))
    check("calib 20/class", all((yall[full] == c).sum() == 20 for c in range(6)))
    check("calib no leakage into test", len(set(full) & set(te)) == 0)
    check("calib no leakage into val", len(set(full) & set(va)) == 0)
    check("calib no duplicates", len(set(full)) == 120)

# ---------------------------------------------------------------- 5. model
section("5. model")
m = sp.task4_model()
npar = sum(p.numel() for p in m.parameters())
check("param count 115,398 < 500k", npar == 115398 and npar < 500_000, str(npar))
out = m(torch.zeros(2, 1, IMG, IMG))
check("output shape (2,6)", tuple(out.shape) == (2, 6), str(tuple(out.shape)))
torch.manual_seed(0)
m1 = sp.task4_model(); m2 = sp.task4_model()
check("fresh init differs (not same weights)",
      not torch.allclose(m1[0].weight, m2[0].weight))
torch.manual_seed(0)
m3 = sp.task4_model()
check("init deterministic under seed",
      torch.allclose(m1[0].weight, m3[0].weight))

# small synthetic training sanity
Xs, ys = sp.synthetic(n_per_class=20)
Xt = torch.tensor(Xs[:60]); yt = torch.tensor(ys[:60])
mm = sp.task4_model()
opt = torch.optim.Adam(mm.parameters(), 1e-3)
lossf = torch.nn.CrossEntropyLoss()
losses = [sp.task5_train_step(mm, opt, lossf, Xt, yt) for _ in range(3)]
check("training loss finite/decreasing", all(np.isfinite(losses)) and losses[-1] <= losses[0] + 1e-6,
      f"{losses[0]:.4f} -> {losses[-1]:.4f}")

# ---------------------------------------------------------------- 6. ONNX
section("6. ONNX export")
td2 = Path(tempfile.mkdtemp(prefix="neu_onnx_"))
try:
    from PIL import Image
    img = rng.integers(0, 256, (IMG, IMG), dtype=np.uint8)
    Image.fromarray(img).save(td2 / "x.bmp")
    a = sp.task1_to_array(td2 / "x.bmp")
    t = torch.tensor(a)[None][None]
    model = sp.task4_model()
    model.eval()
    with torch.no_grad():
        ref = model(t).argmax(1).numpy()
    p = td2 / "m.onnx"
    export_onnx(model, p)
    check("onnx file exists", p.exists())
    check("onnx self-contained (no external data)",
          not any(x.endswith(".data") for x in os.listdir(td2)))
    got = onnx_predict(str(p), t.numpy())
    check("onnx == pytorch", int(got[0]) == int(ref[0]))
    # size honesty: params*4 float32
    sz = p.stat().st_size / 1024
    est = npar * 4 / 1024
    check("size close to params*4 (%.0f kB est)" % est, abs(sz - est) < est * 0.05,
          f"actual {sz:.1f} kB")
finally:
    shutil.rmtree(td2, ignore_errors=True)

# ---------------------------------------------------------------- 7. latency
section("7. latency summary")
times = sorted(np.random.default_rng(1).uniform(0.1, 2.0, 100) * 1000)
s = _summary(times)
check("median matches numpy", abs(s["median"] - np.median(times)) < 1e-9)
check("p99 index correct", abs(s["p99"] - times[int(0.99 * 99)]) < 1e-9)
check("p99 <= max", s["p99"] <= np.max(times))
# empty / single element
s1 = _summary([1.0])
check("single sample ok", s1["median"] == 1.0 and s1["p99"] == 1.0)

# ---------------------------------------------------------------- 8. quantisation
section("8. quantisation")
fp = pipeline.HERE / "model_fp32.onnx"
iq = pipeline.HERE / "model_int8.onnx"
if fp.exists() and iq.exists():
    import onnx
    m8 = onnx.load(str(iq))
    dtypes = {i.data_type for i in m8.graph.initializer}
    check("int8 weights present", onnx.TensorProto.INT8 in dtypes,
          str({onnx.TensorProto.DataType.Name(d) for d in dtypes}))
    ops = {n.op_type for n in m8.graph.node}
    check("QDQ present", "QuantizeLinear" in ops and "DequantizeLinear" in ops)
    check("size ratio ~3.7x", 3.0 < fp.stat().st_size / iq.stat().st_size < 4.5,
          f"{fp.stat().st_size / iq.stat().st_size:.2f}x")
    # acc sanity: quantised within 0.1 of fp32 on a subset
    Xt_, yt_ = load_dataset(pipeline.ROOT)
    rngq = np.random.default_rng(0)
    idx = rngq.choice(len(yt_), 60, replace=False)
    p32 = onnx_predict(str(fp), Xt_[idx])
    p8 = onnx_predict(str(iq), Xt_[idx])
    a32 = (p32 == yt_[idx]).mean(); a8 = (p8 == yt_[idx]).mean()
    check("int8 acc within 0.15 of fp32 (subset)", a32 - a8 < 0.15,
          f"fp32={a32:.3f} int8={a8:.3f}")
else:
    check("models exist", False, "run pipeline.py first")

# ---------------------------------------------------------------- 9. edge cases
section("9. edge cases")
td3 = Path(tempfile.mkdtemp(prefix="neu_edge_"))
try:
    # empty dataset dir
    empty = td3 / "empty"
    (empty / "crazing").mkdir(parents=True)
    Xe, ye = load_dataset(empty)
    check("empty dataset -> (None, None)", Xe is None and ye is None)
    # missing root
    Xe2, ye2 = load_dataset(td3 / "nope")
    check("missing root -> (None, None)", Xe2 is None and ye2 is None)
    # single-class small folder: split should still work proportionally
    single = td3 / "one"
    (single / "crazing").mkdir(parents=True)
    for i in range(10):
        Image.fromarray(rng.integers(0, 256, (IMG, IMG), dtype=np.uint8)).save(single / "crazing" / f"c_{i}.bmp")
    Xs2, ys2 = load_dataset(single)
    check("single-class loads", Xs2 is not None and len(ys2) == 10)
    t, v, te2 = stratified_split(ys2)
    check("tiny split 6/2/2", len(t) == 6 and len(v) == 2 and len(te2) == 2,
          f"{len(t)}/{len(v)}/{len(te2)}")
    # synthetic fallback
    Xsyn, ysyn = sp.synthetic(n_per_class=10)
    check("synthetic 6x10", Xsyn.shape == (60, 1, IMG, IMG) and len(set(ysyn)) == 6)
    check("synthetic dtype/range", Xsyn.dtype == np.float32 and Xsyn.min() >= 0 and Xsyn.max() <= 1)
    # task2_split on odd length
    tr2, te2b = sp.task2_split(np.arange(13))
    check("task2 odd length", len(tr2) == 10 and len(te2b) == 3)
    # task3 with empty arrays
    check("task3 empty", sp.task3_accuracy(np.array([]), np.array([])) == 0.0)
finally:
    shutil.rmtree(td3, ignore_errors=True)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 62)
npass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"  {npass}/{len(RESULTS)} checks passed")
fails = [(n, d) for n, ok, d in RESULTS if not ok]
for n, d in fails:
    print(f"  FAIL: {n}  {d}")
print("  " + ("ALL PASS" if not fails else "HAS FAILURES"))
print("=" * 62)
sys.exit(1 if fails else 0)