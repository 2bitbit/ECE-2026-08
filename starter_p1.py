"""
PROJECT 1 STARTER  -  Edge vision and model quantisation
========================================================

THIS FILE IS FOR STUDENTS. It is your scaffolding.

Most of the code is written for you. There are SIX small gaps marked TODO.
Fill them in one at a time. After each one, run:

    python starter_p1.py --check

If you have never written Python before, read primer_0_python_and_terminal.md
first, then primer_1_machine_learning_in_practice.md.

IF YOU DO NOT HAVE THE DATASET YET, this file still works. It generates
fake images of the same shape so you can make progress while waiting.

Run order:
    python starter_p1.py --check          see what is done and what is left
    python starter_p1.py --task 3         hints and explanation for one task
    python starter_p1.py --run            train and quantise, once tasks pass
"""

import argparse
import os
import time
from pathlib import Path

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

def _find(name):
    """Look for a data file next to this script, then in the working folder,
    then one level up. Saves you from 'file not found' when your editor's
    working directory is not the same as the script's folder."""
    for base in (_HERE, _os.getcwd(), _os.path.dirname(_HERE)):
        p = _os.path.join(base, name)
        if _os.path.exists(p):
            return p
    return name

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
           "rolled-in_scale", "scratches"]
IMG = 96
ROOT = os.environ.get("NEU_ROOT") or _find("NEU-DET")


# =====================================================================
#  TASK 1.  Turn one image file into numbers.
#  ------------------------------------------------------------------
#  A neural network cannot read a JPEG. It needs a grid of numbers.
#  Three things must happen to every image:
#     1. open it and make it grey        (already done for you)
#     2. resize it to IMG by IMG         (already done for you)
#     3. scale the numbers to 0.0 to 1.0 (YOUR JOB)
#
#  Pixel brightness comes out as a whole number from 0 (black) to 255
#  (white). Networks train much better on small numbers near zero, so we
#  divide everything by 255.
#
#  HINT:  arr / 255.0
# =====================================================================
def task1_to_array(path):
    from PIL import Image
    im = Image.open(path).convert("L").resize((IMG, IMG))
    arr = np.asarray(im, dtype=np.float32)

    # TODO: scale arr so its values run from 0.0 to 1.0, and return it
    return arr / 255.0                               # scale 0..255 down to 0..1


# =====================================================================
#  TASK 2.  Split the data honestly.
#  ------------------------------------------------------------------
#  We must NEVER test on images the model trained on, or we are just
#  measuring memory rather than learning.
#
#  'idx' is a shuffled list of positions. Cut it in two:
#     the first 80 percent  -> training
#     the last  20 percent  -> testing
#
#  HINT:  cut = int(0.8 * len(idx))   then use idx[:cut] and idx[cut:]
# =====================================================================
def task2_split(idx):
    # TODO: return two arrays, (train_positions, test_positions)
    cut = int(0.8 * len(idx))                        # first 80% train, last 20% test
    return idx[:cut], idx[cut:]


# =====================================================================
#  TASK 3.  Count how many the model got right.
#  ------------------------------------------------------------------
#  'preds' is what the model guessed, 'truth' is the correct answer.
#  Both are arrays of class numbers, like [0, 3, 3, 1, 5, ...].
#
#  Accuracy is: how many matched, divided by how many there were.
#
#  In numpy,  preds == truth  gives an array of True and False.
#  Taking the mean of that gives the fraction that were True. Neat.
#
#  HINT:  float((preds == truth).mean())
# =====================================================================
def task3_accuracy(preds, truth):
    # TODO: return the fraction correct, as a float between 0 and 1
    if len(preds) == 0:
        return 0.0                                   # no predictions -> no score (avoid NaN)
    return float((preds == truth).mean())            # mean of True/False = accuracy


# =====================================================================
#  TASK 4.  Build the model.
#  ------------------------------------------------------------------
#  Most of the network is written for you. It ends with a Flatten that
#  turns the picture into a long list of numbers.
#
#  The LAST layer must turn that list into one score per defect class.
#  How many classes are there? Count CLASSES at the top of this file.
#
#  nn.Linear(a, b) takes 'a' numbers in and gives 'b' numbers out.
#  The 'a' is already worked out for you as n_features.
#
#  HINT:  nn.Linear(n_features, 6)
# =====================================================================
def task4_model():
    n_features = 32 * (IMG // 4) * (IMG // 4)
    layers = [
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
    ]

    # TODO: add the final Linear layer to 'layers', then return the model
    layers.append(nn.Linear(n_features, len(CLASSES)))

    return nn.Sequential(*layers)


# =====================================================================
#  TASK 5.  One step of training.
#  ------------------------------------------------------------------
#  Training a network is the same four steps, over and over:
#     1. clear last round's gradients      opt.zero_grad()
#     2. make a prediction and score it    loss = lossf(model(xb), yb)
#     3. work out which way to adjust      loss.backward()
#     4. take a small step that way        opt.step()
#
#  Steps 1 and 2 are done. Add steps 3 and 4.
#
#  HINT:  loss.backward()  then  opt.step()
# =====================================================================
def task5_train_step(model, opt, lossf, xb, yb):
    opt.zero_grad()
    loss = lossf(model(xb), yb)

    # TODO: compute the gradients, then take one optimiser step
    loss.backward()
    opt.step()

    return float(loss.item())


# =====================================================================
#  TASK 6.  Measure how long the model takes, fairly.
#  ------------------------------------------------------------------
#  This is the whole point of the project: does it fit the Pi's budget?
#
#  A fair measurement times ONE run and records how many milliseconds it
#  took. The loop and the warm up are written for you. You need the timing.
#
#  time.perf_counter() gives the current time in seconds.
#  Take it before and after, subtract, and multiply by 1000 for ms.
#
#  HINT:  (time.perf_counter() - t0) * 1000.0
# =====================================================================
def task6_time_one(run_once):
    t0 = time.perf_counter()
    run_once()

    # TODO: return how many MILLISECONDS that took
    return (time.perf_counter() - t0) * 1000.0       # seconds -> milliseconds


# =====================================================================
#  BELOW HERE IS WRITTEN FOR YOU.
# =====================================================================

def synthetic(n_per_class=120, seed=0):
    """Fake images, same shape as the real ones. Lets you start early."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    X, y = [], []
    for c in range(6):
        for _ in range(n_per_class):
            b = rng.normal(0.5, 0.14, (IMG, IMG))
            if c == 0:   b += 0.11 * np.sin(xx * 1.7 + rng.uniform(0, 6))
            elif c == 1:
                cy, cx = rng.integers(10, IMG - 10, 2)
                b += 0.24 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 40)
            elif c == 2: b += 0.18 * (np.abs(yy - IMG / 2) < rng.integers(8, 20))
            elif c == 3: b -= 0.30 * (rng.random((IMG, IMG)) < 0.02)
            elif c == 4: b += 0.14 * np.sin(yy * 0.25 + rng.uniform(0, 6))
            else:        b += 0.26 * (np.abs(yy - xx - rng.integers(-20, 20)) < 2)
            X.append(np.clip(b, 0.0, 1.0)[None].astype(np.float32)); y.append(c)
    return np.stack(X), np.array(y)


def load_real(root=ROOT):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files, labels = [], []
    root = Path(root)
    for ci, c in enumerate(CLASSES):
        hits = sorted({p for p in root.rglob(f"*{c}*/*") if p.suffix.lower() in exts}
                      | {p for p in root.rglob(f"{c}_*") if p.suffix.lower() in exts},
                      key=str)
        files += hits; labels += [ci] * len(hits)
    if not files:
        return None, None
    X = np.stack([task1_to_array(f)[None] for f in files])
    return X, np.array(labels)


def full_run(epochs=6):
    X, y = load_real()
    if X is None:
        print("NEU-DET not found, using synthetic images so you can proceed.")
        print("Set NEU_ROOT to your unzipped folder to use the real data.\n")
        X, y = synthetic()
    else:
        print(f"loaded {len(y)} real images\n")

    Xt, yt = torch.tensor(X), torch.tensor(y)
    idx = np.random.default_rng(0).permutation(len(yt))
    tr, te = task2_split(idx)
    dl = DataLoader(TensorDataset(Xt[tr], yt[tr]), batch_size=64, shuffle=True)

    model = task4_model()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()

    print(f"model has {n_par:,} parameters")
    for ep in range(epochs):
        losses = [task5_train_step(model, opt, lossf, xb, yb) for xb, yb in dl]
        print(f"  epoch {ep + 1}  average loss {np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(Xt[te]).argmax(1).numpy()
    acc = task3_accuracy(preds, yt[te].numpy())

    def once():
        with torch.no_grad():
            model(Xt[te][:1])
    for _ in range(20):
        once()
    times = sorted(task6_time_one(once) for _ in range(100))

    print()
    print("=" * 58)
    print("  PROJECT 1 STARTER, FULL RUN")
    print("=" * 58)
    print(f"  test accuracy          {acc:.4f}")
    print(f"  model parameters       {n_par:,}")
    print(f"  float32 size estimate  {n_par * 4 / 1024:.1f} kB")
    print(f"  latency median         {times[len(times) // 2]:.2f} ms")
    print(f"  latency p99            {times[98]:.2f} ms")
    print(f"  budget                 15.00 ms")
    verdict = "FITS" if times[98] < 15 else "TOO SLOW, shrink it"
    print(f"  verdict                {verdict}")
    print()
    print("  Next: export to ONNX and quantise to int8. That is milestone 3")
    print("  and 4 in your brief, and it should make this roughly 4x smaller.")
    print()


def check():
    R = []

    try:
        a = np.array([[0.0, 255.0], [51.0, 204.0]], dtype=np.float32)
        import types
        # emulate task1 body on a known array
        out = a / 255.0
        # test the student's version through a tiny fake image
        from PIL import Image
        tmp = Path("_check_img.png")
        Image.fromarray(np.uint8([[0, 255], [51, 204]])).save(tmp)
        got = task1_to_array(tmp)
        tmp.unlink()
        ok = got is not None and got.max() <= 1.01 and got.min() >= -0.01 and got.max() > 0.5
        R.append(("1  scale pixels to 0..1", ok,
                  "" if ok else f"values run {got.min():.1f} to {got.max():.1f}, expected 0.0 to 1.0"))
    except Exception as e:
        R.append(("1  scale pixels to 0..1", False, f"{type(e).__name__}: {e}"))

    try:
        idx = np.arange(100)
        tr, te = task2_split(idx)
        ok = len(tr) == 80 and len(te) == 20 and len(set(tr) & set(te)) == 0
        R.append(("2  split train and test", ok,
                  "" if ok else f"got {len(tr)} train and {len(te)} test from 100, expected 80 and 20"))
    except Exception as e:
        R.append(("2  split train and test", False, f"{type(e).__name__}: {e}"))

    try:
        v = task3_accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 9]))
        ok = abs(v - 0.75) < 1e-9
        R.append(("3  compute accuracy", ok,
                  "" if ok else f"three of four correct gave {v}, expected 0.75"))
    except Exception as e:
        R.append(("3  compute accuracy", False, f"{type(e).__name__}: {e}"))

    try:
        m = task4_model()
        out = m(torch.zeros(2, 1, IMG, IMG))
        ok = out.shape == (2, 6)
        R.append(("4  build the model", ok,
                  "" if ok else f"model output shape {tuple(out.shape)}, expected (2, 6)"))
    except Exception as e:
        R.append(("4  build the model", False, f"{type(e).__name__}: {e}"))

    try:
        m = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        before = m[1].weight.detach().clone()
        task5_train_step(m, opt, nn.CrossEntropyLoss(),
                         torch.randn(8, 4), torch.randint(0, 2, (8,)))
        ok = not torch.allclose(before, m[1].weight)
        R.append(("5  one training step", ok,
                  "" if ok else "the weights did not change, so no learning happened"))
    except Exception as e:
        R.append(("5  one training step", False, f"{type(e).__name__}: {e}"))

    try:
        ms = task6_time_one(lambda: time.sleep(0.02))
        ok = 15 < ms < 60
        R.append(("6  time one run in ms", ok,
                  "" if ok else f"a 20 ms sleep measured as {ms:.1f}, expected about 20"))
    except Exception as e:
        R.append(("6  time one run in ms", False, f"{type(e).__name__}: {e}"))

    print()
    print("=" * 62)
    print("  PROJECT 1 STARTER  -  progress check")
    print("=" * 62)
    passed = 0
    for name, ok, msg in R:
        print(f"  [{'PASS' if ok else 'TODO'}]  task {name}")
        passed += int(ok)
        if msg:
            print(f"          -> {msg}")
    print()
    print(f"  {passed} of {len(R)} tasks complete   [{'#' * passed}{'.' * (len(R) - passed)}]")
    if passed == len(R):
        print("\n  All tasks pass. Now run:  python starter_p1.py --run")
    else:
        nxt = next(i + 1 for i, r in enumerate(R) if not r[1])
        print(f"\n  Next: task {nxt}.  For help run:  python starter_p1.py --task {nxt}")
    print()


HELP = {
    1: """TASK 1, scaling pixels.

A grey image pixel is a whole number from 0 (black) to 255 (white).
Neural networks train far better when their inputs are small numbers
near zero, so we squash everything into the range 0.0 to 1.0.

Dividing every number by 255 does exactly that:
   0   / 255 = 0.0
   255 / 255 = 1.0
   128 / 255 = 0.502

In numpy you can divide a whole array at once, no loop needed.

Write:   return arr / 255.0""",

    2: """TASK 2, splitting the data.

'idx' is a shuffled list of image positions, for example
[47, 3, 199, 22, ...]. Shuffling matters, otherwise all the crazing
images end up in training and none in testing.

We want the first 80 percent for training and the last 20 for testing.

   cut = int(0.8 * len(idx))

int() throws away the decimal part, because you cannot have half an image.

Then idx[:cut] is everything before the cut, and idx[cut:] everything
after. The function already returns those two, so you only need 'cut'.""",

    3: """TASK 3, accuracy.

preds  = what the model guessed, e.g. [1, 2, 3, 4]
truth  = the right answers,      e.g. [1, 2, 3, 9]

In numpy,  preds == truth  compares element by element and gives
[True, True, True, False].

Python treats True as 1 and False as 0, so the MEAN of that array is
the fraction that were correct: (1+1+1+0)/4 = 0.75.

Write:   return float((preds == truth).mean())""",

    4: """TASK 4, the last layer.

The network so far has turned the image into a long list of n_features
numbers. We need it to end with one score per defect class, so the model
can say 'I think this is class 3'.

There are 6 classes. Count them in CLASSES at the top of the file.

nn.Linear(in, out) is a layer that takes 'in' numbers and gives 'out'.

'layers' is an ordinary Python list, and .append() adds to the end.

Write:   layers.append(nn.Linear(n_features, 6))""",

    5: """TASK 5, one training step.

Training repeats four steps, and two are already written:

  opt.zero_grad()   clear the gradients left over from last time
  loss = ...        make a prediction and measure how wrong it is
  loss.backward()   work out which direction each weight should move
  opt.step()        actually move them a small amount

Forgetting backward() means nothing is ever calculated.
Forgetting step() means nothing ever changes. Both are silent failures:
the code runs happily and the model never learns.

Write:   loss.backward()
         opt.step()""",

    6: """TASK 6, timing.

time.perf_counter() returns the current time in SECONDS as a decimal.
Take it once before, once after, and subtract to get the duration.

Milliseconds are thousandths of a second, so multiply by 1000.

   t0 = time.perf_counter()
   ... the thing you are timing ...
   elapsed_ms = (time.perf_counter() - t0) * 1000.0

Write:   return (time.perf_counter() - t0) * 1000.0

This is the number your whole project is judged on, so it is worth
understanding rather than copying.""",
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    if a.task:
        print("\n" + HELP.get(a.task, "no such task") + "\n")
    elif a.run:
        full_run(a.epochs)
    else:
        check()
