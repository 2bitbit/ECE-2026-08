import sys, importlib

print("python", sys.version.split()[0])

# Pick the line for YOUR project. Delete or comment out the other two.
needed = ["torch", "onnx", "onnxscript", "onnxruntime", "PIL", "numpy", "matplotlib"]              # Project 1, vision

ok = True
for m in needed:
    try:
        importlib.import_module(m)
        print("  OK   ", m)
    except Exception as e:
        ok = False
        print("  FAIL ", m, e)

print("PASS" if ok else "FAIL, fix the lines above")
