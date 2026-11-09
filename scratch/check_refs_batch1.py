import os
import re

txt_dir = "novas_ref/novas_ref_txt/"

def inspect_file(filename, patterns):
    filepath = os.path.join(txt_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        return
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"==================== {filename} ====================")
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        print(f"--- Pattern '{pat}': {len(matches)} matches ---")
        for m in matches[:3]:
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            snippet = text[start:end].replace("\n", " ")
            print(f"  ... {snippet} ...")

print("1. NGUYEN 2020")
inspect_file("Model Predictive Control for Micro Aerial Vehicles A Survey.txt", 
             ["nonlinear", "coupling", "wind", "PID", "inner", "preview", "computational"])

print("2. MAYNE 2000")
inspect_file("Constrained model predictive control Stability and optimality.txt",
             ["PID", "quadratic", "constraints", "preview"])

print("3. KARG 2020")
inspect_file("Efficient Representation and Approximation of Model Predictive Control Laws via Deep Learning.txt",
             ["Cortex", "embedded", "exact", "piecewise affine", "PWA", "dataset", "sampling"])

print("4. DAVID 2021")
inspect_file("TensorFlow Lite Micro Embedded Machine Learning on TinyML Systems.txt",
             ["feedforward", "orders of magnitude", "latency", "sub-millisecond", "fraction"])

