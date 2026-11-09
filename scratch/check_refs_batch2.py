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
        for m in matches[:4]:
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            snippet = text[start:end].replace("\n", " ")
            print(f"  ... {snippet} ...")

print("5. BEMPORAD 2002")
inspect_file("The explicit linear quadratic regulator for constrained systems.txt",
             ["piecewise affine", "exponential", "combinatorial", "storage", "memory"])

print("6. ALSMEIER 2024")
inspect_file("Imitation Learning of MPC with Neural Networks Error Guaranteesand Sparsification.txt",
             ["Lipschitz", "sparse", "memory", "guarantees", "Hoeffding"])

print("7. CHEN 2018")
inspect_file("Approximating Explicit Model Predictive Control Using Constrained.txt",
             ["policy gradient", "maximal", "invariant", "projection", "ReLU"])

print("8. HERTNECK 2018")
inspect_file("Learning an Approximate Model Predictive Controller with Guarantees.txt",
             ["Hoeffding", "Lipschitz", "probability", "statistical"])

print("9. HOSE 2025")
inspect_file("Approximate non-linear model predictive control with safety augmented neural networks.txt",
             ["quadrotor", "0.21", "distribution", "safe candidate", "backup", "recovery", "sequence"])

