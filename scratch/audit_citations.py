import os
import re

txt_dir = "novas_ref/novas_ref_txt/"

def search_text(filename, pattern):
    filepath = os.path.join(txt_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        return
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    print(f"=== {filename} : pattern '{pattern}' -> {len(matches)} matches ===")
    for m in matches[:6]:
        start = max(0, m.start() - 150)
        end = min(len(text), m.end() + 150)
        snippet = text[start:end].replace("\n", " ")
        print(f"  ... {snippet} ...\n")

print("--- NUBERT 2020 ---")
search_text("Safe and Fast Tracking on a Robot Manipulator Robust MPC and Neural Network Control.txt", "KUKA")
search_text("Safe and Fast Tracking on a Robot Manipulator Robust MPC and Neural Network Control.txt", "200")
search_text("Safe and Fast Tracking on a Robot Manipulator Robust MPC and Neural Network Control.txt", "90")
search_text("Safe and Fast Tracking on a Robot Manipulator Robust MPC and Neural Network Control.txt", "25")

print("--- DRGONA 2022 ---")
search_text("Differentiable predictive control Deep learning alternative to explicit model predictive control for unknown nonlinear systems.txt", "Raspberry")
search_text("Differentiable predictive control Deep learning alternative to explicit model predictive control for unknown nonlinear systems.txt", "250")
search_text("Differentiable predictive control Deep learning alternative to explicit model predictive control for unknown nonlinear systems.txt", "sampling")

