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

print("10. AGUIRRE")
inspect_file("INTRODUÇÃO À IDENTIFICAÇÃO DE SISTEMAS Técnicas Lineares e Não Lineares Teoria e Aplicação.txt",
             ["NARX", "FROLS", "ERR", "Razão de Redução do Erro", "ortogonal"])

print("11. JIANG 2022")
inspect_file("Neural Network Based Model Predictive Control for a Quadrotor.txt",
             ["ACADO", "PID", "excitation", "spectral", "persistent", "reduction", "online optimization"])

print("12. JACOB 2018")
inspect_file("Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.txt",
             ["integer", "quantization", "floating-point", "ARM", "latency", "hardware"])

print("13. BANBURY 2021")
inspect_file("MLPerf Tiny Benchmark.txt",
             ["benchmark", "accuracy", "latency", "memory", "perception", "audio", "vision", "control"])

print("14. HEWING 2020")
inspect_file("Learning Based Model Predictive Control Toward Safe Learning in Control.txt",
             ["taxonomy", "three", "dynamics", "controller", "certification", "safety"])

print("15. SANYAL 2024 (RAMP-Net)")
inspect_file("RAMP-Net A Robust Adaptive MPC forQuadrotors via Physics-informed Neural Network.txt",
             ["RAMP-Net", "PINN", "wind", "quadrotor"])

print("16. LI 2024 (2022)")
inspect_file("Learning Model Predictive Control for Quadrotors.txt",
             ["SO(3)", "GPU", "onboard", "heavy", "Jetson", "quadrotor"])

