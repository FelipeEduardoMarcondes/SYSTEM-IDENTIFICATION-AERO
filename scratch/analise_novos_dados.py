import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Dados completos fornecidos pelo usuário
# GCC (Flash total do firmware):
# -O0: 183.05 KB | 4.96 ms
# -O1: 173.70 KB | 2.46 ms
# -O2: 174.14 KB | ~2.15 ms
# -O3: 175.63 KB | 1.98 ms
# -Ofast: 175.57 KB | ~1.96 ms
# X-CUBE-AI:
# Sem Comp: Modelo 139.05 KiB (Pesos 130.5 KiB + RT 8.55 KiB) | 1.58 ms puro / 1.69 ms SIL
# Alta Comp: Modelo 26.31 KiB (Pesos 17.75 KiB + RT 8.55 KiB) | 1.80 ms puro / 1.82 ms SIL

configs = [
    'C Nativo (-O0)',
    'C Nativo (-O1)',
    'C Nativo (-O2)',
    'C Nativo (-O3)',
    'C Nativo (-Ofast)',
    'X-CUBE-AI (Sem Comp.)',
    'X-CUBE-AI (Alta Comp.)'
]

# Flash dos pesos isolados e runtime
# Para C Nativo, pesos = 130.5 KiB, restante é código do firmware
# Para X-CUBE-AI, pesos = 130.5 KiB / 17.75 KiB, runtime = 8.55 KiB
print("Dados prontos para processamento.")
