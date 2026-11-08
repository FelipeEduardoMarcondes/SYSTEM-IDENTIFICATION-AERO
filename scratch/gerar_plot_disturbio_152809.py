import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.85",
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

f_path = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\4_aeropendulo\20261103_152809_degraus_hybrid.csv"
df = pd.read_csv(f_path)
t = df["tempo_ms"] / 1000.0

# Cut PID region: window 5.0s to 69.0s
mask = (t >= 5.0) & (t <= 69.0)
t_w = t[mask].values
y_w = df["angulo_deg"][mask].values
r_w = df["referencia"][mask].values
u_w = df["u_pct"][mask].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True, 
                               gridspec_kw={'height_ratios': [2.2, 1.2]})

# Subplot 1: Ângulo
ax1.plot(t_w, r_w, color="black", linestyle="--", linewidth=1.1, label="Referência $r(t) = 60^\\circ$")
ax1.plot(t_w, y_w, color="#d62728", linestyle="-", linewidth=1.2, label="Ângulo Medido $\\theta(t)$")
ax1.set_ylabel("Ângulo $\\theta$ [°]")
ax1.set_ylim(30, 85)
ax1.grid(True)
ax1.legend(loc="lower right", framealpha=0.95)

# Subplot 2: Controle
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_w, u_w, color="#1f77b4", linestyle="-", linewidth=1.1, label="Controle $u(t)$")
ax2.set_xlabel("Tempo [s] (Janela de Atuação Neural: 5 a 69 s)")
ax2.set_ylabel("Controle $u$ [%]")
ax2.set_ylim(10, 75)
ax2.grid(True)
ax2.legend(loc="lower right", framealpha=0.95)

ax1.set_xlim(5.0, 69.0)
plt.tight_layout()

dest_dirs = [
    r"c:\Users\vicio\Documents\AEROPENDULO\tcc\img\resultados",
    r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\plots_academicos"
]

for d in dest_dirs:
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "rejeicao_disturbio_bancada.png"), dpi=300)
    fig.savefig(os.path.join(d, "rejeicao_disturbio_bancada.pdf"), format="pdf")

plt.close(fig)
print("Plot 152809 gerado com sucesso.")

