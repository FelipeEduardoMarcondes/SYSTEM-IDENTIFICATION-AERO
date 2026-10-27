"""
Script para Gerar os Gráficos Comparativos do PID Clássico vs 100% Rede Neural (Bancada Real)
Gera os 4 ensaios espelhados da Rodada 7:
1. Multisseno 45° (0,50 Hz) - Detalhe 20 a 60 s
2. Multisseno 60° (0,50 Hz) - Detalhe 20 a 60 s
3. Chirp 45° (Amp 35°) - Detalhe 10 a 60 s
4. Chirp 60° (Amp 50°) - Detalhe 10 a 60 s
E versão completa do Multisseno 45° (0 a 150 s)
Autor: Felipe Eduardo Marcondes
"""

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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if not os.path.exists(os.path.join(BASE_DIR, "data", "coletas", "RODADA-7_20260904")):
    BASE_DIR = r"c:\Users\vicio\Documents\AEROPENDULO"

r7_dir = os.path.join(BASE_DIR, "data", "coletas", "RODADA-7_20260904")
real_dir = os.path.join(BASE_DIR, "data", "experimentos_novos", "EXP001_reteste_narx-mpc", "4_aeropendulo")
dest_dirs = [
    os.path.join(BASE_DIR, "tcc", "img", "resultados"),
    os.path.join(BASE_DIR, "tcc", "img"),
    os.path.join(BASE_DIR, "fluxograma-tcc", "10_comparativos_finais", "ann_vs_pid_rodada7")
]

def save_fig(fig, base_name):
    for d in dest_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{base_name}.png"), dpi=300)
        fig.savefig(os.path.join(d, f"{base_name}.pdf"), format="pdf")
    print(f"-> Salvo: {base_name} (.png e .pdf)")

# ==============================================================================
# 1. MULTISSENO 45° - DETALHE (20 a 60 s)
# ==============================================================================
df_pid_m45 = pd.read_csv(os.path.join(r7_dir, "multi-seno-45-050Hz_0904_20-29.csv"))
df_ann_m45 = pd.read_csv(os.path.join(real_dir, "20261023_182532_multi-seno-45-050Hz_ann.csv"))

n_m45 = min(len(df_pid_m45), len(df_ann_m45))
t_m45 = df_ann_m45["tempo_ms"].values[:n_m45] / 1000.0
ref_m45 = df_ann_m45["referencia"].values[:n_m45]
y_pid_m45 = df_pid_m45["angulo_deg"].values[:n_m45]
y_ann_m45 = df_ann_m45["angulo_deg"].values[:n_m45]
u_pid_m45 = df_pid_m45["u_pct"].values[:n_m45]
u_ann_m45 = df_ann_m45["u_pct"].values[:n_m45]

mask_m45 = (t_m45 >= 20.0) & (t_m45 <= 60.0)
rmse_pid_m45 = np.sqrt(np.mean((y_pid_m45[mask_m45] - ref_m45[mask_m45])**2))
rmse_ann_m45 = np.sqrt(np.mean((y_ann_m45[mask_m45] - ref_m45[mask_m45])**2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

ax1.plot(t_m45[mask_m45], ref_m45[mask_m45], color="black", linestyle="--", linewidth=1.1, alpha=0.85, label="Referência")
ax1.plot(t_m45[mask_m45], y_pid_m45[mask_m45], color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.85, 
         label=f"PID (RMSE = {rmse_pid_m45:.2f}°)")
ax1.plot(t_m45[mask_m45], y_ann_m45[mask_m45], color="#d62728", linestyle="-", linewidth=1.3, alpha=0.9, 
         label=f"100% Rede Neural Bancada (RMSE = {rmse_ann_m45:.2f}°)")

ax1.set_ylabel(r"Ângulo $\theta$ [$^\circ$]")
ax1.set_title("Multisseno 45° (0,50 Hz) - PID vs 100% Rede Neural (ANN)", fontsize=10, pad=4)
ax1.set_xlim(20.0, 60.0)
ax1.set_ylim(18, 85)
ax1.grid(True)
ax1.legend(loc="upper right", framealpha=0.95)

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_m45[mask_m45], u_pid_m45[mask_m45], color="#1f77b4", linestyle=":", linewidth=1.1, alpha=0.8, label="PID")
ax2.plot(t_m45[mask_m45], u_ann_m45[mask_m45], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="100% Rede Neural")
ax2.set_xlabel("Tempo [s]")
ax2.set_ylabel(r"Controle $u$ [\%]")
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc="upper right", ncol=2, framealpha=0.95)

plt.tight_layout()
save_fig(fig, "comparacao_rodada7_multiseno45")
plt.close(fig)

# ==============================================================================
# 2. MULTISSENO 45° - VERSÃO ENSAIO COMPLETO (0 a 150 s)
# ==============================================================================
mask_glob = (t_m45 >= 0.0) & (t_m45 <= 150.0)
rmse_pid_glob = np.sqrt(np.mean((y_pid_m45[mask_glob] - ref_m45[mask_glob])**2))
rmse_ann_glob = np.sqrt(np.mean((y_ann_m45[mask_glob] - ref_m45[mask_glob])**2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

ax1.plot(t_m45, ref_m45, color="black", linestyle="--", linewidth=0.9, alpha=0.85, label="Referência")
ax1.plot(t_m45, y_pid_m45, color="#1f77b4", linestyle=":", linewidth=0.9, alpha=0.8, 
         label=f"PID (RMSE = {rmse_pid_glob:.2f}°)")
ax1.plot(t_m45, y_ann_m45, color="#d62728", linestyle="-", linewidth=1.0, alpha=0.85, 
         label=f"100% Rede Neural (RMSE = {rmse_ann_glob:.2f}°)")

ax1.set_ylabel(r"Ângulo $\theta$ [$^\circ$]")
ax1.set_title("Multisseno 45° (0,50 Hz) - Ensaio Completo (0 a 150 s)", fontsize=10, pad=4)
ax1.set_xlim(0, 150.0)
ax1.set_ylim(15, 88)
ax1.grid(True)
ax1.legend(loc="upper right", ncol=3, framealpha=0.95)

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_m45, u_pid_m45, color="#1f77b4", linestyle=":", linewidth=0.8, alpha=0.7, label="PID")
ax2.plot(t_m45, u_ann_m45, color="#d62728", linestyle="-", linewidth=0.9, alpha=0.85, label="100% Rede Neural")
ax2.set_xlabel("Tempo [s]")
ax2.set_ylabel(r"Controle $u$ [\%]")
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc="upper right", ncol=2, framealpha=0.95)

plt.tight_layout()
save_fig(fig, "comparacao_rodada7_multiseno45_completo")
plt.close(fig)

# ==============================================================================
# 3. MULTISSENO 60° - DETALHE (20 a 60 s)
# ==============================================================================
df_pid_m60 = pd.read_csv(os.path.join(r7_dir, "multi-seno-60-050Hz_0904_20-38.csv"))
df_ann_m60 = pd.read_csv(os.path.join(real_dir, "20261023_182910_multi-seno-60-050Hz_ann.csv"))

n_m60 = min(len(df_pid_m60), len(df_ann_m60))
t_m60 = df_ann_m60["tempo_ms"].values[:n_m60] / 1000.0
ref_m60 = df_ann_m60["referencia"].values[:n_m60]
y_pid_m60 = df_pid_m60["angulo_deg"].values[:n_m60]
y_ann_m60 = df_ann_m60["angulo_deg"].values[:n_m60]
u_pid_m60 = df_pid_m60["u_pct"].values[:n_m60]
u_ann_m60 = df_ann_m60["u_pct"].values[:n_m60]

mask_m60 = (t_m60 >= 20.0) & (t_m60 <= 60.0)
rmse_pid_m60 = np.sqrt(np.mean((y_pid_m60[mask_m60] - ref_m60[mask_m60])**2))
rmse_ann_m60 = np.sqrt(np.mean((y_ann_m60[mask_m60] - ref_m60[mask_m60])**2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

ax1.plot(t_m60[mask_m60], ref_m60[mask_m60], color="black", linestyle="--", linewidth=1.1, alpha=0.85, label="Referência")
ax1.plot(t_m60[mask_m60], y_pid_m60[mask_m60], color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.85, 
         label=f"PID (RMSE = {rmse_pid_m60:.2f}°)")
ax1.plot(t_m60[mask_m60], y_ann_m60[mask_m60], color="#d62728", linestyle="-", linewidth=1.3, alpha=0.9, 
         label=f"100% Rede Neural Bancada (RMSE = {rmse_ann_m60:.2f}°)")

ax1.set_ylabel(r"Ângulo $\theta$ [$^\circ$]")
ax1.set_title("Multisseno 60° (0,50 Hz) - PID vs 100% Rede Neural (ANN)", fontsize=10, pad=4)
ax1.set_xlim(20.0, 60.0)
ax1.set_ylim(0, 130)
ax1.grid(True)
ax1.legend(loc="upper right", framealpha=0.95)

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_m60[mask_m60], u_pid_m60[mask_m60], color="#1f77b4", linestyle=":", linewidth=1.1, alpha=0.8, label="PID")
ax2.plot(t_m60[mask_m60], u_ann_m60[mask_m60], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="100% Rede Neural")
ax2.set_xlabel("Tempo [s]")
ax2.set_ylabel(r"Controle $u$ [\%]")
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc="upper right", ncol=2, framealpha=0.95)

plt.tight_layout()
save_fig(fig, "comparacao_rodada7_multiseno60")
plt.close(fig)

# ==============================================================================
# 4. CHIRP 45° (10 a 60 s)
# ==============================================================================
df_pid_c45 = pd.read_csv(os.path.join(r7_dir, "chirp-45-amp35_0904_20-13.csv"))
df_ann_c45 = pd.read_csv(os.path.join(real_dir, "20261023_183336_chirp-45-amp35_ann.csv"))

n_c45 = min(len(df_pid_c45), len(df_ann_c45))
t_c45 = df_ann_c45["tempo_ms"].values[:n_c45] / 1000.0
ref_c45 = df_ann_c45["referencia"].values[:n_c45]
y_pid_c45 = df_pid_c45["angulo_deg"].values[:n_c45]
y_ann_c45 = df_ann_c45["angulo_deg"].values[:n_c45]
u_pid_c45 = df_pid_c45["u_pct"].values[:n_c45]
u_ann_c45 = df_ann_c45["u_pct"].values[:n_c45]

mask_c45 = (t_c45 >= 10.0) & (t_c45 <= 60.0)
rmse_pid_c45 = np.sqrt(np.mean((y_pid_c45[mask_c45] - ref_c45[mask_c45])**2))
rmse_ann_c45 = np.sqrt(np.mean((y_ann_c45[mask_c45] - ref_c45[mask_c45])**2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

ax1.plot(t_c45[mask_c45], ref_c45[mask_c45], color="black", linestyle="--", linewidth=1.1, alpha=0.85, label="Referência")
ax1.plot(t_c45[mask_c45], y_pid_c45[mask_c45], color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.85, 
         label=f"PID (RMSE = {rmse_pid_c45:.2f}°)")
ax1.plot(t_c45[mask_c45], y_ann_c45[mask_c45], color="#d62728", linestyle="-", linewidth=1.3, alpha=0.9, 
         label=f"100% Rede Neural Bancada (RMSE = {rmse_ann_c45:.2f}°)")

ax1.set_ylabel(r"Ângulo $\theta$ [$^\circ$]")
ax1.set_title("Chirp 45° (Amp 35°) - PID vs 100% Rede Neural (ANN)", fontsize=10, pad=4)
ax1.set_xlim(10.0, 60.0)
ax1.set_ylim(-5, 100)
ax1.grid(True)
ax1.legend(loc="upper right", framealpha=0.95)

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_c45[mask_c45], u_pid_c45[mask_c45], color="#1f77b4", linestyle=":", linewidth=1.1, alpha=0.8, label="PID")
ax2.plot(t_c45[mask_c45], u_ann_c45[mask_c45], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="100% Rede Neural")
ax2.set_xlabel("Tempo [s]")
ax2.set_ylabel(r"Controle $u$ [\%]")
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc="upper right", ncol=2, framealpha=0.95)

plt.tight_layout()
save_fig(fig, "comparacao_rodada7_chirp45")
plt.close(fig)

# ==============================================================================
# 5. CHIRP 60° (10 a 60 s)
# ==============================================================================
df_pid_c60 = pd.read_csv(os.path.join(r7_dir, "chirp-60-amp50_0904_20-18.csv"))
df_ann_c60 = pd.read_csv(os.path.join(real_dir, "20261023_183145_chirp-60-amp50_ann.csv"))

n_c60 = min(len(df_pid_c60), len(df_ann_c60))
t_c60 = df_ann_c60["tempo_ms"].values[:n_c60] / 1000.0
ref_c60 = df_ann_c60["referencia"].values[:n_c60]
y_pid_c60 = df_pid_c60["angulo_deg"].values[:n_c60]
y_ann_c60 = df_ann_c60["angulo_deg"].values[:n_c60]
u_pid_c60 = df_pid_c60["u_pct"].values[:n_c60]
u_ann_c60 = df_ann_c60["u_pct"].values[:n_c60]

mask_c60 = (t_c60 >= 10.0) & (t_c60 <= 60.0)
rmse_pid_c60 = np.sqrt(np.mean((y_pid_c60[mask_c60] - ref_c60[mask_c60])**2))
rmse_ann_c60 = np.sqrt(np.mean((y_ann_c60[mask_c60] - ref_c60[mask_c60])**2))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

ax1.plot(t_c60[mask_c60], ref_c60[mask_c60], color="black", linestyle="--", linewidth=1.1, alpha=0.85, label="Referência")
ax1.plot(t_c60[mask_c60], y_pid_c60[mask_c60], color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.85, 
         label=f"PID (RMSE = {rmse_pid_c60:.2f}°)")
ax1.plot(t_c60[mask_c60], y_ann_c60[mask_c60], color="#d62728", linestyle="-", linewidth=1.3, alpha=0.9, 
         label=f"100% Rede Neural Bancada (RMSE = {rmse_ann_c60:.2f}°)")

ax1.set_ylabel(r"Ângulo $\theta$ [$^\circ$]")
ax1.set_title("Chirp 60° (Amp 50°) - PID vs 100% Rede Neural (ANN)", fontsize=10, pad=4)
ax1.set_xlim(10.0, 60.0)
ax1.set_ylim(-5, 135)
ax1.grid(True)
ax1.legend(loc="upper right", framealpha=0.95)

ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_c60[mask_c60], u_pid_c60[mask_c60], color="#1f77b4", linestyle=":", linewidth=1.1, alpha=0.8, label="PID")
ax2.plot(t_c60[mask_c60], u_ann_c60[mask_c60], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="100% Rede Neural")
ax2.set_xlabel("Tempo [s]")
ax2.set_ylabel(r"Controle $u$ [\%]")
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc="upper right", ncol=2, framealpha=0.95)

plt.tight_layout()
save_fig(fig, "comparacao_rodada7_chirp60")
plt.close(fig)

print("Gráficos de PID vs ANN gerados com sucesso!")
