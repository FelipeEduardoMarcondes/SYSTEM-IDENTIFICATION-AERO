"""
Script de Geração de Plots Acadêmicos Padronizados para o Controlador PID
Projeto: Controle de Aeropêndulo com NMPC Aproximado por Rede Neural (TinyML)
Autor: Felipe Eduardo Marcondes

Gera:
1. pid_degrau_30.png e .pdf (Degrau de 30 graus: Simulação Linear vs. Real)
2. pid_degrau_40.png e .pdf (Degrau de 40 graus: Ponto de Linearização)
3. pid_degrau_60.png e .pdf (Degrau de 60 graus: Não-linearidade acentuada)
4. pid_degraus_consolidado.png e .pdf (Comparativo 30°, 40° e 60° em painel unificado)
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAÇÃO DE ESTILO PADRÃO ACADÊMICO
# ==============================================================================
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
    "lines.linewidth": 1.3,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.85",
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BASE_DIR = r"c:\Users\vicio\Documents\AEROPENDULO"
DATA_DIR = os.path.join(BASE_DIR, "fluxograma-tcc", "02_pid_preliminar", "ensaios_degraus")

DEST_DIRS = [
    os.path.join(BASE_DIR, "tcc", "img"),
    os.path.join(BASE_DIR, "tcc", "img", "resultados"),
    os.path.join(BASE_DIR, "fluxograma-tcc", "02_pid_preliminar", "ensaios_degraus")
]

for d in DEST_DIRS:
    os.makedirs(d, exist_ok=True)

# Parâmetros do sistema linear e do PID
Ts = 0.01
Kp = 0.5793
Ki = 0.6647
Kd = 0.20

# G(s) = 6.9522 / (s^2 + 0.8755 s + 6.2537)
num = [6.952198761447751]
den = [1.0, 0.8754771940161032, 6.253652920673461]
sys_d = signal.cont2discrete((num, den), Ts, method='zoh')
b = sys_d[0].flatten()
a = sys_d[1].flatten()
b1, b2 = b[1], b[2]
a1, a2 = a[1], a[2]

steps = [
    (30, "pi-30-1.csv", "pid_degrau_30", "Resposta ao Degrau de 30°"),
    (40, "pi-40-1.csv", "pid_degrau_40", "Resposta ao Degrau de 40° (Ponto de Linearização)"),
    (60, "pi-60-1.csv", "pid_degrau_60", "Resposta ao Degrau de 60°")
]

sim_results = {}

for step_deg, csv_file, out_name, title in steps:
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    t_real = (df["tempo_ms"] - df["tempo_ms"].iloc[0]) / 1000.0
    N = len(t_real)
    
    # Simulação discreta em malha fechada
    y_sim = np.zeros(N)
    u_sim = np.zeros(N)
    e_sim = np.zeros(N)
    u_i = np.zeros(N)
    r_sim = np.full(N, float(step_deg))
    
    for k in range(2, N):
        y_sim[k] = -a1 * y_sim[k-1] - a2 * y_sim[k-2] + b1 * u_sim[k-1] + b2 * u_sim[k-2]
        e_sim[k] = r_sim[k] - y_sim[k]
        u_p = Kp * e_sim[k]
        u_i[k] = u_i[k-1] + Ki * (Ts / 2.0) * (e_sim[k] + e_sim[k-1])
        u_d = -(Kd / Ts) * (y_sim[k] - y_sim[k-1])
        u_sim[k] = u_p + u_i[k] + u_d
        if u_sim[k] > 100.0: u_sim[k] = 100.0
        if u_sim[k] < -100.0: u_sim[k] = -100.0
        
    sim_results[step_deg] = {
        "t": t_real,
        "y_real": df["angulo_deg"].values,
        "u_real": df["u_pct"].values,
        "y_sim": y_sim,
        "u_sim": u_sim,
        "r": r_sim
    }
    
    # Plot individual padronizado (2 subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.12})
    
    # Subplot 1: Ângulo
    ax1.plot(t_real, r_sim, "k--", label=f"Referência ({step_deg}°)", linewidth=1.2)
    ax1.plot(t_real, y_sim, color="#1f77b4", label="Modelo Linear (ZOH)", linewidth=1.4)
    ax1.plot(t_real, df["angulo_deg"], color="#d62728", label="Bancada Real (PID)", linewidth=1.2)
    ax1.set_ylabel("Ângulo $\\theta$ [°]")
    ax1.set_title(title, fontweight="bold", pad=6)
    ax1.legend(loc="lower right")
    ax1.grid(True)
    ax1.set_xlim(0, t_real.iloc[-1])
    
    # Subplot 2: Sinal de Controle
    ax2.plot(t_real, u_sim, color="#1f77b4", label="Controle Simulado", linewidth=1.3)
    ax2.plot(t_real, df["u_pct"], color="#d62728", label="Controle Real", linewidth=1.1, alpha=0.9)
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Controle $u$ [%]")
    ax2.legend(loc="best")
    ax2.grid(True)
    
    # Salvar nos diretórios
    for d in DEST_DIRS:
        fig.savefig(os.path.join(d, f"{out_name}.png"), dpi=300)
        fig.savefig(os.path.join(d, f"{out_name}.pdf"))
    plt.close(fig)
    print(f"Salvo: {out_name}.png e .pdf")

# ==============================================================================
# PLOT CONSOLIDADO DOS 3 DEGRAUS (Painel Acadêmico Unificado)
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.2), sharex="col",
                         gridspec_kw={"width_ratios": [1.4, 1.0], "hspace": 0.18, "wspace": 0.22})

titulos_steps = [(30, "Degrau 30°"), (40, "Degrau 40° (Ponto Nominal)"), (60, "Degrau 60°")]

for row, (deg, label_txt) in enumerate(titulos_steps):
    res = sim_results[deg]
    t = res["t"]
    
    # Coluna 1: Ângulo
    ax_ang = axes[row, 0]
    ax_ang.plot(t, res["r"], "k--", label="Referência" if row==0 else "", linewidth=1.1)
    ax_ang.plot(t, res["y_sim"], color="#1f77b4", label="Modelo Linear" if row==0 else "", linewidth=1.3)
    ax_ang.plot(t, res["y_real"], color="#d62728", label="Bancada Real" if row==0 else "", linewidth=1.1)
    ax_ang.set_ylabel(f"{label_txt}\n$\\theta$ [°]")
    ax_ang.grid(True)
    ax_ang.set_xlim(0, 20.0)
    if row == 0:
        ax_ang.legend(loc="lower right", fontsize=8)
        ax_ang.set_title("Rastreamento Angular $\\theta(t)$", fontweight="bold")
    if row == 2:
        ax_ang.set_xlabel("Tempo [s]")
        
    # Coluna 2: Esforço de Controle
    ax_u = axes[row, 1]
    ax_u.plot(t, res["u_sim"], color="#1f77b4", label="Simulação" if row==0 else "", linewidth=1.3)
    ax_u.plot(t, res["u_real"], color="#d62728", label="Real" if row==0 else "", linewidth=1.1, alpha=0.9)
    ax_u.set_ylabel("Controle $u$ [%]")
    ax_u.grid(True)
    ax_u.set_xlim(0, 20.0)
    if row == 0:
        ax_u.legend(loc="best", fontsize=8)
        ax_u.set_title("Esforço de Controle $u(t)$", fontweight="bold")
    if row == 2:
        ax_u.set_xlabel("Tempo [s]")

for d in DEST_DIRS:
    fig.savefig(os.path.join(d, "pid_degraus_consolidado.png"), dpi=300)
    fig.savefig(os.path.join(d, "pid_degraus_consolidado.pdf"))
plt.close(fig)
print("Salvo: pid_degraus_consolidado.png e .pdf")
