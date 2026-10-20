import os
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Configure Academic rcParams
def setup_academic():
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
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

setup_academic()

exp_dir = r"data/experimentos_novos/EXP001_legado_narx-mpc"
py_dir = os.path.join(exp_dir, "2_mpc_python")
sil_dir = os.path.join(exp_dir, "3_sil_stm32")
real_dir = os.path.join(exp_dir, "4_aeropendulo")

out_dirs = [
    "tcc/img",
    os.path.join(exp_dir, "plots_academicos")
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, base_name):
    for d in out_dirs:
        png_path = os.path.join(d, f"{base_name}.png")
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path, format="pdf")
    print(f"Saved {base_name}.png and .pdf")

# ==============================================================================
# 1. Comparacao Plots (0, 1, 2)
# ==============================================================================
tests = [
    {
        "id": 0,
        "title": "Validação 1: Multisseno + Degraus",
        "py_csv": "simulacao_python.csv",
        "sil_csv": "20261006_161042_referencia_mpc.csv",
        "real_csv": "20261006_181737_referencia_mpc.csv",
        "base_name": "comparacao_plot_0"
    },
    {
        "id": 1,
        "title": "Validação 2: Degraus Aleatórios",
        "py_csv": "simulacao_python_val2_degraus.csv",
        "sil_csv": "20261006_161234_referencia_val2_degraus.csv",
        "real_csv": "20261006_182048_referencia_val2_degraus.csv",
        "base_name": "comparacao_plot_1"
    },
    {
        "id": 2,
        "title": "Validação 3: Senoidal Suave",
        "py_csv": "simulacao_python_val3_senoidal.csv",
        "sil_csv": "20261006_161422_referencia_val3_senoidal.csv",
        "real_csv": "20261006_182839_referencia_val3_senoidal.csv",
        "base_name": "comparacao_plot_2"
    }
]

for t in tests:
    df_py = pd.read_csv(os.path.join(py_dir, t["py_csv"]))
    df_sil = pd.read_csv(os.path.join(sil_dir, t["sil_csv"]))
    df_real = pd.read_csv(os.path.join(real_dir, t["real_csv"]))
    
    t_py = df_py['tempo_ms'] / 1000.0
    t_sil = df_sil['tempo_ms'] / 1000.0
    t_real = df_real['tempo_ms'] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    # Reference
    ax1.plot(t_py, df_py['referencia'], color='black', linestyle='--', linewidth=1.1, 
             alpha=0.85, label='Referência')
    # Python
    ax1.plot(t_py, df_py['angulo_deg'], color='#1f77b4', linestyle='-', linewidth=1.4, 
             alpha=0.9, label='Python (Simulador)')
    # SIL
    ax1.plot(t_sil, df_sil['angulo_deg'], color='#2ca02c', linestyle='-.', linewidth=1.3, 
             alpha=0.85, label='STM32 (SIL)')
    # Bancada Real
    ax1.plot(t_real, df_real['angulo_deg'], color='#d62728', linestyle=':', linewidth=1.3, 
             alpha=0.9, label='Bancada Real')
    
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True)
    ax1.legend(loc='lower right' if t['id'] != 1 else 'upper right', ncol=4, framealpha=0.95)
    
    # Control actions
    ax2.plot(t_py, df_py['u_pct'], color='#1f77b4', linestyle='-', linewidth=1.2, 
             alpha=0.85, label='Python')
    ax2.plot(t_sil, df_sil['u_pct'], color='#2ca02c', linestyle='-.', linewidth=1.2, 
             alpha=0.85, label='SIL')
    ax2.plot(t_real, df_real['u_pct'], color='#d62728', linestyle=':', linewidth=1.1, 
             alpha=0.85, label='Bancada Real')
    
    # Limits and reference lines
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
    
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [\%]')
    ax2.grid(True)
    ax2.legend(loc='upper right', ncol=3, framealpha=0.95)
    
    max_t = max(t_py.max(), t_sil.max(), t_real.max())
    ax1.set_xlim(0, max_t)
    
    plt.tight_layout()
    save_fig(fig, t["base_name"])
    plt.close(fig)

print("Comparacao plots finished!")
